"""Core, offline-testable logic for the Big Broiler feather-pecking event task.

Source: decide-ugent/big-broiler-dataset (AGPL-3.0) annotation CSVs parsed by
`index_behavior_broiler_source_audit.py`. Pure logic only: label parsing,
window/event/coverage flags, episode merging, deterministic whole-video splits,
and causal geometry features derived from released tracking boxes. No network
access, no media, no third-party rows.

Pre-registered design contract (fixed before any model score):
  * Primary event question: state at frame t (one state per second); label =
    source-released action `Feather_Pecking` occurs on >=1 row within
    [t-W+1, t] (W = 250 frames = 10 s). Past/current frames only; no future.
  * Observation-quality status: a window with any `Video_Problem` row, or with
    < 80% of its frames carrying >=1 tracked bird, is UNSCOREABLE for the
    primary question and is separately reported.
  * Missingness: unannotated frames/windows are missing evidence, never
    negative surfaces; a negative is asserted only with complete lookback and
    scorable coverage.
  * Features only from released tracking-box geometry/kinematics; never from
    source action/posture/annotator labels, video/setup/age identity or time.
  * Splits: whole-video disjoint via fixed ordinal slots within each setup;
    documented availability repair uses only per-video FP-window counts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

FP_LABEL = "Feather_Pecking"
VP_LABEL = "Video_Problem"
OOS_LABEL = "Out_Of_Sight"
HARD_NEG_FAMILY = ("Neg_Interact", "Aggression", "Disturbance")  # other damaging interactions
SELF_CONFUSER_FAMILY = ("Preening",)  # visually close self-maintenance behaviour
WINDOW_FRAMES = 250        # 10 s past/current lookback inclusive of the state frame
STRIDE_FRAMES = 25         # one state per second
MIN_COVER = 0.8            # window scorable only if >=80% frames have >=1 tracked bird
MAX_TRACKS_PER_FRAME = 128  # observed source maximum 65 visible boxes/frame
EPISODE_GAP_STATES = 3     # detected positives <=3 states apart merge into one episode
SPLIT_TEST_SLOTS = (3, 15, 27, 39)
SPLIT_CAL_SLOTS = (8, 20, 32)
TEST_MIN_FP_VIDEOS = 2     # whole-collection availability repair trigger
CAL_MIN_FP_VIDEOS = 1
SPEED_MOVE_THR = 0.02
NEUTRAL_NND = 1.0          # imputed neighbour distance when no distance is measurable

FEATURE_NAMES: tuple[str, ...] = (
    "cover_frac", "rows_mean", "rows_max", "tracks_mean", "tracks_max",
    "area_mean", "area_max", "area_std_mean",
    "ycen_mean", "ycen_top", "ycen_bottom", "yspan_mean", "xspan_mean",
    "nnd_mean", "nnd_min", "close_pair_frac_mean", "dense_track_frac_mean",
    "speed_mean", "speed_max", "speed_std_mean", "moving_frac_mean",
)
# Semantics of NaN imputation (documented, deterministic):
#   * speed features and moving_frac: NaN => no matched motion evidence -> 0.0
#   * neighbour-distance features: NaN => cannot measure any pair -> NEUTRAL_NND
#     ("birds far apart"), close_pair_frac/dense_track_frac -> 0.0
NAN_IMPUTATION = {
    "nnd_mean": NEUTRAL_NND, "nnd_min": NEUTRAL_NND,
    "close_pair_frac_mean": 0.0, "dense_track_frac_mean": 0.0,
    "speed_mean": 0.0, "speed_max": 0.0, "speed_std_mean": 0.0, "moving_frac_mean": 0.0,
}
DEFAULT_IMPUTATION = 0.0


# --------------------------------------------------------------------------- #
# Label parsing
# --------------------------------------------------------------------------- #
def parse_action_labels(value: object) -> tuple[str, ...]:
    """Parse source multi-label strings (`;`, `,`, or `|` separated).

    NaN/None/empty means NO RECORDED ACTION on that row - never a negative.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ()
    parts = re.split(r"[;,|]", str(value))
    return tuple(p.strip() for p in parts if p.strip())


def per_frame_label_flags(actions: pd.Series, row_frames: np.ndarray, labels: tuple[str, ...],
                          n_frames: int) -> dict[str, np.ndarray]:
    """flags[label][frame] = True iff >=1 source row at that frame carries `label`.

    Frames with no rows (tracking gaps) are False (no evidence), NOT negatives;
    the mask logic handles them via coverage rather than these flags.
    """
    s = actions.astype("string").fillna("")
    frames = np.asarray(row_frames, dtype=np.int64)
    if len(frames) != len(s):
        raise ValueError("row frame indices must align with the actions series")
    if len(frames) and (frames.min() < 0 or frames.max() >= n_frames):
        raise ValueError("row frame indices outside the video frame range")
    flags = {lab: np.zeros(n_frames, dtype=bool) for lab in labels}
    tokens = set(flags)
    for i, value in enumerate(s.to_numpy()):
        toks = parse_action_labels(value)
        if not toks:
            continue
        f = frames[i]
        for tok in toks:
            if tok in tokens:
                flags[tok][f] = True
    return flags



# --------------------------------------------------------------------------- #
# Windows, event flags, coverage status
# --------------------------------------------------------------------------- #
def state_indices(n_frames: int, window: int = WINDOW_FRAMES,
                  stride: int = STRIDE_FRAMES) -> np.ndarray:
    """State sample frames t such that the whole past window [t-W+1, t] exists."""
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    if n_frames < window:
        return np.zeros(0, dtype=np.int64)
    return np.arange(window - 1, n_frames, stride, dtype=np.int64)


def _window_start(state_idx: np.ndarray, window: int) -> np.ndarray:
    return state_idx - (window - 1)


def cumsum_flags(flags: np.ndarray) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(np.asarray(flags).astype(np.int64))))


def window_any(state_idx: np.ndarray, window: int, per_frame_flags: np.ndarray) -> np.ndarray:
    """True per state iff any frame in [t-window+1, t] carries the flag."""
    cum = cumsum_flags(per_frame_flags)
    i0 = _window_start(state_idx, window)
    hi = cum[state_idx + 1]
    lo = np.where(i0 > 0, cum[i0], 0)
    return (hi - lo) > 0


def window_count(state_idx: np.ndarray, window: int, per_frame_flags: np.ndarray) -> np.ndarray:
    """Number of window frames carrying the flag."""
    cum = cumsum_flags(per_frame_flags)
    i0 = _window_start(state_idx, window)
    hi = cum[state_idx + 1]
    lo = np.where(i0 > 0, cum[i0], 0)
    return hi - lo


def window_coverage(state_idx: np.ndarray, rows_per_frame: np.ndarray, window: int) -> np.ndarray:
    """Fraction of the window's frames that carry >=1 tracked bird row."""
    present = (np.asarray(rows_per_frame) > 0).astype(np.int64)
    return window_count(state_idx, window, present) / float(window)


# --------------------------------------------------------------------------- #
# Episode merging and match accounting
# --------------------------------------------------------------------------- #
def merge_state_runs(positive: np.ndarray, gap_states: int = EPISODE_GAP_STATES) -> list[tuple[int, int]]:
    """Merge flagged states into (start, end) inclusive runs, joining positives
    separated by <= gap_states misses."""
    idx = np.flatnonzero(np.asarray(positive)).tolist()
    if not idx:
        return []
    runs: list[tuple[int, int]] = []
    a = b = int(idx[0])
    for n in idx[1:]:
        n = int(n)
        if n - b <= gap_states + 1:
            b = n
        else:
            runs.append((a, b))
            a = b = n
    runs.append((a, b))
    return runs


def episode_metrics(detected: list[tuple[int, int]], reference: list[tuple[int, int]]) -> dict:
    """Greedy overlap matching (>=1 state overlap; a detected run matches at most
    one reference run; each reference is matched at most once)."""
    used: set[int] = set()
    tp = 0
    for da, db in detected:
        hit = -1
        for ri, (ra, rb) in enumerate(reference):
            if ri in used:
                continue
            if ra <= db and rb >= da:
                hit = ri
                used.add(hit)
                break
        if hit >= 0:
            tp += 1
    fp = len(detected) - tp
    fn = len(reference) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec,
            "f1": f1, "n_detected": len(detected), "n_reference": len(reference)}


# --------------------------------------------------------------------------- #
# Deterministic whole-video split (pre-registered ordinal slots + repair)
# --------------------------------------------------------------------------- #
@dataclass
class SplitRule:
    test_slots: tuple[int, ...] = SPLIT_TEST_SLOTS
    cal_slots: tuple[int, ...] = SPLIT_CAL_SLOTS
    test_min_fp_videos: int = TEST_MIN_FP_VIDEOS
    cal_min_fp_videos: int = CAL_MIN_FP_VIDEOS


def assign_split(stems: list[str], setup_of: dict[str, int], fp_windows: dict[str, int],
                 rule: SplitRule | None = None) -> tuple[dict[str, str], list[str], dict[str, int]]:
    """Whole-video-exclusive train/cal/test assignment, fully deterministic.

    Ordinal slots are fixed constants (never tuned). Availability repair moves
    whole videos from train into a deficient side (most FP windows first, then
    lexical), using only per-video FP-window counts that are known source
    metadata before any model result. Returns (assignment, repairs, fp_counts).
    """
    rule = rule or SplitRule()
    assignment: dict[str, str] = {}
    repairs: list[str] = []
    totals = {"test_fp_videos": 0, "cal_fp_videos": 0}

    for setup in sorted(set(setup_of.values())):
        ordered = sorted(s for s, n in setup_of.items() if n == setup)
        test = [ordered[r] for r in rule.test_slots if r < len(ordered)]
        cal = [ordered[r] for r in rule.cal_slots if r < len(ordered)]
        taken = set(test) | set(cal)
        train = [s for s in ordered if s not in taken]
        if sum(1 for s in test if fp_windows.get(s, 0) > 0) < rule.test_min_fp_videos:
            cand = [s for s in train if fp_windows.get(s, 0) > 0]
            if cand:  # documented deterministic repair
                pick = sorted(cand, key=lambda s: (-fp_windows.get(s, 0), s))[0]
                test.append(pick)
                cal = [s for s in cal if s != pick]
                train = [s for s in train if s != pick]
                repairs.append(f"setup{setup}:test")
        if sum(1 for s in cal if fp_windows.get(s, 0) > 0) < rule.cal_min_fp_videos:
            cand = [s for s in train if fp_windows.get(s, 0) > 0]
            if cand:
                pick = sorted(cand, key=lambda s: (-fp_windows.get(s, 0), s))[0]
                cal.append(pick)
                test = [s for s in test if s != pick]
                train = [s for s in train if s not in set(test) | set(cal)]
                repairs.append(f"setup{setup}:cal")
        t, c, tr = set(test), set(cal), set(train)
        assert not (t & c) and not (t & tr) and not (c & tr), (test, cal, train)
        for s in test:
            assignment[s] = "test"
        for s in cal:
            assignment[s] = "cal"
        for s in train:
            assignment[s] = "train"
        totals["test_fp_videos"] += sum(1 for s in test if fp_windows.get(s, 0) > 0)
        totals["cal_fp_videos"] += sum(1 for s in cal if fp_windows.get(s, 0) > 0)
    return assignment, repairs, totals


# --------------------------------------------------------------------------- #
# Causal geometry features from released tracking boxes
# --------------------------------------------------------------------------- #
def clean_geometry_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """One deterministic box per (frame, track): sort stable, keep first."""
    g = df[["frame", "track", "x1", "y1", "x2", "y2"]].copy()
    g["cx"] = (g["x1"] + g["x2"]) * 0.5
    g["cy"] = (g["y1"] + g["y2"]) * 0.5
    g["area"] = (g["x2"] - g["x1"]) * (g["y2"] - g["y1"])
    g["xspan"] = g["x2"] - g["x1"]
    g["yspan"] = g["y2"] - g["y1"]
    dup = int(g.duplicated(["frame", "track"]).sum())
    # deterministic box selection for (frame, track) duplicates: keep the last
    # row after stable ascending (frame, track, area) sort -> the largest-area box
    g = g.sort_values(["frame", "track", "area"], kind="stable")
    g = g.drop_duplicates(["frame", "track"], keep="last")
    return g, dup


def build_frame_geometry(df: pd.DataFrame, n_frames: int) -> dict:
    """Vectorised per-frame statistics over *released tracking boxes*.

    Label columns are never touched here.
    """
    g, dup = clean_geometry_rows(df)
    if g["frame"].min() < 0 or g["frame"].max() >= n_frames:
        raise ValueError("frame indices outside allowed range")
    if g["track"].nunique() and \
            g.groupby("frame")["track"].size().max() > MAX_TRACKS_PER_FRAME:
        raise ValueError("frame carries more tracked boxes than the pre-registered cap")
    g = g.dropna(subset=["cx", "cy"])

    per_counts = g.groupby("frame").size()
    rows_per_frame = np.zeros(n_frames, dtype=np.int32)
    rows_per_frame[per_counts.index.to_numpy()] = per_counts.to_numpy().astype(np.int32)

    tc = g.groupby("frame")["track"].nunique()
    tracks = np.zeros(n_frames, dtype=np.int32)
    tracks[tc.index.to_numpy()] = tc.to_numpy().astype(np.int32)

    agg_idx = None
    frames_arr = np.full(n_frames, np.nan, dtype=float)
    out: dict[str, np.ndarray] = {"rows_per_frame": rows_per_frame,
                                  "tracks_per_frame": tracks,
                                  "dropped_duplicate_boxes": dup}
    if len(g):
        agg = g.groupby("frame").agg(
            area_mean=("area", "mean"), area_max=("area", "max"), area_std=("area", "std"),
            cy_mean=("cy", "mean"), cy_min=("cy", "min"), cy_max=("cy", "max"),
            yspan_mean=("yspan", "mean"), xspan_mean=("xspan", "mean"),
        )
        agg_idx = agg.index.to_numpy(np.int64)
        for col in ("area_mean", "area_max", "area_std", "cy_mean", "cy_min", "cy_max",
                    "yspan_mean", "xspan_mean"):
            arr = np.full(n_frames, np.nan)
            arr[agg_idx] = agg[col].to_numpy(float)
            if col == "area_std":  # single-bird frame: "no bird variability" -> 0
                arr = np.where(np.isnan(arr), 0.0, arr)
            out[col] = arr
    else:
        for col in ("area_mean", "area_max", "area_std", "cy_mean", "cy_min", "cy_max",
                    "yspan_mean", "xspan_mean"):
            out[col] = frames_arr.copy()

    nnd_min = np.full(n_frames, np.nan)
    nnd_mean = np.full(n_frames, np.nan)
    close_frac = np.full(n_frames, np.nan)
    if len(g):
        for f, block in g.groupby("frame", sort=True):
            if len(block) < 2:
                continue
            coords = block[["cx", "cy"]].to_numpy(float)
            dx = coords[:, None, 0] - coords[None, :, 0]
            dy = coords[:, None, 1] - coords[None, :, 1]
            dist = np.sqrt(dx ** 2 + dy ** 2)
            iu = np.triu_indices(len(coords), 1)
            d = dist[iu]
            nnd_min[f] = float(d.min())
            eye = np.eye(len(coords), dtype=bool)
            nnd_mean[f] = float((dist + eye * 9.0).min(axis=1).mean())
            close_frac[f] = float((d < 0.05).sum() / len(d))
    out["nnd_min"] = nnd_min
    out["nnd_mean"] = nnd_mean
    out["close_pair_frac"] = close_frac
    # "bird density" visible-track saturation proxy (label-free)
    out["dense_track_frac"] = tracks.astype(float) / float(MAX_TRACKS_PER_FRAME)
    return out


def build_track_pivot(df: pd.DataFrame) -> pd.DataFrame:
    g, _ = clean_geometry_rows(df)
    return g


def compute_frame_speed(g: pd.DataFrame, n_frames: int) -> dict:
    """Causal 1-frame-lag matched-ID speeds (median+std across matched birds;
    fraction of birds above SPEED_MOVE_THR). Matching is by exact
    (frame+1, track) lookup, robust to interleaved multi-bird frames."""
    if len(g) == 0:
        empty = np.zeros(n_frames)
        return {"speed_median": empty, "speed_std": empty.copy(),
                "moving_frac": empty.copy(), "n_speed_samples": 0}
    fr = g["frame"].to_numpy(np.int64)
    trk = g["track"].to_numpy(np.int64)
    cx = g["cx"].to_numpy(float)
    cy = g["cy"].to_numpy(float)
    TRACK_MOD = 1_000_003  # safe: observed distinct tracks per video << this
    key = fr * TRACK_MOD + trk
    order = np.argsort(key, kind="stable")
    fr_s, trk_s, cx_s, cy_s = fr[order], trk[order], cx[order], cy[order]
    key_s = fr_s * TRACK_MOD + trk_s
    next_k = key_s + TRACK_MOD
    pos = np.searchsorted(key_s, next_k)
    pos_c = np.clip(pos, 0, len(key_s) - 1)
    hits = (pos < len(key_s)) & (key_s[pos_c] == next_k)
    src = pos_c[hits]
    speeds = np.hypot(cx_s[src] - cx_s[hits], cy_s[src] - cy_s[hits])
    frames = fr_s[hits]
    sp = np.full(n_frames, np.nan)
    sd = np.full(n_frames, np.nan)
    mf = np.full(n_frames, np.nan)
    if len(speeds):
        stats = pd.DataFrame({"frame": frames, "v": speeds}).groupby("frame")["v"].agg(
            median="median", std=lambda s: float(s.std(ddof=0)) if len(s) > 1 else 0.0,
            moving=lambda s: float((s > SPEED_MOVE_THR).mean()))
        fidx = np.asarray(stats.index, dtype=np.int64)
        sp[fidx] = stats["median"].to_numpy()
        sd[fidx] = stats["std"].to_numpy()
        mf[fidx] = stats["moving"].to_numpy()
    return {"speed_median": sp, "speed_std": sd, "moving_frac": mf,
            "n_speed_samples": int(len(speeds))}


def _sliding_mean(values: np.ndarray, state_idx: np.ndarray, window: int) -> np.ndarray:
    valid = np.isfinite(values).astype(np.float64)
    vals = np.nan_to_num(values, nan=0.0)
    bv = np.concatenate(([0.0], np.cumsum(vals)))
    nb = np.concatenate(([0], np.cumsum(valid.astype(np.int64))))
    i0 = _window_start(state_idx, window)
    svv = bv[state_idx + 1] - np.where(i0 > 0, bv[i0], 0.0)
    nvv = nb[state_idx + 1] - np.where(i0 > 0, nb[i0], 0)
    return np.where(nvv > 0, svv / np.maximum(nvv, 1), np.nan)


def _sliding_extreme(values: np.ndarray, state_idx: np.ndarray, window: int,
                     mode: str) -> np.ndarray:
    if len(values) < window:
        return np.full(len(state_idx), np.nan)
    fill = -np.inf if mode == "max" else np.inf
    arr = np.nan_to_num(np.asarray(values, float), nan=fill)
    windows = np.lib.stride_tricks.sliding_window_view(arr, window)
    stat = windows.max(axis=1) if mode == "max" else windows.min(axis=1)
    # view j covers frames [j, j+window-1]; state t covers [t-window+1, t]
    return stat[state_idx - (window - 1)]


def nan_impute(X: np.ndarray) -> tuple[np.ndarray, dict]:
    """Apply the documented per-column NaN policy; report imputed cell counts."""
    counts = {}
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        bad = ~np.isfinite(col)
        if bad.any():
            counts[name] = int(bad.sum())
            col[bad] = NAN_IMPUTATION.get(name, DEFAULT_IMPUTATION)
    return X, counts


def compute_features(df: pd.DataFrame, n_frames: int) -> tuple[np.ndarray, np.ndarray, dict]:
    """Features for every valid state of one video.

    Returns (states x FEATURE_NAMES float32, state frame indices, context dict).
    Deterministic; label columns are never read here.
    """
    state_idx = state_indices(n_frames)
    geom = build_frame_geometry(df, n_frames)
    speeds = compute_frame_speed(build_track_pivot(df), n_frames)
    rows = geom["rows_per_frame"]
    idx = state_idx
    cols = {
        "cover_frac": window_coverage(idx, rows, WINDOW_FRAMES),
        "rows_mean": _sliding_mean(rows.astype(float), idx, WINDOW_FRAMES),
        "rows_max": _sliding_extreme(rows.astype(float), idx, WINDOW_FRAMES, "max"),
        "tracks_mean": _sliding_mean(geom["tracks_per_frame"].astype(float), idx, WINDOW_FRAMES),
        "tracks_max": _sliding_extreme(geom["tracks_per_frame"].astype(float), idx, WINDOW_FRAMES, "max"),
        "area_mean": _sliding_mean(geom["area_mean"], idx, WINDOW_FRAMES),
        "area_max": _sliding_extreme(geom["area_max"], idx, WINDOW_FRAMES, "max"),
        "area_std_mean": _sliding_mean(geom["area_std"], idx, WINDOW_FRAMES),
        "ycen_mean": _sliding_mean(geom["cy_mean"], idx, WINDOW_FRAMES),
        "ycen_top": _sliding_extreme(geom["cy_max"], idx, WINDOW_FRAMES, "max"),
        "ycen_bottom": _sliding_extreme(geom["cy_min"], idx, WINDOW_FRAMES, "min"),
        "yspan_mean": _sliding_mean(geom["yspan_mean"], idx, WINDOW_FRAMES),
        "xspan_mean": _sliding_mean(geom["xspan_mean"], idx, WINDOW_FRAMES),
        "nnd_mean": _sliding_mean(geom["nnd_mean"], idx, WINDOW_FRAMES),
        "nnd_min": _sliding_extreme(geom["nnd_min"], idx, WINDOW_FRAMES, "min"),
        "close_pair_frac_mean": _sliding_mean(geom["close_pair_frac"], idx, WINDOW_FRAMES),
        "dense_track_frac_mean": _sliding_mean(geom["dense_track_frac"], idx, WINDOW_FRAMES),
        "speed_mean": _sliding_mean(speeds["speed_median"], idx, WINDOW_FRAMES),
        "speed_max": _sliding_extreme(speeds["speed_median"], idx, WINDOW_FRAMES, "max"),
        "speed_std_mean": _sliding_mean(speeds["speed_std"], idx, WINDOW_FRAMES),
        "moving_frac_mean": _sliding_mean(speeds["moving_frac"], idx, WINDOW_FRAMES),
    }
    X = np.stack([cols[name] for name in FEATURE_NAMES], axis=1).astype(np.float32)
    X, imputed = nan_impute(X)
    context = {**geom, **speeds, "state_count": len(idx), "imputed_cells": imputed}
    return X, idx, context


# --------------------------------------------------------------------------- #
# Cheap deterministic baselines (CPU)
# --------------------------------------------------------------------------- #
def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int = 7):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
    model.fit(X, y)
    return model


def fit_trees(X: np.ndarray, y: np.ndarray, seed: int = 7):
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(n_estimators=55, random_state=seed, n_jobs=-1)
    model.fit(X, y)
    return model
