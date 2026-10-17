#!/usr/bin/env python3
"""Annotator agreement on the ten triple-annotated Big Broiler videos (092-101).

The released `actions_ground` channel was shown to match annotator 1 exactly on
setup-3 videos, so it is *not* reconciled consensus. This script measures,
exclusively within the handoff's L1 window notion (see
index_behavior_broiler_event_core.WINDOW_FRAMES/STRIDE_FRAMES):

* window-level pairwise feather-pecking agreement between the three released
  simultaneous annotators (F1 over big-broiler window states, stride 1 s),
* windows where >=2 of 3 annotators mark the feather-pecking window:
  the de-facto "consensus" surface we would hand a downstream model,
* ground (actions_ground) agreement with each annotator and with >=2-vote
  consensus, proving the released label column's lineage quantitatively,
* contested windows: exactly 1 of 3 annotators positive (irreducible).

All numbers come from the locally cached parquet files under data/broiler/cache
(no re-download). Sets of windows are aligned across annotators because windows
are defined on the shared frame index (identical fps and length per video).

Output: reports/index_behavior_broiler_annotator_agreement.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import index_behavior_broiler_event_core as core

CACHE = Path("data/broiler/cache")
REPORT = Path("reports/index_behavior_broiler_annotator_agreement.json")
IAA_PREFIXES = [f"{i:03d}" for i in range(92, 102)]
ANNOTATOR_KEYS = ("a1", "a2", "a3")
ANNOTATOR_COLUMNS = ("actions_annotator_1", "actions_annotator_2", "actions_annotator_3")
FP = core.FP_LABEL


def iaa_stems() -> list[str]:
    meta = pd.read_csv(CACHE / "manifest.csv")
    pick = meta.loc[meta["stem"].str[:3].isin(IAA_PREFIXES), "stem"]
    found = sorted(pick.tolist())
    if len(found) != 10:
        raise SystemExit(f"expected 10 IAA videos 092-101, manifest has {len(found)}")
    return found


def window_f1(a: np.ndarray, b: np.ndarray) -> float:
    tp = int((a & b).sum())
    fp_n = int((a & ~b).sum())
    fn_n = int((~a & b).sum())
    prec = tp / max(1, tp + fp_n)
    rec = tp / max(1, tp + fn_n)
    return 2 * prec * rec / max(1e-12, prec + rec)


def video_window_states(df: pd.DataFrame, n_frames: int) -> dict:
    states = core.state_indices(n_frames)
    out = {}
    columns = dict(ground=df["actions_ground"], a1=df[ANNOTATOR_COLUMNS[0]],
                   a2=df[ANNOTATOR_COLUMNS[1]], a3=df[ANNOTATOR_COLUMNS[2]])
    for key, series in columns.items():
        flags = core.per_frame_label_flags(
            series, df["frame"].to_numpy(np.int64), (FP,), n_frames)[FP]
        out[key] = core.window_any(states, core.WINDOW_FRAMES, flags)
    return out


def main() -> int:
    t0 = time.time()
    rows = []
    for stem in iaa_stems():
        pf = CACHE / f"{stem}.parquet"
        if not pf.exists():
            raise SystemExit(f"missing cache for {stem}")
        df = pd.read_parquet(pf)
        mrow = pd.read_csv(CACHE / "manifest.csv").loc[lambda m: m.stem == stem].iloc[0]
        frame_min = int(df["frame"].min())
        if frame_min < 0:  # only video 004 in the broader corpus needs this; keep here for symmetry
            df = df.copy()
            df["frame"] = df["frame"] - frame_min
        n_frames = max(int(mrow.video_nb_frames), int(df["frame"].max()) + 1)
        w = video_window_states(df, n_frames)
        rec = {"stem": stem, "n_states": int(len(w["ground"]))}
        for key in ("ground", "a1", "a2", "a3"):
            rec[f"pos_{key}"] = int(w[key].sum())
        rec["pair_f1"] = {
            f"{a}_{b}": round(window_f1(w[a], w[b]), 4)
            for a, b in (("a1", "a2"), ("a1", "a3"), ("a2", "a3"))
        }
        rec["ground_eq"] = {a: bool((w["ground"] == w[a]).all()) for a in ANNOTATOR_KEYS}
        rec["ground_f1_vs"] = {
            a: round(window_f1(w["ground"], w[a]), 4) for a in ANNOTATOR_KEYS}
        votes = w["a1"].astype(int) + w["a2"].astype(int) + w["a3"].astype(int)
        rec["consensus2_pos"] = int((votes >= 2).sum())
        rec["contested_pos"] = int((votes == 1).sum())
        rec["consensus2_f1_vs_ground"] = None if (votes >= 2).sum() == 0 else round(
            window_f1((votes >= 2), w["ground"]), 4)
        rec["annotator1_f1_vs_consensus2"] = None if (votes >= 2).sum() == 0 else round(
            window_f1(w["a1"], (votes >= 2)), 4)
        rows.append(rec)
        print(stem, json.dumps({k: v for k, v in rec.items() if k not in ("stem",)}, default=str), flush=True)

    pairs = {"a1_a2": [], "a1_a3": [], "a2_a3": []}
    contested, consensus2, both_a_chars = [], [], []
    for r in rows:
        for k, v in r["pair_f1"].items():
            pairs[k].append(v)
        contested.append(r["contested_pos"])
        consensus2.append(r["consensus2_pos"])
    summary = {
        "n_videos": len(rows),
        "mean_pair_f1": {k: round(float(np.mean(v)), 4) for k, v in pairs.items()},
        "mean_pair_f1_over_nonempty_pairs": {
            k: (None if not [a for a, b in zip(pairs[k], rows) if b["pos_" + k.split("_")[0]] + b["pos_" + k.split("_")[1]] > 0]
                else round(float(np.mean([a for a, b in zip(pairs[k], rows)
                                          if b["pos_" + k.split("_")[0]] + b["pos_" + k.split("_")[1]] > 0])), 4))
            for k in pairs},
        "total_consensus2_windows": sum(consensus2),
        "total_contested_windows": sum(contested),
    }
    report = {
        "schema": "openfarm.research.index_behavior_annotator_agreement.v1",
        "source": core_const_meta(),
        "window_frames": core.WINDOW_FRAMES,
        "stride_frames": core.STRIDE_FRAMES,
        "per_video": rows,
        "summary": summary,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("ANNOTATOR_AGREEMENT:" + json.dumps({
        "report": str(REPORT), "elapsed_sec": report["elapsed_sec"],
        "summary": summary}, default=str))
    return 0


def core_const_meta() -> dict:
    return {
        "labels_used": [FP],
        "ground_column": "actions_ground",
        "annotator_columns": ["acts_person_1", "acts_person_2", "acts_person_3"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sys.exit(main())
