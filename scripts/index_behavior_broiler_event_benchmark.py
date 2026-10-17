"""Big Broiler feather-pecking event benchmark (Operator C, index-behavior).

Pre-registered question (documented in the dossier):
    Noul: does the source-released annotation assert Feather_Pecking on >=1
    bird row within the past/current 10 s window ending at this state frame?
plus a separately measured observation-quality status:
    Noul: is this window unscorable (any Video_Problem row in window, or <80%
    of window frames carrying >=1 tracked bird)?

Features: only released tracking-box geometry/kinematics inside the window.
Splits: whole-video disjoint via fixed ordinal slots per setup with the
documented availability repair. Models: cheap CPU baselines only (prior,
always-negative, L2 logistic, small ExtraTrees). No frontier API, no media
download, no third-party row redistribution.

The ten triple-annotated videos (092-101) are excluded from train/cal/test
and measured separately by index_behavior_broiler_annotator_agreement.py.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import index_behavior_broiler_event_core as core

MANIFEST_SHA256 = "c90e25a13bbb373ab9dd4a0411ae56fe76dde180d4c52b6eb2d943ccae222554"
SOURCE_URL = "https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
UPSTREAM_REPO = "https://github.com/decide-ugent/big-broiler-dataset"
IAA_VIDEOS = {f"{i:03d}" for i in range(92, 102)}
KNOWN_LABELS = {
    "Inactive", "Eat", "Explor_Forag", "Out_Of_Sight", "Preening", "Sleeping",
    "Drink", "Locomotion", "Panting", "Resting", "Video_Problem", "Other",
    "Stretching", "Short_Leg_Move", "Feather_Pecking", "Dust_Bathing",
    "Scratching", "Defecating", "Disturbance", "Flapping", "Swallowing",
    "Sneezing", "Play", "Shaking", "Head_Shaking", "Yawn", "Neg_Interact",
    "Body_Shaking", "Aggression", "Fluffy_Feathers", "Running",
}
FAMILY_LABELS = (*core.HARD_NEG_FAMILY, *core.SELF_CONFUSER_FAMILY)


class ConstantModel:
    """Deterministic stateless baseline returning the fixed prior."""
    def __init__(self, p: float):
        self.p = float(p)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.stack([np.full(len(X), 1.0 - self.p), np.full(len(X), self.p)], axis=1)


def load_video_record(pf: Path, rows_json: dict, meta_row: pd.Series) -> dict:
    stem = pf.stem
    df = pd.read_parquet(pf)
    assert len(df) == int(rows_json["rows"]), f"{stem}: cached row count mismatch"
    tokens: set[str] = set()
    for v in df["actions_ground"].astype("string").dropna():
        tokens.update(core.parse_action_labels(v))
    unknown = tokens - KNOWN_LABELS
    if unknown:
        raise SystemExit(f"{stem}: unexpected source action labels {sorted(unknown)}")
    # normalise the frame axis: video 004 carries a 14-frame negative offset
    frame_offset = int(df["frame"].min())
    if frame_offset < 0:
        df = df.copy()
        df["frame"] = df["frame"] - frame_offset
    n_frames = max(int(meta_row["video_nb_frames"]), int(df["frame"].max()) + 1)
    states = core.state_indices(n_frames)
    feat, _, ctx = core.compute_features(df, n_frames)
    label_rows = df["actions_ground"]
    all_flags = core.per_frame_label_flags(
        label_rows, df["frame"].to_numpy(np.int64),
        (core.FP_LABEL, core.VP_LABEL, core.OOS_LABEL, *FAMILY_LABELS), n_frames)
    fp_flags = all_flags[core.FP_LABEL]
    vp_flags = all_flags[core.VP_LABEL]
    oos_flags = all_flags[core.OOS_LABEL]
    fp_window = core.window_any(states, core.WINDOW_FRAMES, fp_flags)
    vp_window = core.window_any(states, core.WINDOW_FRAMES, vp_flags)
    oos_window = core.window_any(states, core.WINDOW_FRAMES, oos_flags)
    cover = core.window_coverage(states, ctx["rows_per_frame"], core.WINDOW_FRAMES)
    masked = vp_window | (cover < core.MIN_COVER)
    family_windows = {}
    for fam in FAMILY_LABELS:
        family_windows[fam] = core.window_any(states, core.WINDOW_FRAMES, all_flags[fam])
    duplicated = ctx.get("dropped_duplicate_boxes", 0)
    return {
        "stem": stem,
        "n_frames": n_frames,
        "states": states,
        "features": feat,
        "fp_window": fp_window,
        "masked": masked,
        "oos_window": oos_window,
        "cover": cover,
        "family_windows": family_windows,
        "dropped_duplicate_boxes": int(duplicated),
        "imputed_cells": dict(ctx.get("imputed_cells", {})),
        "frame_axis_offset": -frame_offset if frame_offset < 0 else 0,
        "vp_window": vp_window,
    }


def assemble(per_video: dict, stems: list[str]) -> dict:
    full = {s: per_video[s] for s in stems}
    lengths = [len(full[s]["states"]) for s in stems]
    X = np.concatenate([full[s]["features"] for s in full])
    y = np.concatenate([full[s]["fp_window"].astype(bool) for s in full])
    masked = np.concatenate([full[s]["masked"].astype(bool) for s in full])
    groups = np.repeat([s for s in full], lengths)
    return {"X": X, "y": y, "masked": masked, "groups": groups, "stems": list(full),
            "lengths": lengths}


def split_block(data: dict) -> dict:
    ok = ~data["masked"]
    y = data["y"][ok]
    return {
        "videos": len(data["stems"]),
        "states": len(data["y"]),
        "scorable_states": int(ok.sum()),
        "masked_states": int(data["masked"].sum()),
        "scorable_positives": int(y.sum()),
        "scorable_prevalence": float(y.mean()) if len(y) else 0.0,
    }


def evaluate(data: dict, per_video: dict, model, threshold: float) -> dict:
    ok = ~data["masked"]
    X, y, groups = data["X"][ok], data["y"][ok], data["groups"][ok]
    prob = model.predict_proba(X)[:, 1]
    pred = prob >= threshold
    tp = int((pred & y).sum()); fp_states = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / max(1, tp + fp_states)
    recall = tp / max(1, tp + fn)
    block = {
        "threshold": threshold,
        "states_scored": int(ok.sum()),
        "tp": tp, "fp_states": fp_states, "fn": fn,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(1e-12, precision + recall),
    }
    if y.sum() and (~y).sum():
        from sklearn.metrics import average_precision_score, roc_auc_score
        block["auroc"] = float(roc_auc_score(y, prob))
        block["auprc"] = float(average_precision_score(y, prob))

    # per-video episode accounting (matching stays within each video)
    ep_detected = 0
    ep_tp = 0
    ep_reference = 0
    hours = 0.0
    det_false_by_video = {}
    for stem in data["stems"]:
        m = groups == stem
        p_bool = pred[m]
        y_bool = y[m]
        reference_runs = core.merge_state_runs(y_bool)
        detected_runs = core.merge_state_runs(p_bool)
        metr = core.episode_metrics(detected_runs, reference_runs)
        ep_tp += metr["tp"]
        ep_detected += metr["n_detected"]
        ep_reference += metr["n_reference"]
        det_false_by_video[stem] = metr["fp"]
        hours += per_video[stem]["n_frames"] / 25.0 / 3600.0
    ep_fp = ep_detected - ep_tp
    ep_fn = ep_reference - ep_tp
    ep_prec = ep_tp / max(1, ep_tp + ep_fp)
    ep_rec = ep_tp / max(1, ep_tp + ep_fn)
    block["episode_metrics"] = {
        "tp": ep_tp, "fp": ep_fp, "fn": ep_fn,
        "precision": ep_prec, "recall": ep_rec,
        "f1": 2 * ep_prec * ep_rec / max(1e-12, ep_prec + ep_rec),
        "false_episodes_per_stream_hour": round(ep_fp / hours, 3) if hours else 0.0,
        "stream_hours": round(hours, 3),
    }
    block["false_episode_rate"] = block["episode_metrics"]["false_episodes_per_stream_hour"]

    # confuser-subset false detections at the same operating point
    pred_full = np.zeros(len(data["y"]), dtype=bool)
    pred_full[ok] = pred
    confuser = {}
    for fam in FAMILY_LABELS:
        hits = 0
        windows_n = 0
        tp_hits = 0
        for stem in data["stems"]:
            m = data["groups"] == stem
            flags = per_video[stem]["family_windows"][fam]
            assert len(flags) == int(m.sum()), f"confuser flag length mismatch for {stem}"
            windows_n += int(flags.sum())
            pred_sub = pred_full[m] & flags
            hits += int(pred_sub.sum())
            tp_hits += int((pred_sub & (data["y"][m] & flags)).sum())
        confuser[fam] = {
            "windows": windows_n, "predicted_positive_states": hits,
            "states_where_target_also_positive": tp_hits,
        }
    block["confuser_windows"] = confuser
    per_setup = {}
    from collections import defaultdict
    for stem in data["stems"]:
        per_setup[str(per_video[stem]["_setup"])] = per_setup.get(str(per_video[stem]["_setup"]), 0) + 1
    block["test_states_by_setup_videos"] = per_setup
    return block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/broiler/cache"))
    parser.add_argument("--report", type=Path,
                        default=Path("reports/index_behavior_broiler_event_benchmark.json"))
    args = parser.parse_args(argv)

    t0 = time.time()
    cache = args.cache
    rows_json = [json.loads(p.read_text()) for p in sorted(cache.glob("*.rows.json"))]
    if len(rows_json) != 102:
        raise SystemExit(f"expected 102 parsed videos in cache, found {len(rows_json)}")
    keep = [r for r in rows_json if r["stem"][:3] not in IAA_VIDEOS]
    excluded = sorted(r["stem"][:3] for r in rows_json if r["stem"][:3] in IAA_VIDEOS)
    if len(keep) != 92 or excluded != sorted(IAA_VIDEOS):
        raise SystemExit(f"IAA exclusion mismatch: kept {len(keep)}, excluded {excluded}")
    meta = pd.read_csv(cache / "manifest.csv")  # checksum verified at audit time
    stem_keep = sorted(r["stem"] for r in keep)
    setup_of = {stem: int(meta.loc[meta.stem == stem, "setup_id"].iloc[0]) for stem in stem_keep}
    age_of = {stem: int(meta.loc[meta.stem == stem, "age"].iloc[0]) for stem in stem_keep}
    rows_by_stem = {r["stem"]: r for r in keep}

    print("computing features and labels per video ...", flush=True)
    per_video: dict[str, dict] = {}
    imputed_total = {}
    for r in keep:
        stem = r["stem"]
        pf = cache / f"{stem}.parquet"
        if not pf.exists():
            raise SystemExit(f"missing cached parquet for {stem}")
        rec = load_video_record(pf, rows_by_stem[stem], meta.loc[meta.stem == stem].iloc[0])
        rec["_setup"] = setup_of[stem]; rec["_age"] = age_of[stem]
        for k, v in rec.get("imputed_cells", {}).items():
            imputed_total[k] = imputed_total.get(k, 0) + v
        per_video[stem] = rec
        print(f"[{stem}] states={len(rec['states'])} fp={int(rec['fp_window'].sum())} "
              f"masked={int(rec['masked'].sum())} dup_boxes={rec['dropped_duplicate_boxes']}",
              flush=True)

    fp_windows = {s: int(per_video[s]["fp_window"].sum()) for s in stem_keep}
    assign, repairs, totals = core.assign_split(stem_keep, setup_of, fp_windows)
    assert totals["test_fp_videos"] >= core.TEST_MIN_FP_VIDEOS, "test lacks FP-positive videos"
    assert totals["cal_fp_videos"] >= core.CAL_MIN_FP_VIDEOS, "cal lacks FP-positive videos"

    per_split = {"train": [], "cal": [], "test": []}
    for s in stem_keep:
        per_split[assign[s]].append(s)
    dataset = {s: assemble(per_video, per_split[s]) for s in per_split}

    report = {
        "schema": "openfarm/index-behavior/broiler-feather-pecking-event-v0",
        "meta": {
            "source_url": SOURCE_URL,
            "upstream": UPSTREAM_REPO,
            "manifest_sha256": MANIFEST_SHA256,
            "license": "AGPL-3.0 source repo; annotations/media remain upstream, "
                       "only aggregate metrics committed",
            "excluded_iaa_videos_092_101": excluded,
            "window_frames": core.WINDOW_FRAMES,
            "stride_frames": core.STRIDE_FRAMES,
            "min_cover": core.MIN_COVER,
            "episode_gap_states": core.EPISODE_GAP_STATES,
            "n_state_per_video_stride": "1 state per second",
            "split_repairs": repairs,
            "split_fp_videos": totals,
            "split_assignment": {k: per_split[k] for k in ("train", "cal", "test")},
            "feature_names": list(core.FEATURE_NAMES),
            "nan_imputation_policy": {k: core.NAN_IMPUTATION.get(k, core.DEFAULT_IMPUTATION) for k in core.FEATURE_NAMES},
            "imputed_feature_cells": imputed_total,
        },
        "blocks": {s: split_block(dataset[s]) for s in per_split},
        "models": {},
    }
    for split in ("train", "cal", "test"):
        print(f"{split:5s}: {report['blocks'][split]}")

    Xtr = dataset["train"]["X"][~dataset["train"]["masked"]]
    ytr = dataset["train"]["y"][~dataset["train"]["masked"]]

    def record_model(name, model, threshold=0.5):
        entry = {"threshold_rule": threshold, "splits": {}}
        for s in ("cal", "test"):
            entry["splits"][s] = evaluate(dataset[s], per_video, model, threshold)
        report["models"][name] = entry
        print(f"[{name}] cal f1={entry['splits']['cal']['f1']:.4f} "
              f"test f1={entry['splits']['test']['f1']:.4f} "
              f"test ep={entry['splits']['test']['episode_metrics']}", flush=True)
        return entry

    prior_p = float(ytr.mean())
    record_model("prior_stream", ConstantModel(prior_p))
    record_model("always_negative", ConstantModel(0.0))
    logit = core.fit_logistic(Xtr, ytr)
    record_model("logistic_l2", logit, threshold=0.5)
    trees = core.fit_trees(Xtr, ytr)
    record_model("extra_trees_55", trees, threshold=0.5)

    # pre-registered calibration threshold selection: max window F1 on cal
    cal_thr = {}
    for name, model in (("logistic_l2", logit), ("extra_trees_55", trees)):
        ok = ~dataset["cal"]["masked"]
        prob = model.predict_proba(dataset["cal"]["X"][ok])[:, 1]
        ycal = dataset["cal"]["y"][ok]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.unique(np.concatenate([np.linspace(0.01, 0.99, 99), [0.5]])):
            pred = prob >= thr
            tp = int((pred & ycal).sum()); fp = int((pred & ~ycal).sum())
            fn = int((~pred & ycal).sum())
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            f1 = 2 * prec * rec / max(1e-12, prec + rec)
            if f1 > best_f1:
                best_f1, best_thr = float(f1), float(thr)
        cal_thr[name] = {"threshold": best_thr, "f1_cal": best_f1}

    for name, model in (("logistic_l2", logit), ("extra_trees_55", trees)):
        entry = report["models"][name]
        entry["threshold_on_cal"] = cal_thr[name]
        thr = cal_thr[name]["threshold"]
        for split in ("cal", "test"):
            entry["splits"][f"{split}@cal_thr"] = evaluate(dataset[split], per_video, model, thr)
        blk = entry["splits"]["test@cal_thr"]
        print(f"[{name}@thr={thr:.3f}] test f1={blk['f1']:.4f} "
              f"false_ep/h={blk['episode_metrics']['false_episodes_per_stream_hour']} "
              f"tp/fp/fn={blk['episode_metrics']['tp']}/{blk['episode_metrics']['fp']}/{blk['episode_metrics']['fn']}",
              flush=True)

    # per-setup state reference counts on the test split
    acc = {}
    dtest = dataset["test"]
    vis = ~dtest["masked"]
    for stem in dtest["stems"]:
        m = dtest["groups"] == stem
        shown = m & vis
        s = per_video[stem]["_setup"]
        acc[s] = acc.get(s, {"states": 0, "scorable": 0, "positive": 0})
        acc[s]["states"] += int(m.sum())
        acc[s]["scorable"] += int(shown.sum())
        acc[s]["positive"] += int(dtest["y"][shown].sum())
    report["models"]["extra_trees_55"]["test_by_setup_states"] = {
        str(k): v for k, v in sorted(acc.items())}

    report["elapsed_sec"] = round(time.time() - t0, 1)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print("BROILER_EVENT_BENCHMARK:" + json.dumps({
        "report": str(args.report),
        "elapsed_sec": report["elapsed_sec"],
        "fits": len(report["models"]),
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
