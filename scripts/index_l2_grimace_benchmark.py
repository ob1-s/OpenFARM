"""GrimACE L2 benchmark.

Subcommands (run from repository root):
  python3 scripts/index_l2_grimace_benchmark.py labels
      stdlib-only: leave-one-rater-out majority ceiling vs every rater, external
      raters vs gt1, round-2 self-consistency (exp1r2), exp2 cross-experiment
      agreement, exp3 hohlbaum vs companion automatic scores. No archive needed.

  uv run --with pillow,numpy,scikit-learn scripts/index_l2_grimace_benchmark.py pixels
      pixel baselines on the frozen exp1 split (train 9 | cal 2 | test 4 mice;
      animal-disjoint). Feature families A: full 64x64 grayscale; C: centre-60%
      crop (approximate face region); D: border-ring background control; B:
      quality-only control. Model: multinomial logistic regression per AU;
      NOT_RATEABLE cells are masked OUT of training (never silently zeroed) while
      staying in the test counts. Reference labels: gt1. Externals (clara/yvonne/
      hohlbaum) and the leave-one-rater-out majority of the remaining three are
      reported on the identical test slice as label-only comparators. Uncertainty
      comes from animal-blocked label permutations (labels exchanged inside each
      held-out mouse's own support), giving a lower-tail p-value
      (1+#{null<=obs})/(n_perm+1) - never the upper-tail of a lower-is-better
      metric.

Committed aggregate report: docs/research/index-l2-perception/GRIMACE_benchmark_report.json
(no per-row label payloads - apgl-v3).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_l2_grimace_lib as lib

PUB_REPORT_LABELS = Path(__file__).resolve().parents[1] / "docs/research/index-l2-perception/GRIMACE_benchmark_report.json"
PUB_REPORT_PIXELS = Path(__file__).resolve().parents[1] / "docs/research/index-l2-perception/GRIMACE_benchmark_report_pixels.json"


# ---------------------------------------------------------------------------
# manifest io
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: Path) -> list[dict]:
    out = []
    with open(manifest_path) as fh:
        for row in csv.DictReader(fh):
            row["frame"] = int(row["frame"])
            row["labels"] = json.loads(row.pop("labels_json"))
            out.append(row)
    if len(out) < 1400:
        raise SystemExit(f"manifest implausibly small ({len(out)} rows); rebuild first")
    return out


def load_splits(splits_path: Path) -> dict:
    return json.loads(splits_path.read_text())["membership"]


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def ordinal_metrics(preds: list, golds: list[int]) -> dict:
    usable = [(p, g) for p, g in zip(preds, golds) if p is not None]
    m = {"n": len(golds), "n_used": len(usable)}
    if usable:
        m["mae"] = sum(abs(p - g) for p, g in usable) / len(usable)
        m["exact_acc"] = sum(1 for p, g in usable if p == g) / len(usable)
        m["masked_count"] = len(golds) - len(usable)
    return m


def binary_auc(scores: list[float], labels: list[int]) -> float | None:
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for sp in pos:
        for sn in neg:
            if sp > sn:
                wins += 1
            elif sp == sn:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def brier_multiclass(probas: list[list[float]], golds: list[int]) -> float | None:
    if not golds:
        return None
    tot = 0.0
    for pr, y in zip(probas, golds):
        for c in (0, 1, 2):
            p_c = pr[c] if c < len(pr) else 0.0
            tot += (p_c - (1.0 if c == y else 0.0)) ** 2
    return tot / len(golds)


def logloss(preds: list, probas: list, golds: list[int]) -> float | None:
    """Mean multinomial NLL using the model's probability mass (or the provided
    prior proba columns); rows without proba support are skipped."""
    import math

    n = 0
    tot = 0.0
    for p, y in zip(probas, golds):
        if p is None or y is None:
            continue
        pv = [float(v) for v in p]
        pred_masked = pv[y] if y < len(pv) else 0.0
        tot += -math.log(max(pred_masked, 1e-12))
        n += 1
    return tot / n if n else None


def permutation_null(ints_pred_or_none: list, golds: list[int], blocks: list | None = None,
                     n_perm: int = 400, seed: int = 0) -> dict:
    """Blocked lower-tail permutation inference for MAE (lower is better).

    One-sided H0: observed MAE is not below chance. Labels are permuted within
    blocks (users pass one block id per gold row, e.g. the animal id) so each
    block keeps its own label marginals and sample size; the held-out mouse's
    label support is therefore respected. Reports the lower-tail permutation
    p-value with the (1 + #null<=obs) / (n_perm + 1) correction, plus null q05
    and median, and the observed MAE for direct comparison."""
    rng = random.Random(seed)
    P, G, B = [], [], []
    for p, g, b in zip(ints_pred_or_none, golds, blocks if blocks else [None] * len(golds)):
        if p is not None:
            P.append(p); G.append(g); B.append(b)
    if len(P) < 5:
        return {}
    groups: dict = {}
    for i, b in enumerate(B):
        groups.setdefault(b, []).append(i)
    obs = sum(abs(p - g) for p, g in zip(P, G)) / len(P)
    maes = []
    for _ in range(n_perm):
        for idxs in groups.values():
            pull = [G[i] for i in idxs]
            rng.shuffle(pull)
            for i, v in zip(idxs, pull):
                G[i] = v
        maes.append(sum(abs(p - g) for p, g in zip(P, G)) / len(P))
    maes.sort()
    return {"obs_mae": obs,
            "p_lower": (1 + sum(1 for m in maes if m <= obs)) / (n_perm + 1),
            "null_mae_q05": maes[int(0.05 * (len(maes) - 1))],
            "null_mae_median": maes[len(maes) // 2],
            "n_perm": n_perm, "n_usable": len(P)}


def maj_of_others(others: dict[str, int | None]) -> int | None:
    """Strict majority over usable raters; None otherwise (NO_CONSENSUS collapsed)."""
    v = lib.per_au_consensus(others, min_raters=2)
    return v if isinstance(v, int) else None


# ---------------------------------------------------------------------------
# labels subcommand
# ---------------------------------------------------------------------------

RATERS = [lib.TRAIN_RATER, "clara", "yvonne", "hohlbaum"]


def run_labels(manifest_path: Path) -> dict:
    manifest = load_manifest(manifest_path)
    by_exp = defaultdict(list)
    for r in manifest:
        by_exp[r["experiment"]].append(r)

    report: dict = {"protocol": {
        "reference": "per-rater labels; blank cell = NOT_RATEABLE (never coerced to 0)",
        "no_consensus_rule": "strict majority of >=2 usable raters; NO_CONSENSUS rows excluded from a metric, counted in n/n_used",
    }}

    # exp1: leave-one-rater-out majority-of-other-3 ceiling, for each rater as reference
    exp1 = by_exp["exp1"]
    loro = {}
    for ref in RATERS:
        rows = {}
        for au in lib.AUS:
            golds, preds = [], []
            for r in exp1:
                g = r["labels"].get(ref, {}).get(au)
                if g is None:
                    continue
                others = {k: r["labels"].get(k, {}).get(au) for k in RATERS if k != ref}
                golds.append(g)
                preds.append(maj_of_others(others))
            rows[au] = ordinal_metrics(preds, golds)
        loro[ref] = rows
    report["exp1_loro_majority_ceiling"] = loro

    # exp1: externals as label-only models against gt1
    ext = {}
    for rat in ("clara", "yvonne", "hohlbaum"):
        for au in lib.AUS:
            golds, preds = [], []
            for r in exp1:
                g = r["labels"].get(lib.TRAIN_RATER, {}).get(au)
                if g is None:
                    continue
                golds.append(g)
                preds.append(r["labels"].get(rat, {}).get(au))
            ext[f"{rat}|{au}"] = ordinal_metrics(preds, golds)
    report["exp1_externals_vs_gt1"] = ext

    # round-2 external self-consistency over the same exp1 frames (103 rows)
    r1map = {(r["session"], r["frame"]): r for r in exp1}
    selfc = {}
    for rat in ("clara", "yvonne", "hohlbaum"):
        tot = same = 0
        deltas = []
        for r in by_exp["exp1r2"]:
            base = r1map.get((r["session"], r["frame"]))
            if base is None:
                continue
            for au in lib.AUS:
                a = r["labels"].get(rat, {}).get(au)
                b = base["labels"].get(rat, {}).get(au)
                if a is None or b is None:
                    continue
                tot += 1
                same += int(a == b)
                deltas.append(abs(a - b))
        selfc[rat] = {"cells": tot,
                      "exact_match": (same / tot) if tot else None,
                      "abs_delta_mae": (sum(deltas) / len(deltas)) if deltas else None}
    report["exp1_round2_selfconsistency"] = selfc

    # exp2 (labels-only): gt2 vs externals (cross-experiment agreement)
    exp2 = by_exp["exp2"]
    cross = {}
    for rat in ("clara", "yvonne", "hohlbaum"):
        for au in lib.AUS:
            golds, preds = [], []
            for r in exp2:
                g = r["labels"].get(lib.EXP2_RATER, {}).get(au)
                if g is None:
                    continue
                golds.append(g)
                preds.append(r["labels"].get(rat, {}).get(au))
            cross[f"gt2_vs_{rat}|{au}"] = ordinal_metrics(preds, golds)
    report["exp2_gt2_vs_externals"] = cross

    # exp3: hohlbaum vs companion automatic posterior scores (matched deposit frames)
    auto_path = manifest_path.parent / "automatic_scores.csv"
    if auto_path.exists():
        auto: dict[tuple[str, int], dict] = {}
        with open(auto_path) as fh:
            for row in csv.DictReader(fh):
                auto[(row["session"], int(row["frame"]))] = json.loads(row["scores_json"])
        e3 = {}
        for au in lib.AUS:
            golds, preds_rounded, preds_raw_delta = [], [], []
            for r in by_exp["exp3"]:
                g = r["labels"].get(lib.EXP3_EXTERNAL, {}).get(au)
                a = auto.get((r["session"], r["frame"]), {}).get(au)
                if g is None or a is None:
                    continue
                golds.append(g)
                preds_rounded.append(round(a))
                preds_raw_delta.append(abs(a - g))
            m = ordinal_metrics(preds_rounded, golds)
            m["raw_abs_delta_mae"] = (sum(preds_raw_delta) / len(preds_raw_delta)) if preds_raw_delta else None
            e3[au] = m
        report["exp3_hohlbaum_vs_automatic"] = e3

    PUB_REPORT_LABELS.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


# ---------------------------------------------------------------------------
# pixels subcommand
# ---------------------------------------------------------------------------

def load_gray(zipref: zipfile.ZipFile, member: str, size: int = 64):
    import io

    from PIL import Image
    raw = zipref.read(member)
    im = Image.open(io.BytesIO(raw)).convert("L").resize((size, size))
    import numpy as np
    return np.asarray(im, dtype="float32") / 255.0


def load_center_crop(zipref: zipfile.ZipFile, member: str, size: int = 64):
    """Approximate face region: centre 60%% x 60%% of the original frame (face box
    geometry is not annotated for the MGS images, so this is a fixed centre window)."""
    import io

    from PIL import Image
    raw = zipref.read(member)
    im = Image.open(io.BytesIO(raw))
    W, H = im.size
    box = (int(0.2 * W), int(0.2 * H), int(0.8 * W), int(0.8 * H))
    im = im.convert("L").resize((size, size), box=box)
    import numpy as np
    return np.asarray(im, dtype="float32") / 255.0


def load_border_ring(zipref: zipfile.ZipFile, member: str):
    """Background/nuisance control: the outer ring with the centre window removed,
    resampled to four 32x32 strips concatenated left/top/right/bottom -> 4096 dims."""
    import io

    from PIL import Image
    raw = zipref.read(member)
    im = Image.open(io.BytesIO(raw))
    W, H = im.size
    X0, X1 = int(0.2 * W), int(0.8 * W)
    Y0, Y1 = int(0.2 * H), int(0.8 * H)
    strips = [
        im.crop((0, 0, X0, H)),        # left
        im.crop((X0, 0, X1, Y0)),      # top (between left/right)
        im.crop((X1, 0, W, H)),        # right
        im.crop((X0, Y1, X1, H)),      # bottom
    ]
    import numpy as np
    feats = []
    for st in strips:
        st = st.convert("L").resize((32, 32))
        feats.append(np.asarray(st, dtype="float32").reshape(-1) / 255.0)
    return np.concatenate(feats)


def quality_features(gray) -> list[float]:
    import numpy as np
    return [float(gray.mean()), float(gray.std()),
            float(np.abs(np.diff(gray, axis=0)).mean()),
            float(np.abs(np.diff(gray, axis=1)).mean()),
            float(gray.max() - gray.min())]



def run_pixels(manifest_path: Path, splits_path: Path, zip_path: Path) -> dict:
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    manifest = load_manifest(manifest_path)
    splits = load_splits(splits_path)
    exp1 = [r for r in manifest if r["experiment"] == "exp1"]
    part_of_animal = {a: p for p, members in splits["exp1"].items() for a in members}

    zf = zipfile.ZipFile(zip_path)
    grays, centers, borders, quals, metas = [], [], [], [], []
    for r in exp1:
        g = load_gray(zf, r["zip_path"])
        c = load_center_crop(zf, r["zip_path"])
        b = load_border_ring(zf, r["zip_path"])
        grays.append(g.reshape(-1))
        centers.append(c.reshape(-1))
        borders.append(b)
        quals.append(quality_features(g))
        metas.append({"animal": r["animal"], "part": part_of_animal[r["animal"]],
                      "labels": r["labels"], "zip_path": r["zip_path"]})
    X = {"A_gray": np.asarray(grays, dtype="float32"),
         "C_center": np.asarray(centers, dtype="float32"),
         "D_border": np.asarray(borders, dtype="float32"),
         "B_quality": np.asarray(quals, dtype="float32")}
    idx_train = [i for i, m in enumerate(metas) if m["part"] == "train"]
    idx_cal = [i for i, m in enumerate(metas) if m["part"] == "cal"]
    idx_test = [i for i, m in enumerate(metas) if m["part"] == "test"]

    fitted: dict[tuple[str, str], object] = {}
    for au in lib.AUS:
        labs = [metas[i]["labels"][lib.TRAIN_RATER][au] for i in idx_train]
        keep = [i for i, v in zip(idx_train, labs) if v is not None]
        y = np.asarray([metas[i]["labels"][lib.TRAIN_RATER][au] for i in keep])
        if len(set(y.tolist())) < 2:
            continue  # degenerate (constant train label); fallback metric only
        for tag in ("A_gray", "C_center", "D_border", "B_quality"):
            fitted[(tag, au)] = LogisticRegression(max_iter=2000, C=1.0).fit(X[tag][keep], y)

    prior = {}
    for au in lib.AUS:
        cnt = Counter(v for i in idx_train
                      if (v := metas[i]["labels"][lib.TRAIN_RATER][au]) is not None)
        prior[au] = cnt.most_common(1)[0][0]

    def predict(tag: str, idx: list[int], au: str) -> tuple[list, object | None]:
        import numpy as np

        clf = fitted.get((tag, au))
        M = X[tag]
        if clf is None:
            return [None] * len(idx), None
        pr = clf.predict_proba(M[idx])
        full = np.zeros((len(idx), 3), dtype="float64")
        for j, c in enumerate(clf.classes_):
            full[:, int(c)] = pr[:, j]
        preds = [k for k in full.argmax(1)]
        return preds, full

    def metrics_vs_gt1(tag: str, idx: list[int]) -> dict:
        out = {}
        for au in lib.AUS:
            preds, probas = predict(tag, idx, au)
            golds, blocks, P, PR = [], [], [], []
            for k, i in enumerate(idx):
                g = metas[i]["labels"][lib.TRAIN_RATER].get(au)
                if g is None:
                    continue
                golds.append(g)
                blocks.append(metas[i]["animal"])
                P.append(preds[k])
                PR.append(list(probas[k]) if probas is not None else None)
            m = ordinal_metrics(P, golds)
            nonnull = [p for p in PR if p is not None]
            bs = [(p[1] + p[2]) if p is not None else None for p in PR]
            pairs = [(s, 1 if g > 0 else 0) for s, g in zip(bs, golds) if s is not None]
            m["auroc_gt0"] = binary_auc([s for s, _ in pairs], [l for _, l in pairs]) if pairs else None
            if nonnull and all(p is not None for p in PR):
                m["brier"] = brier_multiclass([list(p) for p in PR], golds)
            m["logloss"] = logloss(P, PR, golds) if nonnull else None
            m["blocked_null"] = permutation_null(P, golds, blocks)
            out[au] = m
        return out

    def test_support_per_mouse() -> dict:
        out = {}
        for mouse in sorted({metas[i]["animal"] for i in idx_test}):
            rows = {}
            for au in lib.AUS:
                cnt = Counter(metas[i]["labels"][lib.TRAIN_RATER].get(au) for i in idx_test
                              if metas[i]["animal"] == mouse)
                out_v = {str(k): cnt.get(k, 0) for k in (0, 1, 2)}
                rows[au] = {"usable": sum(out_v.values()), "gold_counts": out_v}
            out[mouse] = rows
        return out

    prior = {}
    prior_proba = {}
    for au in lib.AUS:
        vals = [metas[i]["labels"][lib.TRAIN_RATER][au] for i in idx_train]
        vals = [v for v in vals if v is not None]
        cnt = Counter(vals)
        prior[au] = cnt.most_common(1)[0][0]
        n = len(vals)
        prior_proba[au] = [cnt.get(k, 0) / n for k in (0, 1, 2)]

    def prior_proba_baseline(idx: list[int]) -> dict:
        out = {}
        for au in lib.AUS:
            golds = [metas[i]["labels"][lib.TRAIN_RATER].get(au) for i in idx]
            golds = [g for g in golds if g is not None]
            if not golds:
                out[au] = {"n": 0}
                continue
            pv = prior_proba[au]
            # constant proba column -> AUROC undefined by construction
            out[au] = {"n": len(golds),
                       "brier": brier_multiclass([pv] * len(golds), golds),
                       "logloss": logloss([pv.index(max(pv))] * len(golds),
                                          [[pv[0], pv[1], pv[2]]] * len(golds), golds),
                       "auroc_gt0": None, "auroc_reason": "constant-probability baseline"}
        return out

    test_metrics: dict = {}
    test_metrics["prior_baseline"] = _prior_baseline(idx_test, prior, metas)
    test_metrics["prior_proba_baseline"] = prior_proba_baseline(idx_test)
    for tag in ("A_gray", "C_center", "D_border", "B_quality"):
        test_metrics[tag] = metrics_vs_gt1(tag, idx_test)
    test_metrics["external_clara"] = external_slices(idx_test, "clara", metas)
    test_metrics["external_yvonne"] = external_slices(idx_test, "yvonne", metas)
    test_metrics["external_hohlbaum"] = external_slices(idx_test, "hohlbaum", metas)
    test_metrics["loro_majority_of_other3"] = loro_slice(idx_test, metas)
    test_metrics["test_support_per_mouse"] = test_support_per_mouse()

    cal_out = {}
    for au in lib.AUS:
        clf = fitted.get(("A_gray", au))
        if clf is None:
            cal_out[au] = {"n": 0}
            continue
        pr = clf.predict_proba(X["A_gray"][idx_cal])
        full = np.zeros((len(idx_cal), 3))
        for j, c in enumerate(clf.classes_):
            full[:, int(c)] = pr[:, j]
        pairs = [(full[k], metas[i]["labels"][lib.TRAIN_RATER][au])
                 for k, i in enumerate(idx_cal)
                 if metas[i]["labels"][lib.TRAIN_RATER][au] is not None]
        cal_out[au] = {"n": len(pairs),
                       "mean_max_proba": (sum(float(p.max()) for p, _ in pairs) / len(pairs)) if pairs else None,
                       "acc": (sum(1 for p, y in pairs if p.argmax() == y) / len(pairs)) if pairs else None}

    report = {"protocol": {
        "features_A": "64x64 grayscale resample of the exact scored 1280x1024 frame",
        "features_B": "quality-only control: [mean, std, |d/dx|, |d/dy|, max-min]",
        "features_C": "centre 60% crop (approximate face region; no face boxes exist for the MGS frames)",
        "features_D": "border-ring control (outer ring, centre window removed) - background/nuisance statistics only",
        "model": "multinomial logistic regression C=1.0 per AU; NOT_RATEABLE masked in train only",
        "split": "frozen animal-disjoint exp1 layout: train 9 | cal 2 | test 4 mice",
        "counted_on_test": {"train": len(idx_train), "cal": len(idx_cal), "test": len(idx_test),
                            "note": "the 215 test rows are 4 mice, so they are NOT 215 independent units; "
                                    "uncertainty is quantified with animal-blocked label permutations"},
        "null": "permutation_null: lower-tail blocked permutation, p = (1+#{null<=obs})/(n_perm+1), "
                "q05 = 5th percentile of the null MAE distribution; labels permuted within mouse blocks",
        "exp3": "inventory + agreement evidence only (119 human-labelled frames)",
    }}
    report["cal_preview_A_gray"] = cal_out
    report["test_vs_gt1"] = test_metrics

    PUB_REPORT_PIXELS.write_text(json.dumps(report, indent=2, sort_keys=True))
    return report


def _prior_baseline(idx: list[int], prior: dict, metas: list[dict]) -> dict:
    out = {}
    for au in lib.AUS:
        golds, preds, blocks = [], [], []
        for i in idx:
            g = metas[i]["labels"][lib.TRAIN_RATER].get(au)
            if g is None:
                continue
            golds.append(g)
            preds.append(prior[au])
            blocks.append(metas[i]["animal"])
        out[au] = {"n": len(golds), "mae": (sum(abs(p - e) for p, e in zip(preds, golds)) / len(golds)) if golds else None,
                   "exact_acc": (sum(1 for p, e in zip(preds, golds) if p == e) / len(golds)) if golds else None}
        # constant-prediction baseline: its blocked permutation null MAE is identical
        # by construction (q05 == obs), which is why it carries no p-value claim
        out[au]["blocked_null"] = permutation_null(preds, golds, blocks) if len(golds) >= 5 else {}
    return out


def external_slices(idx: list[int], rat: str, metas: list[dict]) -> dict:
    out = {}
    for au in lib.AUS:
        golds, preds = [], []
        for i in idx:
            g = metas[i]["labels"][lib.TRAIN_RATER].get(au)
            if g is None:
                continue
            golds.append(g)
            preds.append(metas[i]["labels"].get(rat, {}).get(au))
        out[au] = ordinal_metrics(preds, golds)
    return out


def loro_slice(idx: list[int], metas: list[dict]) -> dict:
    out = {}
    for au in lib.AUS:
        golds, preds = [], []
        for i in idx:
            gold_i = metas[i]["labels"][lib.TRAIN_RATER].get(au)
            if gold_i is None:
                continue
            others = {k: metas[i]["labels"].get(k, {}).get(au) for k in RATERS if k != lib.TRAIN_RATER}
            golds.append(gold_i)
            preds.append(maj_of_others(others))
        out[au] = ordinal_metrics(preds, golds)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("subcommand", choices=["labels", "pixels"])
    ap.add_argument("--manifest", type=Path, default=Path(".prime_data/indexl2/manifest.csv"))
    ap.add_argument("--splits", type=Path, default=Path(".prime_data/indexl2/splits.json"))
    ap.add_argument("--zip", type=Path, default=Path(".prime_data/grimace/Data.zip"))
    args = ap.parse_args()
    if args.subcommand == "labels":
        rep = run_labels(args.manifest)
        keys = ("exp1_loro_majority_ceiling", "exp1_externals_vs_gt1",
                "exp1_round2_selfconsistency", "exp2_gt2_vs_externals",
                "exp3_hohlbaum_vs_automatic")
        print(json.dumps({k: rep.get(k) for k in keys}, indent=2, sort_keys=True)[:2400])
    else:
        rep = run_pixels(args.manifest, args.splits, args.zip)
        print(json.dumps(rep["test_vs_gt1"], indent=2, sort_keys=True)[:2400])


if __name__ == "__main__":
    main()
