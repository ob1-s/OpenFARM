# GrimACE L2 Benchmark Card (Direct Welfare Perception, Operator B)

Frozen design of the *openfarm-l2* Score/Noul image benchmark built from the GrimACE
deposit (source details: `GRIMACE_SOURCE_DOSSIER.md`). Everything below is
mechanically reproducible from the original Zenodo file set with the committed scripts.

## Scope and labels

- **Family (primary):** exp1 - 773 scored full frames, 15 mice, 4 independent rater families
  (`gt1` + clara + yvonne + hohlbaum), plus 103 second-round external annotations on
  the same frames (`exp1r2`, self-consistency evidence).
- **Label-only evidence:** exp2 - 472 frames with 4 raters (pixels absent from the deposit).
- **Inventory + agreement evidence:** exp3 - 1,725 control-treatment frames
  (`a*_c_*`, 9 mice) with `hohlbaum` human labels on 119 pixel-backed frames
  (eye/ear only) and 1,688 matched companion automatic posteriors (not an eval target).
- **AU tasks:** eye / nose / cheek / ear / whisker ordinal {0, 1, 2}. Blank cell =
  NOT_RATEABLE, preserved as an explicit mask (never coerced to 0); a metric row's
  `n` counts usable references and `n_used` counts usable predictions (NO_CONSENSUS
  pred rows are excluded from scoring and counted).
- **Model-visible state:** the exact 1280x1024 RGB frame the humans rated (no crops,
  no COCO boxes, no filename-encoded metadata exposed to the model).
- **Secondary reference:** strict majority (>= 2 usable raters) per (image, AU);
  all-blank -> NOT_RATEABLE; otherwise NO_CONSENSUS.

## Frozen animal-disjoint splits (no RNG, chosen before examining any test labels)

Rule: within each treatment group, animals sorted ascending (numeric-aware for exp3),
last 2 -> **test**, previous n_cal -> **cal**, remainder -> **train**.

| Family | train | cal | test | digest (`split_membership_digest`) |
|---|---|---|---|---|
| exp1 (m,b) | 3340 3341 3342 3343 3346 3347 3348 3349 3354 | 3355 3369 | 3367 3371 3376 3379 | `f67c8319bebbe20015f863953f3daf3fd40d582971a2b07af77396d5ddc01083` |
| exp3 (c) | 3 4 9 12 13 | 17 18 | 21 26 | `0df49e5e20c01a4e23e927b05f0a94aa03199622119ba3e7ba3f8cd712882399` |

exp1: 454 train / 104 cal / 215 test frames. Membership per animal with image counts:
`GRIMACE_split_membership.csv`. Label-matrix digest (exp1+exp2+exp3+r2 family):
`7063683380647426bf32de3aeec99e3c1e38fc4b8b137cad0a97ad2cb567c7b0`.

## Metrics

Per AU, per (model, reference-rater):
- ordinal MAE over usable rows, exact accuracy, masked count,
- binary AUROC for `score > 0` (probability-mass sum over classes {1, 2}),
- multiclass Brier and multinomial log loss (model vs train-prior-probability baseline),
- **blocked lower-tail permutation null**: 400 permutations, seed 0, labels permuted
  *within each held-out mouse's own row set* (animal-blocked, preserving per-mouse
  label marginals and sample size); one-sided inference for a lower-is-better
  metric is `p = (1 + #{null_MAE <= obs}) / (400 + 1)` with the null q05 reported
  alongside. The 215 test rows are 4 mice - they are NOT 215 independent units,
  and no claim derived from unblocked statistics is supported.
Rater agreement ceilings: leave-one-rater-out strict majority of the other three;
external raters scored as "label-only weak models" on the same test slice.
Per-mouse gold support (usable n, counts by ordinal) is reported in
`GRIMACE_benchmark_report.json` (`test_support_per_mouse`).

## Results (exp1 test slice: 215 frames = 4 unseen mice; 400-perm animal-blocked null)

Shortcut/nuisance ablation (this is the load-bearing row set, per review):

| family | AU | MAE | exact | AUROC `>0` | log loss | p (blocked lower tail) | null q05 |
|---|---|---|---|---|---|---|---|
| **C_center** (centre crop ~face) | nose | **0.342** | 0.658 | **0.712** | 0.648 | **0.0025** | 0.417 |
| C_center | cheek | **0.326** | 0.674 | **0.693** | 0.718 | **0.0025** | 0.436 |
| C_center | whisker | **0.347** | 0.653 | **0.768** | 0.685 | **0.005** | 0.380 |
| C_center | eye | 0.405 | 0.595 | **0.599** | 0.808 | **0.025** | 0.423 |
| C_center | ear | 0.498 | 0.544 | 0.538 | 1.052 | 0.808 (n.s.) | 0.451 |
| **D_border** (background-only ring) | eye | 0.498 | 0.502 | 0.426 | 1.008 | 0.933 (n.s.) | 0.423 |
| D_border | nose | 0.519 | 0.481 | 0.497 | 0.764 | 0.913 (n.s.) | 0.422 |
| D_border | cheek | 0.547 | 0.453 | 0.473 | 0.769 | 0.935 (n.s.) | 0.436 |
| D_border | ear | 0.414 | 0.637 | 0.541 | 1.004 | 0.501 (n.s.) | 0.395 |
| D_border | whisker | 0.446 | 0.554 | 0.399 | 0.968 | 0.247 (n.s.) | 0.446 |
| **A_gray** (full frame) | eye | 0.377 | 0.623 | 0.636 | 0.766 | **0.0025** | 0.423 |
| A_gray | nose | 0.385 | 0.615 | 0.678 | 0.659 | **0.010** | 0.417 |
| A_gray | cheek | 0.348 | 0.652 | 0.664 | 0.700 | **0.0025** | 0.448 |
| A_gray | ear | 0.437 | 0.605 | 0.604 | 0.937 | 0.451 (n.s.) | 0.419 |
| A_gray | whisker | 0.380 | 0.620 | 0.734 | 0.715 | **0.035** | 0.397 |
| B_quality control | (all) | ~prior | - | 0.31-0.42 | ~0.70 | 1.0 (n.s.) | = obs |
| prior-proba baseline (train freqs) | eye | - | - | const | 0.692 (Brier 0.490) | - | - |

Per-mouse gold support (aggregate counts; no test mouse carries ordinal-`2`
support except ear's 1-5 frames per mouse - the `2` extreme is not testable on
this test slice and is excluded from any "severe" claim):

| mouse | eye | ear | nose | cheek | whisker |
|---|---|---|---|---|---|
| 3367 | 32/19/0 | 12/38/1 | 36/10/0 | 33/14/0 | 22/6/0 |
| 3371 | 30/20/0 | 17/32/1 | 23/20/0 | 21/20/0 | 18/6/0 |
| 3376 | 32/22/0 | 4/49/1 | 25/19/0 | 19/24/0 | 12/18/0 |
| 3379 | 29/31/0 | 18/37/5 | 20/34/0 | 19/31/0 | 14/25/0 |

Interpretation (aggregate, no per-row claims):
- The face-region model (C_center) is significantly below its animal-blocked
  permutation null on eye/nose/cheek/whisker (p 0.0025-0.025) and the
  background-only control (D_border) is null on every AU (p >= 0.25), so the
  signal is carried by the face window, not by cage/camera/acquisition geometry.
  The B_quality control stays at chance-or-worse (AUROC 0.31-0.42, p = 1.0).
- **ear is NOT significant under the corrected lower-tail inference in any model
  (p = 0.45-0.81)** - the earlier "beats null q95" claim for ear is withdrawn;
  ear MAE can be as good as the prior purely via the `1`-dominant marginals.
- The weak low-capacity pixel model remains far below the independent expert
  raters on the identical slice (3-state MAE 0.13-0.22; e.g. hohlbaum whisker
  0.132, eye 0.195) and below the rater-majority ceiling (MAE 0.19-0.31), which
  is the honest headroom of the task. Calibration preview on the 104-frame cal
  slice is stable (acc 0.60-0.71), no overfit collapse.
- Brier/log-loss: the models beat the train-prior-probability baseline on
  nose/cheek/whisker but not on ear; eye is near-parity (0.508 vs 0.490).

Round-2 external self-consistency on the same 103 exp1 frames:
yvonne 0.937 exact, hohlbaum 0.895, clara 0.684 (487/474/373 usable cells).

exp2 cross-experiment agreement (gt2 vs externals, 410-472 cells/AU): e.g. ear
exact 0.701 (clara), cheek exact 0.437; the two experiments' rater spaces are
consistent, with clara known to diverge most (matches the round-2 self-consistency).

The label-only family (exp2) is **not** bundled into the image benchmark; it only
contextualizes rater reliability across experiments.

## What would count against the benchmark

- Any metric dependence on session-folder text (filenames are never features).
- Any leakage between mouse ids across splits (machine-asserted in the build).
- Treating the automatic posterior rows as ground truth (they are baselines).
- Quality-artifact shortcutting: the B_quality control (and, more directly, the
  null D_border background-only family) shows acquisition/background statistics
  alone do not produce the face-window AUROCs on the corrected blocked inference.
- Unblocked significance claims: the four test mice contribute 80-215 usable rows
  per AU, not 215 independent units; only the animal-blocked permutation p-values
  above support the "signal beyond chance" statements, and ear carries no
  significant signal in any model family.

## Reproduction

```bash
python3 scripts/index_l2_grimace_build.py --zip .prime_data/grimace/Data.zip
python3 scripts/index_l2_grimace_benchmark.py labels
uv run --with pillow,numpy,scikit-learn scripts/index_l2_grimace_benchmark.py pixels
python3 -m unittest tests.test_index_l2_grimace
```

## Ethics / license

Source data: Zenodo 18299984 (record license apgl-v3); article: PMC13043301 (CC BY).
No source pixels or label rows are committed; this repository carries membership,
hashes, and aggregates only. Mice were treated per the original study's license
permissions (see paper for animal-care approvals).
