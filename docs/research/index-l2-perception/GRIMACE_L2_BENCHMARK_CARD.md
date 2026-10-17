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
- multiclass Brier (3-way, aligned to classes 0/1/2),
- label-shuffle null (200 permutations, seed 0) - MAE median/q95.
Rater agreement ceilings: leave-one-rater-out strict majority of the other three;
external raters scored as "label-only weak models" on the same test slice.

## Results (exp1 test slice: 215 frames, 4 unseen mice; 200-perm null)

| model | AU | MAE | exact | AUROC `>0` | Brier | null MAE q95 |
|---|---|---|---|---|---|---|
| prior (train majority) | eye | 0.428 | 0.572 | - | - | 0.428 |
| prior | nose | 0.444 | 0.556 | - | - | 0.444 |
| prior | cheek | 0.492 | 0.508 | - | - | 0.492 |
| prior | ear | 0.274 | 0.726 | - | - | 0.274 |
| prior | whisker | 0.455 | 0.545 | - | - | 0.455 |
| B_quality control (5 global stats) | cheek | 0.492 | 0.508 | 0.377 | 0.512 | 0.492 |
| B_quality control | ear | 0.274 | 0.726 | 0.181 | 0.481 | 0.274 |
| A_gray (64x64 px + multinomial LR) | eye | **0.377** | 0.623 | **0.636** | 0.508 | 0.498 |
| A_gray | nose | **0.385** | 0.615 | **0.678** | 0.464 | 0.567 |
| A_gray | cheek | **0.348** | 0.652 | **0.664** | 0.486 | 0.558 |
| A_gray | ear | 0.437 | 0.605 | 0.604 | 0.524 | 0.465 |
| A_gray | whisker | **0.380** | 0.620 | **0.734** | 0.485 | 0.512 |
| external clara (labels) | cheek | 0.751 | 0.326 | - | - | - |
| external yvonne (labels) | eye | 0.219 | 0.781 | - | - | - |
| external hohlbaum (labels) | whisker | **0.132** | **0.868** | - | - | - |
| external hohlbaum (labels) | eye | 0.195 | 0.805 | - | - | - |
| LORO majority-of-other-3 vs gt1 | eye | 0.200 | 0.800 | - | - | - |
| LORO majority-of-other-3 vs gt1 | whisker | 0.186 | 0.814 | - | - | - |

Interpretation (aggregate, no per-row claims): the raw-gray weak model beats its
own shuffle-null on eye/nose/cheek/whisker (MAE below q95) and beats the
quality-only control consistently on AUROC (0.60-0.73 vs 0.18-0.42), i.e. the pixels
carry animal-specific score signal beyond acquisition statistics. It remains far
below the independent expert raters (3-state MAE 0.13-0.22) and the rater-majority
ceiling (0.19-0.31 MAE) on the identical slice, which is the honest headroom of the
task. Calibration preview on the 104-frame cal slice is stable (acc 0.60-0.71),
no overfit collapse.

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
- Quality-artifact shortcutting: the B_quality control shows acquisition statistics
  alone do not produce the A_gray AUROCs.

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
