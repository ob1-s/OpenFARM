# Track A — First low-cost learned System-1 decision baselines

**Date:** 2026-10-15 São Paulo local time (CI job completed 2026-10-16 UTC).  
**Status:** Verified source-pinned within-study held-out **engineering test**, **NOT animal welfare monitoring validation**.  
**Reproduction:** `scripts/benchmark_hotpig_system1.py`, `.github/workflows/openfarm-hotpig-system1-baselines.yml` and [successful CI run 38827043666](https://github.com/ob1-s/OpenFARM/actions/runs/38827043666).

## Task and data evidence

- Source: [HotPig Zenodo 17090997](https://zenodo.org/records/17090997), CC BY 4.0, published immutable `series.zip` SHA-256 `1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8`.
- Question: **`Noul`: Did an electronic feeder register intake >0 in this minute, given 12 machine-generated camera posture/event counts?**
- **Features:** `unknown`, `standing`, `seating`, `lying`, `eating`, `drinking`, `cuddling`, `curious`, `idle`, `drink`, `eat`, `mate`. **Excluded:** actual feeder output, source pig ID, time index, experimental condition/period, authors' derived state.
- This is **sensor agreement** with a separate instrument, not animal stress/emotion, not a human adjudicated animal-welfare outcome. Camera input is already processed by the original dataset authors' vision detector.
- Animal-disjoint split: 16 train pigs, four calibration pigs, four never-fit test pigs. Source 24 animals, **one trial/site**, 16 days. Test 92,160 valid minutes, 4,834 electronic-feeder-positive minutes (5.245%).
- Positive and negative minutes from the same pig/day have strong autocorrelation; treat **four independent test animal groups**, not 92,160 independent generalization observations.

## Real-world-looking but limited heldout results

All below are on **same four test pigs**, no test-based model fitting/threshold selection. The learned models were fit on only train pigs. Threshold **0.35** maximized calibration-pig F1 on a prespecified candidate grid; still an unstable estimate with only four calibration animals.

| Method | Decision threshold | Test precision | Test recall | Test F1 | False-positive minutes per pig-day |
|---|---:|---:|---:|---:|---:|
| Fixed author-detector rule: `eating > 0` | >0 | 0.58680 | **0.80761** | 0.67972 | 42.9531 |
| Standardized, L2-regularized logistic regression | 0.35 | 0.79889 | 0.71328 | 0.75366 | 13.5625 |
| Small histogram gradient-boosted tree | 0.35 | **0.80995** | 0.71411 | **0.75901** | **12.6562** |

**Alternate stricter fixed threshold:** At 0.5, logistic gives **0.84373 precision**, **0.63777 recall**, F1 **0.72644**, **8.9219 false-positive minutes/pig-day**. Tree gives **0.84939 precision**, **0.65101 recall**, F1 **0.73709**, **8.7188 false-positive minutes/pig-day**. This threshold trade-off is important: apparent precision wins can hide missed observed feeder events.

### Proper-scoring metrics from probabilities (independent of decision threshold)

| Metric | Logistic | Tiny boosted tree |
|---|---:|---:|
| Brier loss | 0.0203953 | **0.0198794** |
| Log loss | 0.0833469 | **0.0812933** |
| AUROC | 0.9468294 | **0.9498735** |
| Average precision | 0.7598877 | **0.7673783** |

**Probability calibration is not solved.** Example: tree probabilities in the 0.3–0.4 bin average ~0.345, but observed positive frequency there is ~0.487; in the 0.5–0.6 bin probabilities average ~0.551 versus observed ~0.621. The top probability bin is slightly overconfident. No post-hoc Platt/isotonic calibration was fit; four calibration animals aren't enough to justify a deployable calibration claim. Brier/log-loss are descriptive of this one cohort.

### Temporal event/lag diagnostic — not alerts

The **4 held-out pigs × 16 days = 64 pig-days** contain **653 contiguous feeder-active episodes**. A naive >0 decision turns any isolated positive minute into a new event.

| Method | Predicted positive contiguous runs | Predicted runs with no feeder-active run even within ±5 minutes |
|---|---:|---:|
| Fixed eating rule | 2,140 | 1,149 |
| Logistic, threshold 0.35 | 1,195 | 232 |
| Tiny tree, threshold 0.35 | 1,185 | 229 |

This is a **diagnostic tolerance analysis**, not 229 distinct welfare alarms and not a clinical requirement. Some false-positive minutes could reflect feeder-capture limitations, alternative eating activity, or camera alignment. Compare observable video/contact and sensor time stamps before calling them detector errors.

## Why this matters and what not to conclude

**Good result:** Plain CPU-cheap nonautoregressive classifier heads meaningfully improve an author-supplied naive rule on **a held-out four-animal partition** without expensive model inference. They can produce full binary probabilities. This is precisely the kind of System-1 engineering experiment we sought, and the measured error/threshold tradeoffs are more useful than top-1 accuracy.

**Limitations:**
1. Only four test animals and four calibration animals; one facility, fixed setup and treatment phases; uncertain site/camera generalization.
2. No external independent human behavior or clinical welfare label: task is **electronic-feeder agreement** only, not true eating vs no-eating or distress.
3. Camera model's `eating` count is **machine-generated upstream**, and other features may reflect the same upstream model errors. This is a second-stage head, not an end-to-end raw-video model.
4. Individual minute predictions are not independent animals or episodes. Do not quote minute-count normal approximations for CI; use group-level bootstrap or new animals/sites.
5. False-positive minutes and ±5-minute unmatched predicted episodes are **not** prospective false welfare alerts, and no caretaker intervention data exist.
6. Results may be sensitive to treatment, farm, synchronization, missingness, and threshold choices. Compare more independent sites before deciding commercial viability.

## Next falsification experiments (data gate first)

- Verify electronic-feeder vs camera timestamps; audit predicted positive runs against actual observed video frames when authorized; define tolerance based on hardware alignment not score optimization.
- Cluster uncertainty by pig; report per-animal precision/recall and whether one animal dominates results; test simple temporal aggregation trained on independent animals and report **episode-level**, not only minute-level, outcomes.
- If independent sites are available, evaluate leave-facility-out. Otherwise state within-trial limitation and stop before claiming field transfer.
- Introduce calibration-only probabilities (e.g. logistic Platt) cautiously after more independent animals; compare with decision cost threshold.
- For animal-welfare escalation, **acquire separate expert-reviewed adverse events or meaningful biological changes**, with natural time coverage and false alerts per animal-day/pen-day.

**Source transfer fix:** initial learned-baseline CI attempt failed solely due intermittent Zenodo resets; new source downloader supports multiple routes, more retries and a SHA-verified persistent GitHub Actions cache. The successful CI job captured full per-pig statistics and 10-bin reliability arrays as a report artifact, without uploading raw source rows.
