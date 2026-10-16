# OpenFARM — Welfare Decision Evidence Sprint

**Source-audited checkpoint: 16 October 2026.**  
**Mandate:** Stop accumulating weak wrappers; find, independently verify and test original **animal welfare outcomes** using the frozen typed-question compiler and cheap statistical baselines. Preserve unsupported and negative results. Original `main` and the v0.1 benchmark remain unchanged; work is on the draft research PR #2.

> **Learn what deserves attention, from observable behavior and defensible evidence.**

> **16-farm robustness update (same day):** The 5,529-cow retrospective risk pilot has now been tested in **all 16 leave-one-entire-farm-out folds**, not merely a three-farm split. Using **only breed, lactation and days in milk**, L2 logistic beat the train-farm prevalence baseline on **16/16 farms for gait ≥3**, **15/16 for gait ≥4** and **12/16 for BCS <3** (farm-average Brier gains **+0.04494, +0.02236, +0.00804** respectively). Adding milk production offered no meaningful improvement. **Research train/eval admission is limited to cross-sectional welfare risk in this already explored original study, not direct welfare sensing, prospective care, or a blind leaderboard.** [Full evidence and caveats](WELFARE_DECISION_EVIDENCE_SPRINT_LOFO_2026_10_16.md); [passing real-source CI](https://github.com/ob1-s/OpenFARM/actions/runs/38868420045).

## Executive findings

| New original evidence | Actual access and scientific reference | Train/eval test | Admission |
|---|---|---|---|
| **Bran et al. 2019, southern Brazil cows**, [Zenodo 3408818](https://zenodo.org/records/3408818) | **13,716 cows, 50 farms, originally human-scored gait grades 1–5 and body condition 1.5–5**, herd/environment survey available; pinned CSV bytes; **CC BY 4.0** source metadata verified | Successfully compiled **13,716 states; 27,428 valid labels; 4 masks; 13,712 states with TWO different human-reference families** in 30/10/10 farm groups; *management-only risk* learned model LOSES to prior on all three derived outcome screens | **Original clinical labels research-ready**; management-only state not validated as an individual welfare monitor. Use as cross-sectional reference/evaluation stress test, not headline predictive benchmark |
| **Same cows, richer 16-farm subset**, Cowlevel.csv | **5,529 cows, 16 farms, parity/lactation/day-in-milk/milk production/breed, original gait and BCS scores**, **overlapping** 50-farm source, NOT an independent sample | 10/3/3 farm split. Retrospective clinic-lameness L2 logistic AUROC **0.731**, Brier **0.209 vs 0.242 prior**; severe-lameness AUROC **0.749**, Brier **0.124 vs 0.139 prior**. Body-condition question comparatively weak: AUC **0.595** | **Promising bounded retrospective clinical-risk pilot**, not a blind or prospective welfare monitor; same source counts may not be added to original 13,716 |
| **Ollagnier et al., real pig tail-damage treatments**, [Zenodo 5028016](https://zenodo.org/records/5028016), original study [2023 PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0252002) | Actual **65,886 feeder rows, 862 source pig IDs, 65 reconstructed pen cohorts**; outcome is **date of FIRST RECORDED TREATMENT** for tail damage. Original archive **CC BY-NC 4.0** | Paper: **27 cases = Swedish 21 + Swiss 6**; original annotated source: **28 case cohorts = 21 Swedish + 7 Swiss**. Two treatment dates fall outside observed source data; three Swedish TB cohorts have no pre-onset records. **Do not silently rewrite labels.** | Full cross-country source **QUARANTINED** until 28-vs-27 metadata mismatch resolved; never use arbitrary control pseudo-onset dates as events |
| **Swedish-only constrained subset of same pig paper** | Original **21 TB + 21 control pen cohorts**, matching source article; excludes 3 treated cohorts without historical data | **275 causal 14d-history → next-7d first-treatment** pen-week states (171 train / 55 cal / 49 test) and **17 true future onset windows in 17 pen cohorts**. Test 49 windows, **3 positive first-treatment cohorts**, 7 independently grouped pen cohorts. All future feature and pseudo-control leakage excluded. | **RESEARCH EXPLORATORY HISTORICAL PILOT ONLY**; selected case-control source, 3 test events, no population base-rate/operational human-review decision |

### Actual Level-2: two independent clinical questions from one cow state

Original human assessments, not AI guesses:
- `Score(1..5)`: source observer's **gait/locomotion score**.
- `Score(1.5..5)`: source observer's **body condition score**, separately measured.
- Different measured reference families in **13,712 of 13,716** cow states. Exactly **4 body-condition ratings missing** and masked; no fabricated zeros.
- Source-measured gait was 2 in 7,763 cows, 3 in 2,897, 4 in 2,673, 5 in 380, and 1 in 3. By the source's recorded thresholds, **5,950 clinical lameness (gait ≥3)** and **3,053 severe (gait ≥4)**. Original BCS <3 in **5,982** cows.
- Three derived binary target screens are useful evaluation slices, but only **two independent human observations**: general/severe are both threshold rewrites of gait, not new reference labels.
- `scripts/build_brazil_cow_welfare_queries.py` runs the **existing** `openfarm_query_compiler_v02.py` without altering it, emits local separate model states / answer keys / source provenance. [Real-source CI 38859511881](https://github.com/ob1-s/OpenFARM/actions/runs/38859511881) passed.

**Observation/input limitation:** The 50-farm compiled state contains separately surveyed herd management and cow breed, **not a photo or movement signal from the actual cow**. This is a cohort-level risk-prediction task. A high-quality human clinical **label** does not guarantee an observable/camera-ready prediction **input**. Training/evaluation as retrospective welfare epidemiology is appropriate; pretending this measures visual/real-time animal empathy is not.

### Rigorous heldout risk falsification — why observation quality matters

Source original `Herd.csv` has 50 independent farm surveys. Farm-level `LS_1...LS_5`, `Clinical lameness`, `Severe lameness` and any target-derived aggregate values are **excluded from model input**. `farm_ID`, cow identifier, source exams, outcomes, target-derived categories, and assessment order are only hidden provenance. Features restricted to observed herd management (floor, bedding, feed, hoof procedures, facility practices, etc.) plus cow breed. **30 train / 10 calibration / 10 heldout test farms**, randomized by farm type only, fixed seed.

| On 10 unseen farms, 2,410 cows (BCS valid 2,408) | Constant train-prior Brier ↓ | L2 logistic Brier ↓ | Logistic AUROC |
|---|---:|---:|---:|
| Observer-rated clinical lameness (gait ≥3) | **0.25317** | 0.27492 | 0.571 |
| Observer-rated severe lameness (gait ≥4) | **0.18933** | 0.20188 | 0.545 |
| Observer-rated BCS below 3 | **0.23867** | 0.25934 | 0.442 |

**Verdict:** LOGISTIC FAILS all three held-out Brier loss comparisons vs the naive train prevalence. Farm management metadata alone does not give enough independent per-cow evidence for clinical welfare inference.

**Richer state, same original study, overlapping cows:** `Cowlevel.csv` provides independently measured cow milk production, days in milk, lactation/parity and breed for **5,529 cows on only 16 farms**. These are *not* the actual gait/BCS targets and are not derived from the clinical rating. **10 train / 3 calibration / 3 test** entire farms; data are contemporaneous, so NO future prognosis claim:

| On 3 new farms (961 cows) | Train-prior Brier ↓ | L2 logistic Brier ↓ | AUROC |
|---|---:|---:|---:|
| Clinically scored gait ≥3 | 0.24226 | **0.20887** | **0.731** |
| Clinically scored gait ≥4 | 0.13941 | **0.12359** | **0.749** |
| Independently scored BCS <3 | 0.26889 | 0.26808 | 0.595 |

By **each** held-out test farm, severe-gait AUROC was **0.743 / 0.827 / 0.752**, and the model's Brier beat the prior for all three farms. General-lameness AUROC was **0.756 / 0.787 / 0.690**, but Brier was worse than the prior for one farm. These per-site checks matter much more than treating 961 cows as independently sampled farms. **Three sites is a very weak external-domain test**, and this already-inspected test cannot be reused as a sealed blind leaderboard. We cannot assume the measured milk production was available early enough for prospective warnings. [Source-audited CI 38860469564](https://github.com/ob1-s/OpenFARM/actions/runs/38860469564).

### Historical pig treatment: an actual action outcome but not an alert policy

The study defines the first real recorded **tail-damage treatment** in a pen as onset of its tail-biting episode. This is closer to Level-3 than generic social-behavior labels; still not an expert-defined *optimal moment to alert*.

- Dataset ID fields are reused across recording cohorts: grouping only by `(Farm,PenID)` yields **52 bogus groups** and loses treatment events. Reconstructed group must include `(Farm,PenID,PenType,TBSTART_)` and time consistency, all kept in private provenance. Swedish `date` is YYYYMMDD, Swedish onset is Excel-exported irregular M[ M ]DDYY/YYYY, Swiss dates are Excel serials. No IDs or onset dates leak into state.
- Verified **65** original pen/date/status groups; original labels produce **28** positive pens, paper says **27**. Swiss 7 source positive pens vs six published; **not reconciled**, so pooled benchmark remains blocked.
- Three Swedish treated cohorts have **zero** days of before-treatment observation; no possible causal early-warning input. No invented preceding feeder readings.
- Swedish-only pilot uses previous 14 calendar days aggregated over observed pigs to predict FIRST documented treatment during following seven days. Weekly stride to reduce redundant windows, minimum source-history coverage; controls' arbitrarily assigned event anchors **never** count as events or enter model state.
- [Real-source compile CI 38860218762](https://github.com/ob1-s/OpenFARM/actions/runs/38860218762): **275 labeled windows**, with **17 first-treatment positive windows**, train 171/calibration 55/test 49 pen-week states from whole pen-cohort grouping. Only **seven active source pen cohorts and THREE distinct positive treatment-onset cohorts** remain in test, so no robust alert claim.
- [Actual cheap heldout baselines CI 38860418898](https://github.com/ob1-s/OpenFARM/actions/runs/38860418898):

| Swedish-only original-source selected test: 49 windows, three positive cohorts | Test Brier ↓ | AUROC | AP | Positives predicted at 0.5 |
|---|---:|---:|---:|---:|
| Train-only prevalence | **0.05749** | 0.500 | 0.061 | 0 |
| L2 logistic on genuine past feeder observations | 0.06093 | 0.384 | 0.065 | 0 |
| Small random forest | 0.05811 | 0.645 | 0.147 | 0 |

**Verdict:** **NO convincing true-outcome forecasting performance**. As with the 50-farm cow metadata-only approach, the **actual reference is good but observation/state generalization is weak**. The paper's retrospective case-control population and pseudo-dated controls make any present per-pen-day false-alert calculation **not a natural-deployment rate**; the run expressly does NOT measure caretaker alerts. Do not promote this to a validated Level-3 intervention decision benchmark.

## Source status and approval decision

| Tier | What is independently grounded | Status |
|---|---|---|
| **Observe** | Original pig feeding electronic sensor, human cattle behavior observation, human meerkat ethogram | Prior research tasks remain frozen, not modified in this sprint |
| **Interpret — source human reference** | **Original cow gait score and body-condition score** (2 distinct clinical references), permitted CC BY 4.0 | **NEW: source-validated clinical reference corpus** with executable compilation and 50-farm provenance; typed state-to-score evaluation **research use only**, weak current predictor |
| **Interpret — retrospective clinical risk** | Source gait/BCS from richer cow-level parity/lactation/milk data, 16 farms | **Promising limited retrospective research pilot**; significant external/site sample limitation; not a validated individual clinical screen |
| **Historical outcome** | Pig first recorded tail-damage treatment event at source pen/date | **Constrained Swedish research pilot**, validated target chronology and private provenance; 3 treated source pens in heldout test, weak models; NOT reliable caregiver alert |
| **Caretaker intervention action / optimal escalation** | Real human-reviewed decisions, acceptable false alerts per animal-day, consequences of not alerting | **ZERO approved Level-3 decision tasks**. Injury treatment observed is an outcome, *not* a verified policy |
| **Positive welfare affect** | Validated independent indicators of good affect/comfort/motivation with multichannel state | **ZERO new approved positive-valence tasks** this sprint. Do not infer happiness from grooming/play or experimental intervention condition |

**Research discipline:** Success is not a model accuracy number. The new welfare data give us real clinical labels, two independent source families per animal, actual treatment event provenance, and direct demonstrations that source-appropriate *states* matter more than synthetic target proliferation. Source-quality failures remain visible rather than dressed as benchmark wins.

## Files and reproducibility

| Deliverable | Source or exact check |
|---|---|
| Full 13,716-cow forensic four-file raw survey metadata | `scripts/audit_brazil_cow_welfare.py`, [CI 38858299942](https://github.com/ob1-s/OpenFARM/actions/runs/38858299942) |
| Source license, true gait severity/BCS binary derivation verification | `scripts/validate_brazil_cow_welfare_targets.py`, [CI 38858486469](https://github.com/ob1-s/OpenFARM/actions/runs/38858486469) |
| Frozen compiler, TWO independently measured human labels per source cow | `scripts/build_brazil_cow_welfare_queries.py`, [CI 38859511881](https://github.com/ob1-s/OpenFARM/actions/runs/38859511881) |
| Farm-heldout managerial risk baselines | `scripts/benchmark_brazil_cow_welfare.py`, [CI 38858768373](https://github.com/ob1-s/OpenFARM/actions/runs/38858768373) |
| Independent cow covariates per-farm ablation | `scripts/benchmark_brazil_cow_individual_covariates.py`, [CI 38860469564](https://github.com/ob1-s/OpenFARM/actions/runs/38860469564) |
| Verified original 4MB pig feeder/true treatment-date source | `scripts/audit_pig_tail_damage_outcomes.py`, [CI 38859320673](https://github.com/ob1-s/OpenFARM/actions/runs/38859320673) |
| Restored original 65 pen cohorts and clear 28-vs-27 discrepancy | `scripts/reconstruct_pig_treatment_cohorts.py`, [CI 38859917800](https://github.com/ob1-s/OpenFARM/actions/runs/38859917800) |
| Limited real treatment-onset historical states (source-model separation) | `scripts/build_swedish_tail_damage_historical.py`, [CI 38860218762](https://github.com/ob1-s/OpenFARM/actions/runs/38860218762) |
| Low-cost actual injury-outcome baseline/selection falsification | `scripts/benchmark_swedish_tail_damage_warning.py`, [CI 38860418898](https://github.com/ob1-s/OpenFARM/actions/runs/38860418898) |

Source-derived raw rows are NOT committed to GitHub or rehosted. Clean CI downloads original public files, validates immutable checksums, materializes train/cal/test states **locally**, and uploads aggregate metadata, not original studies. Compiler and canonical v0.1 tasks were **not modified**.

## Best new follow-up data criteria (rather than more wrapper code)

**Seek paired individual-level observations with independent welfare labels**:
- Same animal/session: camera-visible gait or body motion + human-rated gait AND clinical lesion/weight trajectory. We have clinical **labels** and metadata risk signals but not the source gait pixels for these 13,716 cows.
- Same animal across time: truly prior feeding/motion/audio signals + actual treatment/clinical deterioration dates and untreated controls *not selected by arbitrary artificial comparison dates*.
- Jointly available channels: synchronized audio+video+motion/accelerometer+temperature AND independently expert-scored behavior or clinical episode; don't call lab-processed future hormones a real-time sensor input.
- Positive welfare indicators: actual expert-assessed voluntary affiliation/motivation/preferences, natural baseline changes and corroborated animal choice/health outcomes, not automatically “happy” from play or a positive experimental arm.

**Do not dilute the suite** with a second derived label that merely rewrites the first. Maintain two distinct capability questions in one state only where genuinely separate reference sources exist.
