# OpenFARM v0.2 — historical and multi-question scientific validation

**Source-audited checkpoint:** 2026-10-16. Source revision and grouping as in [v0.2 query compiler](OPENFARM_QUERY_COMPILER_V0_2.md). Verified direct public original source data and green GitHub Actions. **No clinical or caretaker-action decision has been invented or approved.**

## 1. Hard temporal correction: PigWeb experimental behavior was not pre-existing at PRE

**Critical fix to an earlier OpenFARM assumption.** The PigWeb source behavior CSV carries a timestamp equal to the physiological **PRE** sample time; that shared *row key* does **not** establish that the coded behavioral responses occurring DURING the subsequent experimental stimulus were known at PRE. The [2026 study](https://www.sciencedirect.com/science/article/pii/S1751731126001187) describes behavior measured under experimental conditions and salivary hormones collected before/after. The earlier v0.2 **mistakenly included stimulus-response behavior counts in the predictor state at PRE**. That would be an unavailable-feature/hindsight error.

**Fix implemented in source code (not just documentation):**
- `scripts/build_pigweb_historical_v02.py`: model `features` now contain **only `pre_cortisol_ng_ml`, `pre_oxytocin_pg_ml`**. Human-coded play, escape, inactivity, tail/ear indicators are stored as **`during_stimulus_observations`** for *retrospective association only*, never in a PRE-time forecast.
- `scripts/build_openfarm_decision_queries_v02.py` constructs each PigWeb PRE state from an explicit `pre_only_features` availability whitelist in `scripts/validate_temporal_contract_v02.py`; all human stimulus behavior is isolated in private provenance. `post` must be strictly later than `pre` and never becomes input.
- **No confirmed synchronized multichannel input states remain in the causally valid v0.2 compiled benchmark**. Original experiment contains several modalities, but that is different from demonstrating they are jointly available *as of* one prediction timestamp. The previous claim of `248` valid multimodal predictor states has been withdrawn.
- `tests/test_openfarm_historical_v02.py`: 11 offline adversarial checks, including behavioral feature-after-PRE rejection, unknown-time failures, no false onset, end-of-record and missing-reference censoring. Data-source compiler and integrity workflows run these changes with pinned real rows.

For PigWeb, a separate **post-exposure association** task (observed behavior vs later/presumed-contemporaneous physiology) remains scientifically promising but cannot yet be called prospective without authentic observation-end timestamps. Do not train on the earlier leaked compound `features` export.

**Second availability caveat (operational, not sample collection order):** the *PRE* cortisol and oxytocin values are salivary assays analyzed later in a laboratory. Their **biological sample time precedes POST**, but the source does **not establish when the numeric lab result became available to a monitor**. The pre-only baseline therefore measures an *offline, retrospective prediction from earlier-collected biospecimens*; it **must not be marketed as real-time prospective forecasting**. Operational-as-of-time validity remains **UNVERIFIED** until independent sample processing/result availability times or a corresponding real-time sensor are documented.

## 2. Quantitative validation on genuine grouped heldouts

### A. HotPig: two questions from the *same* physical feeder

- Source: [HotPig Zenodo 17090997](https://zenodo.org/records/17090997), license CC BY 4.0. Author camera-detector features (12 bounded numeric counts) at source-relative minute `t`; independent feeder electronic reading as reference. **Not raw image understanding**.
- Original grouping: **24 pigs, one study site**; 16 train / 4 calibration / 4 untouched test pigs; sample at **15-min stride** to avoid claiming tens of thousands of independent horizons. Future label: *new feeder-active period begins within (t,t+15]*, only when the entire reference window is actually measured (missing/last 15 min → mask). This is event onset, not 'feeding occurs anywhere' or clinical alert.
- **Two labeled questions per state, ONE independently measured feeder reference family**. Masked slots do not become negatives.

| Held-out question (4 pigs) | Test positive / valid | Fixed baseline | L2 logistic | Tiny boosted trees |
|---|---:|---|---|---|
| **Feeder active NOW** | 326 / 6,144 (5.31%) | `eating>0` F1 **0.709**, precision 0.603, recall 0.859 | F1 **0.772**, precision 0.858, recall 0.702, AUROC 0.956, Brier 0.01844 | F1 **0.778**, precision 0.853, recall 0.715, AUROC **0.962**, Brier **0.01812** |
| **NEW feeder bout in next 15 min** | 621 / 6,140 (10.11%) | Train-only prevalence Brier **0.09092**, AUROC 0.5 | Brier **0.09045**, AUROC 0.559, AP 0.122; **no positives at 0.5** | Brier **0.09043**, AUROC **0.567**, AP **0.123**; **no positives at 0.5** |

**Important baseline code correction:** NumPy `int8` accumulated by Python `sum(array)` overflowed and corrupted the initially reported *prior* reference loss. It was corrected to explicit `int64` summation, with an assertion that computed train-only prior matches source training prevalence. **Learned-model scores did not change.** Results above are from [corrected clean CI 38850336437](https://github.com/ob1-s/OpenFARM/actions/runs/38850336437). Source figure decimals rounded for display; full per-pig metrics and reference counts retained as reports.

**Verdict:** `feeder_now` remains a proven *bounded, instrumental observation* research question. `next15_new_feeding_bout` passes causal-reference validity but **FAILS practical predictive discrimination for a one-minute precomputed camera state with these cheap heads**. Keep only as exploratory future-onset stress test, not a welfare monitor/early-warning benchmark. Multiple valid questions do not imply multiple demonstrated capabilities.

### B. PigWeb: PRE-only multi-assay response, four institute-heldout experiments

Actual 2026 source: [AnjaEggert/PigWeb_cross-institutes](https://github.com/AnjaEggert/PigWeb_cross-institutes), CC BY 4.0, pinned `e3dba18edba6e1b214b4bea8ae66109d0f0d1226`. **124 individual pigs, 248 condition sessions** with pre→post assays 21–127 minutes later. Pre hormone state has two scalar channels; context, institute, animal, actual post sampling delay and observed *during-stimulus* human behaviors excluded.

**Protocol:** each of the four institutes AU, INRAE, IRTA, WUR is separately held out as test, with another whole site reserved for future calibration and the remaining **two sites used to train**. Each animal contributes up to two source condition sessions but never crosses sites. Fixed logistic hyperparameters; no test-based hyperparameter or threshold selection. Calibration site is **reserved, not used to fit a probability calibrator**.

| True *later* outcome | Available reference sessions | Pooled source-site-heldout train-only prior Brier | Pre-only logistic Brier | AUROC | Log loss: prior → logistic |
|---|---:|---:|---:|---:|---|
| **Cortisol increase** | 234 | 0.24123 | **0.23737** | 0.642 | **0.67594 → 0.69101 (worse)** |
| **Oxytocin increase** | 210 | 0.25022 | **0.24553** | 0.635 | **0.69359 → 0.70058 (worse)** |

**Transfer failure matters:** at INRAE held out, the learned cortisol Brier **0.30741** exceeds the prior **0.22478**, and oxytocin **0.31277** exceeds the prior **0.25274**. The pooled modest Brier gains obscure an unacceptable site-specific regression. All source measurements have independent lab origins, but independent hormone assays **do not constitute two independent diagnoses of emotional valence or a need for caretaker intervention**. Original experimental contexts may substantially confound both signals. The 40 eye-infrared paired measurements come from one institute only and **cannot** provide a leave-one-institute-out test.

**Verdict:** verified legitimate *future biomarker outcomes with PRE-only input*, but **not admitted as a robust welfare-interpretation benchmark**. Retain in **research hypotheses / reproducibility suite**, demand more independent sites or a different deployable observation state before promotion. Numerical results from [CI 38850336437](https://github.com/ob1-s/OpenFARM/actions/runs/38850336437), `scripts/benchmark_pigweb_future_biomarkers_v02.py`.

### C. EquiFACS: genuine Level-2 expert-vs-independent-vet task **fails the current small-model test**

Separate from source label integrity, we tested whether a tiny logistic model using **expert-observed facial-action-unit counts and temporal occupancy** predicts **separate veterinary clinical pain ratings**. The pain labels from the FACS coder herself were excluded upstream; non-codeable intervals and overlapping non-rateable FACS units are removed, and **entire clinical media** were held out. Fixed L2 regularization `C=0.04`, no test-time tuning; original source `s003` SHA pinned.

| Outcome | Training | Calibration | Held-out test |
|---|---:|---:|---:|
| Codeable independent-vet interval rows | 32 | 10 | **12** |
| Distinct source media | 10 | 3 | **5** |
| Any clinician-rated pain | **5 / 32** | 3 / 10 | **7 / 12** |

| Metric on frozen **12 intervals / 5 media** | Training prevalence-only predictor | Expert FACS → L2 logistic |
|---|---:|---:|
| Brier loss, lower is better | **0.40859** | 0.48116 (**worse**) |
| Log loss, lower is better | **1.09275** | 1.42322 (**worse**) |
| AUROC | **0.5000** | **0.2286** |
| Recall at 0.5 threshold | 0 | 0 |

Real-source [green execution CI 38851016761](https://github.com/ob1-s/OpenFARM/actions/runs/38851016761), `scripts/benchmark_equifacs_vet_binary_v02.py`. **A passing build is not a passing scientific task.** The positive-rate shift (train 15.6%, test 58.3%), only five held-out media, very sparse ordinal categories and pre-annotated facial signals preclude robust welfare generalization. The result does **not** show that facial actions are uninformative; it shows that **this prepared, underpowered evaluation and this low-capacity model do not justify promotion**.

**Verdict: FAIL clinical-model promotion; KEEP small nonranking annotation/rater-integrity pilot only.** No independently validated Level-2 welfare prediction benchmark has passed our current test.

## 3. Source/task/level admission (as of this checkpoint)

| Task | Level | Source label quality | Historical validity | Predictive / eval merit | Admission |
|---|---|---|---|---|---|
| HotPig **feeder now** | 1 observation | Electronic-feeder instrumentation | contemporaneous, valid | Strong bounded within-site pig-heldout F1 0.778 with tiny trees | **APPROVED_RESEARCH_TRAIN/EVAL** (as before) |
| HotPig **next 15-min new feeder bout** | 1→history | Same physical feeder, correctly future/censored | **Pass** | Weak first forecast: AUROC 0.567, AP 0.123 near prior 0.101 | **CANDIDATE ONLY**, no welfare-alert claim |
| PigWeb **later cortisol change** | bio-interpretation proxy (not welfare label) | Lab assay (independent physiological measurement) | **Pass after leak correction** | Tiny pooled Brier gain, INRAE significantly worse | **RESEARCH-ONLY CANDIDATE** |
| PigWeb **later oxytocin change** | bio-interpretation proxy (not welfare label) | Separate lab assay | **Pass after leak correction** | Tiny pooled Brier gain, INRAE significantly worse | **RESEARCH-ONLY CANDIDATE** |
| PigWeb **eye temperature change** | physiology | Instrumented but only 40 source sessions from one site | limited | No cross-site eval | **SITE-LIMITED DIAGNOSTIC ONLY** |
| EquiFACS **vet pain vs expert FACS** | 2 welfare interpretation | Independent vet labels after coder exclusion; 54 intervals, five test media | not a long-horizon forecast | tiny nonranking cohort; heldout FACS→vet logistic AUROC **0.229**, Brier worsens vs prior | **FAILED PROMOTION; pilot only** |
| Cattle **4-way activity** | 1 observed behavior | Human-video ethogram | temporal forecast **not** supported by ambiguous source time keys | approved within-source Choice | **APPROVED_RESEARCH_TRAIN/EVAL** |
| Meerkat **15-way activity** | 1 observed behavior | Human-video ethogram | video-heldout; prediction from expert bbox features | limited observable classes pass independent video test | **APPROVED_NONCOMMERCIAL_RESEARCH_TRAIN/EVAL** |
| **Caretaker escalation now** | 3 care action | **No independent intervention/review outcome** | n/a | cannot measure causal utility | **NO TASK APPROVED** |

## 4. What the multi-question design buys us

- Model should return *different, calibrated answers* to multiple typed questions from the **same allowable as-of-time state**; questions should share computation, not fabricated independent annotations.
- Future onset, current state and physiological outcome can be answered from different states only when source modalities are actually observed by the cutoff.
- Primary reference families remain ONE for a collection of derived human-ethogram questions; two lab assays are genuinely different measurement families, but may still be biologically confounded.
- Multi-modal field monitoring needs **synchronized timestamped streams** (e.g. video + acoustic + wearable) and independent animal/caretaker outcomes. Merely having a CSV with behavior and hormones in a single study does not certify a causal multi-modal state.
- **Abstention/insufficient-evidence**: unknown future record/missing modality is not a healthy or zero event. Probabilistic low confidence cannot substitute for unavailable true labels.

**Next scientific admission target:** independent meaningful welfare outcomes linked to observations (clinical veterinary assessments, verified injuries/lesions, independently documented interventions and reviews), with animal/site-heldout data and a real causal cutoff. Do not auto-label caretaker intervention from experiment condition, elevated cortisol, grimace threshold or generic play. This is the hard remaining Level-2/Level-3 boundary.
