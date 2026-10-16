# OpenFARM — 16-farm welfare decision robustness extension

**Verified:** 16 October 2026 · **Research status:** retrospective *welfare-related clinical risk* evidence only, not a welfare monitoring system.

## Source and experiment

Original [Bran et al. Brazilian cattle cohort](https://zenodo.org/records/3408818), `Cowlevel.csv`, a **5,529-cow, 16-farm subset** of the same already audited 13,716-cow/50-farm study, not a new independent dataset. Upstream source is CC BY 4.0; actual file SHA-256 pinned to `3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529`. The original expert assessments are **gait score** and a separately collected **body condition score (BCS)**.

The previous fixed test had only **three held-out farms**. To test how fragile the apparent signal was, `scripts/stress_test_cow_welfare_leave_one_farm_out.py` now runs **16 separate leave-one-entire-farm-out folds**, using the other 15 farms to fit an L2-regularized logistic classifier and a **train-only prevalence prior** as the baseline. Each fold's source farm is entirely absent from model fitting; no train/test same-farm overlap. This is exploratory *cross-validation*, not a new untouched external test cohort.

All results were produced in [real-source green CI 38868420045](https://github.com/ob1-s/OpenFARM/actions/runs/38868420045). Detailed per-farm class counts, Brier/log loss, AUROC (where both labels present), and signed Brier improvements are available in the `cow-clinical-welfare-16farm-lofo` workflow report artifact. Only aggregates are published here; no third-party cow data were copied into the repo.

### Decision target design

Each state describes contemporaneous cow-level data. The main test deliberately removes milk production to avoid possible health-consequence leakage and dependence on an expensive/late measurement.

**State available to model:** `Breed`, `Lactation` (parity proxy), `DIM` (days in milk), as recorded contemporaneously. One sensitivity ablation additionally includes `MilkProduction`. Source clinical target scores, derived `clin_lam`/`sev_lam`, cow/farm identifiers, and all target prevalence statistics remain unavailable as features.

| Noul question | Measured independent reference | Important condition |
|---|---|---|
| Is observer gait score ≥3 (clinical lameness threshold)? | Human-scored gait 1–5 | Genuinely measured score, but target is a threshold rewrite |
| Is gait ≥4 (severe lameness threshold)? | **Same human gait score** | **NOT** another independent ground-truth family |
| Is body condition score below 3? | **Separate human BCS** | Other clinically relevant observation; not a direct 'hunger' or neglect label |

**Two independent human-reference families** from the same cow, three interpretable binary screens. This is a legitimate welfare-related *retrospective risk* task from tabular state; **no gait video, no future injury, no veterinarian intervention ground truth, no causal claim**.

## Result — all 16 farms tested, not one selected split

**Primary: `Breed + Lactation + DIM`, no milk-production feature.** Brier gain = source-train prior Brier − fitted-model Brier (positive is improvement; all comparisons performed on completely held-out farms).

| Clinically defined question | Farms where model improved over prior | Unweighted mean Brier improvement | Cow-weighted mean improvement | Worst-farm gain |
|---|---:|---:|---:|---:|
| Gait ≥3, clinical lameness | **16 / 16** | **+0.04494** | **+0.04875** | +0.00468 |
| Gait ≥4, severe lameness | **15 / 16** | **+0.02236** | **+0.02145** | −0.01778 |
| BCS <3, lower body condition | **12 / 16** | **+0.00804** | **+0.00892** | −0.01368 |

### Does adding milk production make the signal?

**No.** Including `MilkProduction` changes average farm gains very little, and slightly **reduces** the mean for two of three questions:

| Outcome | No milk: mean Brier gain | With milk: mean Brier gain | Farms improving with milk |
|---|---:|---:|---:|
| Clinical gait ≥3 | +0.04494 | +0.04463 | 16/16 |
| Severe gait ≥4 | +0.02236 | +0.02238 | 15/16 |
| BCS <3 | +0.00804 | +0.00716 | 12/16 |

**Interpretation:** The retrospective risk pattern is not entirely driven by same-day production. However, breed, days in milk and parity are still physiological/demographic correlates, **not sensor evidence of current lameness, pain or emotional state**. A group-heldout clinical-risk signal is useful for probabilistic assessment research; it does not tell a caretaker which specific cow visibly needs examination.

## Task admission and limits

**Decision:** Promote this source task family to **`APPROVED_RESEARCH_TRAIN / APPROVED_RESEARCH_EVAL — retrospective multi-farm risk diagnostics ONLY`**, with **`Breed, Lactation, DIM`** as the primary features. This is *not* an additional independent data source beyond the existing 50-farm corpus; its 16-farm subset overlaps that corpus. The test is no longer truly blind after this source exploration. New external sites are required before headline model comparisons or clinical validation.

- A defensible benchmark asks whether a cheap calibrated predictor expresses **risk of contemporaneous expert-rated gait and independently rated body condition** on a cow not observed during model fitting. It cannot evaluate actual pain recognition, welfare intervention, future deterioration or false alerts per animal-day.
- The two gait screens are **derived** from one score. Use the original ordinal `Score(1..5)` and BCS `Score` as primary tasks in any eventual typed suite; use thresholds as explicit binary questions that **share the same source family**, never advertise three independent judgments.
- Source labels were collected cross-sectionally. Do not rebrand this as a historical or prospective prediction. Do not claim meaningful calibration based only on favorable Brier gains; report reliability/uncertainty by farm and monitor selective risk before deployment.
- This is only **16 farms** in one regional study, with highly unequal farm sizes. Cow-weighted scores are not estimates of independent farm sample size. Farm-level evaluation is essential.
- Without direct video, motion/pose, audio or physiological sensors, its relevance is to **structured welfare risk**, not general multimodal affective inference. This should complement—not replace—more directly observed decision tasks.

### Immediate next gates

1. Evaluate **original ordinal gait and BCS distributions**, not only threshold rewrites, with proper ordinal likelihood/Brier and real multi-rater reliability data if available.
2. Compare how much of the signal comes from **`DIM` / parity alone** versus breed. Flag potentially clinically circular or timestamp-inaccessible features.
3. Test frozen small probabilistic models on independent *new* sites and clinical/sensor state. This cohort's 16-farm cross-validation is now exploratory evidence, not a sealed future benchmark.
4. Keep injury treatment forecasting in a separate pilot; Swedish selected controls and only three held-out treated pen cohorts do not support a deployable Level-3 early-warning policy.

**Scientific result:** A higher-quality welfare reference and an actually informative bounded source state produce a real, reproducible within-study decision task. This advances the **evidence** sprint without pretending it measures raw animal feelings or an intervention policy.
