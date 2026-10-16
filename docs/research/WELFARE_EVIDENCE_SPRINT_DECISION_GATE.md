# OpenFARM welfare-evidence sprint — decision gate

**16 October 2026.** Source of truth for admissions following the first welfare-specific sprint. This gate **does not change** the frozen v0.1 observation suite, the v0.2 compiler or prior evaluation membership.

## Sprint objective
Move from easy observation labels to **independently supervised welfare outcomes** with a defensible, low-cost state available at inference time, per-source legal permission, genuinely held-out subjects/sites, and falsifiable cheap baselines. Admit only what evidence justifies.

**Core rule:** An expert-rated outcome is a valid *target*, but a state comprising herd context is not direct perception, and a historically recorded treatment is not an independently reviewed optimal intervention recommendation.

## Final admission matrix

| Research question | Original label/reference | Input available and genuinely independent of target? | Test integrity | Decision |
|---|---|---|---|---|
| **Contemporaneous cow locomotion risk:** source observer gait score >=3 | Human clinical gait score, *one original reference* | Cow breed, lactation, days in milk; yes as observed covariates, **not** visual gait | Leave one of **16 entire farms** out across all 16 folds. Brier improvement over train-only prior **16/16 farms**; exploratory, no untouched future test. | **ADMIT narrowly: retrospective structured welfare-risk research** |
| **Severe cow locomotion risk:** gait >=4 | **Same** clinical gait reference, not independent supervision | Same 3 covariates | 16-farm LOFO: **15/16** farms improved | **ADMIT as additional typed threshold task, not a second label family** |
| **Low body-condition risk:** BCS <3 | Separate *human BCS* observation | Same 3 covariates; weak-but-real signal | 16-farm LOFO: **12/16** farms improved, mean farm Brier gain 0.00804 | **ADMIT cautiously as second independent reference family**, sensitivity/farm limitations prominently displayed |
| **Historical pig tail damage treatment within next 7 days** | First actual treatment onset date in Swedish source pen | 14 preceding calendar days of observed feeder history, with strict time cutoff | 49 heldout windows, **only 3 positive pen cohorts**; L2 logistic Brier worse than prevalence prior and no positive predictions at 0.5 | **QUARANTINE as exploratory outcome pilot**; NOT a Level-3 action/alert eval |
| **Across-country tail treatment pooled** | Source reconstructed 28 positive pen cohorts vs article's 27 (Swiss discrepancy) | Requires corroboration | Source contradiction unresolved | **REJECT from suite until independently reconciled** |
| **PigWeb cortisol/oxytocin change** | Two separate post-session laboratory assays | Earlier pre-session observations/pre-assays | Four institutes; 124 pigs and 248 paired sessions, large site shifts | **TASK BUILD VALIDATED; prediction performance/construct evaluation pending**; do not call emotional-state truth |
| **Clinical equine pain interpretation** | Two vets' ratings independently separated from AU annotator | Human-coded AU timeline (not raw pixel model) | 54 codeable intervals / 18 media with 5 test media; small and exploratory | **KEEP PILOT ONLY**, no robust clinical rank or deployed pain inference |
| **Direct caregiver action / positive emotional valence** | None sufficiently independently validated and with realistic observation/action cost | None adequate | N/A | **NO APPROVED TASK**; never fabricate from experimental context |

**Reference accounting:** Gait ≥3 and gait ≥4 are two tasks from **one gait assessment**, not two independent clinical truths. BCS is separate. The 5,529-cow 16-farm subset **overlaps** the original 13,716 cows across 50 farms; never add counts.

## Exact evidence

- [Audit source and target construction](WELFARE_DECISION_EVIDENCE_SPRINT_2026_10_16.md).
- [Full 16-farm leave-one-farm-out report](WELFARE_DECISION_EVIDENCE_SPRINT_LOFO_2026_10_16.md) and [real-source green CI](https://github.com/ob1-s/OpenFARM/actions/runs/38868420045).
- [Swedish actual treatment episode chronology and negative baseline evidence](WELFARE_DECISION_EVIDENCE_SPRINT_2026_10_16.md), including [real-source CI](https://github.com/ob1-s/OpenFARM/actions/runs/38860418898).
- [Compiler contract](OPENFARM_QUERY_COMPILER_V0_2.md) and [candidate multi-question expansion](OPENFARM_MULTI_DECISION_TEMPORAL_EXPANSION.md).

## Next scientific falsification steps in executable order

1. **Cow clinical scoring beyond threshold questions:** Evaluate the *original ordinal gait (1–5) and BCS (1.5–5)* with proper ordinal distributions; compare age/parity/DIM-only versus breed versus added cow covariates, report per-farm reliability and abstention. Train/calibration/test groups always distinct; existing LOFO has been inspected so avoid sealed-test claims.
2. **PigWeb assays:** Run low-cost leave-one-institute-out classification with honest missingness and context-only shortcuts; if no robust incremental signal over site-safe baselines, retain its original measured assays as research outcomes but **do not promote** a predictive welfare indicator.
3. **Temporal lesion/treatment episodes:** Preserve first-treatment as a *recorded historical outcome*. Resolve 28-vs-27 original-source mismatch before restoring Swiss cohorts; never convert pseudo-dated control pen events into positives.
4. **Direct perception paired with welfare reference:** Require same-animal observation (actual video/pose/audio/sensor) and independent vet/health labels, explicit rights, independently held-out farm/animal/session. Prioritize freely accessible public material over author correspondence.
5. **Prospective Level-3:** Need observed caretaker review/response and meaningful outcomes, natural baseline prevalence, cost of false alerts per animal/pen-day, and no model features from after decision time. Continue to mark zero approved.

**Sprint outcome:** The *evidence* sprint achieved a bounded welfare-related clinical-risk task with site-heldout verification, preserved negative findings, and restored one limited care-event historical pilot. It did **not** discover a deployable empathetic welfare monitor or calibrated intervention policy. That is the correct scientific boundary.
