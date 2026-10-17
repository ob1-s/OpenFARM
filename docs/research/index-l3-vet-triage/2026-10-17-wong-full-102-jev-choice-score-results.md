# Wong 102-case full held-out Jev comparison — 17 October 2026 (actual API runs)

**Primary research completion: YES.** 408 actual Jev API calls, 102 original case-exclusive test examples × (1.13/1.14) × (Choice/Score). **All 408 valid and scored**. No clinical case text, full workbook, original individual decisions or credential values committed. Only aggregate results and source SHA pin.

Runs:
- Choice (204 calls): https://github.com/ob1-s/OpenFARM/actions/runs/39004705817 — `WONG_JEV_SUMMARY`.
- Ordered Score (204 calls): https://github.com/ob1-s/OpenFARM/actions/runs/39004818969 — `WONG_JEV_SCORE_SUMMARY`.
- Original private data carrier was provided by user through **GitHub Actions repository secret** `WONG_TEST_CASES_B64` (base64(gzip(JSON)) of source-pinned 102 case records and train-only veterinarian prior). Existing `TYPESAFE_API_KEY` provided API authorization. **Base64 is not encryption** but GitHub's secret store, not the encoding, protects the content. Scripts never print source text.

## Source / clinical reference

Wong original deposited clinical workbook `Case_data.xlsx`, 340 historical canine emergency hospital intake cases; 3 independent veterinarian urgency votes per case, 2 nurse opinions (one ambiguous), original median exactly agrees with 3-vet ordinal median on 340/340, SHA256 `4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027`. Original expert reference choices, most-to-least urgent: immediate; 15m; 30–60m; 120m; 240m. Dataset DOI `10.25910/rbk1-p833`, CC BY 4.0 metadata.

Frozen heldout is **102 case-exclusive test cases** (other 171 train and 67 calibration). No animal IDs; patient-disjoint or site-heldout generalization is **not proven**. Test contains **9 majority-immediate reference cases**, **17 unanimous cases**, and **85 cases with some vet disagreement**. Each example has three original veterinarian labels, total 306 vet labels in test.

## Headline API comparison (model outputs are full probability distributions)

| Jev version | Type | valid | exact 3-vet median | five-way Brier vs empirical 3-vet distribution ↓ | vote-weighted ordinal MAE ↓ | vote undertriage | vote overtriage | majority-immediate recognized | median API roundtrip |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.13.0 | Choice | 102/102 | 39/102 (38.24%) | 0.443236 | 0.892157 | 0.101307 | 0.578431 | 8/9 | 71.96 ms |
| 1.14.0 | Choice | 102/102 | 45/102 (44.12%) | 0.380465 | 0.816993 | **0.098039** | 0.526144 | 8/9 | 81.50 ms |
| 1.13.0 | Score | 102/102 | 39/102 (38.24%) | 0.430136 | 0.885621 | **0.091503** | 0.562092 | 8/9 | 73.01 ms |
| **1.14.0** | **Score** | **102/102** | **51/102 (50.00%)** | **0.359701** | **0.738562** | **0.114379** | **0.473856** | **8/9** | 88.78 ms |

**Critical interpretation:** Jev 1.14 Score increases exact agreement and improves Brier and ordinal MAE compared with Jev 1.14 Choice, but its per-vet **undertriage** fraction is **higher** (~11.4% vs ~9.8%), whereas its overtriage fraction falls (~47.4% vs ~52.6%). Therefore optimizing accuracy/Brier alone is not sufficient for a high-stakes action task. Both have only nine majority-immediate test cases; a miss among the nine is not an acceptable deployment claim. These are *agreement differences from retrospectively recorded veterinarian judgments*, not observed patient harm or optimal intervention thresholds.

- Choice 1.14−1.13 Brier delta = −0.062772, case-bootstrap 95% CI [−0.082955, −0.043455] on 102 paired cases.
- Score 1.14−1.13 Brier delta = −0.070435, case-bootstrap 95% CI [−0.089103, −0.052586].
- **Score vs Choice is an exploratory combined primitive/rubric ablation**: question wording and ordered criteria changed together. It does not isolate a causal improvement from the API answer head.
- Repeated use of the same heldout cases to explore variants is a research development set, not a pristine blinded leaderboard. Future independent task references needed for final ranking.

## Strong cheap baselines

- **Training-only veterinarian-vote prior**, applied unchanged to each test state: Brier **0.340640**, exact median 36/102 = **35.29%**, vote undertriage **27.45%**, majority-immediate recall **0/9**.
- **Always-immediate**, per state: Brier **1.309368**, exact median **9/102**, vote undertriage **0%**, vote overtriage **86.93%**, majority-immediate **9/9**.
- The prior still achieves the best Brier, because individual human votes disagree and case-specific predictions from Jev remain imperfect. It also completely fails to identify immediate emergencies as its modal class. Always-immediate is a degenerate overly conservative policy. Calibration plus action-specific recall and resource costs are necessary jointly.
- The Brier used here is squared Euclidean distance of model five-way distribution to **the empirical 3-vet distribution**, not a discrete hard-label Brier against a single panel consensus; it has a zero floor for exactly matching the empirical vote vector. It does not establish medical outcome risk.
- Reported latency is **single-request GitHub-runner wall-clock including network overhead**, not isolated device model latency.

## Research consequences

1. **Wong is a real Level-3 expert *priority recommendation* evaluation candidate**, now with a working authenticated multi-model verifier and human-disagreement-aware metrics. It is not demonstrated safe for real clinical workflow or animal owner advice.
2. There is learnable headroom: train-only constant prior remains best under five-way Brier while Jev carries case-specific emergency sensitivity. Domain-specialist decision heads should target **probability quality without discarding immediate-care discrimination**.
3. The native ordered Score parameterization is promising but needs **same-rubric fixed wording** ablations on independent tasks. Undertriage became worse for 1.14 Score vs Choice.
4. Train/call baseline comparisons must be honest about available state and context. Future learned model baselines must use only 171 train cases, 67 calibration, 102 case-heldout; given 340 total cases and no true patient IDs, specialized tuning claims will be data-limited.
5. A follow-up to original authors for verified privacy/patient linkage and external-site independently collected canine cases would add considerable value.
6. Small-owner-facing Farrow, 30 unique cases, complements Wong with an entirely different **pre-hospital seek-care decision**; both reference expert judgment, not outcome-validated welfare benefit.

## Ready other data tracks (scientific status separately gated)

- **Already approved narrowly for research training/eval** according to `docs/research/APPROVED_TASKS_v0.md`: HotPig instrument-feeder choice/binary task (550,818 valid minute rows, 24 pigs, 4 test pigs), cattle wearable MI → observed 4-way behaviors (9,222 rows, 24 cows; small animal-disjoint test), Meerkat box+motion → 15 observed ethogram behaviors (83,763 train/cal/test sampled label rows across 20 videos, 4 test videos). These are L1 observation, not welfare/clinical action.
- **Prospective benchmark PR candidates**: Big Broiler geometry-based observed feather pecking (29,459 scorable 10-second windows, 92 modeled videos; original MP4s in 32.5GB archive but NO raw video benchmark yet), GrimACE (773 labeled full mouse frames, 215 test frames from 4 mice), Aarhus pig future-tail-injury onset (32,995 windows, 15 heldout independent event onsets), Wong/Farrow expert urgency. Their individual tasks require their own specific source/rights/split review before formal APPROVED_RESEARCH_EVAL. **No causal Level 1→2→3 transfer hypothesis is imposed**.

**Next priority:** admit/source-pin full Wong + Farrow eval tasks as bounded research, freeze this exploratory report, train a small calibrated cheap classifier on 171 Wong cases with 67 calibration cases (no test tuning) or expand independent veterinarian-labeled training; then video-native Big Broiler ingestion if 32.5GB media can be verified under upstream rights. Avoid declaring headroom equivalent to deployment proof.
