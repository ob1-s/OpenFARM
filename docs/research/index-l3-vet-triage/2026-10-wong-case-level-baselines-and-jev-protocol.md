# Wong L3 clinical triage — first reproducible real-case baseline

Date: 16 October 2026 (America/Sao_Paulo). **Evidence level:** original source bytes + 3 independent veterinarian reference votes + reproducible case-level baseline, executed locally. **Source contains no animal IDs and is not an approved prospective/deployment benchmark.**

## Source
- Original user-supplied University of Sydney study XLSX, "Case_data.xlsx"; SHA256 `4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027`, nine exact author columns, 340 case rows.
- All 1,020 veterinarian ratings nonmissing; all 340 median ratings equal ordinal median of 3 clinicians; one ambiguous nurse-2 response `120 mins, 240 mins` masked for single-choice human comparators.
- 58/340 unanimous 3-vet votes, 282/340 disagree. Median distribution: 30 Immediate, 46 15mins, 121 30–60mins, 106 120mins, 37 240mins.
- Assessment is **after hospital presentation** and includes intake vitals. It is not owner-first-contact triage. No date metadata in the model state; one inline `Date:` field initially caught and fixed through offline tests. No source date/treatment information is copied into model input except clinically relevant *prior* history.
- Qualitative leakage spot audit: text occurrences of diagnosis (2), treatment (6), and triage (1) refer to owner-reported **past history**, not source future reference labels. Nonetheless the source is retrospective clinical documentation; no prospective deployment claim.
- Source data do **not** provide canonical underlying animal ID; case-exclusive splitting cannot guarantee animal-disjointness. Heldout duplicates: zero exact duplicate source texts or date-scrubbed states; one quick lexical Jaccard scan found zero pairs over 0.75. This is not an exhaustive near-duplicate / repeated-patient proof.

## Fixed research evaluation protocol

- **Typed output**: five-category `Choice` probability distribution: Immediate, 15 mins, 30–60 mins, 120 mins, 240 mins, ordered from highest to lowest urgency. *Human reference:* distribution of 3 independent vet votes per case; additional median reference only for readily interpretable exact accuracy.
- **Frozen case-stratified SHA split** (not model-score tuned): train 171 cases, calibration 67, test 102. Per median class train/cal/test:
  - Immediate: 15 / 6 / 9
  - 15 min: 23 / 9 / 14
  - 30–60 min: 61 / 24 / 36
  - 120 min: 53 / 21 / 32
  - 240 min: 19 / 7 / 11
- Source-state split digest `d94288a6cd6d43b4cfebce43c4e3aeaa5ce4eaeb87f5ab13a7e971edda63c00c`. Splits by case ID = SHA256(original clinical text), not by individual vet vote.
- **Primary probability metric:** unweighted five-class Brier against empirical 3-vet vote frequencies (original ordinal undertriage remains separate). Also mean ordinal absolute distance to each vet, vet-vote-weighted undertriage/overtriage, exact vs median; do not equate human disagreement with objective medical error. Approx nine median-immediate test cases => high variance on emergency sensitivity.
- **API contract** from official https://docs.typesafe.ai/api : POST https://api.typesafe.ai/v1/systemone , pinned `jev-1.13.0`, state = date-scrubbed original case, single `choice` with five descriptive criteria; capture full probability vector, model ID, latency and tokens. Read `JEV_API_KEY` environment variable only; no public secret. **NO actual Jev inference here** (no valid key or network in session). An offline dry-run verified the shape/102 intended test units and no API calls.
- Archived GPT3.5/GPT4 study outputs are deterministic *labels only*, not native model probabilities. Report one-hot Brier for reference only, not a fair probability-calibration comparison to Jev. Prompting/model vintage also differs.

## Cheap baselines, run on actual 340-row source

All source rows:
| model | n cases | exact-vet-median | 3-vet Brier | vet-vote weighted undertriage |
|---|---:|---:|---:|---:|
| Always immediate | 340 | 0.0882 | 1.346 | 0.000 |
| Empirical veterinarian vote prior fitted on 171-case train | 340 | 0.356 | 0.348 | 0.266 |
| Archived source GPT-3.5 categorical | 340 | 0.365 | 0.905 | 0.136 |
| Archived source GPT-4 categorical | 340 | 0.335 | 0.959 | 0.077 |

Frozen 102-case heldout test:
| model | n cases | exact-vet-median | 3-vet Brier | vet-vote weighted undertriage |
|---|---:|---:|---:|---:|
| Always immediate | 102 | 0.0882 | 1.309 | 0.000 |
| Train-vet-vote prior | 102 | 0.353 | 0.341 | 0.275 |
| Archived source GPT-3.5 categorical | 102 | 0.412 | 0.826 | 0.141 |
| Archived source GPT-4 categorical | 102 | 0.324 | 0.950 | 0.108 |

The policy that never undertriages by expert vote always sends all 340 to immediate care; excessive overtriage makes this unusable in practice. The class prior has relatively good Brier because it expresses uncertainty while categorical archived predictions are one-hot; this is the first demonstration of why the Index needs probability output and rater uncertainty.

## Correct status

`SOURCE_VERIFIED`, `TASK_VALIDATED` and `REAL_BASELINES_EXECUTED`. **Not** `APPROVED_DEPLOYMENT_VALIDATION`, and **model-vs-Jev evaluation still pending authenticated call**. Research evaluation may be meaningful with case-disjoint comparisons and original 340 cases, but the dataset does not prove new-animal/new-site generalization. Code: self-contained standard-library local benchmark archive supplied to user (no original case texts in public PR). This research note is not medical advice.

## Farrow owner-facing reference recovered too

On GitHub Action [38994641057](https://github.com/ob1-s/OpenFARM/actions/runs/38994641057), `scripts/index_l3_farrow_builder.py` fetched both original pinned PLOS/Figshare sources, recovered **30 of 30 unique veterinarian-consensus urgency categories** from the authors' `2.SCORE=Same urgency assessment` relative coding, checked each of six alternative urgency choices for monotonic agreement, reproduced **1497 lower-than-vet urgency owner answers** and exact 48 missing-coded responses. Source 30 cases plus vetted consensus stored in CI artifact (not 14 independent vet votes). Farrow classes 5 within1h, 10 within12h, 6 within24h, 9 within7d. It is **owner seek-care timing**, a distinct six-way construct that should NOT use Wong's five hospital-wait-time labels.
