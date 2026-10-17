# Operator D — live Jev Farrow triage benchmark, 17 October 2026

**Status:** REAL SOURCE / EXPERT CONSENSUS VERIFIED; REAL MODEL API EVALUATION EXECUTED. Exploratory 30-case benchmark, **not deployment validation**, not a protected untouched benchmark after repeated evaluations.

## Reproducible upstream data and target

- Farrow et al. PLOS One DOI `10.1371/journal.pone.0339723`; public original S1 clinical DOCX https://ndownloader.figshare.com/files/61100344 (MD5 `677ba68a3926088d6ac4da52f3010994`); public owner-answer XLSX https://ndownloader.figshare.com/files/60375362 (MD5 `f7816195a5e6c5a379862988cf0d1522`).
- `scripts/index_l3_farrow_builder.py` downloads/validates both originals, extracts **30** independent cases from DOCX S3 Table and **30/30** unambiguous veterinarian-panel consensus urgency labels from survey column `2.SCORE`. Every owner-response alternative strictly obeys `More urgent`/ `Same urgency assessment` / `Less urgent` relative to that label; **0 ordinal contradictions**. 5,316 owner responses, 48 coded missing, 1,497 less-urgent than vet consensus replicated.
- Six-urgency distribution: within 1h **5**, within 12h **10**, within 24h **6**, within 7d **9**, within 14d **0**, self-care **0**. The 14 individual vet vote vectors were **not released/verified**; only original author-coded consensus is supervised. Contrast with Wong (340 real emergency cases and all 3 vet votes).
- All model input uses only source `Dog signalment` and `Vignette` text. Condition names, prevalence, panel votes and owners' answers are excluded. No respondent IDs or original vignette texts committed/logged.

## Actual Jev API runs

Source: original TypeSafe `POST https://api.typesafe.ai/v1/systemone`, GitHub secret `TYPESAFE_API_KEY`; versions explicitly pinned. Official source run [38998318726](https://github.com/ob1-s/OpenFARM/actions/runs/38998318726) contains all four arms (30/30 valid each), aggregate metrics and source-free per-case probability vectors as short-lived artifacts.

Representative all-30 **single run** from source run 38998318726:

| Model | Primitive / rubric | n valid | exact panel consensus | undertriage | overtriage | 6-class Brier ↓ | Ordinal MAE ↓ |
|---|---|---:|---:|---:|---:|---:|---:|
| jev-1.13.0 | Choice, six named options | 30/30 | 11/30 (36.7%) | 7/30 | 12/30 | 0.9543 | 0.70 |
| jev-1.14.0 | Choice, same exact question | 30/30 | 10/30 (33.3%) | 5/30 | 15/30 | 0.8606 | 0.73 |
| jev-1.13.0 | Score, *ordered* 6-level descriptions | 30/30 | 12/30 (40.0%) | 8/30 | 10/30 | 0.7347 | 0.67 |
| jev-1.14.0 | Score, same ordered descriptions | 30/30 | 14/30 (46.7%) | 5/30 | 11/30 | **0.7022** | **0.60** |

**Important:** Score and Choice ablations also vary instructions/criteria wording, not *just* the backend answer type; cannot causally attribute gains to the primitive. The same 30 cases were scored repeatedly, which may compromise a pristine heldout protocol; these are **exploratory comparisons**. Repeat full runs yielded Jev 1.14 Score 14/30 exact with Brier **0.691–0.702**, Jev 1.13 Score 12–13/30 with Brier **0.725–0.735**; Choice 1.14 9–11/30, Brier ~0.858–0.868; Choice 1.13 10–11/30, Brier ~0.940–0.954. Both mean probabilities and some argmax choices vary between calls. Preserve replication variation in reports; do not cherry-pick best runs.
 
Reference calibration:
- Uniform 6-class prior Brier = **0.83333**.
- In-sample gold-frequency prior Brier = **0.73111**, but this exploits the 30 benchmark gold labels and is NOT a legitimate train-only fitted baseline; leave-one-case-out class prior Brier **0.7824** still uses the other eval labels. Use them as analytic references only.
- Thus both Jev **Choice** arms fail to outperform the *uniform* baseline on Brier; **Score** arms beat uniform and perform near the in-sample label-frequency reference. This is meaningful for proof of capability but weak on n=30.

**Matched example-level Jev 1.14 vs 1.13 Choice:** 38998318726 Brier delta −0.0937; 10,000 case-bootstrap 95% interval [−0.1452, −0.0391]; 30 matched cases (case-only bootstrap, not clinic-level external CI), exact-match accuracy not consistently improved.

### Risk-sensitive original human-care target

Script `scripts/index_l3_farrow_urgency_risk.py` checks all 30 case-level outputs, not just aggregate undertriage. On run 38998318726, the reference has just **five** genuinely *within-one-hour* cases:
- Jev 1.13 Choice recognized **5/5**, Jev 1.14 Choice **5/5**, Jev 1.13 Score **4/5**, Jev 1.14 Score **5/5**.
- No model assigned any of the **15 within-12h** reference cases to a 7-day-or-later class in this run. The remaining undertriages largely reflected under-escalation by adjacent urgency bins (not necessarily harmless), while overtriages impose owner/clinic resource burdens. Report category-specific care delays rather than calling all errors equally dangerous.
- There are too few true critical cases to claim near-perfect emergency sensitivity. This is a retrospective expert reference, not observed adverse outcome under deferred care.

**API response integrity:** earlier two long runs failed on 16th or 12th response due strict Choice probability-envelope mismatch. Subsequent full 30-case runs on each model completed with no invalid responses. Builder and runner preserve malformed-response accounting, never conceal missing calls.

**Rough model pricing:** direct provider-published rate is $0.042 per **million input tokens** (zero output charge, subject to account/price changes). Choice arm consumed **15,901** input tokens for 30 cases, nominal ~$0.000668 model input cost; GH runner/network overhead and retry usage excluded.

## Paths and limitations

- Source builder: `scripts/index_l3_farrow_builder.py`.
- Original Jev Choice runner: `scripts/index_l3_farrow_jev_retry.py`; pins `FARROW_JEV_MODEL`, reports coverage, keeps per-vignette probability vectors (no clinical text) in CI artifacts.
- Score ablation: `scripts/index_l3_farrow_jev_score_ablation.py`, pins `FARROW_SCORE_MODEL`, same 30 observations.
- Clinical safety analysis `scripts/index_l3_farrow_urgency_risk.py`; paired uncertainty `scripts/index_l3_farrow_jev_pairwise.py`.
- Source audit + runner workflow `.github/workflows/openfarm-l3-vet-data-access.yml`.
- Historical GPT baselines are not controlled apples-to-apples comparisons here, and any new JEv best-result claims require an independent prospectively curated set.

## Wong — next high-value gate is a single private GitHub Actions dataset secret

- Script `scripts/index_l3_wong_jev_private_runner.py` and workflow `.github/workflows/openfarm-l3-wong-private-jev.yml` are already committed. They compare pinned Jev 1.13 and 1.14 on **102 case-exclusive test cases**, with all original 3-vet reference distributions, a train-vote-prior baseline, always-immediate baseline, under/overtriage, and bootstrap of paired Brier.
- The actual 340-case original Excel was supplied by the user but never put into GitHub. The *prepared 102-case-only* compressed original intake-state + three-vet reference payload is **31,053 bytes** as base64 text, verified locally; it can fit into one GitHub repository secret `WONG_TEST_CASES_B64` (value prepared only in local conversation artifact).
- GH Actions with `TYPESAFE_API_KEY` alone succeeded but printed `WONG_SOURCE_SECRET_MISSING` and did **zero Wong inference**: [run 38998247922](https://github.com/ob1-s/OpenFARM/actions/runs/38998247922). This is a fail-closed guard, not benchmark completion.
- Do **not** paste this base64 clinical data into a public GitHub commit, issue, or workflow log. Once user uploads it as private repo secret, rerun workflow to obtain real Wong results.

**No device/owner health advice, regulatory or commercial approval, or causally beneficial caregiver action inferred from these retrospective human-label agreements.**
