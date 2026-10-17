# Wong original 340-case Excel — byte/label audit (user-supplied source; 16 Oct 2026)

**Status: ORIGINAL CASE-LEVEL BYTES VERIFIED.** This report supersedes the former "data DOI registered but raw file inaccessible" finding in Operator D research. The original spreadsheet has been supplied to ChatGPT and read directly. **No raw clinical text, dates, veterinary votes per case, or original Excel file are committed to GitHub**. No model training or prospective care validation has been performed.

## Original source file
- User-uploaded `Case_data.xlsx` — **111,068 bytes**, SHA-256 `4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027`.
- One sheet `Data`, exactly **341 rows × 9 columns** = **340 real case rows + header**. All 340 `Cases` text records nonempty, all 340 **exact texts unique**, lengths 478–1481 characters.
- Column A: `Cases`, original textual observation available at emergency triage (presenting complaint, history, signalment, often date/time, vital signs). Source vet-note semantics mean this is **already-at-hospital veterinary triage**, not previsit owner assessment. It would be unsafe to expose date/time, outcome or diagnosis if correlated with triage labels; medical historical diagnosis is different from future diagnosis. No patient IDs in this workbook, so **unique cases ≠ proven unique animals**.
- Column B: `Veterinary triage - median vet` **reference only** (never model input).
- C/D/E: `Veterinary triage - vet 1/2/3` **independent reference votes**, all 1,020 present and parseable.
- F/G: `Nurse triage - nurse 1/2`, secondary human comparators, **1 anomalous cell** in nurse 2 row 303: `120 mins, 240 mins`; mark ambiguous/missing as a single-choice rating rather than silently coercing.
- H/I: `ChatGPT-3.5 triage`, `ChatGPT-4 triage`, historical **model comparators only**; never enter state or reference.

## Verified reference construct

Five ordinal categories, highest to lowest urgency: `Immediate`, `15 mins`, `30-60 mins`, `120 mins`, `240 mins`.

- All **340/340** provided `median vet` labels **exactly equal the ordinal median** of veterinarian columns C–E. Zero mismatches.
- **Only 58/340 cases (17.1%) are unanimous across the three vets.** 282/340 have some disagreement; max/min vote span 0 steps: 58 cases, 1 step: 162, 2 steps: 101, 3 steps: 19.
- Median-vet distribution (n=340): Immediate **30**, 15 mins **46**, 30–60 mins **121**, 120 mins **106**, 240 mins **37**.
- Vet1/vet2 exact agreement: **160/340** (47.1%). Vet1/vet3: **91/340** (26.8%). Vet2/vet3: **128/340** (37.6%). Treat the 3-vet empirical distribution as a first-class target and report agreement as a limit of the reference, not pure factual ground truth.
- No missing vet votes, no unknown vet categories, no exact duplicate `Cases` strings.

## Cheap real-source reference comparison (not a new model run)

Scored preexisting historical prediction columns vs median of 3 vets:
- ChatGPT-3.5: **124/340** exact (36.5%); **32/340** urgency undertriage; **184/340** urgency overtriage; 0.841 mean 5-rank absolute difference.
- ChatGPT-4: **114/340** exact (33.5%); **9/340** urgency undertriage; **217/340** urgency overtriage; 0.835 mean 5-rank absolute difference.
- These are historical source column outputs (models and prompting fixed by study); **NOT** re-run against contemporary GPT models. Always-immediate would give zero undertriage but overtriage on 310/340 median reference cases, so no responsible evaluation should report only sensitivity or raw accuracy.

## Scope and critical remaining gates

- **SOURCE_ACCESS_VERIFIED** and **ORIGINAL_EXPERT_REFERENCES_VERIFIED**. This qualifies as a strong `TASK_VALIDATED_CANDIDATE` for a research-only expert-veterinarian care-priority `Choice` task, **not** an approved autonomous intervention or actionable care policy.
- Build a source-specific XLSX-to-case transformer using the exact 9 columns; SHA pin; fail closed on unexpected category, duplicate texts or median disagreement, preserve ambiguous nurse vote as `not_rateable`.
- Model input strictly only column A (optionally scrub calendar dates and other retrospective recording artifacts after case audit); *never* supply columns B–I. Group by case; underlying unique patient identity absent, so patient-disjoint split cannot be asserted.
- Freeze deterministic train/cal/test split by case-text hash before model benchmarking, stratifying without inferring groups that do not exist. Include full 3-vet reference distribution and ordinal-consensus as alternative.
- Confirm **rights of the exact original file** and the correspondence between this SHA-pinned upload and the University of Sydney deposited file if feasible; DOI metadata says CC BY 4.0, but local upload alone does not certify it is exactly the deposited revision.
- Inspect 340 case prompts for explicit diagnosis/outcome/triage leakage beyond field names; observe subtype/breed/time variability. Exercise a low-cost label-independent prior and a clinician disagreement baseline.
- Do not reinterpret a 5-bin hospital **waiting time** target as choosing medications, treating an animal on a farm or advising a pet owner who has not contacted a veterinarian.

## Concrete proof of success

The previous research bottleneck **"Wong spreadsheet inaccessible" is resolved by the uploaded original, inspectable `Case_data.xlsx`**. `SOURCE_BYTES_VERIFIED` and `REFERENCE_VOTES_VERIFIED` now hold. Pending task builder/source rev/rights and leakage audit rather than accessibility.
