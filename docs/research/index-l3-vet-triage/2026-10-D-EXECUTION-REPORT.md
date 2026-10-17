# Operator D execution report — direct Level-3 expert veterinary urgency (16 October 2026)

**Role:** undertaken in ChatGPT, not delegated. **Branch:** `research/index-l3-veterinary-triage-2026-10`, draft PR #8. **Outcome:** runnable **source ingestion + clinician-disagreement/evaluation tooling** with 16 synthetic offline tests; **zero original veterinary patient cases actually retrieved**, hence **zero Level-3 benchmark admissions**. This is progress on engineering and source verification, not a completed empirical benchmark.

## 1. What I verified independently from original publications

### Wong et al. — preferred case source

- [Veterinary Record DOI 10.1002/vetr.6126](https://doi.org/10.1002/vetr.6126) (online first published **4 December 2025**; in 2026 issue). 340 canine emergency-hospital retrospective patient presentations, **five clinical system strata of 68**, triaged from clinical signs and history by **3 veterinarians + 2 nurses** (plus 2 historical LLMs). Five categories: red/immediate (0m), orange (15m), yellow (30–60m), green (120m), blue (240m).
- Publisher explicitly says full **cases and triages** are deposited under **https://doi.org/10.25910/rbk1-p833**. This claim is author-provided, NOT a verified source-file listing: attempts to open DOI and DataCite endpoint in this environment fail; direct network access in container fails DNS. The publisher also lists a 23.3KB `vetr6126-sup-0001-SuppMat.docx`, which could be a separate supplemental file, but its bytes could not be read here.
- Dataset rights (actual deposited cases) remain **unverified**. Do not assume paper open access permits copying anonymized clinical data.
- **Scientific distinction:** these are real emergency cases, but the question is how quickly an **already-presenting hospital patient should be seen**, not when an owner should initiate first contact. Three independent clinical opinions are not proof of correct/optimal timing.

### Farrow et al. — owner-facing external seed

- [PLOS One DOI 10.1371/journal.pone.0339723](https://doi.org/10.1371/journal.pone.0339723) published **16 January 2026** (CC BY 4.0). 30 distinct approximately 50-word vignettes **constructed from anonymized real VetCompass clinical histories**, not 30 naturally collected owner messages. Reviewed by **14 RCVS veterinary surgeons** from 7 practices for likely condition and recommended first-care timeframe. Urgency was chosen by majority, conservative earliest timeframe if votes tied. The 71% (10/14) threshold in paper was for **condition recognition**, not a required unanimity on urgency. Four ambiguous vignettes underwent revisions.
- Source appendix `pone.0339723.s001.docx` (~337KB) includes all 30 vignette texts and survey; PLOS link https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0339723.s001&type=supplementary . The web crawler follows a signed Google Storage redirect that was inaccessible in this environment, and container has no external DNS. **No original vignette/individual expert vote matrix was downloaded**; do not reconstruct 30 gold labels from article condition tables.
- Paper reports **1,772 distinct owner participants × three scenarios = 5,316 responses**, but these are only **30 independent physician/veterinary-validated clinical decision prompts**. 28.4% owner responses assigned too low urgency vs vet consensus (source-level statistic, not model result).
- Public supplementary owner survey https://doi.org/10.6084/m9.figshare.30884855 is likewise inaccessible from this environment. Cannot know whether each raw 14-vet urgency response is published; do not assert it.

### Bio-Vet original rules — machine-readable audit *but no patient outcome evidence*

- Verified original raw text via connected GitHub: https://github.com/Bio-Vet/pet-symptom-triage/blob/main/data/symptom-triage.json (source Git blob `ac552f073112024cb0673a508fd0843621da17df`).
- Exactly **27 unique sign rules**: 15 red/immediate, 12 orange/same-day. By applicable species: dog 10 red + 8 orange; cat 9+8; bird 7+4; reptile 7+5; rabbit/rodent 9+6; exotic 9+6. Claims of August 2026 veterinarian review are upstream author statements; not independently certified. Repo states data CC BY 4.0.
- This supports a **static interpretable safety-policy baseline**. **No original patient vignettes, negatives, measured benefits or independent clinician votes**. Never split these rules into faux training/test samples or count this as admitted Level 3.

## 2. Engineering delivered

### `scripts/index_l3_vet_build.py`

- Standard library only, accepts explicitly **human-inspected source mapping** from original case CSV/JSON into `case_id`, triage-time-only `state_fields`, independent `vet_votes`, separate `nurse_votes` (if released).
- Case ID uniqueness and exact duplicate text hard fails; SHA256 of actual source bytes required and checked; unknown urgency values error rather than guessed; missing votes preserved; individual vet vote distribution and tied votes retained, no majority vote tie-breaking invented.
- Fails if an input state column is named as an obvious future diagnosis/treatment/outcome/triage vote. **Not an NLP-level guarantee**: if post-event information was in a plausible-looking free-text history, humans must inspect actual cases and remove that field/phrase.
- Emits local-only `cases.json` and `audit.json` (contains patient data; **do not git commit**).

### `scripts/index_l3_vet_evaluate.py`

- Proper **case-level**, rater-distribution-aware multiclass Brier, vote-weighted ordinal MAE, fraction of vet opinions potentially undertriaged or overtriaged, sensitivity on cases where >50% of vets say immediate. Explicit zero-undertriage/maximum-overtriage **always-red** reference baseline.
- Requires five normalized probabilities per case, correct joins and valid sum-to-one. No evaluation on unreleased cases; only synthetic examples tested.
- These metrics are agreement against **retrospective expert vote preferences**, not actual mortality, prospective outcomes, causal intervention value, treatment/timing policy safety or deployment validation.

### Tests

- `tests/test_index_l3_vet_build.py` (10 synthetic tests) and `tests/test_index_l3_vet_evaluate.py` (6 synthetic tests).
- **16/16 passed locally** via `python -m unittest discover -s tests -p 'test_index_l3_vet*.py'` in isolated test folder, no external data, no GPU, no API costs.
- Smoke checks include: source checksum, no future-diagnosis state, duplicate patients/identical case text, preserved missing/tied vet votes, no fabricated negatives, alerting all-red undertriages zero but overtriages, strict prediction joins.

### Reproduce once **real source files are obtained**

1. Obtain/inspect **actual** Wong deposited case file by visiting https://doi.org/10.25910/rbk1-p833 (source author provided), or Farrow supplementary via:
   ```bash
   curl -fL 'https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0339723.s001&type=supplementary' -o farrow_s1.docx
   ```
   NOTE: source DOCX is not in the normalized parser format. Inspect its tables/text locally and convert to case-level CSV/JSON **without manufacturing missing votes**; record conversion provenance and bytes SHA256. E.g. `sha256sum source.csv`.
2. Create **actual source-specific mapping.json only after seeing original columns**; never treat the following schematic as a representation of the original:
   ```json
   {
     "case_id": "REAL_CASE_IDENTIFIER",
     "state_fields": ["PRESENTING_SIGNS_COLUMN", "TRIAGE_AVAILABLE_HISTORY_COLUMN"],
     "vet_votes": ["ORIGINAL_VET1_COLUMN", "ORIGINAL_VET2_COLUMN", "ORIGINAL_VET3_COLUMN"],
     "nurse_votes": ["ORIGINAL_NURSE1_COLUMN", "ORIGINAL_NURSE2_COLUMN"],
     "source_doi": "10.25910/rbk1-p833",
     "source_sha256": "REPLACE_WITH_REAL_64_CHAR_SHA256",
     "expected_cases": 340
   }
   ```
3. Run `python scripts/index_l3_vet_build.py --source <real-normalized-cases.csv> --mapping <source-specific-mapping.json> --out-dir data/index_l3_vet/...` . Store source/derived files locally, not in repo.
4. Run `python scripts/index_l3_vet_evaluate.py --cases data/index_l3_vet/.../cases.json` to evaluate **always-red** safety baseline. Supply `--predictions predictions.json` only after evaluation scope, untouched patient splits, and rights approvals are established.
5. Confirm the exact 340 case texts, independent rater columns, per-body system counts, any inappropriate future diagnosis/after-treatment text, deidentification, file license, case duplicates and adjudication protocol. If no genuine independent vet opinions were released, **stop**.

## 3. Scientific admission gates unresolved

| Gate | Wong | Farrow | Bio-Vet |
|---|---|---|---|
| Primary paper and protocol | VERIFIED | VERIFIED | Self-published rules verified |
| Case-level original bytes retrieved | **NO** | **NO** | Yes, but rules only |
| Label-to-case alignment | NO | NO | No cases |
| Independent vet votes in publicly released source | Author reports votes; **unverified file schema** | 14 raters in paper, **unverified public raw vote vector** | No |
| Rights for derivative ML training/eval of individual records | UNKNOWN | Source PLOS CC BY; supplementary data scope to confirm | Source JSON CC BY 4.0 |
| Label validity/action construct | Retrospective hospital queue priority (potential L3) | Retrospective owner first-contact recommendations (potential L3) | Static rule knowledge only |
| **Index status** | `DISCOVERED / PAPER_DATA_DEPOSIT_CLAIM_ONLY` | `DISCOVERED / SUPPLEMENT_ACCESS_BLOCKED_HERE` | `POLICY_BASELINE_ONLY` |
| `TASK_VALIDATED` / research eval approved | **NO / NO** | **NO / NO** | **NO / NO** |

**Next scientific bottleneck is obtaining original source bytes**. Code scaffolding now exists and passed tests but it is explicitly NOT evidence that the 340 cases are actually accessible/usable. Do not add artificial urgency vignettes or score synthetic records. This lane can stop unless the DOI files can be retrieved from a normal network or author correspondence yields the actual data.
