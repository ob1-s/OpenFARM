# Operator D original file access — **VERIFIED** (17 October 2026 UTC)

Status: **The two original Farrow file downloads succeeded on a networked OpenFARM GitHub Actions worker, and both archives were parsed. Wong is a registered CC BY XLSX dataset, but its original repository response was HTTP 403 on this worker.** This updates and supersedes the earlier speculative/no-access parts of the Operator D 16 October handoff and execution report. **No Level-3 benchmark is admitted.**

## Networked verification

- GitHub Actions run **38991777958**: https://github.com/ob1-s/OpenFARM/actions/runs/38991777958 — initial proof of the original file bytes.
- GitHub Actions run **38991920599**: https://github.com/ob1-s/OpenFARM/actions/runs/38991920599 — original Farrow DOCX opened as a real Office ZIP with internal tables, author Figshare identifiers listed, Wong DataCite DOI and CC BY status confirmed.
- GitHub Actions run **38992000172**: https://github.com/ob1-s/OpenFARM/actions/runs/38992000172 — Farrow DOCX full table header and complete 2.2MB owner XLSX column/sheet audit.

### Farrow 2026 — both originals actually downloadable

Official PLOS supplementary material, Figshare article **31090093**:
- Official repository https://plos.figshare.com/articles/journal_contribution/_p_Study_advertisement_poster_full_survey_text_clinical_vignettes_example_accuracy_scoring_responses_per_vignette_univariable_analyses_p_/31090093
- API metadata https://api.figshare.com/v2/articles/31090093
- **Verified original download** https://ndownloader.figshare.com/files/61100344
- File `pone.0339723.s001.docx`, 345,381 bytes, MD5 **677ba68a3926088d6ac4da52f3010994**, HTTP 200 from two separate upstream servers (Figshare S3 and PLOS Google storage), structurally verified Word DOCX.
- S3 Table (second numbered body table) has **31 rows**: 30 original cases + header, with columns `Vignette #`, `Condition`, `Condition prevalence(%)`, `Presentation type`, `Dog signalment`, `Vignette`. The condition/presentation columns are **not allowable model input**, but the original `Vignette` and dog signalment text are.
- Other DOCX tables include 3 owner survey examples, answer-accuracy rubric, per-vignette response counts and study statistics. **No separate original 14-veterinarian raw-vote matrix was verified**. The DOCX's example **urgency answer options** are questions, NOT expert urgency gold labels.

Original companion owner response survey, Figshare article **30884855**:
- API https://api.figshare.com/v2/articles/30884855
- **Verified download** https://ndownloader.figshare.com/files/60375362
- File `251215 DogWeb Dataset Open Access.xlsx`, **2,228,252 bytes**, MD5 **f7816195a5e6c5a379862988cf0d1522**, SHA256 **4b246b6c9f37254eaee7c47dc415318ff706b71db5554a0fd84a31d990eba12c**, HTTP 200, actual Excel parsed with `openpyxl`. Sheet `Data sheet`: **5317 rows (1 header + 5316 owner responses), 95 columns**.
- Columns start `VIGNETTE/ RESPONDENT`, `Respondent ID`, `VIGNETTE`, `1.CONDITION`, `1.SCORE`, `2.URGENCY`, `2.SCORE`, `2.BinaryScore` and continue through demographic/information-source questions. **`2.URGENCY` records the OWNER's answer, not an original veterinarian answer**. `2.SCORE` and `2.BinaryScore` are author-derived scoring variables; their relationship to the 14-vet consensus must be independently proven before trying to infer the reference.
- **Do not treat 5,316 owner answers as 5,316 independent veterinarian-approved benchmark states**: they all repeat 30 case texts. Deduplicate on vignette ID; 30 unique cases are an upper bound for independent expert-scored tasks.

### Wong 2024 deposit / 2025 online / 2026 issue — genuinely registered original spreadsheet, repository blocked here

- Paper https://doi.org/10.1002/vetr.6126 ; authored open-data declaration `Full dataset of the cases and triages are available as Supporting Information data`.
- Data DOI https://doi.org/10.25910/rbk1-p833 is **actually registered** in DataCite: HTTP **200** on API `https://api.datacite.org/dois/10.25910/rbk1-p833`; exact registered title **Use of Artificial Intelligence Models for Veterinary Triage**; rights **Creative Commons Attribution 4.0 International**; actual destination **https://ses.library.usyd.edu.au/handle/2123/32901**.
- FAO AGRIS https://agris.fao.org/search/en/providers/122674/records/674841787625988a37198e21 independently catalogues the dataset as `application/vnd.openxmlformats-officedocument.spreadsheetml.sheet` (XLSX), CC BY 4.0; linked to University of Sydney.
- **Access blocker is at the Sydney repository, not DOI registration:** Github Actions DOI redirect reaches official `/handle/2123/32901`, returns HTTP **403**. Wiley's separate publisher supplementary file was also HTTP 403 from this runner. **Actual 340-case XLSX contents/schema have NOT been read.** A user browser (not bot) may succeed. If not, ask repository or corresponding author for the already-published CC BY spreadsheet.

## Exact local acquisition commands

```bash
mkdir -p openfarm-vet-source && cd openfarm-vet-source
curl -fL --retry 2 'https://ndownloader.figshare.com/files/61100344' -o pone.0339723.s001.docx
curl -fL --retry 2 'https://ndownloader.figshare.com/files/60375362' -o farrow-owner-answers.xlsx
md5sum pone.0339723.s001.docx farrow-owner-answers.xlsx
# Expected respectively:
# 677ba68a3926088d6ac4da52f3010994
# f7816195a5e6c5a379862988cf0d1522
```

Wong browser URL: https://ses.library.usyd.edu.au/handle/2123/32901
If that page is blocked, original authors' University of Sydney dataset record is at https://doi.org/10.25910/rbk1-p833 and the article lists corresponding author **Arlene Wong**, `awon9610@uni.sydney.edu.au`. A request for a direct alternate download of the **already-published** XLSX is appropriate.

## Scientific next gate / no manufactured admission

1. Farrow DOCX: parse S3 row fields into 30 `vignette_id, dog_signalment, vignette_text` rows; strip explicit diagnosis, prevalence and source-condition labels from model-visible state; preserve medical sign text intact.
2. Farrow XLSX: inspect (owner `2.URGENCY`, derived `2.SCORE`, `2.BinaryScore`) aggregated by vignette and response option. **Only reconstruct veterinarian consensus urgency if scoring methodology uniquely establishes it and there are no equivocal same-score choices.** Original individual 14-vet vote vectors may not be publicly published. If unavailable, retain 30 vignettes as *input-only* until expert reference obtained.
3. Wong: obtain actual source XLSX; check 340 cases, individual 3 veterinarian + 2 nurse opinions, triage-time-only inputs, data rights and model-output leakage.
4. No model training or claims based merely on successful downloads. No original respondent IDs/content or patient data were committed or logged in the workflow; temporary networked bytes were discarded after safe aggregate/structural verification.

**Current verdict:** `FARROW_SOURCE_BYTES_VERIFIED`, `WONG_DATASET_METADATA_VERIFIED_BUT_REMOTE_403`, `ZERO_TASKS_ADMITTED`.
