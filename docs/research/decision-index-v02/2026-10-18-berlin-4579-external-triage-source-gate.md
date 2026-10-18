# New independent veterinary L3 source gate — Berlin 4,579 visits (2026 paper)

**Date:** 18 October 2026. **Status:** `ORIGINAL_PAPER_VERIFIED; CASE_LEVEL_DATA_NOT_ACQUIRED; NOT_BENCHMARK_ADMITTED`

## Independent clinical cohort

- Wizenty, Leoni & Doherr, Marcus G. (2026), *Clinical Urgency of After-Hours Presentations of Dogs and Cats to a Private Small-Animal Practice in Germany: A Retrospective Study of 4579 Visits*, `Veterinary Sciences` **13(9):899**, DOI [10.3390/vetsci13090899](https://doi.org/10.3390/vetsci13090899). Published 2 September 2026.
- **4,579** visits from one Berlin private practice's year-2021 after-hours EHR, **3,189 dogs and 1,390 cats**; original 4,888 extracted, 219 non-dog/cat and 90 missing demographic details excluded.
- Five Manchester-inspired retrospective urgency colors **red** (immediate), **orange**, **yellow**, **green**, **blue**. Cats had 24.8% red, dogs 7.4% red *per article* (percentages cannot replace case-level raw labels). More than half of all visits were green/blue.
- Original medical records include presenting history/exam and later diagnosis/diagnostic/treatment/billing elements. **The reference urgency was determined by one researcher looking at the whole visit record, including final suspected diagnosis, later tests and treatment**; it is not blind prospective emergency priority and not comparable to Wong's 3-vet independently rated intake cases without a redesign.
- The anonymized study dataset does not preserve identifiers to reliably match repeated patients. One-visit-per-row is **not** animal-disjoint; 2021 visits from one site are **not** cross-site validation.
- Public study appears to expose aggregate tables, **not 4,579 original individual intake presentations**. Original authors/clinic hold rights and likely ethical duties to owner data; obtain affirmative source permission before research access or API upload.

## Proper task if case-level access is granted

A suitable new OpenFARM L3 task would ask **based only on the information demonstrably available to the triaging veterinarian before tests/treatment**: which 5-step urgency tier is warranted? The manuscript's single full-record retrospective grade can only be a provisional soft reference. For a strong new benchmark, blinded veterinarians should rate intake-only presentations independently, then be compared to adjudication and observed downstream outcomes (if recorded). Record actual timestamp boundaries in the compiler and never use clinician decisions, therapies, eventual diagnosis, discharge, or costs as input shortcuts.

Minimal evaluation: no overlap with Wong/Farrow; versioned source hash and site; group by privacy-preserving animal or visit IDs if authors can supply them; 3+ rater disagreement when possible; predeclare critical undertriage costs, calibration, abstention and per-species slices; clinician-reviewed consent/licensing for research, training and hosted API use separately. **Do not convert the paper's color percentages into synthetic case data.**

## Specific collaborator outreach draft (not sent)

Subject: Research collaboration on independent veterinary emergency triage evaluation (4579-visit Berlin study)

Dear Dr. Wizenty and Professor Doherr,

I'm developing OpenFARM, an open research evaluation program for compact, calibrated animal-welfare decision models. Your recent 4,579-visit after-hours triage study is particularly relevant because our existing retrospective canine emergency cohort has only 340 presentations and very few independent critical cases.

Would you be open to discussing restricted, privacy-preserving access to the de-identified case-level study dataset, or an independently expert-rated subset, for a noncommercial research benchmark? We are particularly interested in **intake-time-only** complaints/history/initial examinations, separated from later tests, treatments and diagnoses. I noticed that the published triage grades were assigned retrospectively from the full clinical record; we'd explicitly avoid claiming prospective decisions without blind intake-only clinical reference ratings. We would also respect any ethics, data-use, patient re-identification, licensing, commercial-use and third-party-model restrictions you specify.

A small controlled collaboration with aggregate-only published evaluation results would be valuable if original clinical text cannot be redistributed. I can share our current scientific task/split and calibration protocols for review. Please let me know if there is an appropriate access pathway or colleague to contact.

Sincerely,
OpenFARM research collaboration

**Do not send without selecting an actual verified contact address and reviewing the request. No data access implied.**

## Additional examined sources (distinct limitations)

- [Bio-Vet pet-symptom-triage](https://github.com/Bio-Vet/pet-symptom-triage): 27 veterinary-reviewed symptom rules under CC BY 4.0. Not thousands of independent original clinical cases; useful only as explicit rule/policy coverage.
- ACVECC Veterinary Committee on Trauma registry: thousands of authentic canine/feline trauma patient outcomes and triage statistics in publications, but individual de-identified record release has **not** been verified. Could be a restricted collaboration source for adverse outcomes, not a current public benchmark.
- [Issue #10](https://github.com/ob1-s/OpenFARM/issues/10) holds the high-priority independent L3 source-acquisition gate.

**Scientific priority:** independently sampled original case diversity and prospective temporal integrity over manufacturing more variants of Wong's 102 or Farrow's 30 reused tests.
