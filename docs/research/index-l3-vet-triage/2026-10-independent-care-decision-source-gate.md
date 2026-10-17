# OpenFARM Decision Index — Direct Level-3 Veterinary Care Triage Discovery (16 Oct 2026)

**Status: PRIMARY PAPER/PROTOCOL + PUBLIC SOURCE MANIFEST verified, source dataset bytes NOT downloaded/read in this environment. NO benchmark admission, training, or model score claims.**

This source search is separate from pig-onset Operator A, GrimACE Operator B, and Big Broiler Operator C. Goal: real, independent recommendations about when a caretaker should obtain professional care (L3 action policy) and, separately, genuinely recorded interventions with timestamps. The difference is essential: **vet-recommended urgency** is a Level-3 *decision reference*; a logged staff treatment/removal is an *observed action*; neither is independently proven to optimize real animal outcomes.

## Ranked source findings

### 1. Wong et al., Veterinary Record 2026 — PRIMARY L3 CASE-BASED BENCHMARK CANDIDATE

- Primary article https://doi.org/10.1002/vetr.6126 ; publisher text https://bvajournals.onlinelibrary.wiley.com/doi/abs/10.1002/vetr.6126
- **Data Availability Statement explicitly names public supporting data** https://doi.org/10.25910/rbk1-p833 . As of this session **DOI landing/actual file bytes were not accessible through current web/container environment**; availability claim from publisher != verified downloaded records, so status `ACCESS_DECLARED_PUBLIC / BYTES_UNVERIFIED`.
- 340 real canine cases retrospectively selected from a private emergency referral hospital, balanced by **five clinical body-system categories** (68 cases each, paper method). Clinical signs and history provided as triage input. **Three independent veterinarians**, two veterinary nurses, and old ChatGPT-3.5 / GPT-4.0 judged each real case. Original full case+triage dataset reportedly shared (must prove with actual file inspection).
- Five original action timing **Choice/ordinal** categories: red immediate (0 min), orange 15 min, yellow 30–60 min, green 120 min, blue 240 min. Target = **expert judgment of safe target waiting time to see vet**, NOT intervention administered to the dog.
- Paper reports severe-emergency sensitivity 80–90% for earlier LLMs; (~60% overtriage nonurgent, not a calibration metric). These are study results, **not OpenFARM benchmark results**.
- **Acceptance critical path**: obtain real source file; inspect all patient vignette texts, each original independent veterinarian score and nurse score, case IDs, actual health-care timing if released. Verify whether model-output columns/diagnosis/future charting are present and **exclude them from state**. Determine if 3 independent veterinarian ratings survive or only consensus. Check body-system grouping, duplicates, deidentification/license; multiple rating rows for SAME case are not independent data points.
- Outcome evaluation: clinician rater distribution + LORO vet holdout; confusion of undertriage/overtriage by class/body system and cost, proper multiclass probabilities/Brier, selective abstention. Rater agreement and baseline 'always red' (undertriage=0 but unacceptable overtriage) essential. Split by patient case; optional body-system-heldout stress test. With 340 actual cases, meaningful pilot **if source verified**.
- **Main limitation**: 340 pre-presented emergency-hospital dogs do not represent pet owners deciding whether to visit at all; referral spectrum/site shift. No raw video/audio. No prospective clinical impact/action utility or measured harm from waiting. Reviewer uses explicit human triage protocols: potential input contamination via lists/prompts/diagnoses.

### 2. Farrow, O'Neill & Packer, PLOS ONE Jan 2026 — VALIDATED OWNER-FACING ACTION-URGENCY CANDIDATE

- Paper https://doi.org/10.1371/journal.pone.0339723 (CC BY 4.0).
- Original S1 supplementary **DOCX**: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0339723.s001&type=supplementary . Paper says S1 contains S3 Table of **30 clinical scenario texts** and original survey/rubric; file ~337KB. Public related owner-survey data https://doi.org/10.6084/m9.figshare.30884855.
- 30 distinct, roughly 50-word **constructed but real-record-derived** dog vignettes (from anonymised VetCompass original visits), validated by **14 RCVS registered general-practice veterinary surgeons from seven practices**, 3–41 years qualified. Each recorded condition identification and **recommended timeframe for seeking veterinary care**. Consensus urgency selected by majority of 14, conservative shortest-wait in ties. Initial diagnosis agreement gate 10/14; four vignettes revised. **Do not assume all 14 raw vote vectors were deposited**.
- 1,772 owners each saw three vignettes → **5,316 owner responses**, but **only 30 independent scenarios**, not 5,316 independent L3 expert outcomes. Owners underestimate vet-recommended urgency in 28.4% of responses (authors' estimate); a good human-behavior *undertriage reference*, not 5k supervised training rows.
- Need source S1 DOCX parsed and exact urgency choices / expert consensus for all 30 confirmed. Figshare owner data must be inspected for exact vignette ID ↔ answer key; do not infer from disorder name, or leak diagnosis into input.
- This target is more directly **caretaker-facing** (when to seek professional care) than hospital triage after admission, but tiny, standardized, clinical scenarios are not field patient observations. Useful independent 30-case **research evaluation seed only** and cross-construct contrast with Wong.

### 3. Bio-Vet pet-symptom-triage — ORIGINAL RULES ACCESS VERIFIED, NOT AN EVAL SET

- Live code/data https://github.com/Bio-Vet/pet-symptom-triage/blob/main/data/symptom-triage.json . Source blob SHA `ac552f073112024cb0673a508fd0843621da17df`, opened/parsed with GitHub connector on 16 Oct 2026.
- **27 unique source red/orange records**: 15 red (immediate vet) and 12 orange (same-day). No duplicate source IDs; dog has 10 red+8 orange applicable, cat 9+8, bird 7+4, reptile 7+5, rabbit/rodent 9+6, exotic 9+6. Repo's source description says these rules are veterinarian reviewed, last 16 Aug 2026, Moscow clinic network, license CC BY 4.0; this reviewer claim is upstream self-description, **not independently validated by us**.
- This is a machine-readable *expert operational policy/ontology* and safety check. It has **no actual patient-level cases, no ordinary negatives, no uncertainty, no counterfactual action outcomes**. Never call the 27 known urgent sign strings an independent AI evaluation or demonstrate intelligence by regurgitating them. No automatic "green" clinical safety guarantee for all other symptoms.

### 4. INRAE cow activity & health — SOURCE PUBLIC, NOT CARE-DECISION LABELS

- https://doi.org/10.15454/52J8YS ; French public Dataverse. Four underlying datasets, across **386 cows**, >2.35 million hourly cow records, real activity features (EAT/REST/IN_ALLEYS/ACTIVITY_LEVEL), multiple event flags. Dataset4 alone **300 cows / 2,177,208 cow-hours / ~1 year**, events include mastitis, lameness, other disease, accidents, and 'disturbance' (heterogeneous mild interventions such as injections, claw trimming, late feeding and handling). Dataset3 is only oestrus.
- Events recorded by DAY but repeated as hourly boolean—**never interpret as genuine hour-exact event/action timestamps**, nor count positive hourly rows as independent adverse events. 'Disturbance' collapses nonmedical actions and general management noise. Dataset1 includes **randomly assigned NSAID vs saline following experimental LPS injection**, not an independent optimal vet-recommended treatment.
- Public platform shows Etalab Open License 2.0, downloadable files (Dataset4 `.tab` ~150 MB, Dataset1 ~7.5 MB and `Table1` 391 bytes). But we **did not download/read originals**, so status public ACCESS_METADATA only. This is probably an excellent *real sensor -> later independent disease event* research target (L2→L3), NOT our requested pure L3 action policy.
- Potential next step (separate future lane): source-byte audit dataset1/2/4, true positive cow-days and first disease dates, leave-one-farm-out, chronological labels and censored event segmentation, validation against random-hour split papers. Do not distract veterinary triage Operator D.

### 5. Other important leads / reject traps

- `https://doi.org/10.1111/vec.70068`: VetTriS 2025 prospective **164 dog/cat emergency patients**, substantial rater agreement kappa=0.69 and undertriage 0.6% published. **Source row-level data access not established**, do not claim usable. Supporting/author outreach only after public sources.
- `https://doi.org/10.1002/vetr.6126` (Wong above) is stronger than the popular 'VetTriageBench-45' https://zenodo.org (search by title): this latter bank explicitly states **AI-assisted gold labels / expert veterinarian validation only planned for v2**. Reject as independent expert reference as currently published.
- `https://pmc.ncbi.nlm.nih.gov/articles/PMC7411665/` and https://pmc.ncbi.nlm.nih.gov/articles/PMC7922278/ describe authentic **recorded mastitis/lameness treatments** in cows with sensor histories, 167 animals and 80,307 cow-days in the earlier single-farm study; **underlying joined patient-level source bytes not shown public**, so cannot count as approved dataset.
- Full natural clinician actions and outcomes remain scientifically valuable even if retrospective consensus triage succeeds; do not conflate **action recommendation at emergency intake** with **preinjury alerting or deciding when to inspect a farm animal**.

## Decision Index proposed Level-3 *care action* task contract (not yet admitted)

```json
{
  "task": "vet_triage_wait_time_5way",
  "state": {
    "species": "dog",
    "presenting_signs": "<source case's triage-time visible clinical signs>",
    "history": "<source history available to triage assessor>"
  },
  "question": "Choose appropriate maximum waiting-time urgency for veterinary assessment.",
  "output": {
    "type": "Choice",
    "options": [
      "red_immediate",
      "orange_15min",
      "yellow_30to60min",
      "green_120min",
      "blue_240min"
    ],
    "allow_abstain": true
  },
  "reference": "3 actual veterinarian triage votes if present, otherwise explicitly identified consensus and its uncertainty",
  "unit": "unique patient encounter, NOT rater-vote row",
  "leakage": "exclude future diagnosis, treatment, hospital outcome, baseline model outputs and clinician assessments from state"
}
```

The JSON is an **illustrative task schema**, not an actual downloaded case or asserted file schema. Urgency definitions from paper must be verified against deposited case files.

## Concrete admission gate

1. Open the Wong underlying **DOI 10.25910/rbk1-p833**, enumerate actual files, download bytes and hash. Check license and ethics/deidentification. Confirm 340 unique cases and 3-vet rating vectors. If unavailable or only summary counts, stop; do not fake.
2. Parse the 30 Farrow clinical vignettes + consensus urgency from S1 supplementary DOCX, verify expert case/join. Source reader must NOT invent missing votes. Independent 30-case evaluation if original reference visible.
3. Build `Score/Choice` task with original five colors/time bins, preserve missing/contradictory raters, clean earliest-time available clinical features. Rank and report undertriage/overtriage, uncalibrated overtriage ('always red'), veterinary rater agreement and clinic-type shift. The independent evaluation is by **case**.
4. Obtain robust rights for each proposed use. A fully shared paper is not proof of full clinical raw-data redistribution license.
5. If sources pass, propose `TASK_VALIDATED` and `APPROVED_RESEARCH_EVAL` separately, not deployment or commercial release without validation.

**No actual source case-specific records or clinical vote vectors were read here. No model inference was run.**
