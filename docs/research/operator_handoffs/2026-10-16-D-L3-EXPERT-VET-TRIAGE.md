# OPERATOR D — Real Expert Care-Decision Benchmark (Level 3)

**Branch:** `research/index-l3-veterinary-triage-2026-10`
**Own paths:** `docs/research/index-l3-vet-triage/`, `scripts/index_l3_vet_*.py`, `tests/test_index_l3_vet_*.py`. Leave Operator A's pig `index-l3-care`, Operator B GrimACE, Operator C Big Broiler, and existing frozen benchmark/compiler alone.

## Mission
Unlike our old outcome forecasting tasks, produce at least one *originally expert-supervised decision about whether/how urgently an actual animal should receive care*. **Priority 1: Wong 2026 emergency hospital cases (340 real dogs × 3 veterinarians).** Priority 2: Farrow 2026 owner-facing vignettes (30 scenarios × 14 vets). Validate actual source data and train/eval rights first; build a typed research `Choice` (or source ordinal `Score`) with reference uncertainty and realistic harms from delayed vs unnecessary escalation. No generic 'feeling pain' → 'seek care' remapping.

## START HERE
Read our detailed source audit:
`docs/research/index-l3-vet-triage/2026-10-independent-care-decision-source-gate.md`.
And OpenFARM `AGENTS.md`, `DATA_APPROVAL_POLICY.md`, `WELFARE_EVIDENCE_SPRINT_DECISION_GATE.md`.

### Primary paper and source
- Real historical dogs, 340 cases, paper DOI https://doi.org/10.1002/vetr.6126, full patient/triage supporting dataset DOI https://doi.org/10.25910/rbk1-p833 (author asserts public; **NOT yet byte-verified**). Distinct triage ratings from three vets and two nurses plus old ChatGPT model judgments. First gate is actual file listing, byte access/checksum, record schema, exact 340 case uniqueness, per-clinician independent vote vectors, deidentification and data licensing. If the deposit is gated, log evidence, pivot promptly.
- 5-class wait-time urgency from human veterinarians: RED 0min, ORANGE 15min, YELLOW 30–60min, GREEN 120min, BLUE 240min. Vet disagreements should be preserved as targets or human ceiling, not collapsed silently. Hypothesis: a small calibrated typed head can triage with low undertriage and acceptable overtriage without autoregressive generation.

### Backup — caretaker first contact, not emergency-room prioritization
- 30 vetted (14 veterinarians) clinical vignettes, based on real VetCompass cases, https://doi.org/10.1371/journal.pone.0339723, full S1 https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0339723.s001&type=supplementary, public owner answers https://doi.org/10.6084/m9.figshare.30884855 .
- Note there are ONLY 30 independent vignette questions; 5,316 responses are from 1,772 owners presented with repetitions, not 5,316 different veterinarian labels. Verify actual scenario+vet-consensus action urgency exists as a paired source record; if no original votes released, report consensus uncertainty as unknown.
- Optional machine-readable policy reference: https://github.com/Bio-Vet/pet-symptom-triage/blob/main/data/symptom-triage.json (27 vet-reviewed signs, 15 red +12 orange, CC BY 4.0; source blob SHA `ac552f073112024cb0673a508fd0843621da17df`): **rule-only safety baseline**, not an independent eval case bank.

## Execution contract
1. Download ORIGINAL publicly released file; SHA256+source revision; inspect real individual cases, original clinical questions, reference votes and whether diagnoses/outcomes are AFTER triage time. Never fabricate examples; report 2 real anonymized field-mapped cases within redistribution rights.
2. Build provenance manifest: `case_id, source, presenting_signs, history, time_of_triage (if available), clinician_ref_id, clinician_rating, target_max_wait, clinical_body_system, source_revision, rights, missingness`. The literal file columns need not match; this is target schema.
3. Construct case-disjoint train/cal/test if source robust; prefer an independent untouched evaluation if possible. If scores from 3 vets on same case, keep all ratings grouped with that case; **never split independent rater votes across train/test**. Stress test body-system shift (5 categories from paper); de-duplicate near-identical case vignettes or templated notes. If 30 Farrow vignettes, eval-only no misleading train split.
4. No post-hoc gold construction: independent vet rating(s) only, separately nurse and older GPT response baselines. Exclude the later diagnosis, hospital treatment, mortality, evaluator field, model prediction and explicit urgency field from model-visible case.
5. Metrics: multiclass Brier/log loss (if probabilities), ordinal MAE, dangerous undertriage rate stratified by emergency, overtriage burden, selective coverage, disagreements among vets, per-body-system uncertainty. Baselines: always RED (zero undertriage but maximum resource burden), train-only class prior, obvious-symptom rule, clinician agreement. Explicit costs; no assumption all overtriage equally safe or harmful.
6. Anti-shortcut audit: don't let section labels, case title, specialty/triage code, name of disease, or synthetic prompt response become label cues. Detect if initial public case text has outcome/retrospective diagnosis unavailable at actual intake and mask it.
7. Preserve ethics and limits: two use cases are **after arrival hospital scheduling** (Wong) and **before seeking care owner action** (Farrow). Not interchangeable. Neither is a natural farm vision/video monitoring intervention dataset. No autonomous clinical recommendation/deployment claims.
8. Scope: prefer one complete validated 340-case research benchmark and 30-case eval seed over broad surveying. If the first source cannot be accessed, prove that, move to backup; other leads (VetTriS 164 patient prospective, Ruys 2012 485 triage cases) source data currently UNKNOWN.

## Output
- Pinned `source_gate.csv` or compact source manifest + a scientific dataset card.
- Small `index_l3_vet_*.py` parser/builder, provenance, deterministic tests and zero-cost reference/prior/rater-disagreement baseline; practical budget ~1–2 day local investigation maximum. If no source bytes, still provide thorough negative proof rather than synthetic data.
- Open draft PR against `research/2026-10-data-prospecting`; don't merge, don't change frozen v0.1/v0.2; report original group counts, real source rights, tasks that are READY vs CANDIDATE, PR and reproducible commands.
- **Do NOT train large generative or frontier models** before scientific approval.

### Explicit success criterion
At least one real-source, expert-grounded **care-priority decision** whose reference is truly independent clinician judgment and whose observation state is available to the assessor *before* care priority is chosen. If source supports it, report case-level clinician disagreement as an uncertainty-aware decision target. If not, a scientifically correct failure is preferable to claiming unearned Level 3.
