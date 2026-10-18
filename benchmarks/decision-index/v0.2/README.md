# OpenFARM — Next phase after Decision Index v0.1

Date: 18 October 2026. **Research only, no veterinary deployment clearance.**

## Freeze: Decision Index v0.1

- Frozen branch: `research/decision-index-v01-frozen`
- Frozen commit: `132618bc21c1cb6254b5c40c74c1a305c13ce3d2`
- Source PR: [#9](https://github.com/ob1-s/OpenFARM/pull/9), stacked on existing clinical PR #8. This branch was created from exactly PR #9's validated head and must not be rewritten or used to tune future models.
- v0.1 includes eight typed task contracts across six species; 15 authenticated/verified **L3 developmental** model-score records, generic scoring for binary and ordinal multiclass decisions, private Wong three-veterinarian disagreement audit, Farrow format-forensics, and verified Big Broiler original MP4 ingestion.
- All Wong and Farrow case sets were repeatedly queried; keep `EXPLORATORY` status. Do **not** turn them into pristine independent leaderboards by reshuffling identical original cases.
- Existing PRs #4 Aarhus, #5 independent welfare source gates, #6 GrimACE, #7 Big Broiler, #8 Wong/Farrow and #9 v0.1 remain separate; no unreviewed raw clinical/source or copyrighted pixel data moved into Git.

## Actual multimodal decision research

Real Liquid `d1-omni-600M-GGUF Q8` and exact Q8 multimodal projector via CPU `llama.cpp` on source video frames:
- [Completed Big Broiler original video-to-decision run](https://github.com/ob1-s/OpenFARM/actions/runs/39035562752).
- Six *label-enriched* 10-second windows: one positive and one negative in each of three camera setups; four time-ordered original frames/window; `Noul` probabilities, **3/6** correct at .5, binary Brier **0.45537**. Low sample and sparse frames mean it does not establish real video-event recognition.
- Source video and annotations not redistributed. Input representation is a **multi-frame sampled sequence**, not a dedicated continuous-video transformer or pretrained temporal video model.

## New L2 external source — Vidal 2022 orbital-tightening mouse images

- Study DOI: `10.1016/j.mlwa.2022.100312`, separate experiment from 2026 GrimACE.
- Original annotations: [author repository](https://github.com/AndreaVidal/WhiteFurredMice_Dataset), **1,087 scored images from 8 mice**, 552 score 0, 342 score 1, 193 score 2; annotation SHA256 `754d3442b056b14aee137643adab101c166dee27f42338397ef7a1f869cb9ad2`.
- Original author-hosted `Eyes_detection.zip` downloaded **493,247,093 bytes**, SHA256 `57c8a2dc112232544b25726a9d4a594795c44c2c528956128d9a2ad647c4064e`. All **1,087 original PNG filenames matched annotation keys**, original image decoding/box plausibility verified across all eight animals. [Original source and image gate passed](https://github.com/ob1-s/OpenFARM/actions/runs/39103207180).
- Scope is **ordinal orbital tightening**, not the five-action-unit GrimACE output, and not clinical nociceptive outcomes.
- **Rights**: public archive access is not itself an explicit grant for redistribution, external API uploads, model training or commercial use. Research-original-image source gate is complete; license assessment still pending. Work tracked in [issue #11](https://github.com/ob1-s/OpenFARM/issues/11).
- An original-image local-CPU evaluation is implemented in `scripts/index_v02_vidal_liquid_ordinal.py` with source bytes pinned; 96 deterministic label-enriched images across all eight mice. No image file or embedding is committed.

## Original clinical physiology candidate — 2026 Dryad 48 calves

- `10.5061/dryad.k3j9kd5qt`: original publication declares 48 identified calves in 12 balanced replicates, 2,050 mechanical nociceptive tests and 1,502 tissue assessments.
- **Source bytes not verified**: both official individual stream and dataset API download yielded HTTP 403 from GitHub runners ([original attempt](https://github.com/ob1-s/OpenFARM/actions/runs/39102344240), [API fallback](https://github.com/ob1-s/OpenFARM/actions/runs/39102445652)).
- No L2 trained pain model admitted, no animal/source-matched rows in Git; obtain original permitted files, verify source hash/IDs, then design temporal independent-outcome question that excludes future observations and treatment shortcuts.
- Access and permission blocker tracked in [issue #11](https://github.com/ob1-s/OpenFARM/issues/11).

## Small specialist development: new attempted rater-conditioned model

- Reproducible `scripts/wong_benchmark.py` and `scripts/index_v02_train_rater_conditioned.py`; original `Case_data.xlsx` is required **outside** the repository.
- Exactly 171 original training cases (3 vet labels per case), 67 calibration cases, 102 **already reused** developmental cases.
- Train char ngram TF-IDF regularized logistic with **optional veterinarian identity intercept**, original votes never discarded. Test a *fixed calibration-only* 24-setting grid (four rater-strength values, two regularizations, three training-vote-prior shrinkage values).
- Winner selected **only on calibration Brier**, tie-break by fewer immediate escalations: pooled (rater identity strength 0), C=3, shrink 0. Calibration Brier **0.28176**. This negative result means *separate rater intercepts did not improve this small-data objective*, **not** that experts share thresholds.
- Existing repeatedly used test: Brier **0.29263**, 45.1% argmax exact with median; using cal-selected urgent action threshold 0.1753: **6/9 majority-Immediate** among 13 immediate escalations; vote undertriage 30.4%, overtriage 31.4%. **Not clinically safe**, and not superior to Jev's 8/9 immediate under separate settings.
- No pretrained model weights, patient text, source spreadsheet, or trained token vocabulary committed. Exact aggregate metadata: `benchmarks/decision-index/v0.2/rater-aware-specialist-results.json`.
- CPU environment, source hash and synthetic source-freeze tests pass [v0.2 CI](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-v02-specialist-ci.yml).

## Explicit next scientific gates, not hidden unfinished tasks

1. **Fresh independent clinical L3 evaluation**: new expert-rated cases/clinics and genuinely held-out animal/patient IDs, preferably observed downstream outcomes, before claiming model ranking or owner-care efficacy. [Issue #10](https://github.com/ob1-s/OpenFARM/issues/10).
2. **Mouse image rights and source annotation provenance**: before open dataset release/commercial training or uploading original pixels to proprietary APIs. [Issue #11](https://github.com/ob1-s/OpenFARM/issues/11).
3. **Calf original XLSX access**: historical 48 calves and independent MNT need original source-byte verification; then temporally causal input/target boundary and cohort split. [Issue #11](https://github.com/ob1-s/OpenFARM/issues/11).
4. **Training strategy**: do not select specialists by reusing test scores; collect new independent reference cases or train on new sources and validate against held-out original animals/sites. If optimizing clinical policy, jointly report probability Brier, any-vet immediate recall, severity-weighted undertriage, number of escalations, and abstention/human review volume.
5. **Animal video**: scale beyond six positives/negatives, use continuous frames/temporal models, camera-family heldouts and natural-prevalence episode-level false alarms.

This checkpoint is intentionally stringent: a research benchmark can be useful before it becomes a valid deployment evaluation, but those claims must be separated.
