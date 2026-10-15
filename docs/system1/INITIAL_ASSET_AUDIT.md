# OpenFARM System-1 Reboot — Initial Asset Audit and Triage

**Date:** 2026-10-15  
**Status:** **PRELIMINARY / NOT DATA-CERTIFIED**. This is a first-pass repo-code, environment README, HF-card and public-literature review. No complete dataset download, notebook execution, sample-level media inspection, source reannotation, or group-overlap proof has yet been performed. Claims in existing README/cards are **claims to verify**, not certifications.

See [research charter](RESEARCH_CHARTER.md).

## Surviving assets in GitHub

- Repository: https://github.com/ob1-s/OpenFARM ; checked default branch `main` (126 tree entries, including 6 evaluation environments, 9 data-preparation notebooks, 2 model-training notebooks, and supporting scripts/species guides).
- Root `README.md` is **zero bytes**. This does **not** mean the project or dataset documentation is missing: environment README files, Hugging Face dataset cards, notebooks, and the old source-probing and data-preparation skills survive.
- Six environments: `openfarm_aww`, `openfarm_bioacustics`, `openfarm_dog_pain_triage`, `openfarm_horse_grimace`, `openfarm_sheep_apr`, `openfarm_zoo_arousal_eval`.
- Bioacoustics is a registry containing **eight distinct task definitions**. Existing training code mostly asks for XML answers and often optional generative reasoning/judging; this is **not** a native probabilistic decision benchmark yet.
- Nine data notebooks: AWW synthetic; CatMeows; Dog Pain; Horse Grimace; laying-hen stress; Sheep APR; SoundWel; ungulate valence; zoo valence/arousal. Two training notebooks: Gemma 4 text RL and Gemma 4 audio SFT.
- Existing prep/probe skill files already encode good instincts: source-shaped media, grouping, metadata screening, HF checks. Verify actual execution, not instructions alone.

## Critical code-level checks

### P0: bioacoustics silently falls back to training split on *any* split-load error

File: [openfarm_bioacustics.py](../../environments/openfarm_bioacustics/openfarm_bioacustics/openfarm_bioacustics.py), `load_source_dataset`, currently lines ~119–132.

`load_dataset(dataset_name, split=split, revision=...)` is wrapped in `except Exception`, which logs a warning and loads `split="train"` instead. **Conditional train-as-test contamination risk**, even for transient network/decode errors. This is an observed *implementation defect/risk*, not proof any prior evaluation used the wrong split.

**Code mitigation committed (2026-10-15):** `load_source_dataset` now raises on any failed requested split rather than substituting `train`. Six mock-backed regression tests were added under `environments/openfarm_bioacustics/tests/`, including split identity and strict alias/answer parsing. **Tests are authored but not yet executed in a fully installed project runtime; no historical evals were reverified.** Remaining gate: prove group boundaries and hashes from actual source samples; pin immutable revisions and test live Hub access.

### P1: tasks currently optimize answer matching rather than probability quality

Many existing environments reward exact normalized XML answers and some optional LLM-judged explanations. This is useful historically but **cannot evaluate calibrated probabilities**. A new typed interface and a scoring/eval layer should preserve distributions and compute Brier, log loss, class-conditional calibration, ECE/reliability, and risk-coverage, not only exact-answer reward.

Dog-pain code's `accuracy_reward` includes substring acceptance (`ans_clean in pred`) and needs a strict, tested label parser. Do **not** accidentally improve benchmark scores by accepting malformed or mixed answers.

### P1: audio modality contracts must be tested end-to-end

The bioacoustics env documents native audio, generated spectrogram, joint audio+spectrogram, and metadata modes. Check actual model adapter support; log preprocessing, sample rate, spectrogram settings, padding, truncation and what input the model received. Previous OpenFARM runs informally found spectrograms were often *faster and better* for models tested; reproduce, with timestamps/cost and paired samples.

## Task-by-task audit queue (preliminary, not model results)

| Legacy source/task | What the label actually supports | Plausible typed questions | First issue to test | First verdict |
|---|---|---|---|---|
| **AWW-bench** (232 train / 40 test, synthetic, literature notes) | Constructed judgments of welfare states under confounders; no real media | Choice over defined scenario labels; separate Noul for confounders; explicit uncertainty if underdetermined | Synthetic answer shortcuts, template overlap, annotation provenance; possible circular LLM judge | **Research sandbox**, not validated clinical ground truth |
| **Horse grimace-region** (cropped images; scores 0/1/2) | **Facial-region pain-associated cue intensity**, not horse-level clinical pain | Score 0–2; cue present Noul; optionally region Choice **only if not supplied in state** | Small number of horses; crops correlated with same photo/subject; balanced-vs-natural splits; rater disagreement | **Priority vision candidate** after audit |
| **Sheep APR** (web-sourced facial images; binary proxy label) | Facial expression proxy based on scale cues, not diagnosis | Cue-present Noul; specific facial-feature judgments only if source supports labels | Near duplicates, web/source identity, subject provenance, synthetic appearance/background shortcuts | **Conditional candidate** |
| **Dog Pain Triage** (clinical metadata, 68/18 balanced; 441/116 natural) | Source pain category from clinical observations, not direct video sensing | Noul for particular clinical indicators; bounded risk/triage Choice only when labels support it | Clinical diagnosis or scenario text could reveal target; case mix overwhelmingly nociceptive (407/441 raw train; 107/116 raw test) | **Investigate label shortcut before training** |
| **SoundWel pig bioacoustics** (valence/context/call type/age) | Source experimental/context and acoustic classes; not all are welfare states | Call type Choice; separate valence Choice; age/context only as ancillary tasks | Subject/session/file grouping; valence vs experimental context; acoustic-only baselines; metadata leaks | **High-value acoustic source; prioritize measurable calls** |
| **CatMeows** (440 clips, 21 cats) | Elicitation context (brushing, unfamiliar isolation, awaiting food), **not pain/stress diagnosis** | Three-way context Choice; call/no-call or vocalization feature targets only with new annotations | Cat/owner/recording correlations, recording conditions; 8 kHz Nyquist = 4 kHz | **Good confounder-test candidate** |
| **Laying hen audio stress** (102 source recordings; short excerpts) | Four experimental treatment/phase contexts; derived `stress_binary` marks only poststress treatment | Context Choice as source-recognition task; observational vocal event questions need new labels | **Label defined from experimental design; excerpts lack event timestamps**. A 15s excerpt may not contain target event | **High-priority label audit; NOT clinical stress detection** |
| **Ungulate valence** (3,181 calls, 7 listed species/subgroups) | Source study valence linked to experimental context; not independent clinical pain | Valence Choice as study-label prediction, sound-type Choice where supported | Species/animal/context confounding, in/out-of-study transfer, class imbalance and overlap with SoundWel/PVWB | **Priority audio candidate** if split/label validity holds |
| **Zoo affect** (15 muted clips, 3 species) | Expert-coded valence and arousal from visual stimuli | Arousal Noul, valence Choice, combined Choice only if valid classes | Only five stimuli per species; expert disagreement; clip/stimulus processing; no independent generalization estimate | **Qualitative probe, not headline statistical benchmark** |

Sources: environment READMEs and `task_registry.py` in repo; prepared dataset cards:
- AWW: https://huggingface.co/datasets/oliveirabruno01/AWW-bench
- Horse: https://huggingface.co/datasets/oliveirabruno01/openfarm-horse-grimace-region
- Sheep: https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark
- Dog: https://huggingface.co/datasets/oliveirabruno01/openfarm-dog-pain
- SoundWel: https://huggingface.co/datasets/oliveirabruno01/soundwel-pig-vocalizations
- CatMeows: https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows
- Hen: https://huggingface.co/datasets/oliveirabruno01/openfarm-laying-hen-stress
- Ungulate: https://huggingface.co/datasets/oliveirabruno01/openfarm-ungulate-valence
- Zoo: https://huggingface.co/datasets/oliveirabruno01/openfarm-zoo-valence-arousal

## P0 update — confirmed ungulate cross-split audio duplication (2026-10-15)

**This supersedes the preliminary statement that the animal-heldout ungulate split is merely unverified. Its released raw train/test pools are demonstrably media-contaminated.**

At the immutable Hugging Face revision `9fd70b71c9f2bfa5058b9480f587f365bfd82851`, full-data GitHub Actions verified:

- `train_raw`: 1,859 rows, `test_raw`: 496 rows. **94 distinct `audio_sha256` values occur in both** despite disjoint claimed `source_group_id` values.
- Independent SHA-256 hashing of stored audio bytes matches the published `audio_sha256` value in **all** scanned entries: 506/506 `train`, 136/136 `test`, 1,859/1,859 `train_raw`, 496/496 `test_raw`, 826/826 `pig_family_heldout` — **3,823/3,823 scanned rows across views**.
- Repeated identical audio is present within several splits, including 227 extra same-hash occurrences in `train_raw` and 24 in `test_raw`; repeated hashes with contradictory labels occur in `train_raw` and `pig_family_heldout`. These could reflect duplicate recordings incorrectly assigned to different source/subject labels.
- Example matching 12-char audio digest prefixes: `0298a0301dfa`, `02e826f9ceb3`, `0582932fa2ee`. The same content hashes map to **different** claimed group IDs across splits.

**Action:** Quarantine **raw** ungulate train/test evaluation immediately and treat balanced splits as **not yet cleared** pending independent cross-view overlap checks. Do not train and benchmark on these as if clean heldout samples. Reconstruct original audio provenance / clip IDs; form connected components across identical audio bytes, recording source/session and animal, then redesign splits, preserving class support and external holdouts. Check exact and near-duplicate acoustic content, not merely filenames. Do not overwrite the published source dataset from this audit.

Auditable CI: https://github.com/ob1-s/OpenFARM/actions/runs/38719744768 (ungulate job). Reproducible checker: `scripts/audit_openfarm_integrity.py`.

### Completed cross-view integrity and connected-component diagnostics

Expanded analysis on immutable revision `9fd70b71c9f2bfa5058b9480f587f365bfd82851` established **exactly which pairings are contaminated**:

| Training view | Test view | Shared byte-identical audio hashes |
|---|---|---:|
| `train` | `test` | **4** |
| `train_raw` | `test_raw` | **94** |
| `train` | `test_raw` | **15** |
| `train_raw` | `test` | **18** |

A connected-component diagnostic unions rows linked by **either** equal audio SHA-256 **or** equal `source_group_id`, across all 3,181 unique raw source rows (`train_raw`/`test_raw`/`pig_family_heldout`):

- 2,825 distinct byte-level audio hashes across 3,181 source rows; 332 nonempty distinct source-group IDs, 10 rows missing group identity.
- 291 connected components; **three components span both legacy raw train and raw test**.
- **Largest component has 1,091 rows (46.33% of the 2,355 non-pig raw pool), 781 distinct audio hashes and 31 nominal source-group IDs**, with 841 `train_raw` and 250 `test_raw` rows. All are source-labeled `Negative`.
- This component is far too large for conventional 80/20 subject-heldout sampling without severe distribution constraints; a clean split may require revisiting the source label/group provenance rather than a simple RNG resplit.
- Pig-family holdout has **no matching exact audio hashes with non-pig raw views**. This establishes only byte identity, not full study/source independence.

Source: `scripts/diagnose_ungulate_groups.py`; GitHub Actions run https://github.com/ob1-s/OpenFARM/actions/runs/38720784988, artifact `openfarm-ungulate-component-graph`. The graph is a diagnostic, **not a published repaired dataset**.

The new environment raises a quarantine error for `ungulate_valence` by default; `allow_uncertified_ungulate_eval=True` exists **only for explicit forensic debugging**, never valid comparison claims.

## Split semantics requiring reconstruction — sheep

At HF revision `4a8e9a6a6296be4ad12584be8bd0fec0c822e5fe`:

- Balanced `test`: 37 label=1 out of 74; `test_raw`: **29** label=1 out of 225. Therefore balanced `test` **cannot** be a no-replacement subset of `test_raw`, regardless of sample identity policy.
- Comparing `source_filename::label` provenance keys, 32 balanced train keys were absent from `train_raw`, and 60 balanced test keys absent from `test_raw`. Expanded exact-byte matching confirms **32 identical images in balanced `train` versus `test_raw`, and 60 in `train_raw` versus balanced `test`**; matched `train`/`test` and `train_raw`/`test_raw` show **zero exact-image overlaps**. **Mixing balanced/raw views therefore causes proven training/evaluation contamination.**
- Original `source_filename` **repeats across classes** and must **not** be used alone to identify a sample. An early scanner erroneously used it, reporting spurious cross-split media leakage; those assertions are withdrawn.
- Exact published image bytes were checked in the second pass. No cross-train/test *byte-identical* images were detected. Image re-encoding, crops, and near-duplicates still require perceptual checks, and there are no trustworthy animal group IDs.
- This is **proven incompatible split-view semantics**, but **not yet proven train/test image leakage**. Reconstruct the notebook's exact sampling and raw-split derivation and separate any benchmark using balanced and raw pools until checked.

**Action:** Do not treat the sheep balanced/raw views as subsets or mix them freely for training/eval. Reconstruct split provenance and perform perceptual near-duplicate checks.

## Other full-data findings (bounded checks, not label certification)

- **Horse** revision `ba5799d2c29886e238f8a069fb3501d1d91444f5`: claimed sample hash + horse ID train/test separation checks passed, but `source_sha256` repeats 1,035 times in `train_raw` and 116 times in balanced `train`. Investigate whether this intentionally denotes shared parent images across facial-region crops, not necessarily duplicated crop pixels.
- **CatMeows** revision `57fe220b22eae46eb555af9f4169bd4677584cea`: declared cat-ID and `audio_sha256` split boundaries and view subsets passed.
- **Dog pain** revision `fb1223f435a78ab2ea37ddf5f6d3e66e5e1e3941`: declared `source_subject_id` + `source_row_index` boundaries and subset checks passed. Clinical field/label circularity remains open.
- **Hen** revision `89004914b245f8a594cec4a7401bf315dab9b1c8`: declared day-group and audio-hash split boundaries passed. No proof of independent chicken-level subjects or time-localized distress labels.
- **SoundWel** revision `4164e39798c2e65bce5ef8d56159d98e1267f577`: 1,392 training, 337 test, 6,887 `raw` rows; sample identity checks found no exact cross-split fingerprint match, but **animal/session holdout unverified**.
- **Zoo** revision `e8a7555ee800239a84e83be50f0173f7ddff0a5b`: 15 test-only cases, cannot establish independent generalization.
- **AWW** currently returns HTTP **401** for the existing `oliveirabruno01/AWW-bench` HF ID (no unauthenticated recovery). This is an access/recovery blocker, not proof the content was deleted.

**Importantly:** Labels encoded in source filenames do not constitute model-answer leakage if those filenames are excluded from all model-accessible inputs. Evaluate the actual rendered state, not a dataset filename in isolation.

## Strict re-audit protocol (before head training)

For **every source**, materialize a mini-manifest (`dataset_id`, immutable revision, DOI, license statement, source size, media hashes, source_subject_id, collection_session, farm/location if available, source recording, original author annotations, preparation transforms). Preserve whether it is *observed*, *derived*, or *not provided*.

1. Trace publication → raw artifact → notebook → HF row → environment state/question → target → metric.
2. Check the raw bytes and source label semantics; do not use existing card as sole truth.
3. Inspect multiple representative and difficult samples visually/aurally. Archive non-sensitive QC indices/review notes; do not redistribute source media without rights.
4. Check duplicates and near-duplicates; subject/owner/session/site group leakage; split resolution; filename leakage; source-derived proxy labels; class-prior exploitation.
5. Run **metadata-only, context-only, simple-feature, and wrong-modality controls**. If a label is predictable from a collection artifact, flag it as a confounded capability claim.
6. Build a **typed-question catalog** per source with: input state; available modality; answer type; precise semantics; label provenance; observed vs inferred; group split; missing evidence; ability to abstain; supporting annotation; evaluator.
7. Mark each proposed task: **gold** (source-grounded observable), **silver** (proxy requiring qualifiers), **exploratory**, or **reject**. These are decisions to be earned, not assumed.
8. For deployment targets, add site-heldout data and event prevalence. Report false alerts/animal-day and missed events as well as benchmark accuracy.
9. Avoid producing a large pile of one-hot derivative tasks from the same unreliable label: new questions need independent semantic grounding.

## First external source leads — DISCOVERY ONLY

These are **not verified additions** to OpenFARM datasets. Rights, sample-level quality, holdout groups, and sources must be checked before ingestion.

1. **CowRumIA (2026)** — 624 cattle vocalization recordings across calm/stressed and beef/dairy categories; public Zenodo ZIP ~152 MB; includes field recordings and materials from archives/platforms. **Rights shown as copyright, no clear open content license on record**; trace per-record provenance and label criteria. https://zenodo.org/records/20406266
2. **Multi-Species Animal Vocalization Raw Dataset (2026)** — 8 species, reported cat/dog fine-grained emotion labels; ~964 MB archive. Split claims and emotion validation should be checked; public landing page does not by itself establish commercial training permission. https://zenodo.org/records/20571465
3. **Computer-vision bovine lameness system (2026)** — paper says 12,182 annotated images and pose/locomotion measurement. **Paper is not equivalent to accessible annotated dataset**; verify code/data and reuse rights. https://doi.org/10.1016/j.atech.2026.101952

### Synthetic-data research queue (separate from primary dataset recovery)

- Anuran call synthesis, diffusion augmentation: https://doi.org/10.1016/j.ecoinf.2025.103322
- Killer whale calls, hybrid deep generative + classic augmentation: https://doi.org/10.1111/mms.70186
- Sow estrus vocalizations, lightweight recognition with spectrogram-conditioned diffusion: https://doi.org/10.1093/jas/skag075
- Frog CLAP synthetic embedding transfer / zero-shot: https://doi.org/10.1111/2041-210x.70384
- BirdDiff controllable bird-call generation: https://doi.org/10.1016/j.ecoinf.2026.103917

These studies support a real literature sweep, not a claim that synthetic data helps OpenFARM. Require **held-out real recordings, subject/site separation, matched augmentation budgets, and generator-artifact controls**.

## Next actions, in order

1. **Fix/check the train-fallback path** and add split/manifest integrity checks; investigate equivalent risks in the other five envs.
2. Freeze a reproducible manifest of actual available HF datasets and preparation notebook revisions; identify missing original raw data, source mapping, licenses and QC files after SSD loss.
3. Inspect *actual sample media and annotations*, starting horse/ungulate/hen; then run subject/source/label audit; log disagreements and failed claims.
4. Rewrite task contracts *after* auditing input-to-label validity. Create multiple meaningful questions only when independently grounded.
5. Only then do model comparisons and minimal decision-head training.
6. Continue systematic external prospecting, annotating access and rights separately from scientific usefulness.

## Verification log

- [x] Enumerated GitHub `main` recursive tree and verified six environments / nine data notebooks / two training notebooks.
- [x] Inspected six environment READMEs and bioacoustics task registry.
- [x] Inspected relevant sections of four environments' implementation, including bioacoustics split fallback.
- [x] Inspected Hugging Face dataset cards for key source/label/split claims.
- [x] Located initial 2026 external leads and synthetic-data papers.
- [ ] Executed existing prep notebooks / validated output artifacts.
- [ ] Downloaded decoded source sample bytes and visually/aurally audited them.
- [ ] Proven split non-overlap from source IDs, hashes or embeddings.
- [ ] Verified all upstream rights/license chains for commercial use.
- [ ] Completed comprehensive current data/literature sweep.
- [ ] Trained or benchmarked any System-1 model.

## October 15 follow-up: direct HF viewer observations (metadata-only)

The public Hugging Face dataset viewers (not downloaded media) currently show:

- **Horse grimace:** `train` has 945 rows; card also lists `test` 279, `train_raw` 3931, `test_raw` 916, all claimed animal-heldout. Displayed samples include `source_filename` strings such as `M1_C2 (1).jpg` and `M1_B0 (154).jpg`, encoding horse/region/**answer**. Existing environment does not put filenames in prompts; nonetheless, future exported task builders must maintain this isolation. No independent proof of holdout correctness.
- **Ungulate valence:** source card reports 3181 clips (2508 Negative, 673 Positive), non-pig balanced train/test 506/136 and natural test 496; overlap risks and underlying `species_slug::animal_id` boundaries remain to verify.
- **Laying hen:** the viewer exposes four-class experimental treatment/phase targets and derived binary target; one visible row has a 15-second assembled excerpt from **five three-second windows** drawn across a 1060-second source recording. This is not a time-localized stress event. `train` and `test` show 48/16 rows; split integrity remains to be proven from full manifests. The source dataset card itself candidly documents these limits.

Direct primary pages:
- https://huggingface.co/datasets/oliveirabruno01/openfarm-horse-grimace-region
- https://huggingface.co/datasets/oliveirabruno01/openfarm-ungulate-valence
- https://huggingface.co/datasets/oliveirabruno01/openfarm-laying-hen-stress

This is **schema/metadata/visible-row inspection**, not media decoding. Signed HF viewer image URLs were unavailable for actual visual inspection during this pass.

See [typed question and calibration contract](TYPED_QUESTION_CONTRACT.md) for the updated System-One/Jev interpretation: multiple atomic questions per shared state, Noul vs Choice vs Score, probability distribution vs derived confidence vs empirical calibration, and why continuous streams are required to estimate false-alerts/animal-day.

**Reminder:** We are auditing historical work made with earlier agents, not certifying it. This note is a starting map, not the result of the exhaustive forensic review.
