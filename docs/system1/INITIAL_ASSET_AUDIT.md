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

**Fix gate:** fail closed on missing/unavailable evaluation split; explicitly permit an intentional all-train exploratory mode only via a named flag. Assert resolved split, dataset revision, source group boundaries, example count, and unique sample hashes in eval manifest. Check all prior task split names against live HF revisions.

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

**Reminder:** We are auditing historical work made with earlier agents, not certifying it. This note is a starting map, not the result of the exhaustive forensic review.
