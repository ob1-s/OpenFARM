# OpenFARM evaluation readiness — checkpoint 2026-10-15

**Status:** This is a measured data-integrity triage, not a veterinary-competence benchmark, commercial product certification, or an exhaustive scholarly survey.

## Decision table

| Source | Verified integrity checks | Blocking gaps | Allowed use today |
|---|---|---|---|
| **Ungulate Valence** | 3,823 published hashes exactly match stored audio bytes across views; full pairwise test and group connectivity diagnostics | **BLOCKED**: 4 same-media IDs in balanced train/test, 94 in raw pair, 15 and 18 across mixed views; 1,091-row component spans 31 nominal groups and both raw partitions; label conflicts on exact duplicated audio | Forensic diagnostics only, with explicit opt-in. **No honest independent heldout model comparison on published partitions.** |
| **Sheep APR** | Full 1,369 row/view instances' exact-byte split matching; provenance confirmed in preparation notebook | **BLOCKED for mixed views**: 32 exact image overlaps train/test_raw; 60 overlaps train_raw/test; balanced/raw independently generated. Two close dHash=4 image pairs between balanced train/test warrant further inspection, including visually near-identical images with opposing labels; no animal IDs | Dataset/photo-label quality assessment only until a near-duplicate-safe, cross-view-compatible partition exists. |
| **Horse Grimace Region** | No identical configured source IDs or horse IDs shared between train/test; balance/raw view relationships verified | 1,035 repeated source IDs in raw train, 116 in balanced train; gallery confirms several byte-identical crop pairs; only a few horses, substantial lighting/visibility variability; rater reliability unverified | Candidate image task; do not treat repeated rows as independent evidence or derive clinical pain diagnosis from crop cue labels. |
| **CatMeows** | Pinned revision: exact audio-hash and cat-ID disjoint train/test; balanced view subset; actual dataset inspectable | Labels are elicitation *context*, not welfare stress; owner/session/microphone effects need checking | Conditional *context-classification* baseline after further shortcuts audit. |
| **Dog Pain Triage** | Pinned revision: reported source-subject and source-row IDs nonoverlapping between views; subset relationships verified | Clinical text/diagnosis shortcut prediction; very imbalanced natural test (107 negative nociceptive / 9 neutral) | Conditional clinical-metadata ablation, not sensor-vision welfare detection. |
| **Laying Hen Stress** | Pinned revision: audio-hash and experimental-day-group partition checks passed; balanced subset checks passed | Label derived from experimental treatment/phase, not timestamped animal stress. Five disjoint source windows concatenated into 15s excerpts | **Treatment-context** classification only; no real-world stress detector claim. |
| **SoundWel Pig** | Pinned revision: all splits accessible (train 1,392 / test 337 / raw 6,887) | No certified subject/session grouping in current scanner; may contain source-context confounders | Explore acoustic call type and dataset context; no subject-heldout claim yet. |
| **Zoo Valence/Arousal** | 15 stimuli accessible; source clip/filmstrip implementation survives | Tiny eval-only corpus, 3 species, no generalization inference; expert interpretation not objective diagnosis | Qualitative probes only. |
| **AWW Synthetic** | Source notebook survives in GitHub | HF dataset currently responds 401 unauthenticated; `ob1-s/animal-data` older reference is inaccessible/404; synthetic label rationale and generated source examples not independently recovered | Blocked from reproducible public data evaluation until recovered. |

**Not green for production:** No checked dataset yet demonstrates real-world alert precision, missed clinically meaningful events, calibrated probabilities, false alerts per animal-day, or site-heldout generalization.

## Evidence and how to reproduce

- End-to-end immutable HF audit scanner: `scripts/audit_openfarm_integrity.py`.
- Separate full raw-media/animal graph: `scripts/diagnose_ungulate_groups.py`; source pinned at `9fd70b71c9f2bfa5058b9480f587f365bfd82851`.
- Actual media previews: `scripts/audit_openfarm_media.py`, producing annotated horse/sheep contact sheets and ungulate spectrograms in CI; look for job `openfarm-media-audit-contact-sheets`.
- Historic transformation provenance: `scripts/audit_openfarm_notebooks.py`. **Sheep source notebook explicitly uses separate stratified splits:** one over `clean_df` filtered + deduplicated + balanced and another over full `df` for raw. This directly explains why balanced and raw partitions cannot be freely combined.
- **Ungulate root-cause investigation in progress**: audio names are parsed and merged with `Data S1.xlsx` based on `animal_id|species_code|context_code|valence`, with Pandas `validate="many_to_one"`. The merge therefore is *not* an unchecked many-to-many multiplication; investigate repeated original WAV bytes under different source filenames and/or later preparation output.
- Current GitHub Actions: https://github.com/ob1-s/OpenFARM/actions
- Draft research PR: https://github.com/ob1-s/OpenFARM/pull/1

## Media QA observations, explicitly not clinical judgments

Sample sheets show very dark horse-eye crops, inconsistent framing/lighting and selected repeated crops. Sheep samples vary by face angle, animal appearance, background and image provenance. One stock-photo watermark was visible in a sample; **verify the actual rights of each upstream photo for commercial training**, rather than assuming a dataset-level CC license settles every embedded source. In balanced sheep train/test, dHash Hamming distance 4 for two closest pairs and one appears to represent the same photograph with conflicting labels (train 0 vs test 1). This requires source photo/near-duplicate verification before being treated as proven leakage.

The spectrogram previews are visualization-only and use a 1024-point FFT in short fixed windows; native sample rate varied (10 of 12 inspected at 44.1 kHz and 2 at 48 kHz). These previews are **not** validated model inputs, and spectrogram scaling should be calibrated for relevant species-frequency ranges.

## Process gates remaining

1. Confirm perceptual image identity/annotation conflicts using original source photos, rather than just a perceptual hash.
2. Reconstruct corrected ungulate source IDs; cluster identical audio and true subject/session groups, and evaluate whether a useful new holdout is feasible given the giant component.
3. For every dataset: validate original author label protocol against representative raw examples, split grouping, annotation ambiguity/inter-rater reliability, licensing.
4. Verify metadata-only and irrelevant-modality controls, then create typed decision interfaces for *defensible questions*, with abstention and probability calibration.
5. Only after data/source integrity: train/test decision-head probes and compare to matched generative baselines.

Root README is empty by design of the surviving repository; actual prepared HF cards, environment README files and notebooks are separate and do survive.
