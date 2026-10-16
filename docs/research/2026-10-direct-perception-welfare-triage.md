# OpenFARM — Direct Welfare Perception Source Triage (16 October 2026)

**Status:** Primary-publication/metadata + code-level forensic reconnaissance. **NOT** a source-byte audit, not a model benchmark, not an admission of new data to training. Existing canonical v0.1 and query compiler v0.2 remain unchanged.

## Mandate

Find directly observable animal welfare targets with original sensory inputs and independent source references. Prioritize authentic animal/image or video/pose/spectrogram -> welfare-related human observations, rather than demographic risk proxies or relabelled experimental conditions. First establish scientific validity, source bytes, permissions, group splits and trivial baselines; only later benchmark pretrained decision models. No email/outreach on the critical path.

## Triage verdicts

| Source | Input and independently labeled target | Verified here | Missing gate | Disposition |
|---|---|---|---|---|
| **GrimACE, mice (2026)** | Mouse face images -> five expert-scored Mouse Grimace Scale facial-action ratings 0/1/2/**not rateable**, 1,245 expert-scored images / **31 mice** in training experiment, LOAO protocol; separately acquired top/front videos and pose data in system | Primary peer-reviewed paper; Zenodo supporting archive lists `Data.zip` (~3.8 GB), code and training scripts publicly indexed. Inspected actual GitHub training code: 31-fold iteration, HDF5 images/labels/bboxes, reference label class 3 treated as missing/not rateable; `fold_ranges` define folds | Have **not downloaded/parsed Data.zip** or checked whether released HDF5 matches manuscript counts/animal grouping. Check per-AU/per-animal support, scoring authorship, frame adjacency, raw and expert label alignment, whether front/top video is actually source-available and synchronized, release-specific data license. GitHub code license does **not** certify archive image license | **Top Level-2 perception candidate; ACCESS_METADATA_VERIFIED / CODE_INSPECTED, not approved training/evaluation** |
| **HAF 2024, horses (2024/2025 releases)** | 605 cropped horse-face images from 22 videos / **11 horses**, source expert judged `pain`/`nopain` *face*; high-FPS alternative includes `ID=` filename field | Original public 233.8 MB archive and MD5 `be09a4ac6c1d3db3c4908cad1aeca075` listed on Zenodo 14541977; 25 fps alternative 5.5 GB listed at Zenodo 17389614 | No archive bytes inspected. 1-fps filename has *video ID*, not animal ID; 25-fps filename includes an ID but its meaning / horse identity map unverified. Must group all source videos per horse, test class/horse/site/time confounds, whether labels assigned per image versus source session, annotator independence and disagreement, data license (not legible in accessible landing page). Do not call binary label verified pain state | **Small, image-first Level-2 pilot candidate; no eval admission** |
| **Big Broiler, chickens (2025)** | Original video and per-track/frame human ethogram, includes observed `Feather_Pecking`, `Out_of_sight`, other behaviors | Existing OpenFARM audit **actually sampled original CSVs** from official ~33GB archive in CI 38837878581: 10 source clips, 3 independent annotators, 5,984 fixed-stride sampled track/frames. `Feather_Pecking` mean human pair F1 **0.824**, but only **45 reference-positive sampled rows across six videos**. `Out_of_sight` F1 **0.791**. Official GitHub provides 102 videos, paired annotations; authors label feather pecking as beak manipulation of conspecific feathers | Causal video/track splitting (do not trust track-only canonical split), temporal episode support and distinct source groups, actual video pixel/source frame parity, rarity under naturally observed duration, source license/AGPL and media use, separation of gentle feather interaction from damaging feather pecking/injury. No actual lesion/clinical outcome. IAA is not model performance | **Nearest farm-focused welfare-*relevant observation* candidate (Level 1); not a pain or caregiver-alert benchmark** |
| **PainFace, mice (2024)** | Paper describes >70,000 human-scored source images from 214 mice and MGS facial-action ratings | Peer-reviewed source; Figshare dataset ID 23907651 publicly labels **37.25 MB** of *validation data*, **CC BY 4.0** | The available 37.25MB validation deposit is **not established to contain the entire training image/label archive**. Need list actual file members, assess if independent external validation with original images and group IDs; don't claim all 70k are downloaded | **Data discovery lead, not source ready** |

## Scientific task contract: GrimACE (proposed, not yet admitted)

- **State:** an actual cropped front-camera face image from a specified mouse at a source timestamp, with model-visible pixels only. Optional later: image quality assessment and paired pose *if independently confirmed aligned*.
- **Primary target:** `Score(0,1,2)` for each of the five **expert-rated facial action units** (eye/orbital, nose, cheek, ears, whiskers), each retaining explicit `not_rateable`/mask. Use multivariate reference rather than declaring the summed MGS to be five independent biological outcomes.
- **Reference family:** human expert assessment of the Mouse Grimace Scale. Ratings are welfare-proximal pain-associated *cues*, not direct proof of pain, distress, need for treatment, or welfare valence.
- **Split:** mouse/recording/session heldout with a documented mapping; examine source HDF5 `fold_ranges` and original identifiers to confirm no adjacent frames or duplicated scenes leak. Source manuscript says 31-mouse LOAO for experiments 1 and 2, but **cannot certify actual file splits without original HDF5 bytes**.
- **Mandatory negative controls:** class priors, rater disagreements, between-condition background artifacts, animal-identity memorization, crop selection, unrateable AU frequency; do **not** use surgery/analgesia context, source labels or timepoint metadata as shortcuts.
- **Potential useful downstream task:** `Noul(visible pain-associated grimace feature present)` or multivariate ordinal prediction; **not** `Noul(needs_caretaker)` absent independent clinician intervention/adverse outcomes.
- **Gate to `TASK_VALIDATED`:** original raw media plus labels opened, counts/IDs/time matched, license confirmed, actual image montage inspected, animal-group leakage checks and source-human consistency passes. Gate to `APPROVED_RESEARCH_EVAL`: independent group-heldout evaluation plus minimum per-feature animal support and frozen clinical interpretation.

## Concrete priority order

1. **Inspect original GrimACE `Data.zip` and README at Zenodo 18299984:** list file members, find expert-scored image HDF5 or annotation tables, verify 1,245 original images / 31 animal IDs / five ordinal plus non-rateable channels, fold mapping, source video and license. Keep bytes upstream; record hashes/schema/examples only.
2. **Inspect HAF 233.8MB zip at Zenodo 14541977:** file list, per-video positive/negative counts, random images, exact duplicates, source/video/horse map using the 25fps counterpart only if justified, license and independent face judgments. If horse IDs cannot be recovered, block any subject-generalization evaluation.
3. **Existing Big Broiler:** use original annotated video segments for `Feather_Pecking` and `Out_of_sight`, group by full video/setup, quantify rare positive *episodes* and consensus timing before claiming an observation task. Do not silently label every feather-pecking behavior harmful.
4. Deprioritize PainFace training source until the promised 70k-image original archive is positively located; distinguish 37.25MB validation files from training corpus.
5. When a Level-2 visual task passes source validation, run only cheap image/pose baselines and ablations before investing in alternative system-1 architectures.

## Sources

- GrimACE manuscript: https://pmc.ncbi.nlm.nih.gov/articles/PMC13043301/
- GrimACE original source deposit: https://zenodo.org/records/18299984 (3.8GB Data.zip; source files not downloaded in this reconnaissance)
- GrimACE code: https://github.com/ETHZ-INS/GrimACE_manuscript ; checked `GrimaceModelTraining/big_grimace_loao_bb.py`, `dataset/kfold_hdf5_bb_dataset.py`, `dataset/hdf5_bb_dataset.py`
- HAF small release: https://zenodo.org/records/14541977
- HAF 25fps release: https://zenodo.org/records/17389614
- Big Broiler: https://github.com/decide-ugent/big-broiler-dataset ; OpenFARM actual source audit: https://github.com/ob1-s/OpenFARM/blob/research/2026-10-data-prospecting/docs/research/2026-10-video-heldout-sprint.md
- PainFace paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC11287051/
- PainFace Figshare validation deposit: https://figshare.com/articles/dataset/_b_Development_of_PainFace_software_to_simplify_standardize_and_scale_up_mouse_grimace_analyses_b_/23907651

**Known access limitation of this checkpoint:** web/source metadata and GitHub code were retrieved, but large original ZIP media were not fetched/decoded in this environment. No model trained, no source promoted, no source rights guessed.
