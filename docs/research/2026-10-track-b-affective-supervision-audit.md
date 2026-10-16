# OpenFARM Track B — actual affect/welfare source inspection

**Checkpoint: 2026-10-15 · Scientific status: source audits, not model results.**

**Mission:** Affective System-1 empathy requires **observable behavior → independently defensible welfare indicator → appropriately calibrated human-attention decision**, across audio/video, pose, physiology and bounded temporal/clinical states. Context manipulation is not an emotion label, and many source rows are not many independent animals.

We are deliberately *not* extending the approved training/evaluation suite just because a source is large, CC0 or recently published. Our separate [approval contract](DATA_APPROVAL_POLICY.md) still applies.

## Source A — new 2026 four-facility PigWeb: **ACCESS_VERIFIED, indicator construct audit in progress**

**Primary study:** Reimert et al., *Teaming up for welfare: a cross-institute approach to identify robust indicators of emotion in pigs*, 2026, DOI [10.1016/j.animal.2026.101871](https://doi.org/10.1016/j.animal.2026.101871). Paper explicitly links source [AnjaEggert/PigWeb_cross-institutes](https://github.com/AnjaEggert/PigWeb_cross-institutes), data and R statistical analyses under **CC BY 4.0**. **Pinned** source commit `e3dba18edba6e1b214b4bea8ae66109d0f0d1226`.

**Actually read** three source CSVs: `PigWeb-behavior.csv` (256 rows), `PigWeb-physiology.csv` (496 rows), and `PigWeb-PCA.csv` (256 rows), plus the source analysis notebooks and license.

- Behavior records: **128 animals across four institutes**, each with positive/negative experimental condition → 256 paired session records. **248** have actual locomotor-play/escape count observations; one site has eight missing behavior-field observations. The labels are **behavior counts**, not independently certified happiness/distress.
- Physiology: **124 animals × positive/negative × pre/post = 496 rows**. A verified keyed join on (institute, pig_id, context) matches **248 of 256** behavior records to **post-exposure physiology**; eight rows from four animals at one institute have no matching physiology. Source CSV names/identities are preserved in the forensic script; no individual animal rows rehosted in our repository.
- Per-facility behaviors **differ materially**. In one institute (AU) escape events were **zero in both** contexts. Others show differential play/escape; cross-institute indicator transfer must be **explicitly tested**, not assumed.
- Source physiology includes cortisol, oxytocin and infrared temperature, but **cortisol/oxytocin are post-exposure measures** and instrument availability differs by site. Do **not** label cortisol as generic distress/negative valence, and do not build a fictitiously causal prediction of earlier behavior using future physiology.
- No raw synchronized behavior video is published in the code repository. This source is suitable to study **indicator validity and heterogeneous behavior↔physiology associations**, **not yet an approved video/audio inference benchmark**.

**Audit script:** `scripts/audit_pigweb_cross_site.py`; verified [GitHub Actions run 38823982604](https://github.com/ob1-s/OpenFARM/actions/runs/38823982604). Group keys and missingness are tested. Before admitting a typed prediction task: identify a real prospective observation/input aligned to the human behavior reference, and prove leave-one-institute-out validity; if nothing useful remains after controlling facility/experimental context, **reject that proposed task**.

## Source B — CC0 pig social nosing: **ACCESS_VERIFIED, labels semantically interpretable, not a vision set**

**Source:** [Figshare 27960810, version 2](https://figshare.com/articles/dataset/Dataset_on_pig_affiliative_social_interactions/27960810), DOI `10.58073/SRUC.27960810.v2`, source platform reports **CC0**.

**Actually downloaded** both XLSX files; `All Pen Behaviours Revised` contains **19,942 logged interaction rows**, across **eight pens, two observation days, four coarse time intervals, and one recorded batch**. `All Pen Attributes` has **117 animal records**. The comment embedded in cell G1 supplies the observer's ethogram:
- NNC = nose-nose contact
- NHC = nose-head contact
- NBC = nose-body contact
- NNP = nose-nose proximity
- NHP = nose-head proximity
- AG = aggression.

Source behavior code counts: NHP 8,161, NNP 3,233, NBC 3,062, AG 2,491, NHC 2,030, NNC 941, MT 21; two whitespace-padded labels and one malformed `3NNP` require adjudication. The event workbook has **9,305 byte-identical field rows**—likely repeated indistinguishable events within a coarse recorded interval, **not disposable image duplicates**. **Do not call `drop_duplicates()`** without author protocol.

**Crucial blocker:** subject/pen IDs differ by notation between the files (e.g., `P1` / `FP1`; pig ID numeric vs `P1`). Preliminary literal joins fail. A candidate normalized key matched **29 of 117** observed actor keys; **88 remain unmatched**. This means the actual participant-to-pen and treatment metadata must be reconciled with source codebook/author before any subject-heldout train/eval. Do not infer that unmatched IDs are fictitious animals.

There are no paired video clips. Source supports **rich social behavior ontology and possibly temporally structured *tabular* decisions** with pen/day/time grouping; it does **not** provide images for vision training or prove animal subjective happiness. Future tasks require genuine *predictor state* rather than simply feeding the target behavior code back to the model.

**Audits:** `scripts/audit_positive_welfare_sources.py`, `scripts/audit_pig_social_nosing_deep.py`; real-source CI [38823463194](https://github.com/ob1-s/OpenFARM/actions/runs/38823463194), [38823941180](https://github.com/ob1-s/OpenFARM/actions/runs/38823941180), identity normalization [38824146440](https://github.com/ob1-s/OpenFARM/actions/runs/38824146440).

## Source C — Agroscope pig social interactions/lesions: **PARTIALLY ACCESS_VERIFIED, don't claim predictive lesion outcome**

**Source:** [Zenodo 5920843](https://zenodo.org/records/5920843), *Pen mates' interactions, potential precursors of damaging behaviours*, 2023.

Recovered raw `animal.txt` (**95 pigs**), `conspecifics.txt` (**380 observation rows = four observations/animal**), and `objects.txt` (**380 rows**) by actual Zenodo downloads. Fields count ~150 specific social-encounter/reaction variables, including manipulating/bite/seize on head, body, ear and tail. **Only three session rows have `tail_bite_active>0`** (two rows with 1 and one row with 3 events), so this is **extremely sparse** support for a standalone tail-biting prediction benchmark. Observers recorded social/interaction counts directly, not from videos provided in the deposit.

Original `lesions.txt`, `activity.txt`, `confrontations.txt`, `rooting.txt` downloads suffered transient Zenodo timeouts, so **a lesion-linked outcome evaluation is NOT yet approved**. Even after recovery, lesions were assessed *after* the behavior observations—not necessarily an independent future-onset event label. Study setting is one experimental farm with protein-reduced feed; no cross-farm test available.

**Audit:** `scripts/audit_pig_social_ethogram.py`, actual-file [CI 38823522382](https://github.com/ob1-s/OpenFARM/actions/runs/38823522382).

## Source D — positive play with independent health outcomes: **ACCESS_METADATA VERIFIED; file-level clinical join pending**

[Steinerová et al., 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC11443507/), experimental play enrichment versus standard pig rearing, published source [Dryad 10.5061/dryad.76hdr7t55](https://datadryad.org/dataset/doi:10.5061/dryad.76hdr7t55), **CC0**. Nine source litters (five intervention, four control), 28 challenged pigs (14/14), with separate XLSX for play/exploration/feeding observations, clinical scores, lung/skin lesions, immune response and viral loads. This is **the rare independent outcome source** relevant to positive welfare hypotheses, but the treatment condition is not an expert happiness score and the tiny **nine-litter** independence matters enormously.

File metadata and naming were verified in [CI 38823463194](https://github.com/ob1-s/OpenFARM/actions/runs/38823463194). `scripts/audit_pig_play_clinical.py` is separately attempting actual clinical/behavior workbooks. **Do not** admit a disease-resilience prediction benchmark unless verified animal/litter/time joins, pre-onset predictors and independent clinical outcome labels survive and leakage checks pass. No source footage is presently verified.

## Source E — EquiFACS 2020: **ANNOTATION ACCESS VERIFIED, video-linked task not approved**

Separate from old Horse Grimace: [Rashid et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7608869/). Two actual XLSX supplements downloaded. Experimental pain/no-pain labels: **12 videos = 6 horses × baseline/ischemia**, with timed facial-action coding. Clinical data: 21 horses per paper; `PAIN` tab contains **167 interval rows**, **22 distinct `Media` values**, at least **77 non-codeable entries** plus whitespace-inconsistent severity labels. Don't treat a row as an animal and don't interpret “Non-codeable” as “No pain.” The ~880MB experimental video ZIP is indexed, but **video/annotation synchronization and animal-level split have not been verified**; source clinical videos are not publicly distributed in the same supplements.

Candidate questions **only after synchronization**: `Noul(AU active in this interval)`, independent ordinal expert-rated pain with non-codeable/abstention, multiple modalities (visual/time). With only six experimental horses, leave-one-horse-out is primarily a **small exploratory generalization exercise**, not a calibrated deployment metric.

Prior results: [actual-file report](2026-10-initial-source-approval-results.md).

### EquiFACS actual structured task build (not a certified clinical benchmark)

Follow-up data build **passed clean GitHub CI** [run 38825087764](https://github.com/ob1-s/OpenFARM/actions/runs/38825087764). `scripts/prepare_equifacs_structured.py` parsed **1,346 valid expert AU intervals** across **21 clinical media IDs**, and joined them to all **167 veterinary rating intervals** by media/time. Of the original ratings, 77 were **non-codeable**; only 90 had a stated clinical category.

Strict independent-annotator filter: **exclude Alina's 50 pain annotations**, since Alina also did the FACS annotation. Johan and Camilla together leave **64 codeable rating intervals (47 no-pain, 12 mild, 5 severe)** across 21 media; *none are moderate*, so this does **not** support reliable Score(0–3) supervision. Excluding **10** of these intervals with overlapping facial-action `Uncodable` or `Non-scoreable` quality flags yields **54 binary reference intervals across 18 media IDs**.

`scripts/build_equifacs_clinical_v0.py` produces three **video/media-disjoint local partitions** and stores original labels/rater/media timestamps as separate provenance, never as model features. Data build output (research pilot only):

| Partition | Independent media groups | Rating intervals | Pain-positive intervals | No-pain intervals |
|---|---:|---:|---:|---:|
| Training | 10 | 32 | 5 | 27 |
| Calibration | 3 | 10 | 3 | 7 |
| Test | 5 | 12 | 7 | 5 |

**Admission decision:** `TASK_VALIDATED/PILOT CONSTRUCTED`, **NOT** admitted as a reliable independent veterinary benchmark or general-purpose calibrated welfare model scorecard. The **test has only five clinical media** and twelve rated intervals; its positive prevalence is different from train, and the original manuscript's 21 clinically affected horses must be cross-verified against source `Media` identities before claiming animal holdout. Validation/threshold tuning on just three media is especially fragile. This is a legitimate *System-1 structured perception-to-expert-judgment testbed* for exploratory low-budget comparison, not an answer to false alerts per horse-day, clinical diagnosis, or raw-video recognition. The source videos and independent site samples would be needed to graduate.

**Meaningful scientific win:** the question is now based on **separate human observations**, not a binary threshold of our old HGS source scores. It is subject to genuine censoring, rater independence, media grouping and quality-gating rather than invented labels.

## Additional positive/subjective evidence prospects (do not equate positive experimental situations with positive emotion)

- [2024 horse affiliative indicators review](https://www.sciencedirect.com/science/article/pii/S0168159124000844): adult horse **voluntary social proximity** with context is better supported than assuming all play or social grooming indicates welfare. Physical affordance matters (cannot approach if confined).
- [2024 Rothamsted beef-cattle group QBA](https://data.rothamsted.ac.uk/en/dataset/comparison-of-in-person-and-video-qualitative-behaviour-assessments-of-beef-cattle): **40 live sessions, re-rated from matched 10-minute video by trained observers; 20 qualitative descriptors on 0–125 scales**, both positive and negative. The data landing page paradoxically declares public CC BY 4.0 but currently displays a **private-dataset warning**; original recorded video also requires its own access and reuse check. **Do not claim publicly downloadable training files.**
- [2026 positive-welfare research project](https://www.vetinst.no/en/research-and-innovation/ongoing-research-projects/eupahw-positive-animal-wel) explicitly states there is no direct gold standard for animal emotion. Validation strategies rely on preferred situations, judgement-bias assessments and indicators that converge with independent measures. This reinforces our evidence-layer approach.

## Honest task disposition

| Candidate | Data access | Meaningful decision supervision | Train/eval admission |
|---|---|---|---|
| PigWeb cross-institute play/escape | **Verified, pinned source** | Human behavior counts + independent physiology but no paired video | **Not yet**: define noncircular state; site-heldout pilot |
| Pig social nosing CC0 | **Verified, actual XLSX** | 19,942 event observations with ontology; no media | **Not yet**: resolve animal IDs, repeated-event semantics, predictive state |
| Agroscope interaction/lesion | **Partially verified** | Real expert 5-minute ethogram, few tail bites | **Not yet**: lesion files/access and time ordering |
| Dryad play plus health outcomes | **Metadata verified** | Independent clinical outcomes could ground welfare proxies | **Not yet**: verify individual actual XLSX join, litter split |
| EquiFACS | **Expert XLSX verified** | Timed action units and clinical score (non-codeable) | **Not yet**: verify synced video and holdout |
| Beef-cow QBA | **Paper+landing page** | Trained assessor affective descriptors | **Not yet**: source access says private |
| HotPig feeding | **Actual files, split CI** | Independent feeder event from bounded sensor-state | **Already approved narrow research train/eval** (not clinical/affective) |

**Next decision:** Prospecting is justified but **no Track B source has earned welfare-affect train/eval approval yet**. The strongest immediate paths are (a) paired expert FACS/video; (b) independently measured pig welfare/health endpoints matched to earlier behavior; (c) access to expert rater scored gait clips; (d) PigWeb cross-site indicator validity, using source data without emotion-label overclaim.

Do not enlarge the suite with a task that measures only the study's assigned condition or that gives a model the very source observation it is supposed to predict.
