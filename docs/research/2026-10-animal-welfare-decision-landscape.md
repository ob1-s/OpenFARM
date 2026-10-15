# OpenFARM: animal welfare and behavior *decision* data — landscape survey v0

**Date:** 2026-10-15 · **Status:** externally source-backed *discovery sweep*, **not** a cleaned data release or full audit of source files.  
**Related:** [37-entry source register](2026-10-animal-data-source-register.csv); [OpenFARM legacy audit](../system1/FINAL_PRODUCT_FIT_AUDIT.md).

> **Product/research objective:** Can a small, cheap, non-autoregressive, calibrated model infer defensible, action-oriented animal welfare and behavior decisions from bounded or compact longitudinal states, matching useful generative-model performance and eventually lowering alert costs?

**Crucial outcome:** The legacy audit gave us one narrow horse Grimace cue seed but **zero validated commercial monitoring datasets**. This fresh sweep finds more plausible supervision in *ethology, veterinary clinical scoring, behavior-event annotation, wearable/feeder time series, and causal interventions* than in web-labeled sentiment/animal emotion benchmarks.

## Epistemic standard: what has/hasn't been verified

- **Verified** at this stage: source landing pages, paper summaries, stated file listings/download availability, stated study designs and label protocols where identifiable. We have not downloaded and decoded all 37 sources, nor certified raw annotations, inter-rater consistency, sample/subject/site splits, actual source media/license chain, or training/data rights.
- Dataset linked ≠ dataset complete; paper linked ≠ source media accessible; paper CC-BY ≠ rights to every video/image deposited; public watchable video ≠ commercial training rights.
- Distinguish **gold** (independently expert-observed time-localized event/clinical score), **silver** (source-defined experimentally manipulated condition or machine-generated sensor proxy), **bronze** (weak context/derived classification), **ontology-only**, **index-only**, and **unavailable**. These labels are *assessment goals*, not yet assigned as certified grades.
- **Observation**: measurable event/posture/vocalization. **Interpretation**: indicator or clinical welfare score with a defensible rubric. **Action**: escalation to a human after decision analysis and false-alert costs. Avoid deriving “emotion” from context, behavior stereotypes or absence of activity.
- Time windows **must be causal** for prospective warning tasks. Do not show models future outcomes, collection context that encodes targets, videos of treatment itself, neighboring holdout sessions, or repeated frames from the same bout.

## First shortlist: where to inspect actual rows and annotations first

| Rank | Source and exact evidence | Attractive atomic decisions | Major flaw / required check | Accessible? |
|---|---|---|---|---|
| **1A** | **[Two-farm pig feeding and tail-biting](https://zenodo.org/records/5028016)**: 65 pens, 27 outbreaks, daily feeder metrics and outbreak timestamps | `Noul(tail-biting-event-next-7d)`; change-from-pen-baseline; alert calibration at real prevalence | Only **2 farms**; artificially assigned control pseudo-event dates; no post-outbreak leakage; many overlapping windows; stratify by pen and leave-one-farm-out | Public small text deposit; license verify |
| **1B** | **[Cow locomotion / 10 raters / Brazil](https://animalwelfarehub.org/cows-project/)**: **215 paired examples, 430 videos**, full ten individual locomotion scores per paired observation | `Score(lameness:0–3)`, probability over experts’ ordinal ratings, visibility/abstention; low-dimensional gait pose decision | 215 **paired observations, not 430 independent animals**; farm and animal IDs need investigation; MP4 and score spreadsheet *described*, but **direct public download URLs not independently verified**; 3D files on request; rights unresolved | Website describes repository; access needs further verification |
| **1C** | **[HotPig](https://zenodo.org/records/17090997)**: 24 pigs, 9 days thermal-neutral, 6 days 32°C and 1 day recovery, manually annotated training images; minute-level model-generated posture and event series, independent feeder data | `Noul(behavior-changed-from-baseline)`; `Choice(posture)`; event-rate/temperature risk; per-pig day-level states | **Do not treat 16-day YOLO model outputs as independent human labels**; intervention known but not direct stress label; 24 animals and one experimental facility | 2.6MB series, 621MB manual frames, media/demo; licensing check |
| **1D** | **[GrimACE](https://zenodo.org/records/18299984)**: multimodal mouse cage-side expert facial units, perioperative changes, pose, analgesia, ~9GB deposit | quality gate + independent facial action `Score(0–2)`, clinical review decision after human-validated threshold; baseline change | Some ratings are non-rateable; ~31 mice, domain/implant effects; rater agreement, video grouping, and license first | Large open archive, code AGPL3; data permission separately |
| **1E** | **[EquiFACS pain](https://pmc.ncbi.nlm.nih.gov/articles/PMC7608869/)**: six experimental horses, 21 clinical horses; timed facial-action labels; 3-vet clinical ratings; experimental-video supplementary | `Noul(facial-AU-active)`; ordinal facial cue; compare AU pattern to blinded clinical outcome | Experimental sample extremely small; clinical media availability/rights uncertain; no prospective deployment base-rate | Experimental videos public supplementary; clinical media unverified |
| **1F** | **[Cow acoustic grazing/rumination](https://www.nature.com/articles/s41597-023-02673-3)**: 708h audio, 392h expert bouts, >6200 labeled jaw movements | `Noul(jaw-chew)`; `Choice(grazing/rumination/other)`; compact daily eating-time deviation | Five cows, animal-heldout uncertainty, wearable microphone vs environment, no guaranteed direct disease events | Paper/data publication; exact media access verify |

These are not a single ranked leaderboard: **tail-biting and HotPig** best align with low-cost existing feeder/temporal sensor monitoring; **cow lameness/GrimACE/EquiFACS** best align with expert-rated *welfare-related* supervision; **cow grazing audio** gives attractive inexpensive perception and compact event histories.

## Second tier: action recognition, ethograms, posture and difficult domains

### Behavior and temporal events (better observable labels than generic emotion)
- **[Edinburgh Pig Behavior](https://homepages.inf.ed.ac.uk/rbf/PIGDATA/)**: eight pigs in one pen, full multi-day recordings and **manually annotated minority** (16 behaviors incl fighting/tail-biting). Multiple cheap `Noul` event decisions; enormous temporal autocorrelation and one-pen domain limitation.
- **[MABe 2025](https://www.kaggle.com/competitions/MABe-mouse-behavior-detection/data)** and [CalMS21](https://data.caltech.edu/records/s0vdx-0k302): mouse interval-level social ethogram with already-compressed pose trajectories; excellent *System-1 temporal* probe, not direct affect/welfare. Verify source overlaps across competitions, lab holdout and unannotated windows.
- **[Meerkat behavior](https://meerkat-dataset.github.io/)**: multi-action grooming, playfighting, guarding, foraging. Observable **affiliative behavior** but *not* calibrated happiness. 20 videos from zoo enclosure; **CC BY-NC-SA** blocks straightforward commercial usage.
- **[ChickenVerse](https://github.com/amirivojdan/ChickenVerse)**: 15,250 human-labeled broiler behavioral clips and 6,539 boxes/masks from five facilities, with facility-heldout *detection* evaluation. Check **behavior clip** split independence separately. **CC BY-NC-SA 4.0**, research only without license clearance.
- **[NESTLER poultry](https://zenodo.org/records/20924893)**: 1.2GB six videos with walking/standing/lying/pecking bounding-box and pose annotations from Bulgaria and Rwanda. Positive geographically shifted evaluation opportunity; tiny recording/session sample and rights to validate.
- **[HotPig image and temporal components](https://zenodo.org/records/17090997)** can support both image-level posture and time-series change decisions, but these are different label-validity claims.

### Veterinary locomotion and clinical measures
- **[2026 cow keypoint trajectories](https://github.com/hrussel/lstm-lameness-detection)**: **272 released trajectories with locomotion scores**. Particularly strong cheap structured System-1 candidate for typed gait severity; must find identities/farm splits and dataset/code licenses.
- **[CowScreeningDB](https://github.com/Shahid-Ismail/CowScreeningDB-A-public-database-for-lameness-detection)**: dairy lameness videos downloadable **only after signed license/email**, not a free public data dump; worth author outreach if needed.
- **[Southern Brazil bovine lameness](https://zenodo.org/records/3408818)**: cross-sectional bovine welfare clinical data across **50 farms** (38 freestall, 12 compost), valuable domain-rubric/structured predictor test, but **no synchronized video**. Beware predictors which include the clinician's conclusion.
- **[Dog Pain Database 2025](https://zenodo.org/records/15304081)**: 200 clinical dogs with real veterinary scale ratings and annotations; spreadsheets accessible, paired **videos by author contact**. Do not accidentally recreate old target-predictor circularity.
- **[2026 pig cough/footpain/healthy videos](https://www.sciencedirect.com/science/article/pii/S221431732600065X)**: paper claims large farm cohort and triple-vet labels. **No verified public download**. High-impact collaboration lead, not training-ready evidence.

### Relevant ethology/quality-gating and novel welfare observations
- **[Agroscope pig interaction + lesion dataset](https://zenodo.org/records/5920843)**: 95 pigs, four 5-minute observed sessions each, rich **human ethogram** with 150 social-interaction behaviors, other events and subsequent lesions. Six tables **not raw video**. Useful for ontology, structured risk hypotheses and temporal supervision; no image perception claim.
- **[Mouse Pose Analysis](https://google.github.io/mouse-pose-analysis-dataset/)**: 455 manually action-labeled 0.5-second video clips and technician keypoints. Cheap pose/ethogram signal.
- **[Cow posture 2026](https://zenodo.org/records/20426189)**: standing/lying bboxes; useful *perception* (not welfare outcome).
- **[Cow Behavior Video Dataset CBVD-5](https://www.nature.com/articles/s41598-024-65953-x)**: multi-action dairy video and source study, **downloadable data not verified**.
- **[Sheep activity video dataset 2024](https://www.sciencedirect.com/science/article/pii/S2352340924000015)**: grazing/running/standing etc with annotated clips. Better input modality than old Sheep APR, but no subjective welfare ground truth.
- **[Google/AP10K/Animal Kingdom](https://openaccess.thecvf.com/content/CVPR2022/html/Ng_Animal_Kingdom_A_Large_and_Diverse_Dataset_for_Animal_Behavior_CVPR_2022_paper.html)**: broad action/pose representation seeds, **weak as welfare decision GT**, copyright and video provenance barriers.
- **[Farm pig PigLife](https://data.aifarms.org/view/piglife)** (restricted research use) and [Newcastle pig image datasets](https://outputs.ncl.ac.uk/articles/dataset/Automated_recognition_of_postures_and_drinking_behaviour_for_the_detection_of_compromised_health_in_pigs/13042619) (**CC BY-NC-ND**): strong research-only baselines, not cleared for commercial data training.

## Survey multipliers and indicator ontologies

A deep sweep cannot search only for filenames like `animal-emotion-dataset`. Search **human experimental welfare literatures** and their supplementary deposits:
- [Welfare Quality assessment manuals](https://www.welfarequalitynetwork.net/en-us/reports/assessment-protocols/) and [AWIN](https://cordis.europa.eu/project/id/266213/reporting): validated observation *definitions*, not datasets.
- [EFSA-based animal-welfare indicators review](https://www.frontiersin.org/journals/veterinary-science/articles/10.3389/fvets.2025.1661470/full): a broad map of indicator types, **not** labeled media.
- [Livestock computer-vision systematic source index](https://github.com/Anil-Bhujel/Public-Computer-Vision-Dataset-A-Systematic-Survey) (67 dataset leads), and [poultry dataset review](https://pmc.ncbi.nlm.nih.gov/articles/PMC11762189/) (~20 studies): **search amplifiers**; their inclusion is not proof that datasets are public, legally reusable or behaviorally valid.
- [Brazilian UNESP animalpain](https://animalpain.org/) veterinary scale definitions and clinical demos are candidate collaboration/annotation protocols; embedded demonstration video is not presumptively commercial ML training material.
- **[V-TRACE (2026)](https://www.biorxiv.org/content/10.64898/2026.04.14.718392v2.full)** and [DeepEthogram](https://github.com/jbohnslav/deepethogram): **existing behavior-perception baselines/annotation tools**, not stand-ins for original event ground truth.
- **[Welfare Footprint](https://welfarefootprint.org/framework)**: a framework for intensity/duration/prevalence; helps articulate outcome utility without declaring welfare labels for arbitrary footage.

## Product-oriented opportunity map (hypotheses, no feasibility claim)

| Product wedge | Sensor/perception cost | What event is independently observed? | First viable decisions | Hardest blocker |
|---|---|---|---|---|
| **Pig pen tail-biting precursor warning** | Very cheap *inference* on installed feeder/pen event metrics; instrument itself may be expensive | Recorded tail-biting onset dates; lesion follow-up | Risk in next 7 days, owner/pen baseline deviation, human-review threshold | Only 2 farms; very few outbreaks; highly correlated windows; event definitions |
| **Farm pig behavior/heat response** | Camera posture detector once, then compact minute stats | Manual event/posture for frames; thermal manipulation and electronic feeding corroboration | Detect unusual feeding/lying patterns vs baseline; issue review when sustained | Only 24 pigs/one setup; proxy thermal exposure ≠ distress; detector error |
| **Dairy cow mobility/lameness triage** | Passage camera + pose encoder (or trajectory stream), sparse per-animal samples | 10 human expert ordinal locomotion ratings | `Score(0–3)` with disagreement, Noul(abnormal gait), abstain/ask new clip | Independent animals/sites, minority severe cases, rights and real costs |
| **Research mouse postoperative monitoring** | Cage-side cameras, animal bbox/face model plus sparse landmarks | Expert facial units, procedure + analgesia treatment | Facial-unit scores, changes from baseline, triage for handler | 31 mice/one lab; no deployment generalization; research-specific buyers |
| **Dairy feeding/rumination anomalies** | Wearable acoustic/accelerometer capture, cheap streaming features | Expert-labeled feeding/rumination intervals, jaw events | Activity and daily budget, Noul(anomaly to check) | True clinical outcomes and multi-farm recordings |

**Caveat:** Ground truth for *whether to alert* is not simply animal discomfort. It requires expert-defined review criteria, intervention costs, prevalence, and a prospective record of actual missed vs false alarms.

## Proposed System-1 task contract and benchmarking ladder

### State → questions → probabilities → calibrated actions

**Example A: pig pen/day compact state:** prior 7–14 days only, feeding-time distribution, visit counts, variability, number of pigs, ambient context where independently measured, observation quality. Questions:
- `Noul(Is feeding time abnormally decreased relative to this pen's baseline?)` — requires baseline and sensor quality checks.
- `Noul(Will a documented damaging tail-biting episode begin in the next 7 days?)` — validated on held-out pens/farm and properly censored negatives; label is observed future event, **not a claim of current emotion**.
- `Noul(Should a caretaker review the pen now?)` — **later**, only after human-cost/intervention thresholds and independently checked event/false-alert data exist.

**Example B: cow passing a camera:** short gait clip or pose trajectory plus animal/optical quality. Questions:
- `Noul(Is the locomotion assessment sufficiently visible?)` — needs independent quality labels.
- `Score(0,1,2,3): locomotion under the published DairyNZ rubric` — preserve distributions of expert ratings and **rateability**. Ordinal mean alone is not full uncertainty.
- `Noul(Is the animal's locomotion measurably changed from a preceding valid observation?)` — requires actual paired time points/animal IDs, not synthetic pairs.

**Example C: monitoring positives:** grooming, play, rest, feeding, exploration, affiliative interactions are **observable ethological events**. Inference that they imply *positive emotional valence* requires independent welfare evidence and context, so do not train `happy=yes` from `grooming=yes` by fiat.

### Mandatory scientific and engineering tests

1. Ingest immutable source revisions; inspect real media/tabular examples and primary annotation code/agreements.
2. Train/test partition by subject, pen, site, session/bout and publication where necessary; exact+perceptual duplicate scanning; *calibration/thresholds fitted on independent validation groups*. If only one farm, no cross-farm claims.
3. Artifact-only baselines: study/source metadata, equipment ID, record date, breed/species, clock time, event prevalence and numeric sensor baseline. Run a naive reference rule and regularized logistic classifier/gradient-boosted trees for tabular.
4. Frozen pretrained modality encoders+linear heads, lightweight temporal models; matched prompts/inputs for frontier generative zero-shot+; explicitly note if a model receives metadata that another doesn't.
5. Retain *full per-option probabilities*, not just top classes or model verbal “confidence”; Brier/log loss, proper ordinal metrics, classwise ECE where sample sizes allow, bootstrap by animal/pen/farm and risk-coverage/abstention.
6. For a prospective product, require **episodes and time** (not balanced clips), event recall, alert precision, **false alerts per animal-day or per pen-day**, lead time before adverse event, review work, peak compute, joules and expense per monitored animal-day.

## Access, copyright, consent and commercialization

**Source register distinguishes access and license.** Openly indexed ≠ commercially reusable. Examples:
- Research-only/noncommercial: Meerkat **CC BY-NC-SA**, ChickenVerse **CC BY-NC-SA**, Newcastle pig **CC BY-NC-ND**, PigLife custom research-only.
- Source available only by negotiating terms or requesting media: CowScreeningDB (signed agreement), dog pain video, 2026 pig clinical footage.
- Public Zenodo archives with licenses not unambiguously rendered in initial web results are **rights pending**, not “CC0” or free commercial assets.
- Commercial training rights, redistributing frames/labels, publishing weights trained on disputed media, subject/owner/farm privacy, source website terms and downstream licensing are **separate** review questions. Seek author authorization before using raw public YouTube/TikTok/social images/video, even if visible without payment.
- Never infer pain, distress or “high welfare” from video caption alone, or from a happy-looking animal. Expert labels and observable discriminators matter, and “insufficient evidence” is a valid decision.

## Prospecting sequence: high-quality data per compute-hour

1. **First inspect the *small cheap sources***: pig feeder/outbreak tables (few MB), HotPig **series.zip** (~2.6MB), pig social interaction lesion tables (<1MB), ten-rater cow gait spreadsheet, 2026 cow lameness 272 pose trajectories.
2. **Then inspect representative *sensor ground truth***: verify cow locomotion paired MP4 and rating-file download path with USP-linked team (its landing page describes files but exposes no independently verified direct links), then dairy grazing/rumination audio, HotPig manual frames, mouse GrimACE rater masks and per-animal grouping. Validate annotations and source rights before bulk downloads.
3. **Then build actual decision task candidates**, not rebranded source labels: their exact typed propositions, raw observation state, verifier and abstention/quality policies, subject/site holdout, and expected deployment prevalence.
4. **Contact source authors only for high-value missing pieces**: individual IDs/farm IDs, consent for commercial use, source clips, clinical validators, and ongoing continuous streams. Paid/commercial licenses last resort after feasibility proof.
5. **Extend to the literature horizon**: deep backward/forward citation graph from these seeds; EFSA/Welfare Quality taxonomies and 67+20 public dataset-review listings; links to methods and existing specialists (V-TRACE, GrimACE), and separately existing synthetic bioacoustics/pose augmentation literature.

## Not done in v0

No bulk datasets downloaded; no paired media/license legal sign-off; no reannotation or manual view of full clips; no full systematic snowballing of all papers in all species; no trained-model, calibration, or prospective product results. This is a **substantive source-verification and prioritization pass**, not a complete data forensic audit. Continue into source-by-source gates; keep every rejected source in the register with why.

**Tracking:** [Machine-readable source register](2026-10-animal-data-source-register.csv). Source pages and original publications are linked inline throughout this document.
