# Public cattle observation — second *research* train/eval admission

**OpenFARM System-1 · 2026-10-15 · source-first, no emails or restricted data.**
**Decision:** APPROVED_RESEARCH_TRAIN and APPROVED_RESEARCH_EVAL **only for bounded within-study bovine behavior `Choice`**; NOT approved as clinical/welfare inference, continuous event-duration evaluation, affect, or calibrated deployed probabilities.

## Evidence / source

Original: Perea et al., *Labeled sensor dataset of beef cattle behavior grazing desert rangelands* (Figshare DOI [10.6084/m9.figshare.28801181.v2](https://doi.org/10.6084/m9.figshare.28801181.v2)), CC BY 4.0. **Actual published bytes accessed** using the Figshare API; `cattle_behavior_data_time_V2.csv`, MD5 `7cd143e8cccc79a5e159eaa83428529f`, SHA-256 `b4d6e14d6ff029dbf497287a774aaa27ec471e7ed1ef5884e9cf0207c784fe37`.

- **24 identifiable cattle**: four breed groups, six cows each; **12 Jornada animals / 12 Chihuahuan Desert animals**. Breed and location are **confounded**; cross-site comparisons are not pure site effects.
- **9,222 one-minute behavior observation records**, no missing values in Breed/ID/MI/State/Behavior.
- Labels: **3,928 GR** grazing; **2,286 WA** walking; **2,032 RE** resting; **976 RU** ruminating. Authors state these were coded from review of corresponding animal video by a trained operator and assistant, thresholding the majority behavior within each minute.
- Model input: ONLY **MI**, the source wearable accelerometer's proprietary *motion index* integer 0–30. Source `Behavior` is human coded. **Do not expose `State`** (AC/ST), which is a deterministic rewrite of the target. Do not expose Breed/ID/site/Time.MTD as features.
- **Timestamp caveat**: `Time.MTD` is NOT a validated unique global timestamp. **594 source rows repeat a cow/time key; 258 key groups have nonidentical MI/state/behavior**. Original source has 34 full-row exact repeats. Do not arbitrarily drop conflicting rows or fabricate unique minutes/episodes. Evaluation *as individual labeled observations on distinct cows* is allowed; **longitudinal chronology and alerts per cow-day are NOT**. Source-level provenance investigation remains a requirement for a time-series task.
- Other original file `Cattle_behavior_data.csv` drops Time.MTD and creates **8,105 identical rows**. Use only timestamped V2 for this task, even though its timestamp remains ambiguous.
- Experimental sensor scalar MI is **already preprocessed** by proprietary firmware. This task does not measure raw accelerometer feature extraction, eye/video recognition, pain, distress, clinical welfare, positive emotional state or intervention thresholds.

## Reproducible research task

`scripts/build_public_cattle_behavior_v0.py` downloads only source-author bytes and validates both original checksum and CC BY 4.0 metadata; performs identity, label and source time-key checks; locally materializes:

- `train/calibration/test.features.csv.gz` — only MI
- `train/calibration/test.labels.csv.gz` — independently coded GR/WA/RE/RU
- `train/calibration/test.provenance.csv.gz` — Breed/ID/site/Time.MTD/derived State, never seen by models

**Three distinct research protocols:**
1. Breed-balanced **16 cow training / 4 cow calibration / 4 cow test**, with one test cow per breed, 1,503 test observations.
2. **Jornada → Chihuahuan**: nine train cows and three cal cows from Jornada, all 12 Chihuahuan cows unseen at evaluation.
3. **Chihuahuan → Jornada**: reverse direction, nine train + three calibration from Chihuahuan and 12 heldout Jornada cows.

No animals cross any individual protocol's groups; however **all three protocols use the same underlying source data**. Do not count them as independent external datasets. Repeating model choice on all three test outcomes constitutes exploratory design work, not final sealed blind evaluation.

[GitHub Actions first source audit](https://github.com/ob1-s/OpenFARM/actions/runs/38830460260) and [group-heldout baseline result](https://github.com/ob1-s/OpenFARM/actions/runs/38830879270) were **successful**, as was the later [actual local train/cal/eval export check](https://github.com/ob1-s/OpenFARM/actions/runs/38831042355).

## Baselines: no GPU, no source-context leakage

| Protocol | Train-prevalence accuracy | L2 logistic accuracy | Tiny boosted tree accuracy | Tree macro-F1 |
|---|---:|---:|---:|---:|
| 16/4/4 cow holdout | **41.18%** | **81.57%** | **82.83%** | **0.643** |
| Jornada → Chihuahuan | 42.20% | 81.44% | 81.44% | 0.643 |
| Chihuahuan → Jornada | 42.94% | 81.55% | 82.10% | 0.640 |

**Do not headline accuracy alone.** The held-out overall majority of GR makes a large class imbalance; macro-F1 around 0.64 reveals significant minority-class errors. The test **log loss of the tiny tree is an alarming ~4.03**, despite >82% accuracy (logistic ~8.99), suggesting severe overconfidence or systematic near-zero probabilities on some correct labels. **Probabilistic calibration has NOT been achieved**, and we should not deploy these outputs as well-being decision confidence.

**Scientific interpretation:** We have a *real, legally reusable, independently human-annotated four-way observable behavior decision dataset*. A tiny learned System-1 classifier is useful but still poorly calibrated, and only one scalar input makes some classes inherently ambiguous. Group splits are sound by cow; source-reported timestamps are not suitable for event-level longitudinal inference.

**Source rights:** CC BY 4.0 applies to the Figshare tabular dataset; provide source author attribution. Underlying reviewed video is **not** in this release; do not republish or imply we can train from source video. Output a manifest + runnable loader, not a clone of the original source rows in our Git repo.

## What this does and does not advance

- **Adds one real, directly usable, source-verified, group-heldout `Choice` task** to the research suite alongside HotPig instrument-feeding `Noul`. Different species, observations, labels, sensors, and studies.
- Preserves EquiFACS expert-pain interpretation as a tiny **pilot** pending more independent animals.
- Does not provide independently validated positive valence/comfort, actual clinical state or caretaker escalation ground truth. Pursue those separately, but **never let access-dependent outreach block progress on public tasks**.

Next scientific work: reliability curves and calibration fit on independent cows; inspect misclassification by class/breed/site; recode a strict unknown/abstain option only with actual uncertainty criteria; seek additional sites/datasets *via public sources*, not emailing authors as routine.
