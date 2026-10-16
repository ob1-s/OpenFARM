# OpenFARM task admissions — first new-source candidate approved for bounded research

**Date:** 2026-10-15. This is *not* the product benchmark or a model-ranking result.

## HotPig: instrument-verified feeding observation (v0)

**Admission:** **APPROVED_RESEARCH_TRAIN** and **APPROVED_RESEARCH_EVAL — single-site, animal-heldout, instrumental observation only.** Not approved for a welfare/stress/affect benchmark, commercial product validation, cross-farm transfer claims, or caretaker alerts.

**Why this task is admitted:** a real 16-day monitoring archive with 24 distinct pigs; a source-independent electronic feeder measurement provides a concrete timestamp-aligned observable target, while the supplied image-derived posture/event streams form bounded low-cost states. We did not invent a welfare label from exposure condition or emotional interpretation. The raw source file was downloaded, cryptographically verified, all 552,960 minute rows parsed, and an actual split-preserving dataset builder executed successfully in clean GitHub CI.

- **Primary source:** [HotPig — Zenodo 17090997](https://zenodo.org/records/17090997), DOI 10.5281/zenodo.17090997; record-level **CC BY 4.0**; attribute source creators. Exact ZIP SHA-256 `1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8`.
- **Question:** Noul: “Did the independently instrumented feeder register intake greater than zero during the current minute?” The label is derived deterministically from the **independent feeder reading**, not from a camera-generated activity label or experimentally assigned heat-stress condition.
- **Model inputs:** 12 numeric camera-derived posture/event count features (`unknown`, `standing`, `seating`, `lying`, `eating`, `drinking`, `cuddling`, `curious`, `idle`, `drink`, `eat`, `mate`), all author machine-generated, not manual frame GT. **NEVER** expose feeder output (`feed`), treatment (`conditions`), time (`period` or minute index), source pig identifier, or derived source `state` to the learner.
- **Invalid rows:** Drop only source-minute records with missing/non-finite target or features. Across all pigs, 552,960 minutes -> **550,818 valid**. Do not turn missing feeder data into zero or sample synthetic negatives.
- **Grouping:** source pigs **1–16** train, **17–20** calibration/threshold selection, **21–24** independently evaluated test. No animal overlap. These pigs share *one trial/facility and experimental timeline*, so this is **NOT** cross-farm generalization; overlapping minutes within a pig/day are correlated, and number of independent animals is 24, not 550,818.
- **Train:** 368,508 valid minute samples; 17,538 positive (~4.76%).
- **Calibration:** 90,150 valid minute samples; 4,440 positive (~4.93%).
- **Evaluation:** 92,160 valid minute samples; 4,834 positive (~5.25%).
- **Frozen membership digests:** train `5bf5361d8d64b5598b3e55199aafe328451c367c814d3166f112b5899574d99c`; calibration `41ea4fbf98c4f293aa62c9bc6bc35cb89f60c72471871f82cd3ad230e8df41c4`; test `ddc0d59f42f6bf87c35db8f0b626746231b4c738e47dd5c37c2d56ef79aa133e`.
- **Evidence/implementation:** `scripts/validate_first_sources.py`, `scripts/evaluate_hotpig_feeder_candidate.py`, `scripts/build_hotpig_feeder_v0.py`. Reproducible build verified in [GitHub Actions run 38820879969](https://github.com/ob1-s/OpenFARM/actions/runs/38820879969). Data are generated as separate training/labels/provenance files **locally**, with only the source author’s license and immutable links in Git; the CI artifact contains **manifest only**, not redistributed source rows.

### Mandatory sanity baseline

A frozen rule that declares a feeder event when source video model `eating > 0` was present gave:

| Split | Precision | Recall | False-positive **minutes** per pig-day |
|---|---:|---:|---:|
| Train (16 pigs) | 0.5901 | 0.8780 | 41.80 |
| Calibration (4 pigs) | 0.5688 | 0.7291 | 39.20 |
| Test (4 pigs) | **0.5868** | **0.8076** | **42.95** |

**Do not report those 42.95 false-positive minutes as welfare alerts**; an event-deduplication/temporal aggregation and independent reviewed animal-welfare consequences would be needed first. The majority always-negative baseline has 94.76% **test accuracy but zero feeder-event recall**. The trivial-rule test scores were inspected before downstream model training, so **they are exploratory and do not constitute a blind model selection contest**.

### Research evaluation success criteria

Compare source prior and always-negative, this published `eating>0` rule, regularized logistic/trees on the 12 features, cheap trained/non-AR typed heads, and zero-shot generative models **on semantically equivalent inputs**. Do not allow a richer model to see the actual target or heat condition while simpler baselines cannot. Keep raw probabilities; evaluate Brier/log loss, AUROC/AUPRC, per-pig recall/precision, calibration using **only the four calibration pigs**, animal-weighted uncertainty and OOD limitations. Source-derived machine activity features are not independent human behavior truth. **No welfare, affect, or deployment claims** may be drawn from this task.

### Separate permission/use statuses

| Intended use | Status |
|---|---|
| Source access | **Verified** |
| Source media/annotation and license metadata | **Verified for these machine-generated count and independent feeder series** |
| Research training for this exact question | **Approved — narrow and attributed** |
| Research within-study pig-heldout evaluation | **Approved — limited; only four test animals** |
| Cross-farm / clinical well-being outcome validation | **Not approved / not provided** |
| Commercial training or product deployment | **Not cleared by this audit** (though record-level CC BY is favorable; no product validation) |

## Cattle wearable MI → expert video-observed behavior (v0)

**Admission:** **APPROVED_RESEARCH_TRAIN + APPROVED_RESEARCH_EVAL (bounded, 24 cows / two locations)**. A genuine low-cost multimodal *observation* supervision task; **not** independently certified clinical welfare, affect or caregiver escalation.

- **Task:** `Choice(GR grazing, WA walking, RE resting, RU ruminating)` from scalar wearable motion index `MI` alone, *video-reviewed human behavior* as reference.
- **Source:** [Figshare 28801181 version 2](https://doi.org/10.6084/m9.figshare.28801181.v2), **CC BY 4.0**, timestamped `cattle_behavior_data_time_V2.csv`, SHA256 `b4d6e14d6ff029dbf497287a774aaa27ec471e7ed1ef5884e9cf0207c784fe37`. Download and actual-file source SHA validated.
- **Observations:** 9,222 source rows from **24 distinct cows**, with six animals in each breed and twelve at each of two ranch sites. Breed and site are confounded.
- **Research splits:** 16 train/4 calibration/4 test cows, one test cow per breed; and separate directional all-animal **site-heldout** protocols (9 train/3 calibration/12 test cows). Use only `MI` as model input; keep `Breed`, `ID`, `Time.MTD`, `site`, source `State` and label in separate provenance.
- **Verified CPU models:** On 4 cow-heldout test animals, prior accuracy **41.18%**, logistic **81.57%**, tiny tree **82.83%** (macro-F1 **0.643**); site-transfer trees **81.44% / 82.10%**. **Log-loss is severe (~4.03 tree, ~8.99 logistic); do not claim calibrated decision confidence or deployability.**
- **Data integrity caution:** `Time.MTD` isn't uniquely identified per cow: 258 repeated cow/time keys carry differing values, and 34 full-row duplicates exist. We preserved records rather than deleting plausible different-session observations. **Not an approved temporal event/longitudinal label**. All research evaluation split by cow.
- **Scripts:** `scripts/audit_open_cattle_behavior.py` and `scripts/build_public_cattle_behavior_v0.py`; [actual downloadable files audit](https://github.com/ob1-s/OpenFARM/actions/runs/38830460260), [CPU baselines green](https://github.com/ob1-s/OpenFARM/actions/runs/38830879270), [local research train/eval artifacts verified](https://github.com/ob1-s/OpenFARM/actions/runs/38831042355).
- **Full evidence:** [Cattle source and decision admission](2026-10-public-cattle-choice-admission.md).

## EquiFACS: constructed scientific pilot, NOT approved benchmark

**Status:** **TASK_VALIDATED / PILOT CONSTRUCTED**, **not** approved for a reliable held-out veterinary benchmark, calibrated pain screening, or deployment. `scripts/prepare_equifacs_structured.py` aligned source facial-action intervals to independent clinical ratings. Strictly excluding Alina's pain ratings because Alina also coded the facial actions leaves **64 codeable intervals from Johan/Camilla**. Excluding 10 overlapping unscorable-AU intervals leaves **54 binary clinical judgments across 18 media**, with source-group-disjoint 10 train / 3 calibration / 5 test media. Only **12 held-out rating intervals** and 5 held-out clinical videos; `Score(0–3)` is not supported because no independently retained moderate ratings. [Real-source build passed](https://github.com/ob1-s/OpenFARM/actions/runs/38825087764). See [Track B dossier](2026-10-track-b-affective-supervision-audit.md) for the crucial rater, missingness, and animal-group caveats.  
 
Previous annotation-only source status is superseded. The original 2020 study has 12 experimental videos (six distinct horses, six baseline and six ischemia/pain) and 21 clinical horses. The clinical XLSX has 167 pain interval rows, **22 distinct media identifiers**, and 77+ non-codeable entries. Those rows/IDs do **not** demonstrate 167 independent video samples or 22 separate clinical horses. Wide FACS annotations need source-video/timestamp joining, explicit abstention on non-rateable intervals, and horse-level test design. Video supplement ~880 MB; not automatically downloaded or redistributed. Not the old 2021 Lencioni Horse Grimace crops.

## Pig tail-biting and pig social-ethogram tables

Pig feeder source **CC BY-NC 4.0**, access open in metadata but raw 4MB file timed out; social ethogram API/file inspection timed out. Both remain **ACCESS_PENDING for actual source data**. Never use artificial TBSTART pseudo-dates as observed control outbreaks. No approved alerts or research evaluation until true temporal outcomes and animal/pen/site splits are validated.

## Reminder

One **bounded nonclinical instrumental behavior task** is now approved for research train+eval. **No new validated affective or welfare-intervention benchmark** has yet been approved. This is a valid first step in the larger System-1 empathy suite—not evidence the full product anchor is solved.
