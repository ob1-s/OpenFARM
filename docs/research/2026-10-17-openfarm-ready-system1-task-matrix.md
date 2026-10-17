# OpenFARM System‑1 ready tasks — consolidated research matrix (17 October 2026)

**Taxonomy, not pipeline:** L1 observations, L2 expert welfare-relevant interpretation and L3 intervention-priority/attention or future adverse-event risk are *independent* task complexity/use-value classes. Neither training transfer nor a causal L1→L2→L3 ladder is presupposed. Favor L2/L3 commercial value, but preserve L1 data as useful general animal/motor-intelligence training.

## Original source-grounded tasks

| Source/task | Level / natural modality | Raw usable labeled instances | Main independent evaluation groups | Reference construct | Current scientific admission |
|---|---|---:|---|---|---|
| **Wong** expert vet ER triage | L3 / text clinical state | 340 cases × 3 vet ratings; 171 train/67 cal/102 test cases | 102 case-only test (animal identity unavailable); 9 majority-immediate | Each of 3 veterinarians independently ranks 5 hospital waiting priorities; disagree on 282/340 | Original file SHA-verified, live Jev and Julia CPU benchmark done; research candidate, **no deployment** |
| **Farrow** owner care-seeking | L3 / text owner vignette | 30 independent vignettes; 5,316 owner judgments are **NOT** 5,316 cases | 30 text scenarios; only 5 immediate cases | One original vet-panel consensus uniquely reconstructed from author-scored owner rows; raw 14 expert-vote matrix unavailable | Original PLOS DOCX/XLSX downloaded with hashes; Jev and Julia real model comparisons done; exploratory reused eval set |
| **Aarhus** first recorded new pig tail injury | L2→L3 risk / time-causal clinical numeric state | 32,995 prediction windows with later clinical label/censor masks | 15 independent heldout first-onset events | Later fresh-tail-injury clinical assessment within 7 days | Proposed time-causal event task; limited independent onset count; no timestamped caregiver actions |
| **GrimACE** mouse facial action units | L2 / original 1280×1024 images, **five questions per image** | 773 scored images / 15 mice, up to 3,865 image-AU decisions | 215 frames from 4 test mice; raters and missing masks | 4 human rater families × eye/nose/cheek/ear/whisker Score(0–2) | Source and animal-disjoint benchmark built; AU-specific reliability differs; pending final admission |
| **Big Broiler** observed feather-pecking | L1 / source tracking boxes, **video-native media available but not modeled yet** | 29,459 scorable 10-second windows, 92 labeled modeling videos (original source 102) | 9 diagnostic test videos across 3 camera setups; weak camera-family generalization | Released human video action label (single-coder lineage; IAA subsets) | Geometry→behavior validated as limited research candidate, source MP4 video ~32.5GB untested in own model; no clinical outcome |
| **HotPig** feeder intake presence | L1 / computer vision-derived 12 count features → independent feeder sensor | 550,818 valid minute rows, 24 pigs | 4 test pigs, 92,160 test minutes (time-correlated) | Independently measured feeder intake >0 | Approved bounded research train/eval; not welfare outcome |
| **Cattle MI** four recorded behaviors | L1 / wearable motion index | 9,222 source rows, 24 cows across 2 sites | 4 cow-heldout test; site-heldout protocols | Human video-observed graze/walk/rest/ruminate label | Approved bounded research train/eval; no clinical welfare label |
| **Meerkat** 15-action ethogram | L1 / human boxes+motion, original video-linked | 83,763 sampled train+cal+test label rows, 20 videos | 4 whole heldout videos, no proven animal independence | Human-coded 15 observable behaviors | Approved bounded noncommercial research, not raw pixel task |

**Do not add** count-derived positives across species/tasks and call them independent welfare events. Approved status is task/revision/split/license specific.

## Research model lanes and checkpoints

1. **Reference baselines before big models** — priors, clinical decisions with safety cost, label disagreement, simple supervised heads trained only on train cases, threshold/abstention calibrated only on cal.
2. **Hosted decision reference** — Jev 1.13/1.14 Choice+Score; compare probability distributions under exact task definition and reports; maintain prompt/rubric identifiers.
3. **Local original decision weights** — Julia‑1 (144M, Apache‑2.0, PyTorch CPU, text; run 39005324887) and Liquid d1‑omni‑600M (587M, LFM1.0, llama.cpp Q8 CPU, genuine text/vision/audio head; run separately). Check CPU latency including and excluding load; model downloads are not training.
4. **Video-native next** — verify at least a source MP4 decode and frame/timestamp alignment with labels; sample contiguous multi-frame windows (temporal video) without using future frames when target is prospective; report memory cost, video-disjoint/ camera-family-heldout behavior performance. Comparison with source box geometry must preserve same label/episode denominator.
5. **True benchmark admission** — freeze unexposed final subsets when possible; avoid iterating criteria on a repeatedly probed 30-case eval; add cross-source/new-site clinical and video sources rather than reusing narrow test ad infinitum.

## Critical scientific distinction learned from Wong

The train-only clinical label-frequency prior has lower Brier than zero-shot Jev 1.14 despite failing to rank any majority-immediate case first. A **trained TF-IDF+logit** selected on 67 calibration cases improves Brier again (0.305694 vs constant prior 0.340640) while *also recognizing 0/9 majority-immediate cases*, with 40.5% veterinarian-vote-weighted undertriage. This is not a leaderboard win for triage. Use **proper scoring rules and risk-sensitive clinical utility + abstention jointly**, with prospective clinician signoff if contemplating patient advice.

**No product is approved for live animal care, veterinary intervention, clinical advice or autonomous treatment** by these retrospective benchmarks.
