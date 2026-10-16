# OpenFARM sprint: first source-audited, whole-video-heldout human ethogram evaluation

**Sprint acceptance contract:** (1) source-audited real human behavior annotations, (2) **video-level** independent evaluation, (3) cheap reproducible baselines, (4) behavior-by-behavior verdicts supporting or rejecting further welfare research.  
**Checkpoint:** 2026-10-15 local / 2026-10-16 UTC. **All four criteria achieved at a bounded exploratory level**. Does **not** establish clinical welfare decisions or independent animal/site transfer.

## A. Meerkat Behaviour Recognition Dataset: source-backed, full working benchmark

**Source:** [Rogers et al., CV4Animals 2023](https://arxiv.org/abs/2306.11326); original [University of Auckland archive](https://meerkat-dataset.github.io/Annotations.zip), linked from [official dataset website](https://meerkat-dataset.github.io/) and [Figshare record](https://auckland.figshare.com/articles/dataset/Meerkat_Behaviour_Recognition_Dataset/23538261). Dataset media/annotation rights: **CC BY-NC-SA 4.0**. **Research-only/noncommercial, preserve attribution.**

**Actual-file evidence:** Downloaded full **24,999,708-byte** annotations ZIP, SHA-256 `7ad7b369d5a9fc90a29c28dabfc0503c7c67af10d304d97ba1b5413260890463`, containing exactly **20 distinct video-named CSVs** and `behaviours.json`. Parsed the nine source fields `frame, object_type, track_id, behavior, occluded, x1, x2, y1, y2`. Source records provide 15 named ethogram behaviors plus `None`. Object IDs last only *within* one video; animal independence across videos is **not** established.

### Leakage/quality contract

- Deterministic original-file SHA check; source format, category, animal type and occlusion codes checked.
- Screen **meerkats only**, with non-occluded and known human-coded behavior rows. Do not equate occlusion/`None` to a behavior negative or a clinical outcome.
- Sampling every **50th frame** is fixed before training. Derive 6 static bbox features and 6 **causal past-only** motion features; source video ID, camera, date, time/frame, track ID and human behavior label **never become model features**.
- **Whole source videos are indivisible groups**, separately hashed/sorted within each camera (C2/C3) using frozen split key. 12 training videos, 4 calibration videos, **4 unseen held-out test videos**. No random frame or track partition, no source video overlap.
- **52,440** visible sampled training annotations; **14,775** calibration; **16,548** test. Each is an annotation row, not an independent animal or episode.
- Source videos come from **one Wellington Zoo enclosure** and may show repeat meerkat individuals. Hence this is **video-heldout ONLY**, not animal-heldout, new-site, natural-prevalence, or clinical welfare generalization.

### Cheap reproducible System-1 baseline results

Code: `scripts/benchmark_meerkat_video_heldout.py`. Run:
```bash
python -m pip install 'numpy<3' 'pandas>=2.2,<3' 'scikit-learn>=1.5,<2' requests
python scripts/benchmark_meerkat_video_heldout.py --report reports/meerkat-video-ethogram.json
```
Independent clean CI run: **[38837664706](https://github.com/ob1-s/OpenFARM/actions/runs/38837664706)**. Detailed all-class confusion, classwise precision/recall and per-video accuracy are in the GitHub Actions `meerkat-video-level-heldout-metrics` artifact. Only aggregates, not original human rows, are uploaded.

| Baseline (CPU) | Heldout accuracy | 15-class macro F1 |
|---|---:|---:|
| Prior from training class frequencies | 26.13% | 0.0276 |
| L2 logistic, **static bounding box** | 48.37% | 0.1856 |
| L2 logistic, box + **past-only motion** | 58.73% | 0.2440 |
| 55-estimator small extra trees, box + motion | **63.71%** | **0.2657** |

**Model selection** used calibration-video macro F1; the extra-trees model won (cal macro F1 0.3100). The test macro F1 stays low because the geometry-only state cannot identify many fine-grained social behaviors. This is the *intended truth-revealing diagnostic*, not a result to inflate into affect classification.

### Behavior-by-behavior gate

For a behavior to pass the exploratory *model-recoverability* gate we declared before evaluating, require at least **3 positive train videos and 2 positive test videos**, at least **50** held-out positive annotated rows, test F1 ≥ 0.55, precision ≥ 0.60 and recall ≥ 0.50, with no test-based threshold fitting. Passing = viable **further behavioral perception research**, not validated positive/negative emotional indicators and **not certified human inter-rater label reliability**.

| Behavior (source ethogram) | Test positives | Precision | Recall | F1 | Verdict |
|---|---:|---:|---:|---:|---|
| Raised guarding (vigilant) | 1,821 | .792 | .873 | **.830** | **Explore perception** |
| Foraging | 4,324 | .655 | .964 | **.780** | **Explore perception** |
| High sitting / standing (vigilant) | 1,526 | .755 | .626 | **.684** | **Explore perception** |
| Moving | 2,910 | .614 | .694 | **.652** | **Explore perception** |
| Dig burrow | 1,709 | .496 | .782 | .607 | **Not yet: imprecise** |
| Low sitting/standing | 1,822 | .591 | .240 | .341 | **Not yet: misses most** |
| Interaction with pup | 740 | .331 | .053 | .091 | **Not yet** |
| Lying/resting | 102 | .000 | .000 | .000 | **Not yet** |
| Grooming | 39 | .000 | .000 | .000 | **Not yet, rare** |
| Allogrooming | 3 | .000 | .000 | .000 | **Not yet, extremely rare** |
| Playfighting | 11 | .000 | .000 | .000 | **Not yet, very rare** |
| Carrying pup | 412 | .000 | .000 | .000 | **Not yet** |
| Sunbathing | 113 | .000 | .000 | .000 | **Not yet (only one test video)** |
| Human interaction | 62 | .000 | .000 | .000 | **Not yet (one train/test video)** |
| Interaction with foreign object | 954 | .000 | .000 | .000 | **Not yet** |

**Critical interpretation:** Bbox geometry + motion being insufficient for grooming, play and pup interactions **does not imply those behaviors are scientifically ill-defined**. These actions likely require raw pixels, mouth/paw pose, social partners, and longer time windows. Conversely high F1 for guarding/foraging does **not** prove independently annotated positive/negative valence. The source has one released human-coded label per row; **inter-annotator agreement not measured** on this dataset.

## B. Big Broiler: independent human-agreement extension, source access verified

A separate source is [Cardoen et al., Big Broiler (CV4Animals 2025)](https://github.com/decide-ugent/big-broiler-dataset), published as **AGPL-3.0** code/data. The official archive is ~33 GB, but HTTP **range reads succeeded** and the remote ZIP index has **102 individual annotation CSVs and 102 paired videos**. Real CSVs were inspected at video IDs `000` and `101`; the later heldout record contains `actions_annotator_1/2/3` and `actions_ground`, including preening and inactivity. **Do not use the official canonical _track_ splits as if they ensure heldout videos.**

This is an unusually good **human annotator reliability** benchmark: ten source video IDs **092–101** were coded by *three* annotators concurrently, and the reference `actions_ground` can be compared without treating it as an independent fourth annotator. The source-level video data live on the authors' server; OpenFARM scripts do not republish them. Reproducible script: `scripts/audit_broiler_human_reliability.py`, storing only per-behavior pairwise statistics. **This scientific reliability check is distinct from the successful Meerkat prediction benchmark.** Its independent outcome is to be recorded in this document when verified.

## Acceptance and next scientific implications

**Acceptance achieved:**
1. **Actual human-coded source inspected**: 20-file source archive + ethogram, checksummed.
2. **Video-level split defended**: full 12/4/4 source videos kept disjoint; no sample/track splitting.
3. **Cheap baseline result**: prior, logistic static, logistic with causal motion, very small extra-trees; concrete heldout numbers.
4. **Behavior evidence table**: four further-perception candidates; explicit rejections for rare/complex behaviors and no false claims of welfare.

**Remaining scientific work:** inter-rater evidence on external sources (Big Broiler), new-camera/site/animal generalization, positive-behavior pixels and audio/pose, natural time-based monitoring outcomes, and distinguishing *observed behavior* from *welfare-related interpretation* and *human-attention decisions*. Do not replace good ethology questions with context-derived emotion pseudo labels.

**No new model weights or licensed third-party videos are published, and the existing OpenFARM suite remains unchanged.** 
