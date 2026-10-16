# OpenFARM Decision Suite v0.1 — canonical research baseline

**Status:** Candidate canonical *research* suite, documented on `research/2026-10-data-prospecting` pending PR review. Not a validated animal-welfare product benchmark.  
**Mission:** Small, affordable, typed, calibrated System-1 decisions that **learn what deserves human attention from observable animal behavior and defensible evidence**. Support multimodal and interleaved observations without pretending that datasets contain modalities they do not.

This is the **first fixed, bounded inventory** to compare cheap specialist decision heads and generative zero-shot+ baselines; source-specific scientific assertions and limitations are part of the specification, not optional footnotes.

## Evidence ladder and admissions

| Status | ID | Tier | Exact question and real observed inputs | Verified scope | Not established |
|---|---|---|---|---|---|
| **Research train + eval** | `OF-OBS-001` | **Observe** | `Noul`: Did the independent electronic feeder record intake during this minute? Input: 12 **already machine-generated** camera posture/event counts. | **HotPig** CC BY 4.0; **24 pigs**; **368,508 / 90,150 / 92,160** train/cal/test minute records from **16 / 4 / 4 pigs**. | Visual model quality from raw pixels, feeding welfare/disease, farm transfer, caregiver alerts |
| **Research train + eval** | `OF-OBS-002` | **Observe** | `Choice(GR, WA, RE, RU)`: grazing, walking, resting or ruminating? Input: one wearable **motion-index scalar**, independently compared to *human video-reviewed* behavior. | **Public cattle** CC BY 4.0; **9,222** labeled observations, **24 cows**, source cows disjoint in 16/4/4 and two directional site-heldout protocols. | Raw accelerometer understanding, clinical welfare or emotion, reliable video access, validated timestamp chronology |
| **Research train + eval (NC)** | `OF-OBS-003` | **Observe** | `Choice(15 ethogram behaviors)`: visible meerkat behavior? Input: **human bounding-box geometry plus prior-only motion**. | **Meerkat** CC BY-NC-SA 4.0; **20 source videos**; **12 / 4 / 4 entire videos**, with **52,440 / 14,775 / 16,548** sampled labeled track-frame records. | Raw video/pixel classification, held-out animals/sites, interpretable affect, reliable rare social behaviors |
| **Exploratory seed** | `OF-WLF-001` | **Interpret** | `Score(0,1,2)`: graded *facial-region grimace cue* from an actual cropped horse image. | **Horse Grimace** source-verified byte dedup; **829** balanced training crops (five horses) and **279** test crops (two horses). | Clinical diagnosis, independent multi-rater agreement, field generalization |
| **Exploratory pilot** | `OF-WLF-002` | **Interpret** | `Noul`: does an independent veterinarian's interval judgment indicate **any pain**? Input: preannotated **expert FACS AU counts/time occupancy**, not raw video. | **EquiFACS** source XLSX; coder's own clinical ratings and unscorable AU intervals removed; **54 rated intervals** over **18 media**, split **32 / 10 / 12 intervals** and **10 / 3 / 5 media**. | Generalizable clinical pain assessment, independently verified animal IDs/site, reliable 0–3 ordinal calibration, autonomous face perception |

**No Level-3 caretaker-attention task is approved.** In particular, positive behaviors alone do not certify happiness; a camera/feeder mismatch is not a welfare alert. Source review and human outcome evidence must precede any `needs_human_attention` target.

### Strongest existing cheap baselines, not a single comparable leaderboard

- `OF-OBS-001`: fixed `eating > 0` rule: test F1 **0.680**, ~**42.95 false-positive *minutes*/pig-day**; tiny boosted tree (12 numeric features, calibration-group-selected threshold): test F1 **0.759**, precision **0.810**, recall **0.714**, ~**12.66** false-positive minutes/pig-day. This is **sensor disagreement**, not welfare alarms. [Evidence](2026-10-track-A-learned-baselines.md).
- `OF-OBS-002`: source four-way behavior `Choice` from motion index: tiny-tree cow-heldout test accuracy **82.83%**, macro-F1 **0.643**; **severely poor probability log loss (~4.03)**. Directional site transfer is confounded by breed. [Evidence](2026-10-public-cattle-choice-admission.md).
- `OF-OBS-003`: camera-balanced, entire-video-heldout, 15-way ethogram: tiny ExtraTrees held-out accuracy **63.71%**; macro-F1 **0.266**. Raised guarding, foraging, high-sitting/standing and moving are promising *observable* targets, while grooming, allogrooming and playfighting have insufficient support/perception from bounding boxes. [Evidence](2026-10-video-heldout-sprint.md).
- `OF-WLF-001` and `OF-WLF-002` are **not calibrated baseline claims**; runnable source-data preparations exist, but test animal/media counts are too small to publish generalized welfare model rankings.

### Multimodal/interleaved task interface — invariant, not a promise of present coverage

A future OpenFARM model receives a **typed observation state**, possibly multiple/interleaved channels (images, image sequences, audio/spectrograms, keypoints, wearable streams, physiological/tabular measurements, previous observed events, source-backed clinical metadata), and returns one or more **typed probabilities/scores**. Individual tasks may use just **one** channel or a small bundle; do **not** invent multimodal pairs by combining unrelated animals or studies.

Minimal conceptual example, **not a requirement to serialize this exact JSON**:

```json
{
  "observations": [{"modality": "structured_motion", "window": "one minute", "value": {"motion_index": 12}}],
  "questions": [{"id": "observed_behavior", "primitive": "Choice",
                 "options": ["grazing", "walking", "resting", "ruminating"]}]
}
```

An output is a full probability distribution over the task's source-defined choices, not a one-word response dressed up as confidence. `Noul` produces `p_yes`; ordinal `Score` preserves the rating distribution. **Abstain/review/insufficient evidence is a separately justified policy**: source records with unknown/unscorable labels are not silently coerced to `no`.

### Freeze rules for v0.1

1. **Exactly three bounded research train/eval tasks and two *nonranking* welfare probes.** Never aggregate them into one welfare-intelligence number. Report evidence tiers and modality-family scores separately.
2. Source revisions/byte hashes, animal and whole-video membership, exclusions, real labels, fixed heldout splits, and permitted uses must remain unchanged for this candidate version. Any corrected target, split or dataset requires **v0.2** or an explicitly noted erratum.
3. Train and calibration are distinct; never choose threshold, rubric or models by their held-out scores. Previously inspected heldout results are **exploratory**, not a new concealed leaderboard.
4. **Disclose the input pipeline:** human bbox → geometry for Meerkat; machine camera counts → sensor agreement for HotPig; precomputed wearable index for cattle; horse crop pixels for Grimace; *expert-coded* AU timelines for EquiFACS.
5. Report by independent animal/video/site as supported. Thousands of highly correlated minutes and frame annotations **do not equal thousands of independent animals**.
6. Three levels remain separate: **Observe → Interpret → Decide whether to escalate**. Level 3 has **no independent labels** yet; do not construct one by heuristically relabeling Level 1 or by thresholding a grimace score.
7. Public code, source pointers and hashed manifests are reproducible; **do not bundle third-party data** beyond its stated redistribution rights. Meerkat is noncommercial research-only. Distinguish research evaluations from commercial rights and deployment validation.
8. The accompanying machine-readable [v0.1 manifest](OPENFARM_DECISION_SUITE_V0_1.json) is an **index to existing verified scripts and audits**, not a replacement for actual source-media loaders, a release tag, or five new training implementations.

## Existing source-verified implementation and reference documents

- **HotPig:** `scripts/build_hotpig_feeder_v0.py`, `scripts/benchmark_hotpig_system1.py` · [evidence](2026-10-track-A-learned-baselines.md).
- **Cattle:** `scripts/build_public_cattle_behavior_v0.py`, `scripts/audit_open_cattle_behavior.py` · [admission](2026-10-public-cattle-choice-admission.md).
- **Meerkat:** `scripts/benchmark_meerkat_video_heldout.py`, `tests/test_meerkat_video_benchmark.py` · [video-heldout audit](2026-10-video-heldout-sprint.md).
- **Horse:** `scripts/prepare_horse_unique.py`, `environments/openfarm_horse_grimace/` · [legacy repaired-source verdict](../system1/FINAL_PRODUCT_FIT_AUDIT.md).
- **EquiFACS:** `scripts/prepare_equifacs_structured.py`, `scripts/build_equifacs_clinical_v0.py` · [Track B dossier](2026-10-track-b-affective-supervision-audit.md).

**Next research step:** establish additional clinically/behaviorally defensible positive welfare supervision from *public* sources, and independent caretaker-outcome annotations when available. But v0.1 is already useful **without waiting for outreach**.
