# Big Broiler observed feather-pecking event detection — L1 (PROPOSED, FAMILY-ROBUSTNESS FOLLOW-UP ADDED)

Operator C, 2026-10-16 (rev 2 after main-operator review of PR #7).
Branch `research/index-behavior-to-care-2026-10`.
Status: **PROPOSED TASK_VALIDATED (observation-only L1)**; `APPROVED_RESEARCH_EVAL` deferred until family-robustness reporting fully reviewed; L2/L3 explicitly blocked.

## Four-line PASS/FAIL/DEFER verdict

- **PASS (L1, limited claim):** geometry→event inference — released tracker-box features predict *observed feather-pecking* (welfare-relevant precursor, not proven injurious) windows on the named enriched diagnostic holdout (test F1 0.31–0.37; per-video mean F1 0.13–0.19) and in 2 of 3 leave-one-setup-out families (held-out F1 0.12–0.21).
- **PASS (label quality surfaces):** released `actions_ground` equals exactly one annotator's coding per video (single-coder lineage; consensus≥2/3 F1=1.00 where it exists, not independent ground truth); 29 contested windows preserved separately.
- **FAIL (family robustness caveat, now quantified):** 80.1% of diagnostic-test positives come from the 6-video setup-3 family; on the same models at the same threshold, setup-1/2 test slices score F1 0.00–0.06, and LOSO-setup3 (i.e., training on setups 1+2, evaluating on setup 3) transfers at F1 0.44 (trees, over-fire threshold 0.01) but **0.00 for logistic** — cross-camera generalization is weak/mixed. The old split is therefore explicitly a **deliberately enriched diagnostic holdout**, not natural-prevalence external validation.
- **DEFER/L2-L3:** no clinical/care/treatment join surface exists in the source; no raw video was inspected or benchmarked here (inputs are released human/tracker boxes); 16.6–40.6 false episodes per *test-clip hour* are not deployable false alerts per commercial farm-hour.

## 1. Source record

- Record: decide-ugent Big Broiler Dataset, GitHub `decide-ugent/big-broiler-dataset`.
- Media+annotations ZIP: `https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip` (~32.5 GB; annotation CSVs extracted by HTTP range reads; **video media never downloaded or inspected by this work**).
- Manifest pinned: `data/broiler/cache/manifest.csv`, SHA256 `c90e25a13bbb373ab9dd4a0411ae56fe76dde180d4c52b6eb2d943ccae222554`; 102 rows; per-video row/frame counts verified against loaded data.
- Upstream license state: repo AGPL-3.0 labels+code; raw media and annotation content from Ghent/WUR remain upstream obligations; only aggregate metrics, pinned pointers, and scripts are committed.
- **Unit of age: days** (upstream manifest `age` column spans 10–40 days; setup 3 records age 32). An earlier revision wrote "weeks" — corrected per operator review.

## 2. Question (pre-registered before scoring; constants in `scripts/index_behavior_broiler_event_core.py`)

- Prediction state at each integer second `t` (stride 25 frames at 25 fps), window `W=250` frames (10 s) ending at `t` inclusive.
- Primary target: `Noul(≥1 source row labeled Feather_Pecking in [t−249, t])`. Upstream ethogram: pecking/manipulating a conspecific's feathers. This target is an **observed feather-pecking event**, not a separate injurious/gentle distinction, not lesion severity, and not required care. No injury claim is made beyond the label text.
- Mask (not negative): any `Video_Problem` row in the window OR window coverage (rows per covered frame / 250) < 0.8. `Out_Of_Sight` stays inside the scorable set where coverage otherwise suffices (co-label with FP at row level: zero of 6.67 M rows).
- Features: 21 geometry/kinematics features from **released tracking boxes** (tracker+human-derived) only — no action labels, no posture labels, no annotator IDs, no setup/age/time metadata. The input modality is therefore released box geometry, not camera pixels; all resulting scores are geometry→event inference.
- Episodes: state runs separated by ≤3 negative states merge into one episode (≥1-state-overlap greedy matching per video); `false_episodes_per_test_clip_hour` is the stream metric over the evaluated clips only.
- Splits: **whole-video groups** by camera-setup family; per setup by source stem order with pre-registered ordinal slots (test {3,15,27,39}, cal {8,20,32}) and a deterministic positive-window-count repair (`setup3:test`, `setup3:cal`). Because the repair uses per-video positive counts (source metadata, no model scores) to select moved videos, **the resulting split is outcome-informed sample selection**; it is reported below as the **enriched diagnostic holdout** and is not used as evidence of natural-prevalence external validation. K-fold-independent robustness is instead provided by leave-one-setup-out (§5b).
- Counts: 76/7/9 videos, 31,074/3,242/3,814 states → scorable 23,678/2,269/3,512, positives 1,383/345/813, prevalence 0.058/0.152/0.231 (train/cal/test). 651/813 (80.1%) test positives sit in setup 3.

## 3. Source integrity checks (audit `scripts/index_behavior_broiler_source_audit.py`, output `reports/index_behavior_broiler_source_audit.json`)

- All 102 CSVs parsed to the exact released schema (`frame, track, x1, y1, x2, y2, actions_ground, postures_ground, actions_annotator_1..3`); per-video row counts and frame ranges cross-checked against the pinned manifest; benchmark fails closed on any mismatch.
- 6,670,416 rows total; 25 fps; 3 camera setups (49/47/6 videos); ages 10–40 days (setup 3 all age 32).
- Label inventory: Feather_Pecking 37,130 rows / 35,248 unique frames / 54 videos (present in all 3 setups); Out_Of_Sight 424,706; Video_Problem 84,178 across 37 videos; Preening 362,454; Neg_Interact 690 (7 videos); Aggression 420 (2 videos); Disturbance 3,924. 31 distinct label tokens — hard fail-closed on any unseen token.
- Quirks recorded, not silently fixed: 7 videos carry duplicate `(frame,track)` rows (dedup keep-last-largest applied deterministically); 51 videos have bbox values slightly outside the unit range; 84 videos have frame-index gaps (handled via coverage); video 004 carries a 14-frame negative annotation offset (axis-shifted, offset recorded; features invariant to the shift); maximum visible boxes/frame = 65 (feature cap 128).
- **Zero FP ∩ Out_Of_Sight co-labels** on any `(frame, track)` row; 26 of the 54 FP videos also carry Video_Problem rows — co-occurrence exists at video level, handled by masks.

## 4. Label quality (simultaneous annotators; `scripts/index_behavior_broiler_annotator_agreement.py`, output `reports/index_behavior_broiler_annotator_agreement.json`)

The 10 dual-purpose IAA videos `092–101` (setup 2, ages 20–40 days, ~60 s each, 51 window-states per video, 3 simultaneous annotators + released ground) show:

- Released `actions_ground` equals *at least one single annotator's* coding exactly in every video with positives, sometimes two when two annotators agree fully (exact-match: 092 → a1≈a3, 093 → a3, 095 → a1≈a2, 096 → a1, 097 → a2≈a3, 099 → a1, 101 → a1≈a2 both zero while a3 marks 14 windows).
- Consensus(≥2/3), where it exists, matches ground exactly → F1 1.00 on 092, 093, 095, 096, 097. **This is observer-agreement evidence, not proof of error-free ground truth**: disjoint disagreements exist (29 contested windows where exactly 1 of 3 annotators marks FP: 11 on 099 a1-only, 14 on 101 a3-only, 4 partial splits), and ground-vs-a2/a1/a3 non-exact matches above are included in the reports. We do not treat consensus windows as independent ground truth; they are preserved as a label-quality surface, and the 29 contested windows never enter negatives.
- Pairwise window F1 restricted to pairs with any positives: a1_a2 0.81, a1_a3 0.56, a2_a3 0.65; overall means including the four empty-FP videos: 0.49/0.39/0.39.
- A **segment-IoU** style comparison (as in the upstream paper) is acknowledged as more aligned with the source's span nature than our 1-s windows: it is now reported for every evaluation block (`segment_iou_detected_mean` / `segment_iou_reference_mean`, per-video values in the benchmark report; currently 0.01–0.09, i.e. spans overlap weakly at the operating points found — consistent with the window-F1 numbers, and the span-level metric is a remaining improvement target if this slice proceeds).
- Model pipeline trains/predicts against `actions_ground` per the release's own use-case; neither annotator channel enters any feature (geometry only).

## 5. Model results (`scripts/index_behavior_broiler_event_benchmark.py`, output `reports/index_behavior_broiler_event_benchmark.json`)

### 5a. Enriched diagnostic holdout (the pre-registered split, explicitly outcome-repaired)

All:
- Thresholds selected on cal (max window F1 over grid 0.01..0.99 + 0.5); test untouched until final numbers. Uncalibrated models at 0.5 report F1 0.0 (probability imbalance) — recorded as base case.

Diagnostic test (9 videos, 3,512 scorable states, 813 positives):
- `logistic_l2` (C=1): cal F1 0.4187 (p 0.267 / r 0.971) at thr 0.07; **test F1 0.3076** (p 0.300 / r 0.316).
- `extra_trees_55`: cal F1 0.3437 (p 0.219 / r 0.800) at thr 0.04; **test F1 0.3704** (p 0.254 / r 0.686).
- Baselines `prior_stream` and `always_negative`: F1 0, recall 0 → feature-based signal is not a prevalence artifact.
- **Per-video distribution** (uncertainty over videos, not windows): test@cal_thr per-video F1 mean 0.1302 (logit) / 0.192 (trees), median 0.0 / 0.057, SD 0.218 / 0.257, max 0.5576 / 0.6712, one zero-positive video; episode FP concentrated in a few videos (logit [2,0,0,0,0,7,5,2,2]; trees [10,0,4,8,0,0,17,2,3]).
- Episodes: logistic test tp/fp/fn = 10/18/14, 16.6 false episodes per test-clip-hour (1.083 clip-hours); trees 17/44/7, 40.6 per test-clip-hour.
- Segment IoU at these points: detected-mean 0.047 (logit) / 0.060 (trees); reference-mean 0.042 / 0.087 — weak span alignment, matching the F1 regime.
- Confuser windows at the same operating points: test split `Neg_Interact` 10/42 flagged with 0 also-FP; `Disturbance` 27/114 with 0 also-FP; `Preening` 587/1934 flagged, 194 also-FP (self-confusion partially real); cal split `Aggression` 15/28 flagged, 5 also-FP.
- Diagnostic per-setup breakdown (same models at cal_thr, per setup slice of the diagnostic): **setup 3 (2 videos) test F1 0.5035 (logit) / 0.6309 (trees); setup 1 (4 videos) F1 0.0201 / 0.0613; setup 2 (3 videos) F1 0.0000 / 0.0376** — the headline diagnostic F1 is setup-3-dominated, as the operator review predicted.

### 5b. Leave-one-setup-out family-blocked generalization (operator follow-up, no outcome use)

Train on the other two setups' videos (threshold picked by max-F1 over the training windows only — no eval-family outcome info); evaluate on the entire held-out family (video-disjoint by construction):

| held-out family | videos | scorable/positives (prevalence) | logistic thr / F1 (p/r) | trees thr / F1 (p/r) |
|---|---|---|---|---|
| setup 1 | 49 | 13,227 / 746 (0.056) | 0.17 / **0.2060** (0.124/0.602) | 0.01 / 0.1304 (0.070/1.000) |
| setup 2 | 37 | 12,686 / 790 (0.062) | 0.26 / **0.1536** (0.087/0.662) | 0.01 / 0.1213 (0.065/0.992) |
| setup 3 | 6 | 3,546 / 1,005 (0.283) | 0.18 / **0.0000** (0/0) | 0.01 / **0.4357** (0.281/0.967) |

- Setup-3 does **not transfer** for the linear family at all (model never fires above threshold); trees transfer at an over-fire threshold (0.01) chosen on train-side prevalence; LOSO trees per-video F1 mean 0.453 (SD 0.271) over its 5 positive videos.
- LOSO train-side threshold choices collapse to ≤0.26 everywhere (train-side max-F1 favors recall at rare prevalence): a natural-prevalence deployment search remains open, but family-blocked generalization is demonstrated at F1 0.13–0.21 for setups 1 and 2 with real held-out prevalences, and **fails** for the small setup-3 family.
- Interpretation for the gate: the L1 claim is the *geometry→observed-FP-event* learnability fact (within-family, diagnostic holdout; and moderate family-blocked transfer for setups 1–2), **not** cross-camera robustness. Setup-3-only scoring power (6 videos, dense low-canopy flock at age 32 days) plus the failure to transfer into it from other families both assert a real independence risk and justify keeping the enriched split labeled diagnostic only.

## 6. Positive states vs positive windows (L1 hygiene note)

- Work registered at window state level (1-s `window_any`). The 0.30% natural corpus prevalence over all 102 videos rises to 5.84/15.2/23.1% by construction of the enriched split and the FP-heavy composition of setup 3 inside the test pool. Pre-registered constants; no post-hoc split qualifier, no label swap; the operating points come from cal only, and LOSO gives zero eval-family information.

## 7. Why this is L1 (not L2/L3) after the follow-up

**L1 (established):** an independently hand-labeled, real-source welfare-relevant observable event (observed feather-pecking) is learnable from released tracking-box geometry with video-disjoint evaluation — quantified on the enriched diagnostic holdout and on leave-one-setup-out family blocks for setups 1–2. Label-quality surfaces (single-coder ground lineage + consensus + 29 irreducible contested windows) are quantified and preserved. No raw video bytes were inspected or benchmarked: the input modality is released human/tracker boxes, so an end-to-end camera model and real farm false-alarm rates remain open work downstream.

**L2/L3 blocked, exactly why:**
1. **No care/clinical join surface exists in the source** — no medication, vet, injury or intervention columns (camera/setup/age only). Any caregiver/clinical linkage would be fabricated. Per the handoff contract, care/clinical linkage belongs to Operator A; our deliverable is the join schema and candidate question (`flock_id | camera/setup | age_days | timestamp/frame-index | episode span | episode score | obs-quality flags`). Upstream has no recorded care events to join; this L3 lineage is a documented dead end from this source unless an independent source appears.
2. **No deployable alert-rate evidence** — 16.6–40.6 false episodes per *test-clip hour* on the diagnostic, LOSO episode FP 3.7–22.5 per held-out clip-hour; these are clip-derived numbers, not commercial farm-hours, and sit far above any realistic caretaker alert budget.
3. **Family robustness is the open scientific blocker for EVAL admission** — 80.1% of diagnostic-test positives from 6 setup-3 videos; setup-1/2 diagnostic slices score ≈0; setup-3 does not transfer from other families (logistic F1 0.00). Until a family-balanced replication or span-IoU-optimized operating point is published, `APPROVED_RESEARCH_EVAL` stays deferred (per operator disposition).
4. **Label lineage caution** — single-coder ground + contested windows imply any alert evaluation downstream of these labels needs multi-annotator consensus replication first.

## 8. Repro

```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy pandas scikit-learn remotezip pyarrow
.venv/bin/python scripts/index_behavior_broiler_source_audit.py   # once, ~16.6 min; requires the ZIP reachable
.venv/bin/python -m unittest tests.test_index_behavior_broiler_event_core    # 18 offline tests, no network
.venv/bin/python scripts/index_behavior_broiler_event_benchmark.py          # ~7 min offline (LOSO included)
.venv/bin/python scripts/index_behavior_broiler_annotator_agreement.py      # <1 s offline
```
Data under `data/broiler/cache/` is gitignored; only aggregate JSON/CSV reports (untracked) plus code/docs are version-controlled.

## 9. Next gate (proposal)

1. Reviewer confirmation of the updated framing: enriched diagnostic holdout + LOSO family blocks as the robustness surface for the L1 claim.
2. Family-balanced replication option (pre-register a second split where no single setup family may dominate the test pool) before any `APPROVED_RESEARCH_TRAIN`/`_EVAL` move.
3. Span-IoU operating-point optimization and calibration by camera family (best remaining improvement lever for a real L2 slice, per this follow-up's segment-IoU report).
4. Multi-annotator consensus replication (92 consensus windows across 5 IAA videos) as the next label-replication cutoff.
5. Operator A handoff of the join schema + evidence package; no linkage without A.

## 2026-10-18 full 92-video LOSO source-clip stress test — completed

[Original 102-CSV source verification and 92-video model run](https://github.com/ob1-s/OpenFARM/actions/runs/39131052484), source [reproducible workflow](../../../.github/workflows/openfarm-broiler-natural-clip-loso.yml), [aggregate script](../../../scripts/index_behavior_broiler_natural_loso.py).

- **Population:** all 92 source clips not reserved for annotator-agreement study, comprising **10.8335 annotated clip-hours**. The 3 LOSO models each hold out an entire camera-setup family, producing one prediction per source clip from a model trained on the other two setups. No event-informed eval-video *selection* or label-informed split repair in this evaluation. Clips themselves were curated at collection, so prevalence is **not natural farm prevalence**; geometry is released tracked boxes, **not raw video pixels**.
- **Reference window positive prevalence:** 8.63% over heldout scorable states. Episode matcher follows existing fixed script (overlapping positive-state runs, gap up to 3 states).
- **ExtraTrees:** episodes TP/FP/FN = **47/84/74**, F1 **0.3730**, 7.754 false episodes per annotated clip-hour; stratified by source setup, 400 video-cluster bootstrap estimate of 95% interval **[5.68, 9.99]** false episodes/clip-hour. Window F1 0.1736.
- **Logistic L2:** episodes TP/FP/FN = **35/145/86**, F1 **0.2326**, 13.384 false episodes per annotated clip-hour; 95% video-cluster bootstrap interval **[9.02, 18.19]**. Window F1 0.1597.
- Thresholds in LOSO are selected only on the **other two source camera setups' training windows**; not tuned on the target setup. However, the source dataset and evaluation program have been explored previously; these are new descriptive developmental statistics, not untouched prospective outcomes.
- **Practical judgement:** geometry-only detection fails as unattended care alerting: substantial false episodes and missed episodes; no pig/chicken lesion, independent physiological welfare or caretaker-action endpoint. Real farm-hour false-alarm burden must be measured on uncensored continuous deployment footage at ordinary prevalence before claiming commercial welfare-monitor usability.
- **Follow-on:** run the actual video-native model over a large unselected all-camera subset or all 92 original MP4 clips, with timestamps/episodes, persistent model version, adaptive event-level screening permitted only on training/cal footage. Do not compare six chosen Liquid windows to this geometry LOSO as if inputs/eval units were matched.
