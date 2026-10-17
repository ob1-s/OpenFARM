# Big Broiler feather-pecking event detection — L1 welfare-observation task (PROPOSED)

Operator C, 2026-10-16. Branch `research/index-behavior-to-care-2026-10`.
Status: **PROPOSED TASK_VALIDATED at L1** (not APPROVED until review; commercial/train/eval rights separate decisions; L2/L3 not claimed).

## Four-line PASS/FAIL/DEFER verdict

- **PASS (L1):** real welfare-relevant harmful-behavior event detection learned from released tracking boxes only, video-disjoint, test window F1 0.31–0.37 vs 0.0 baselines at calibrated thresholds; prevalence 0.23 test.
- **PASS (label quality):** released `actions_ground` equals exactly one (not all) simultaneous annotators per video and window-level consensus(≥2/3) matches ground where it exists; human pairwise window F1 0.94–1.00 on the 5 videos with ≥2-annotator positives.
- **FAIL (L2/L3):** no medication/vet/treatment/clinical/join surface in the source, so caretaker-time linkage to L3 is not manufacturable here; 16.6–40.6 false episodes/stream-hour is not a deployable alert rate.
- **DEFER:** sharper families (calibrated trees by camera setup, aggressive recall/precision tradeoff search) and join-schema handoff to Operator A in `openfarm-l3` (broiler camera date ≈ 2021, age 10–40, hospital pen treated as censored).

## 1. Source record

- Record: decide-ugent Big Broiler Dataset, GitHub `decide-ugent/big-broiler-dataset`.
- Media+annotations ZIP: `https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip` (~32.5 GB; annotations processed by range reads, only CSV bytes pulled; **media never downloaded here**).
- Manifest pinned: `data/broiler/cache/manifest.csv`, SHA256 `c90e25a13bbb373ab9dd4a0411ae56fe76dde180d4c52b6eb2d943ccae222554`; 102 rows, exact per-video row counts and frame counts match loaded data (see §3).
- Upstream license state: repo AGPL-3.0 labels+code; raw media and annotation content from Ghent/WUR remain upstream obligations; we commit only aggregate metrics, pinned pointers and scripts (no media/annotations in git).

## 2. Question (pre-registered before scoring; constants in `scripts/index_behavior_broiler_event_core.py`)

- Prediction state at each integer second `t` (stride 25 frames at 25 fps), window `W=250` frames (10 s) ending at `t` inclusive.
- Primary target: `Noul(≥1 source row labeled Feather_Pecking in [t−249, t])` — feather pecking is a damaging interaction event in the source's action taxonomy; no label merge, no claim of injurious contact severity.
- Mask (not negative): any `Video_Problem` row in the window OR window coverage (rows per covered frame / 250) < 0.8. The mask keeps `Out_Of_Sight` inside the scorable set when coverage is otherwise adequate; `Out_Of_Sight` was otherwise the biggest confounder family (424,706 rows; co-label zero with FP at the row level — no row in the 6.67 M has both).
- Features: 21 geometry/kinematics features from released tracking boxes only (`n tracks, area/ysize/xspan stats, center-of-mass moments, nearest-neighbor distance stats, speed stats`) — **no action labels, no posture labels, no annotator IDs, no setup/age/time metadata** so label leakage by construction is impossible (~10 model-visible features ≈ box shapes only, documented per-feature in code).
- Episodes: state runs separated by ≤3 negative states merge into one episode, evaluated with ≥1-frame overlap greedy match per video for onset/offset style metrics (`false_episodes_per_stream_hour` is the stream metric).
- Splits: **whole-video groups**, per setup by source stem order with pre-registered calibration/test ordinal slots (test {3,15,27,39}, cal {8,20,32}); videos 092–101 (the ten triple-annotated ones) are **excluded from all three model splits** and reserved for label-quality measurement (§4). Repair rule (registered before scoring; deterministic, FP-richest by count with lexical tiebreak): `setup3:test`, `setup3:cal` fired → 76/7/9 videos, 31,074/3,242/3,814 states → scorable 23,678/2,269/3,512, positives 1,383/345/813, prevalence 0.058/0.152/0.231 (train/cal/test).

## 3. Source integrity checks (audit `scripts/index_behavior_broiler_source_audit.py`, output `reports/index_behavior_broiler_source_audit.json`)

- All 102 CSVs parsed to identical schema (`frame, track, x1, y1, x2, y2, actions_ground, postures_ground, actions_annotator_1..3`); per-video row counts and frame ranges inconsistent with the manifest → fail-closed on load if mismatch.
- 6,670,416 rows total; 25 fps; 3 setups (49/47/6); ages 10–40 weeks (setup 3 all age 32).
- Label inventory: Feather_Pecking 37,130 rows / 35,248 unique frames / 54 videos (present in all 3 setups); Out_Of_Sight 424,706; Video_Problem 84,178 across 37 videos; Preening 362,454 (self-confuser family); Neg_Interact 690 (7 videos); Aggression 420 (2 videos); Disturbance 3,924. 31 distinct label tokens — hard fail-closed on any unseen token (guard in benchmark).
- Quirks recorded, not silently fixed: 7 videos carry duplicate `(frame,track)` rows (dedup keep-last-largest applied deterministically); 51 videos have bbox values slightly out of the unit token range; 84 videos have frame-index gaps (states' masks handle these — flagged via coverage); video 004 carries a 14-frame negative annotation offset (axis-shifted, offset recorded in provenance; feature invariance preserved); maximum visible boxes/frame = 65 (feature cap 128).
- **Zero FP ∩ Out_Of_Sight co-labels** on any `(frame, track)` row; 26 of the 54 FP videos also carry Video_Problem rows — co-occurrence exists at video level, handled by masks.

## 4. Label quality (simultaneous annotators; `scripts/index_behavior_broiler_annotator_agreement.py`, output `reports/index_behavior_broiler_annotator_agreement.json`)

The 10 videos `092–101` (setup 2, ages 20–40, one ~60 s clip each — 1,504–1,524 frames, i.e. 51 window-states per video, with 3 simultaneous annotators + released ground) show:

- Released `actions_ground` equals *at least one single annotator's* channel exactly in every video with positives, sometimes two when those two annotators agree fully (exact-match: 092 → a1≈a3, 093 → a3, 095 → a1≈a2, 096 → a1, 097 → a2≈a3, 099 → a1, 101 → a1≈a2 both zero while a3 marks 14 windows).
- Consensus(≥2/3), where it exists, matches ground exactly → F1 1.00 on 092, 093, 095, 096, 097 — the only observable consensus surface is windows where ≥2 annotators agree; the release's ground otherwise reflects a single-coder (session-leader) lineage.
- Pairwise window F1 restricted to pairs with any positives (092/093/095/096/097): a1_a2 0.81, a1_a3 0.56, a2_a3 0.65; overall means including the four empty-FP videos: 0.49/0.39/0.39.
- Consensus2 surfaces total 91 FP windows across 5 videos; **29 contested windows exist** where exactly 1 of 3 annotators marks FP (11 on 099 — a1-only; 14 on 101 — a3-only; and 4 split windows elsewhere) — these are irreducible single-coder windows and are preserved as `not rateable` quality markers, never collapsed into negatives.
- Consequence: the model pipeline trains/predicts against `actions_ground` per the release's own use-case (single-coder lineage), while label-quality commentary uses the consensus surfaces; **neither hidden channel is exposed to any model** (features are box-geometry only — no annotator column enters).

## 5. Model results (pre-registered standard models; `scripts/index_behavior_broiler_event_benchmark.py`, output `reports/index_behavior_broiler_event_benchmark.json`)

All:
- Train features on train split only; select threshold on cal split (max window F1 over grid 0.01..0.99 + 0.5); test is untouched until final numbers. Un-calibrated models at threshold 0.5 report F1 0.0 (over-predict-probability imbalance) — recorded as the base case.

Windows:
- `logistic_l2` L2-multinomial-type logistic (C=1): cal F1 0.4187 (p 0.267 / r 0.971) at cal_thr 0.07; **test F1 0.3076** (p 0.300 / r 0.316).
- `extra_trees_55` (55 single-threaded trees): cal F1 0.3437 (p 0.219 / r 0.800) at cal_thr 0.04; **test F1 0.3704** (p 0.254 / r 0.686).
- Baselines `prior_stream` (constant train prevalence) and `always_negative` both score F1 0 and recall 0 on the window metric: **feature-based signal is real, not prevalence artifact**.
- Per-setup test positives: setup1 128/1109, setup2 34/1221, setup3 651/1182 scorable windows — **the test positives concentrate in the 6-camera setup-3 family**, a distribution-shift risk logged for the next gate (family-fold replication, not the current pre-registered split).

Episodes/stream metric:
- Logistic at cal_thr: test episode `tp=10, fp=18, fn=14, 16.6 false episodes per stream-hour` over 1.08 stream-hours (nine 10-min videos concatenated as one anonymous stream — no cross-video episode merging).
- ExtraTrees at cal_thr: `tp=17, fp=44, fn=7, 40.6 false episodes per stream-hour` (recall-favoring).
- Episode metrics are supplementary: FP/hour at deployable levels (≤1–2 fp/h also targeted by adjacent work is vastly better than this; L2/L3 clearly not justified by this number family).

Confuser windows (operating at the same cal_thr on the same windows):
- Test split: `Neg_Interact` 10/42 predicted-positive with 0 also-FP windows (false-fire rate 10/42 in hostile-interaction windows where the target is absent by label); `Disturbance` 27/114 similarly 0 also-FP (distractor family near caretaker/meal times); `Preening` 587/1934 flagged, 194 also-FP (preening co-occurs with FP ~1/3 of the time it fires — self-confusion partially real).
- Cal split: `Aggression` 15/28 flagged, 5 also-FP — the rarest family (420 rows total) still contributed some genuine co-positive structure; the rest is residual false-fire noise.
- Conclusion: modest false-fire burst remains; the expensive family to control is `Preening`/`Disturbance` which the trees family prefers.

## 6. Positive states vs positive *windows* (L1 hygiene note)

- Work registered at window state level (`window_any` over 1-s states). The 0.30% natural corpus prevalence on all 102 videos rises to 5.84%/15.2%/23.1% by design of the split and by the FP-heavy composition of setup 3 within the test pool. Pre-registered splits, no post-hoc split qualifier, no label swap.

## 7. Why this is L1 (not L2/L3)

**L1 (cleared):** an independently hand-labeled real-source welfare event is learnable purely from released 2D tracking boxes with video-disjoint evaluation, video-level prevalence preserved, and label-quality surfaces quantified (single-coder ground lineage; pairwise agree-high consensus surfaces). That fact alone is a meaningful research finding and a candidate `PROPOSED APPROVED_RESEARCH_TRAIN` research slice once reviewers confirm.

**L2/L3 blocked, exactly why:**
1. **No care/clinical join surface exists in the source** — no medication, vet, injury or intervention columns; camera/setup/age only. Any claimed linkage to a caretaker decision (e.g., "treat the grabbed birds") would be fabricated. The handoff's separate clinical/care action event would need a different public source (tail-bite + caretaker audio data are explicitly not published; screamer/extract events are paper-restricted).
2. **The operational alert rate is not there yet** — 16.6–40.6 false episodes per stream-hour, above any realistic caretaker alert budget; at 0.3% natural prevalence a deployable L2 would need precision/episode-rate improvements (score calibration by camera family, temporal smoothing, aggressive hard-neg mining of Preening+Disturbance windows) — next round, not now.
3. **Label lineage caution at L3** — because `actions_ground` is single-coder and FP windows are known contested in 29 windows, any alert evaluation downstream of this label would inherit annotator-pair-partition noise; a multi-annotator consensus replication is a prerequisite for any medical/care claim.
4. Per the handoff contract, the care/clinical linkage belongs to Operator A; what we hand over is the **join schema + candidate question**: `flock_id | camera/setup | age_weeks | timestamp(UTC-anchored frame-index) | episode span | episode score | obs quality flags`. Upstream has **no** recorded care events to join ON this schema; therefore no L3 slice is spun off — the L3 lineage from this source is a documented **dead end** unless an independent source appears.

## 8. Repro

```
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python numpy pandas scikit-learn remotezip pyarrow
.venv/bin/python scripts/index_behavior_broiler_source_audit.py   # once, ~16.6 min; requires the ZIP reachable
.venv/bin/python -m unittest tests.test_index_behavior_broiler_event_core    # 18 offline tests, no network
.venv/bin/python scripts/index_behavior_broiler_event_benchmark.py          # ~4 min offline
.venv/bin/python scripts/index_behavior_broiler_annotator_agreement.py      # <1 s offline
```
Data under `data/broiler/cache/` is gitignored; only aggregate JSON/CSV reports (untracked) plus code/docs are version-controlled.

## 9. Next gate (proposal)

1. Reviewer confirmation of PROPOSED admission at L1 (window level, prevalence-annotated protocol).
2. Family-fold replication: forbid any whole-setup family to be the sole positive concentration of test (replicate 086–091-only positive pool into test/cal across setups via group-balanced random assignment at video level) — then report the same model family since pre-registration is done.
3. Multi-annotator FP window consensus-replication (label reconstruction on 92 windows available) as the next-gate cutoff for APPROVED_RESEARCH_TRAIN.
4. Operator A handoff: candidate join schema + evidence package, no linkage made without A.
