# L3 care-event lane — source audit and Aarhus historical builder (Operator A)

**Date:** 2026-10-16. **Branch:** `research/index-l3-care-events-2026-10`. **Role:** independent prospecting for real train/eval animal-welfare care-event tasks. This document does **not** approve anything: statuses below are PROPOSED for review per the data approval policy (`docs/research/DATA_APPROVAL_POLICY.md`).

**Corrective pass (2026-10-16, after PR #4 scientific review):** (1) non-modal-pen rows are now actually quarantined from states and pen histories, with an adversarial regression test; (2) baseline metrics renamed (`true/false/missed_positive_windows`) and a separate event-level evaluation added (distinct first-onsets covered, first-alert lead, 3-day-cooldown alert episodes); (3) prevalence denominators now report both total-window and evaluable-window rates (test: 0.422% vs **0.556%**; best-rule lift ~4.0×); (4) false-positive unit relabeled to per-1,000-evaluable-weekday-scoring-windows; (5/6) construct-cost and generalization-sensitivity section added (2.7b). Changed numbers only; no models trained.

## Executive result (four-line verdict)

- **PASS (L2→L3 candidate, PROPOSED, not APPROVED):** Coutant et al. 2026 Aarhus dataset (Zenodo 18785731, CC BY 4.0): all bytes md5/SHA-verified, 37,093 real timestamped animal-day clinical rows (37,091 used after quarantining 2 non-modal-pen rows), 810 pigs, 60 (batch, pen) cohorts, 282 causal 7-day first-onset warning windows with 24,774 clean-at-risk true negatives; deterministic builder + 16 offline unit tests + fixed-threshold baselines reporting window-, onset- and episode-level results on real bytes.
- **PASS (access + structure, verified):** Chou et al. 2020 PLOS ONE supporting S1 XLSX (CC BY 4.0; SHA `250522604e6dfaf78ef135d2085d1ce5957d163f39f734556af67e18668d5b52`) — 672 pigs, 48 pens, date-stamped per-pig ordinal tail/ear scores with 154 at-risk wound-escalation windows; caregiver removals/treatments recorded only as pen-level totals (no per-event timestamps), so it cannot be an approved L3 action task as released.
- **FAIL (quarantined, not realizable as a time-forecast event task):** Agroscope ethogram+lesion Zenodo 5920843 — all 10 files byte-verified and cleanly joined (prior join blocker resolved); only 3 pigs ever performed active tail bites across 380 focal rows; released tables contain **no observation dates**, so prediction times and horizons cannot be pinned; only 2 farrowing groups exist for separation.
- **CONFIRMED NEGATIVE (still true):** zero public datasets with per-action timestamped staff care events (treatments/removals) found in this lane; the two scientifically-described action sources (Heseker 2024 staff biter removal; Eisermann 2025 Bite-o-Mat) release data only *on request*.

---

## 1. Ranked source gate (full table)

See `docs/research/index-l3-care/source_gate.csv` (rank 1–9, ≤12 rows, STATUS per evidence ladder). Summary:

| rank | id | source | status in gate | core evidence |
|---|---|---|---|---|
| 1 | L3-A01 | Coutant et al. 2026 Aarhus extensification, Zenodo 18785731 | PROPOSED_L2TO_L3 | record + zip md5 + member SHA-256 verified; builder + tests + baselines on real bytes below |
| 2 | L3-A02 | Chou et al. 2020 PLOS ONE e0241619 S1 XLSX | PROPOSED_NextGate | 7 sheets parsed; per-pig per-date ordinal clinical scores; actions are pen-aggregates only |
| 3 | L3-A03 | Roch/Kasper Agroscope 5920843 | QUARANTINED | byte-verified, joins fixed, but event-poor, no dates, 2 farrowing groups |
| 4 | L3-A04 | Heseker et al. 2024 Sci Rep (pig screams + staff biter removal) | ACCESS_BLOCKED_PUBLIC | paper states data only on request (privacy); verified in commit `413f7b8` |
| 5 | L3-A05 | Eisermann et al. 2025 Bite-o-Mat (SA&T 11:101010) | ACCESS_BLOCKED_PUBLIC | article (CC BY) states "Data will be made available on request"; no public event rows |
| 6 | L3-A06 | Swedish/Swiss feeder first-treatment (Zenodo 5028016) | QUARANTINED | previously established (28-vs-27 conflict; pseudo-dated controls; 3 heldout positives) |
| 7 | L3-A07 | CowScreeningDB cow lameness video | ACCESS_PENDING_GATED | requires signed license/registration; rights not yet obtained; no bytes inspected |
| 8 | L3-A08 | Chou et al. 2019 Animals 9(8):582 multi-step outbreak interventions | DISCOVERED_NOT_ACCESS_VERIFIED | protocol paper only; no released per-event rows found in this sprint |
| 9 | L3-A09 | Aarhus stomach-ulcer endpoint (same record) | DISCOVERED (secondary) | 368 scored stomachs (151 ulcer / 19 scar / 198 healthy); only after onset task review |

---

## 2. Deep audit #1 — Aarhus/Coutant 2026 (Zenodo 18785731, CC BY 4.0)

### 2.1 Provenance and byte verification

- Record: https://zenodo.org/records/18785731 (published 2026-02-26; connected paper Coutant, Hedemann, Pedersen, Larsen 2026, *Livestock Science* 307:105934, DOI 10.1016/j.livsci.2026.105934, article CC BY 4.0 per Crossref license list).
- Zip: `Data - Effects of single extensification factors ... finisher pigs.zip`, **md5 `7d07b5597b935abea7999c26adec9869` verified on download** (local copy also SHA-256 `58cfdb3260b022abd9f820fcd8a0f83bc79dc439ad3cc56886ae405a73df188c`).
- Members pinned in builder: tail CSV SHA-256 `6ad08458e705c66c94e3f95091e0594f88ec8d307053c09abc1d7b4348dc749e`; fouling CSV SHA-256 `2da365f7530fb25d75b51ce5a1cdbf8214f12f0005eb54bb3a500db690d1a270`.
- READMEs (authored by the dataset creators, inside the zip) define the protocol: RFID-linked daily pen entry, one experimenter scores all pens per day, reference-photo protocol, consensus training; tail/ear "fresh damage" is a per-animal binary per scoring day; `RegDateTime` is the per-row scan timestamp. Pen fouling is scored first, then pigs are handled for the damage assessment.

### 2.2 Original schema (3 CSVs)

- `tail-ear-damage.csv`: 37,093 rows; per (Batch, PenID, AnimalID, Day): Treatment, Period, Week, Weekday, Day, Observer initials, Sex, arrival tail status (shortened/damaged), `IntactTail_arrival`, `DamagedTail_arrival`, `RegDateTime`, `taildamage_fresh`, `eardamage_fresh`.
- `pen-fouling.csv`: 2,880 rows; pen-day cleanliness Score 0–600, `PenFouling` binary, Severity.
- `stomach-ulcer.csv`: 368 rows; per selected animal at slaughter: ulcer index 0–10 and class (Healthy/Ulcer/Scar).

### 2.3 Structural findings that a correct task must respect (all verified in bytes)

1. **PenID axis (801–830) is reused across both batches with different treatments** — e.g. pen 801 is "Space 9" in batch 1 and "Control" in batch 2. Cohort identity must be `(batch, pen)` = 60 cohorts; grouping by PenID alone confounds treatment and time.
2. **Two rows conflict with the animal's modal pen** (batch 2: 54598 has one day-3 row in pen 827 while the remaining days are in 828; 54605 has one day-3 row in pen 831 while rows 4–67 are in 821). The release does not document pen moves. **Builder quarantines both rows**: they are listed in provenance under `quarantined_rows_detail` and excluded from every state window AND from every pen history (they can never inform pen-mate features or a candidate scan). Regression-tested adversarially: a contaminated pen history would flip pen-mate features on other animals.
3. **19 batch-2 animals stop being scored after 1–16 days** (row counts 1–9). Some of them had fresh damage days before disappearing. The release explains nothing (paper full text is not accessible from this environment; Elsevier page bot-gated). Their disappeance is **informative censoring of unknown cause** — the builder treats any window whose later coverage ends >36 h before the horizon boundary as `censored`, never negative.
4. **No scoring Sat/Sun** (day gaps in every animal); only batch-2 pens 831–836 partially deviate (18 animals have non-45-row counts, e.g. 51/50-row histories).
5. **Paper totals sum to 824 animals vs 810 in the release** — small residual discrepancy; reconciliation requires the paper (DEFER to next gate). Irrelevant to task construction because the released table is the source of truth for labels, but reported honestly here.
6. 110 animals ever register ≥1 fresh-tail day (82 batch 1 / 28 batch 2), 112 ever ≥1 fresh-ear; 700 animals never any fresh tail damage => genuine hard negatives; fresh flags heal (195 1→0 transitions, 33 1→1), so "first onset" is a well-defined cohort boundary.
7. 39 animals are flagged already on their first observed day (no clean-before evidence) — excluded from the at-risk cohort by the builder (reported in provenance), never counted as "onset positives after prediction".

### 2.4 Builder contracts (`scripts/index_l3_care_build_aarhus_historical.py`)

- Source checksums enforced on load (zip md5 + member SHA-256); any mismatch fails closed.
- `prediction_ts` = the animal's own scan time of a clean scanned day. Model-visible state = this scan and everything earlier (own arrival status, treatment, days-in-farm, own ear-damage days so far, pen-mate fresh-tail days and distinct flagged penmates in the prior 7 days from strictly earlier scans in the same `(batch, pen)`, pen fouling score from the latest prior fouling day, `PigsInPen`, weekday). Target = fresh tail damage recorded on later scans within 7 days. Non-modal-pen rows are quarantined before any state or history is built.
- Treatment columns (Control/Enrichment/Roughage/Space 9/Space 6) are the assigned husbandry conditions of the source trial — never a treatment administered to an injured animal; the task's clinical surveillance features (own-ear and pen-mates' fresh-damage histories) require expert human inspection, so this construct validates scientifically on welfare outcomes, not the usefulness of a low-cost autonomous warning system.
- Positives only from real scored rows (event_ts strictly later than prediction_ts). No pseudo-onset dates for controls ever exist in this design.
- Statuses preserved: `positive` / `negative` (complete coverage of the 7-day horizon within 36 h grace) / `censored` (coverage ends too early: unknown removal) / `not_measured` (no later scans at all). Censored/not-measured are never turned into negatives.
- Splits: per batch, the 30 cohorts are ranked by `sha256("openfarm-index-l3-care-aarhus-v0|batch-pen")`; bottom ~20% per batch = test, next ~10% = calibration, rest train. Result: 42 train / 6 calibration / 12 test cohorts; both batches represented in test. Membership digest embedded in provenance; cohort names in `data/index_l3_care/aarhus/provenance.json`.

### 2.5 Real built numbers (rebuilt after the quarantine fix; states.csv SHA-256 `93cd9be4de361589abc116afeb4e92fbc7eb0804430b56ac544f7347b901de09`)

Source rows: 37,093 → 37,091 used (2 anomalous rows quarantined: batch 2, animal 54598, day 3, pen 827; animal 54605, day 3, pen 831). The two quarantined rows had produced 2 clean-negative windows in the pre-fix build; positives, censored and not_measured counts are unchanged.

| split | windows | positive | negative | censored | not_measured | cohorts | animals |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 21,028 | 233 | 15,748 | 4,601 | 446 | 42 | 515 |
| calibration | 3,670 | 14 | 2,768 | 809 | 79 | 6 | 83 |
| test | 8,297 | 35 | 6,258 | 1,829 | 175 | 12 | 190 |
| **total** | **32,995** | **282** | **24,774** | **7,239** | **700** | 60 | ~810 minus 39 first-day-flagged |

<rb>Test split extra denominators (per the review): evaluable windows 6,293 (35 pos / 6,258 neg); evaluable-window prevalence 35/6,293 = **0.556%**; total-window prevalence 35/8,297 = 0.422%; distinct first onsets in test = **15**.

Invariant checks on real output: zero positives with `event_ts <= prediction_ts`; zero censored/non-measured rows with a non-null target; zero duplicate `(batch, animal, prediction_ts)` keys; zero negatives carrying an event timestamp; the 2 quarantined rows appear in no state and in no pen history (regression-tested: pen-mate features of other animals cannot see them).

No-effect tests for leakage: an onset occurring exactly at the horizon boundary (+7 d) is positive for predictions made ≤7 d earlier and a **negative** for the +6-day prediction made earlier than that (unit-tested); prediction rows after an animal's first flag do not exist at all (onset-cohort semantics, unit-tested); pen-history features only see scans strictly earlier than `prediction_ts` (unit-tested).

### 2.6 Fixed-threshold baselines (`scripts/index_l3_care_benchmark_aarhus_baselines.py`, JSON in repo)

No model training and no threshold tuning on held-out data. Window metrics use the evaluable subset only; a separate onset-level evaluation de-duplicates overlapping windows of the same first-onset `(batch, animal, event_ts)`; episode metrics collapse repeated per-animal alerts with a 3-day cooldown. False-positive rates are per **1,000 evaluable weekday scoring windows** — a unit distinct from calendar pig-days or deduplicated post-suppression alerts.

Test split (12 cohorts; 6,293 evaluable windows = 35 pos / 6,258 neg; evaluable prevalence 0.556%; **15 distinct first onsets**):

| rule | precision (windows) | window recall | false-positive windows per 1000 evaluable scoring windows | mean lead of alerted positive windows (d) | onsets covered | onset recall | mean first-alert lead (d) | alert episodes (3 d cooldown) | animals ever alerted |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| constant train prior, threshold 0.5 → no alerts | — | 0.000 | 0.0 | — | 0/15 | 0.000 | — | 0 | 0 |
| ≥1 pen-mate fresh-tail day in prior 7 d | 0.0107 | 0.286 | 147.5 | 3.52 | 5/15 | 0.333 | 3.55 | 332 | 127 |
| ≥2 distinct flagged pen-mates in prior 7 d | 0.0224 | 0.286 | 69.3 | 3.52 | **5/15** | **0.333** | 3.55 | 100 | 71 |
| ≥3 pen-mate fresh-tail days in prior 7 d | 0.0282 | 0.286 | 54.7 | 3.52 | 5/15 | 0.333 | 3.55 | 82 | 56 |
| own ear fresh ≥1 prior day | 0.0085 | 0.171 | 110.8 | 3.55 | 2/15 | 0.133 | 4.52 | 174 | 26 |
| max(ear, pen-mate) ≥1 | 0.0073 | 0.314 | 237.6 | 3.47 | 5/15 | 0.333 | 3.76 | 458 | 132 |

Honest interpretation (window vs event): 35 positive test **windows** collapse to 15 distinct recorded onsets; the best rules touch only **5 of the 15** (33%), yet are attributed 28.6% "window recall" — window recall is heavily repeated-window inflated and must not be read as event coverage. Evaluable-prevalence lift is **~4.0×** for the ≥2-pen-mate rule (0.0224/0.005562) and ~5.1× for ≥3, not the earlier loose "5–7×". Every rule alerts at false-positive rates of 54.7–237.6 per 1,000 evaluable scoring windows — grossly above any plausible caretaker budget — and alerts spread over 56–132 animals (i.e., many pigs warned without later recorded damage). In-sample (train) the same rules look far stronger (e.g., ≥3 rule precision 0.130, onset recall 0.362 on 69 onsets), confirming a real train→heldout generalization gap across pen cohorts. Calibration-split metrics are unstable (e.g., ≥2-pen-mate rule: 0 detections on 4 onsets; earlier own-ear precision flip), so these candidate detectors are **not calibrated across cohorts**. No learned model was trained (handoff stop rule).

### 2.7 Level assessment and admission verdict

- This is **not** an approved Level-3 caregiver-action task: the release contains **no timestamps of staff actions** (no treatment/removal/dose rows). It is a genuine, time-causal **future adverse clinical outcome** (first fresh tail damage) task — precisely the "strongest defensible future adverse clinical outcome" fallback contemplated by the handoff.
- **PROPOSED (PROPOSED_L2TO_L3 in gate):** APPROVED_RESEARCH_TRAIN/EVAL candidate scope = binary 7-day first-fresh-tail-damage warning restricted to clean-at-risk animals, `(batch, pen)`-cohort-heldout, noncommercial research, attribution to Coutant et al. (2026). Never claim deployment/alert-benchmark validity from it.
- Conditions before any further admission: (a) reconcile 824-vs-810 and the fate of the 19 short-history animals with the paper/authors; (b) confirm published paper's protocol text (fresh-damage definition; removal handling so "first **recorded**" onset can be read despite weekday-only scoring possibly missing transient lesions); (c) pre-registered alert budget with event-level evaluation (onset recall, per-pig alert episodes) at a defined caretaker follow-up review, before any APPROVED_DEPLOYMENT_VALIDATION discussion (not applicable now).

### 2.7b Construct cost and generalization sensitivities (review items 5–6)

- **Feature cost:** the state features require expert clinical inspection of other pigs in the pen and of this pig's ear (fresh-damage surveillance). This is measured clinical surveillance, not low-cost sensor telemetry; the task validates the scientific welfare-outcome question, not an autonomous low-cost warning system.
- **Censoring heterogeneity:** test-split unevaluable windows (1,829 censored + 175 not_measured of 8,297 = 24.2%) are cohort-dependent; window-generation censoring interacts with removal timing (animals removed within days of first damage are silently lost). A follow-up analysis must compare censor rates per cohort and by eventual outcome before any admission.
- **Recorded ≠ true onset:** "first fresh tail damage" is first **recorded** on weekday scoring; transient damage occurring and healing between scans is unobservable in this release.
- **Generalization boundary:** two batches of ONE experimental study at one study site; `(batch, pen)`-heldout is **not** farm-heldout; batch/treatment windows differ materially (test prevalence 0.15–1.3% per `(batch, treatment)` cell). Any admission text must state this scope limit explicitly.

### 2.8 Reproduce

```bash
python3 -m pytest tests/test_index_l3_care_aarhus_build.py        # 16 offline tests
python3 scripts/index_l3_care_build_aarhus_historical.py --download \
  # or --zip <local copy with md5 7d07b5597b935abea7999c26adec9869>
python3 scripts/index_l3_care_benchmark_aarhus_baselines.py \
  --states data/index_l3_care/aarhus/states.csv \
  --output docs/research/index-l3-care/aarhus_baseline_report.json
```

Outputs (states.csv, provenance.json) stay outside Git under `data/`; only aggregate numbers above are committed.

---

## 3. Deep audit #2 — Chou et al. 2020 PLOS ONE S1 XLSX (CC BY 4.0)

- File: `https://journals.plos.org/plosone/article/file?type=supplementary&id=10.1371/journal.pone.0241619.s001` (PMC mirror serves an HTML interstitial; download from PLOS works). SHA-256 `250522604e6dfaf78ef135d2085d1ce5957d163f39f734556af67e18668d5b52`, 567.9 KB.
- Sheets: `Lesion` (4,368 rows × 20 cols), `Behaviour` (1,008), `Production` (48), `Carcass` (672), `Wood` (220), `Toy` (216), `Removal` (48).
- `Lesion`: per pig (672 IDs; one pen only per pig; 14 pigs/pen exactly) × scoring event (12 real dates 2020-09-02 → 2020-12-14); per row: start/date/day/week/stage/pig/sow/block/pen/treatment and ordinal scores `FT1` (tail damage 0–3), `FT2` (blood 0–3), `Tailmissing` (amputation level 0–3), `Ear` (0–4; missing on 1,393 rows preserved as `.`), plus 66 unscorable `.` rows on FT1/FT2 (31 pigs) — these must remain `not measured`, never negative.
- Integrity: 0 duplicate pig-week rows; one treatment per pig; real calendar dates (per-batch scoring days differ — batch split must not be mixed).
- Events (computed): 3,619 consecutive-scoring pairs with both scores; 154 transitions 0/1 → ≥2 (wound), 71 into 3; 626 first-blood transitions; 3303 at-risk windows with baseline <2; 228/672 pigs ever wound; 635 ever blood; 446 partial tail loss; 9 pigs permanently dead/dropped (pen-level 'Death or dropped' totals 9).
- `Removal`: **pen-level only** — victims 194 (161 individual pigs, some removed repeatedly), biters removed 58, in-pen injections 52, outbreaks in 22/48 pens. Numbers match the published Table 3 totals (161 individual removals / repeated counts / 9 permanent removals). There are no per-event timestamps or per-pig movement rows in the release; the "recorded in detail" hospital-pen movements are not in the S1 file.
- Verdict: **ACCESS_VERIFIED with a real animal-week ordinal clinical-escalation task potential** (e.g., "does this pig worsen to wound score ≥2 by the next scheduled scoring, median 13 days"), independent clinical labels, genuine negatives, exact dates and censoring structurally available. It stays PROPOSED and unbuilt in this sprint (handoff: one builder). The pen-level action aggregates are evidence of real caregiver actions but **cannot** validate an action-prediction task.

## 4. Quarantine — Agroscope ethogram+lesions (Zenodo 5920843, CC BY-NC 4.0)

- All 10 record files downloaded and md5-verified (lesions.txt `322540016a2d322ee6284b200a02751a` etc.); the previously un-joined `lesions.txt` now parses and joins cleanly on `pigID + observation` (380 rows, 95 pigs, 0 duplicate keys, keys complete across animal/conspecifics/lesions).
- Chronology confirmed from README/abstract: within each session the 5-min focal counts are recorded first and wounds/cleanliness are scored *after* the behavioural observation (same-visit causal order; consecutive observations are on different days but **no dates or gaps are in the released tables** → cross-session prediction times/horizons undefined).
- Event poverty: only **3 pigs** ever show active tail biting across all 380 focal rows (paper: 3 biters); tail-severity escalation (Cat0 → ≥1) occurs for ~19 pigs over 4 sessions; body-part scratch fluctuations make "any escalation" near-universal (93/95) and not an adverse event construct. Bulk of lesion signal = scratches (Cat1) on head/body.
- Group structure: only 2 farrowing groups (48/47 pigs) for any separation; no pens in released tables.
- Verdict: **QUARANTINED** for the care-event lane: real clinical labels but far too few events for heldout event metrics, no session dates to pin prediction times, and NC license. Keep as L1/L2 ethogram/lesion-association evidence, not as a forecasting task.

## 5. Confirmed persistent blockers in this lane

1. Heseker et al. 2024 (real staff identification/removal of 7 biters): paper's data availability = request only (privacy), verified by earlier commit. Not retried.
2. Bite-o-Mat / Eisermann et al. 2025: open-access CC BY article mirrored at openagrar (SD2025137) states **"Data will be made available on request."** No public event timeline rows. The paper about an alarm device is not itself an alarm dataset.
3. CowScreeningDB requires signed license/registration — not downloaded (rights gate).
4. The longitudinal "action-logged" style studies either do not release per-event action timestamps (this bit of Chou 2020) or are fully private (Ontario case study; mushroom-farm GDPR sources).

Therefore, after this sprint: **zero APPROVED Level-3 caregiver-action tasks; zero public per-action-timestamp care-event datasets confirmed.** The two best real reproducible outcomes in this branch are the Aarhus clinical-outcome builder (PROPOSED) and the fully verified Chou clinical-escalation structure (PROPOSED next gate).

## 6. Ownership statement

All changed/added files in this branch belong to Operator A paths: `docs/research/index-l3-care/`, `scripts/index_l3_care_*.py`, `tests/test_index_l3_care_*.py`, plus `data/index_l3_care/` (Gitignored). No changes to frozen v0.1/v0.2 suites, compiler, other operators' files, or handoff docs beyond prior commit `413f7b8`. Original source bytes remain outside Git (cached under `data/index_l3_care/aarhus_source/`).
