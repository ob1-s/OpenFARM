# OpenFARM — canonical state → multiple decision questions (with actual evidence contracts)

**2026-10-16 · v0.2 compiler hardening.** This document describes transformations implemented in `scripts/build_openfarm_decision_queries_v02.py` and checked by `scripts/openfarm_query_compiler_v02.py`. **Numerical state values below are illustrative**; real labels come from the pinned source files, never synthetic defaults.

## 1. HotPig: current behavior and future onset from one observation

One source pig and a source-relative minute `t`. The **model input** is the twelve numeric counts inferred by the dataset creators' vision detector *at t only*, e.g.:

```json
{"state":{"modality":"computer_vision_precomputed_behavior_counts",
          "observed_counts":{"eating":1,"standing":1,"lying":0,"...":"9 other counts"}},
 "questions":["hotpig.feeder_now","hotpig.next15_new_feeding_bout"]}
```

**Private reference (not input):** minute-aligned electronic-feeder readings `feed[t]`, `feed[t+1]` ... `feed[t+15]`.

- `Noul(feeder_now)`: `1` iff valid `feed[t] > 0`. `None` + explicit mask when the measured feeder signal at t is unavailable.
- `Noul(new_feeder_bout_next_15min)`: `1` iff a feeder transition `feed[i-1] <= 0 and feed[i] > 0` occurs for `t < i <= t+15`; `0` iff no transition and **the entire window t..t+15 is observed**; otherwise masked.
- 15-min stride, **whole-pig** 16/4/4 train/cal/test, same one-farm cohort. Independent **reference family count = 1**, even with two distinct time questions.
- Held-out test: **6,144 current minute labels**, **6,140 observed future onset labels**, four pigs. Source-only current tiny-tree F1 ≈ **0.778**; future onset tree AUROC ≈ **0.567**, barely better than prior, so **do not promote the forecast as animal-welfare early warning**.

## 2. Cattle: multiple behavioral questions, one human ethogram

**Actual input:** the source-provided wearable motion index (a scalar, not whole raw accelerometer or video), derived from a cow-minute.

```json
{"state":{"modality":"wearable_preprocessed_motion_index","motion_index":12},
 "questions":["cattle.behavior.choice","cattle.is_grazing","cattle.is_walking",
              "cattle.is_resting","cattle.is_ruminating"]}
```

**Private independent human-video-reviewed reference:** a single class from `GR`, `WA`, `RE`, `RU`. The `Choice` question is the **primary supervision**, and each `Noul` is a deterministic `choice_equals` projection that must agree with the original human-coded class. If source says grazing, `is_grazing = yes` and the other three = no. These are useful for per-behavior **probability calibration**, but **not five independently annotated welfare outcomes**.

The original source has **9,222 cow-minute rows from 24 animals**, source-grouped by cow; the time key is not uniquely identified and cannot validate historical forecasting. Use only contemporaneous behavior questions. Two source sites are confounded with breed. Zero injury/clinical/affect labels.

## 3. PigWeb: separately assayed continuous values and derived binary questions

**As-of-time input for an *offline* historical analysis:** collected PRE cortisol and oxytocin lab *values*, when actually available after laboratory processing (real-time availability NOT verified). Behavior recorded DURING experimental exposure is **not** a predictor state at the PRE collection timestamp.

```json
{"state":{"modality":"pre_session_biomarker_values",
          "pre_cortisol_ng_ml":6.7,"pre_oxytocin_pg_ml":11.1},
 "questions":["pigweb.cortisol.delta","pigweb.cortisol.increased",
              "pigweb.oxytocin.delta","pigweb.oxytocin.increased"]}
```

**Private source reference:** two different later POST assays. `Value(post - pre)` is **the measured primary target** for each assay; `Noul(increased)` is a `delta_positive` projection and must equal the actual sign of its parent delta, *not* a second physiological outcome. The model never sees the post outcome, source institute, experimental context, during-exposure human-coded behavior or post delay.

The source has **248 paired sessions from 124 pigs across 4 institutes**. Independent assay channels are available for subsets (234 cortisol outcome sessions, 210 oxytocin). Even with independent biological assays, hormones are **not independently verified emotional states**. Leave-one-institute-out predictive transfer failed some sites. **Do not label as generalizable emotion or live welfare forecast**.

## 4. EquiFACS: ordinal vet rating and binary presence from one vet

Input: expert human-coded facial action-unit intervals (not raw video). Reference: ordinal judgment by a **different veterinary annotator** who did not create the action-unit coding. A primary `Score(0,1,2,3)` can yield `Noul(any_pain)` via the mechanically tested `score_positive` derivation.

The compiler excludes the FACS coder's own clinical annotations, clinically non-codeable intervals and unscorable AU segments. **54 source intervals** across 18 clinical media survive; only **12 rating intervals in 5 held-out media**, so it remains a **nonranking integrity pilot, not an approved generalizable pain classifier**. The initial tiny logistic pilot failed against the prevalence reference (AUROC ~0.229).

## The executable guardrail

`scripts/openfarm_query_compiler_v02.py` now rejects:
- A binary behavior slice inconsistent with its original human `Choice`;
- A claimed pain binary inconsistent with its measured ordinal `Score`;
- A hormone `increased` label inconsistent with the measured `delta`;
- A derived label without an actual primary label in the **same state**, or with mismatched reference family/time;
- Replacing a missing primary source measurement with an invented valid derived answer.

Censoring and masking are not negative labels; probabilities are reported for measured questions separately and reference-family counts are kept apart from question-slot counts. Compiler source regeneration and all offline contract tests passed [GitHub Actions run 38852020797](https://github.com/ob1-s/OpenFARM/actions/runs/38852020797).

## Scientific boundary

We can compile **many questions from one state**, but that is not automatically a world model, independent affective supervision, or caretaker escalation. A future Level-3 action needs new independent staff/vet review or intervention-outcome references. The current candidate suite is best described as a verified **multi-question observation + limited interpretation testbed**.
