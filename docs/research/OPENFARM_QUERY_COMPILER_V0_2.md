# OpenFARM v0.2 — real multi-decision query compilation
> **Temporal science correction (2026-10-16):** [Historical validation report](OPENFARM_HISTORICAL_TASK_VALIDATION_V0_2.md) supersedes older PigWeb behavioral state/forecast claims below. Source `pre` timestamps do **not** show that stimulus-response behavior was available then. The executable builder now exposes **pre biomarkers only** in PigWeb forecast input; human behavior is separately retained for *retrospective association*. The current compiled suite therefore has **ZERO validated multichannel model-input states**, not 248. All **130,005 states, 623,174 valid reference labels and 546 evidence masks remain unchanged**. First heldout forecasts are weak; do not call them approved welfare outcomes.


**Checkpoint:** 16 October 2026 · **Research branch:** `research/2026-10-data-prospecting`. This implements the priority described in [multiple decisions and historical task audit](OPENFARM_MULTI_DECISION_TEMPORAL_EXPANSION.md). The frozen [canonical v0.1 decision inventory](OPENFARM_DECISION_SUITE_V0_1.json) remains unmodified.

> **Scientific aim:** one actual observation state → several scientifically warranted typed questions → one measured reference per underlying evidence channel, including validity masks and future-only labels where there is a real observation order. *Do not mistake newly generated propositions for newly obtained independent supervision.*

## Actual compiled source data — reproducible, not aspirational

These counts come from real pinned source downloads and green CI builds; the final aggregated CI summary independently recomputes them from source manifests. Includes only allowed modalities (not claimed raw video/audio processing when the input is already an expert-coded track or model-derived summary).

| Source | Actual states | Q labels / state | Valid Q labels | Underlying independent reference families / state | Contract and decision status |
|---|---:|---:|---:|---:|---|
| **HotPig** | **36,718** (15-min sampled historical windows from 24 pigs) | 2, except censored future | **73,410** | **1.00** electronic-feeder instrument | Current feeding: approved bounded research. Future new-bout onset: newly constructed *historical candidate*, not welfare alert |
| **Cattle MI** | **9,222** (24 cows) | 5 | **46,110** | **1.00** human-video ethogram | Original 4-class Choice approved bounded research; four one-vs-rest Noul questions are **dependent diagnostic slices** |
| **Meerkat geometry** | **83,763** (20 original source videos) | 6 | **502,578** | **1.00** human-coded ethogram | Original 15-class Choice approved for bounded **noncommercial** research, 12/4/4 whole-video holdout; five binary behavior slices are dependent diagnostics |
| **PigWeb physiology** | **248** (124 pigs, two study contexts) | 6 candidate slots | **968** | **1.95** source-limited independent assays | Actual pre-observation + pre hormone state; subsequent independently measured cortisol, oxytocin and eye infrared temperature; candidate **not** validated affective welfare model |
| **EquiFACS** | **54** independently vet-rated, quality-gated intervals | 2 | **108** | **1.00** independent vet score | Ordinal original rating plus derivative binary presence; **tiny nonranking pilot**, expert facial-action preannotation (not raw video inference) |
| **Total (different source types)** | **130,005** | — | **623,174** | **~1.0018** source-weighted | Different studies, not 130,005 independent animals or clinical episodes |

There are **623,720 attempted question slots and 546 valid masks**: 520 PigWeb assay-missing slots (because binary and delta projections are masked together) and 26 HotPig future target windows where source coverage is incomplete or ends. `623,174` is the number of **valid question labels**, **not the amount of independent ground truth**.

Of those valid labels, **456,241 are dependent rewrites** of existing reference information: 418,815 Meerkat binary diagnostic slices, 36,888 cattle binary diagnostic slices, 484 PigWeb continuous deltas sharing the corresponding binary biomarker assay, and 54 EquiFACS binary derivatives. The remaining 166,933 are **non-derived atomic labeled propositions**, *still sharing original sensors* in historical HotPig. The actual count of **independently measured reference families across states** is only **130,241**; a subset of these are repeated minutes/intervals/animal contexts. This is a strict, intentionally conservative *reference-family* measure, not a claim of statistically independent observations.

**Corrected:** The original PigWeb *source* includes human-observed stimulus behavior and physiological samples, but their causal availability is not simultaneous at the **PRE** forecast cutoff. The corrected compiler retains PRE saliva assays only, so **0 states** have scientifically certified multiple input modalities *at prediction time*. Retrospective multimodal analysis may be possible as a separate protocol, but must not be described as a prospective forecast. HotPig's target feeder remains a reference, not an input. The compiler does **not** claim raw image, full audio or video understanding.

## Concrete questions from one real observation

### PigWeb — pre-only historical state with independently measured future outcomes

```text
Observation state (at PRE sample, all truly available then):
  pre-session cortisol; pre-session oxytocin
  missing pre-sample assay channels marked as null
  [IMPORTANT: exclude play, escape, inactivity, tail and ear actions:
   those were observed DURING experimental exposure, later than PRE.
   They can be studied separately as retrospective associations.]
  [never pass facility, animal ID, treatment/positive-negative context,
   future post sample, sample date or post elapsed minutes as model input]

Typed questions:
  Noul: Did the next measured cortisol sample exceed the previous one?
  Noul: Did the next measured oxytocin sample exceed the previous one?
  Noul: Did infrared eye temperature subsequently increase?
  Value: How much did each measurement change? [ALTERNATIVE scoring of
         the same three assays; not three more biological references]
  Each question has its OWN available/missing evidence mask.
```

Verified original data: 234 cortisol paired increases (143 positive / 91 not); 210 oxytocin paired labels (102 positive / 108 not); **40** temperature pairs from one research site. All future assay sample timestamps occur **later than the pre-state**. The compiler hides the realized delay from the model. Four-institute source splits are configured with entire institutes as test and a different one for calibration; repeat pigs never cross the split. **Caveat:** treatment/session context may determine measured physiology; hormone change does not equate to pain, joy, or need for care.

### HotPig — true historical decision (source index, no invented calendar times)

```text
State at original relative minute t:
  12 ORIGINAL CAMERA-DETECTOR behavior/posture counts; all observed by t
  (No source feeder values, pig ID, treatment period or future counts)

Q1 Noul: Did the independent feeder register activity now?
Q2 Noul: Will a NEW feeder-active bout begin during (t,t+15min]?
   reference derived from verified consecutive feeder minutes after t
   if future window is incomplete => MASK, never assume "no event"
```

A 15-minute state stride is used to reduce overlapping forecast windows. **Both questions derive from one electronic-feeder instrument**, so the measured-family count is one; onset is a legitimate *future temporal formulation*, not an independently diagnosed welfare outcome. No calendar timestamps are invented to support target causality: source-relative ordinal minute indices are recorded only as private audit provenance.

### Cattle / Meerkat — Choice with transparent diagnostic projections

One human-observed `Choice` ethogram permits several binary questions (`Noul(foraging)`, `Noul(guarding)`, `Noul(grazing)`, etc). It **doesn't add independent labels**. The derived `Noul` questions are useful for inspecting sensitivity/precision for specific animal behaviors but are never counted as new ground truth, tasks from a new source, or independent welfare outcomes.

Cattle model state is **one author-provided wearable motion-index scalar**. Its source time key is ambiguous: no new future behavior tasks are built from it. Meerkat input is **human-drawn bounding-box geometry with causal past-only motion**, NOT raw frames. Its partitions remain 12/4/4 original video files; animal independence across videos is not established. Rights remain noncommercial.

### EquiFACS — the independent-rater welfare interpretation pilot

Original clinician `Score` is retained as a reference and `Noul(any clinician-rated pain)` is an explicitly dependent projection. Current-source media-level 10/3/5 holdout has **54** intervals from 18 videos after excluding the facial coder's pain ratings and overlapping unscorable facial actions. **No robust ordinal calibration** (no moderate scores in retained data), no reliable clinical generalization from five heldout media, and clinical video identity might not independently equal one horse. This remains *nonranking pilot*, not a promoted welfare benchmark.

## Compiler safety / artifact layout

**Implementation:** `scripts/openfarm_query_compiler_v02.py` defines `Observation`, `Question`, `Compiler`, native typed outputs `Noul`, `Choice`, `Score` and experimental numeric `Value`. The last is a source-referenced scalar delta contract, **not a claim that Jev's official runtime currently exposes a `Value` primitive**. Adapt it to an appropriate supported output type during actual model training.

A question includes original reference family, precise typed proposition, measured vs derived origin, valid/masked status with reason, current/future relation and **real ISO datetime or source-relative ordinal index**, never synthetic source timestamps. Multi-question aggregation counts independently measured families per observation and explicitly separates *valid question labels* from *primary questions* and derived equivalents.

For `--local-dir`, the builder outputs three separate files per source/split:
- `states.jsonl`: **opaque state ID**, observable features, modality tags, question IDs. No answer, group, study context, rater, original source ID or timestamp.
- `references.jsonl`: original labels, null/masked reasons and source evidence channels (kept out of model prompts).
- `provenance.jsonl`: original subject/video/site keys, relative or real timestamp, missingness, source/license information (for auditors, not model input).

There is also a question catalog with types/options/evidence origins, no reference answers. An evaluator can join by the opaque hash ID, score the model, and report by source group. **Do not supply the merged reference/provenance JSONL as the model state.**

Fail-closed invariants and adversarial unit tests include: no label or future data as features; no source subject/site metadata as features; no opaque ID containing treatment, animal or calendar time; no train↔test reuse of independent groups; no mixed-source state duplication; true strict future labels; inability to call missing values healthy/no-event; recognized Choice/Score/Noul references; alternate projections of one original label do not increase measured-family counts.

## Run and verify

```bash
python -m pip install pandas numpy requests scikit-learn openpyxl

python -m unittest discover -s tests -p 'test_openfarm_query_compiler_v02.py' -v

python scripts/build_openfarm_decision_queries_v02.py --source pigweb --report reports/v02-pigweb.json --local-dir /tmp/openfarm-v02
python scripts/build_openfarm_decision_queries_v02.py --source hotpig --report reports/v02-hotpig.json --local-dir /tmp/openfarm-v02
python scripts/build_openfarm_decision_queries_v02.py --source cattle --report reports/v02-cattle.json --local-dir /tmp/openfarm-v02
python scripts/build_openfarm_decision_queries_v02.py --source equifacs --report reports/v02-equifacs.json --local-dir /tmp/openfarm-v02
python scripts/build_openfarm_decision_queries_v02.py --source meerkat --report reports/v02-meerkat.json

python scripts/aggregate_openfarm_queries_v02.py --reports 'reports/v02-*.json' --output reports/v02-suite.json
```

**Official CI:** `.github/workflows/openfarm-query-compiler-v02.yml` performs source-free adversarial tests, a five-source pinned real-data matrix (separately cached/checksummed where appropriate), isolated local model/label/provenance exports for four sources, and a final suite aggregation. Only **manifest summaries** are uploaded as artifacts, never third-party row-level data or source video.

## Scientific status and next admission decisions

This is **real, functioning data infrastructure**, with empirical question counts and evidence accounting across the existing research sources. It is *not* five new clinically validated benchmarks.

- **Observation:** HotPig current feeding, cattle 4-class ethogram and Meerkat 15-class video-heldout ethogram remain narrowly approved for *research* under original respective terms; behavioral binary slices are diagnostic views, not more independent tasks.
- **Interpretation:** EquiFACS is **tiny nonranking pilot**; PigWeb future hormonal outcomes are **source-joined, scientifically interesting candidate forecasts**, but they are *not* independently validated positive/negative emotional labels.
- **Action:** **ZERO** tasks currently carry independently verified caretaker attention, treatment choice or measured welfare-intervention outcome. Never derive an “alert now” answer from a cortisol threshold, pose score or behavior anomaly by fiat.
- **Multimodality:** The PigWeb study combines behavioral and physiological observations **retrospectively**, but source evidence does not establish that stimulus behavior was available at PRE. Therefore **0 v0.2 causal forecast input states are certified multimodal**. Next sought-after sources: truly synchronized video+feeder/accelerometer and vocalization+physiology with independent veterinary outcomes.
- **Positive welfare:** source expert observed grooming, social behavior, foraging and play actions can define observation-level tasks, but **positive emotional valence still requires independently validated contextual/outcome evidence**.
- **Jev connection:** a query-based decision workload with shared observation state, typed outputs and reproducible evidence masks is model-family-agnostic. No assumptions about compatibility of raw videos/Box with a particular hosted model API. Confirm runtime-specific input/output types when implementing an inference adapter.

**Important:** An example count from overlapping time points or repeating the same human label is not an independent evaluation sample. For confidence intervals and science claims, cluster by original animal/video/site, with separate heldout test cohorts. The v0.2 compiler enforces split contamination safeguards; it does not create independent subjects.
