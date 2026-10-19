# Stage B scientific decision gate — 2026-10-19

**Decision: BLOCKED.** The present trained two-source shared-trunk architecture does **not** justify replacing task-specific trained decision models. We reached this result by *successfully executing* source-verified training and inference, **not** by interpreting an infrastructure outage as a model failure.

## Execution and provenance

- [Completed original-source CPU model workflow (success)](https://github.com/ob1-s/OpenFARM/actions/runs/39194149742)
- Original run artifact: `openfarm-stage-b-two-original-independent-source-gate`, artifact ID `12160674769`, SHA256 of artifact ZIP `823bf5a64764003935b104de1757ede33f0984284314d36199b3da6d69a25e90`. This artifact has its own retention period; the script and reported summary survive.
- Source SHA256 HotPig series.zip: `1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8`.
- Source SHA256 Brazilian cow original Cowlevel.csv: `3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529`.
- Run file `scripts/index_alpha_stage_b_auth_sources.py`. Source bytes remain outside Git; task inputs exclude gold and group IDs.
- Pigs train 1–16 / calibration 17–20 / development test 21–24. Cow farms split using deterministic random seed 20261019: train 13,16,18,20,26,28,30,32,34,39; cal 7,31,47; development test 23,40,45.
- Tasks use **different reference families and sources**. HotPig's 12 *previously computed* camera counts predict feeder current-minute positive intake; cow breed/lactation/DIM predict a contemporaneous independent human gait score ≥3. **Neither task is direct perception of pain or a validated welfare alert.**
- HotPig eval 92,160 correlated minute rows in **four test animals**; clinical cows 1,335 rows in three heldout farms. Farm grouping, not row grouping, is the independence unit.
- Classical train-only prior, regularized logistic and tiny boosted tree; two trained tiny neural variants with source-specific input adapters and binary output heads, either two independent trunks or one shared 24→32→16 trunk; both independently fit to real original data. Six fixed training epochs, AdamW, seed 20261019, early checkpoint selected using *calibration* prior-normalized Brier, never test.
- Independent neural: **3,146 parameters**, 1.29s fitting CPU. Shared: **1,818 parameters**, 1.18s fitting CPU. These timings measure local fitted neural loops on one GitHub runner only; not comparable to hosted pretrained model inference.

## Predeclared Stage B success gate

Shared must reduce *held-out Brier* by **≥2% relative to the best trained specialist** (regularized logistic, tiny tree, independent neural) on **BOTH** sources, and improve subject/group Brier on **≥3/4 pigs AND ≥2/3 test farms**.

This is already a *developmental* evaluation, not a pre-registered clinical/publishing trial; test cases and source have been studied previously. Nevertheless, the within-run threshold was fixed before inspecting scores and not retuned afterward.

## Actual results

| Source task | Train-only prior Brier | Logistic Brier | Tiny boosted tree Brier | Independent neural Brier | Shared neural Brier |
|---|---:|---:|---:|---:|---:|
| HotPig feeder current minute | 0.049725 | 0.020392 | **0.019879** | 0.020619 | 0.020611 |
| Cow human gait ≥3 (3 heldout farms) | 0.232182 | 0.192482 | 0.190647 | 0.189600 | **0.189457** |

**HotPig:** the shared network has **3.68% higher Brier** than the best tiny boosted-tree specialist and is worse on **all four test pigs**. The tiny tree remains the best trained model in this comparison. Even the separate neural is worse than the tree. This is a meaningful decision-model capability/cost baseline, not evidence a new neural generalist is needed.

**Clinical cow:** shared is about **0.075% better relative Brier** than the separate neural, a minuscule improvement; it wins on just **1 of 3** farms relative to the separate network. It does not meet either predeclared clinical-cow condition.

**Overall gate:** FAIL both source tasks. Stage B **blocking failure, not generic model success**. No OpenFARM generalist model should be claimed from this evidence.

## What this does and does not falsify

This **does falsify the specific engineering claim** that the tested small shared-trunk model offers sufficiently better real-source decisions than simple specialists on these two tasks under fixed protocols. We have working source ingestion, baselines, a trained 1,818-parameter joint head and matched heldout evaluation.

This does **not** prove that a better multimodal encoder, more varied supervised tasks, pretrained model, other training objective, or new field data cannot enable beneficial shared decisions. A shared model with separate input/output adapters is not truly zero-shot task generalization anyway. The original expert outcome and the feeder instrument do not validate positive emotions, pain, welfare, or caregiver attention.

A fair *same-input pretrained foundation-model comparison* remains absent. Liquid's six sampled Big Broiler video windows and the veterinary triage text experiments use different sources, states and tasks; they must not be included in this matched Brier table.

## Independent Index alpha task status

[Original-source Index alpha runner](../../scripts/index_alpha_runner.py) validated **HotPig** and **16-farm clinical cow** independently on real source files in [CI 39193407661](https://github.com/ob1-s/OpenFARM/actions/runs/39193407661) and [CI 39193742632](https://github.com/ob1-s/OpenFARM/actions/runs/39193742632). Third source (cattle wearable MI human behavior) was **BLOCKED at download** by provider HTTP 500 from original Figshare download and alternate official public API routes. The file source hash and declared rights were not silently bypassed.

**Therefore:** runnable Index alpha infrastructure exists with two original-source passing scorecard evaluations and a third explicitly unavailable; Stage B has a measured, completed model-comparison **negative result** using the two authenticated sources. It is not a public blind leaderboard.

## Next scientific decision

**Do not scale this specific shared trunk now.** Keep the tiny HotPig specialist and plain Cow gait logistic/trees as reference baselines. To revisit the shared-model hypothesis, find a new dataset or model-visible input type that actually supplies transferable information or a new independent welfare label, freeze unseen groups and all training decisions, then test the hypothesized transfer rather than merely reuse two tiny unrelated tabular prediction tasks. Index can launch as a **developmental alpha**, clearly separated from approved blind benchmark or deployment status.
