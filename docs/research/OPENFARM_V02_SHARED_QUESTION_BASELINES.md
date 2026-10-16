# OpenFARM v0.2 — the first *single-model, multi-question* heldout experiment

**Verified on actual pinned HotPig series; GitHub Actions run [38852338294](https://github.com/ob1-s/OpenFARM/actions/runs/38852338294) succeeded on 2026-10-16.** Script: `scripts/benchmark_hotpig_shared_queries_v02.py`; offline mask/shape tests: `tests/test_openfarm_shared_queries_v02.py`.

### Two questions from each identical sensor-state input

- Input: 12 camera-model-derived posture/event **numeric counts at the observed minute**, not raw video; no feeder, pig ID, condition, timestamp or future information.
- Question A: current independent electronic-feeder activity `Noul(feeder_now)`.
- Question B: truly NEW feeder-active bout within the next 15 minutes `Noul(new_bout_in_next_15_minutes)`, masked if the reference horizon isn't entirely observed.
- **One physical electronic feeder reference family**, not two independent labels. 24 source pigs from one trial; whole-pig splits **16 train / 4 calibration / 4 frozen test**. 15-minute state sampling, 49,116 valid train question instances, 12,284 test instances. Test outcomes: 326/6144 positive current and 621/6140 positive future onset.
- All thresholds in this table are the **prespecified 0.5** cutoff. **No tuning on test**; calibration data reserved, and no post-hoc probability fit made.

### Three tiny CPU-only architectures + prior

1. **Train-only prevalence** per question: a strict baseline.
2. **Two independent L2 logistic heads**, one for each question, trained from its valid train states.
3. **One shared L2 logistic with one common set of 12 camera-feature slopes and question-specific intercept** (`shared_one_head`). This is the strictest state-sharing hypothesis.
4. **One joint L2 logistic with common 12 slopes plus question-specific residual slopes** (`shared_with_question_residuals`). Still **one fitted probabilistic model**, 12 shared observed variables, one question ID bit, and feature interactions; it can learn task-dependent feature effects.

| Test task / method | Brier ↓ | AUROC ↑ | F1 @0.5 | Average precision |
|---|---:|---:|---:|---:|
| Current feeding, train prior | 0.05028 | 0.500 | 0 | 0.053 |
| Current feeding, shared slopes only | **0.03109** | 0.956 | 0 | 0.780 |
| Current feeding, joint with question-specific residuals | **0.01846** | 0.956 | 0.771 | 0.785 |
| Current feeding, separate logistic head | **0.01844** | 0.956 | **0.772** | 0.785 |
| Next 15-min bout, train prior | 0.09092 | 0.500 | 0 | 0.101 |
| Next 15-min bout, shared slopes only | 0.09946 **(worse)** | 0.556 | 0.063 | 0.118 |
| Next 15-min bout, joint with question-specific residuals | **0.09044** | 0.556 | 0 | 0.122 |
| Next 15-min bout, separate logistic head | **0.09045** | 0.559 | 0 | 0.122 |

### Interpretation

- **Verified architecture failure:** merely asking two questions of one shared sensor state does not imply one common feature mapping works. Pure shared slopes produce severely distorted probabilities and zero detected current-feeding positives at a 0.5 cutoff, despite decent AUROC. This is why we must evaluate **calibration and decision metrics**, not just rankings.
- **Small architecture recovery:** explicit task-conditioned interaction slopes in a *single* joint model nearly recover the accuracy and Brier loss of two independent heads. This demonstrates actual computational/state sharing with separate conditional decision behavior; it does **not** demonstrate a latent animal world model or a new welfare capability.
- **Historical outcome still weak:** next-bout AUROC remains ~0.56 and average precision ~0.122 at 10.1% natural class prevalence, with no positives predicted at 0.5 under the better-behaved models. Merely generating the question from source timestamps does **not** establish useful early warning.
- Source samples are 15-minute series observations, **not independent 6,000 animals**; only four heldout pigs. The two outcomes are correlated reference windows from the same instrument and must not be averaged as independent welfare scores.
- Correct conclusion: **retain contemporaneous instrumental observation**, mark future onset **causally valid but not predictively validated**, and use this as the first strict multi-question harness quality check.

### Compiler-lineage gate (independent of model results)

Separately, [five-source compiler rebuild run 38852020797](https://github.com/ob1-s/OpenFARM/actions/runs/38852020797) passed. Deterministic diagnostic questions now carry executable `derivation` operations and are checked against the same state's primary human/measurement reference, with matching temporal provenance and masks. The PigWeb continuous biomarker delta is primary, with sign/increase derived. The 623,174 existing valid label slots and 546 masks were **not inflated** by this change. See [worked transformations](OPENFARM_DECISION_TRANSFORMATIONS_AND_CHECKS.md).

### Stop conditions before enlarging model complexity

Do not turn HotPig future-onset into a pseudo-Level-3 caretaker alert; no independently rated welfare outcome supports it. Better model training requires additional animals/farms or genuinely richer **historically available** temporal input. For a zero-shot+ animal-welfare monitor, secure an independent welfare outcome/review reference first. Multimodal input requires *jointly timestamped and actually available* observation channels, not the presence of unrelated columns in a study.
