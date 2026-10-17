# OpenFARM Decision Index v0.1 — executed research sprint, 17 Oct 2026

**Disposition:** Main index/forensics/data-authenticity gates **COMPLETE**, newly observed L3 decision-model failures described below. **Not** a released public leaderboard, independently blinded prospective study, or authorized clinical decision service.

## Artifact evidence and verified CI

| Deliverable | Result | Provenance |
|---|---|---|
| Machine-readable source/task index | 8 source-verified typed tasks across 6 animal species; 3 previously approved narrowly for research, 5 candidacies requiring task-specific final review | [tasks.json](../../../benchmarks/decision-index/v0.1/tasks.json) |
| Real decision-model registry | 15 genuine scored model/task/primitive results across Wong and Farrow, with source run URLs | [observed-results.json](../../../benchmarks/decision-index/v0.1/observed-results.json) |
| General scorer | Binary Noul, ordinal Score, categorical Choice, masked/abstained states, true expert vote distributions, under/over urgency and group-bootstrap Brier | [score script](../../../scripts/index_v01_score.py) |
| Schema/score/source CI | **PASS** all 5 index structural tests and 6 scoring tests + private Wong 102-reference audit | [GH Actions 39035300467](https://github.com/ob1-s/OpenFARM/actions/runs/39035300467) |
| Original full clinical rater audit | **340 original cases, 1,020 vet decisions**, SHA pinned; no clinical source data committed | [rater dossier](2026-10-17-wong-human-disagreement.md) |
| Jev 1.14 vs per-vet references | **204 authentic API calls**, 102 cases × Choice/Score, with separate expert and urgency-risk curves | [GH Actions 39035637253](https://github.com/ob1-s/OpenFARM/actions/runs/39035637253) |
| Farrow prompt forensics | **120 Liquid Q8 CPU + 120 Jev hosted decisions** across four task rubrics on original public vignettes | [GH Actions 39033689489](https://github.com/ob1-s/OpenFARM/actions/runs/39033689489) |
| Big Broiler original video access | 102 original MP4 members; **3 true MP4 videos**, one from each camera setup, SHA checked + their matching annotation CSV and 25 fps; 9 actual decoded frames | [GH Actions 39033955908](https://github.com/ob1-s/OpenFARM/actions/runs/39033955908) |
| Small specialist training | **Six actual CPU variants** on 171 training /67 calibration /102 reused exploratory test cases; word/character/combo TF-IDF × categorical/ordinal decision heads | Reproducible private-input-free user deliverable, original XLSX not committed |

## Result 1 — veterinarian disagreement is structural

On the **original 340 clinical cases**, the three vets chose Immediate **23**, **26**, and **71** times. All three agree on only **58 cases**; all three choose different ordinal priorities on **77 cases**. Vet1/vet2 exact agreement **47.1%**, vet1/vet3 **26.8%**, vet2/vet3 **37.6%**. This is not necessarily sloppy annotation: care thresholds and clinical risk tolerance plausibly differ. We cannot diagnose rater quality from ratings alone.

On the frozen repeatedly-used 102-case test, **17 unanimous /27 all-different /9 majority-Immediate**; vet3 chose Immediate **26** times versus vet1 **6** and vet2 **8**. A single “expert consensus” hides these dramatic expert-specific differences.

A mathematical distinction that now belongs in all Index scoring:
- `Brier_q = Σ(p_i − q_i)²` compares the probability forecast `p` to the empirical three-vet distribution `q`.
- `E_vet[Brier_onehot] = Brier_q + [1 − Σ_i q_i²]` for a randomly selected one of the original three labels.
- Test mean additive variability **0.42919**; full-source 340-case mean **0.41895**. This is the measured disagreement noise component, not an algorithmic safety bound.

**Jev 1.14 Score** matches vet1 on **28.4%**, vet2 on **43.1%**, vet3 on **47.1%** of the same 102 cases; its decisions skew more urgent than vet1 on **66.7%**, more urgent than vet2 on **54.9%**, but more urgent than vet3 on **27.5%**. It broadly aligns more with the aggressive vet3's risk threshold, not an independently verified optimal urgency. Model-human entropy correlation **0.327**: mild positive relationship between model uncertainty and human disagreement.

## Result 2 — Liquid Farrow failure is a measurable output collapse, not just low accuracy

On the original **30 independent** Farrow owner-facing scenarios, all variants had **30/30 valid native probabilistic results**. No unseen source data were added; these are repeated exploratory forensic probes.

| Genuine Liquid d1-omni Q8 CPU primitive/rubric | “Self-care” predictions | Exact panel consensus | Undertriage |
|---|---:|---:|---:|
| Six-way named Choice, original question | **26/30** | 1/30 | 90.0% |
| Same exact content but reversed option iteration order | **28/30** | 0/30 | 93.3% |
| Explicit owner-at-home rather than hospital queue | **23/30** | 0/30 | 93.3% |
| Ordered six-level Score | **13/30** | 4/30 | 73.3% |

Liquid's failure has a *consistent low-urgency/self-care bias*, not a mere indexing mistake or a particular position in the option map. The ordinal primitive reduces collapse but still fails **all 5 within-one-hour references** in this forensic run. Does **not** justify claims about veterinary advice safety, true medical outcomes, or general animal reasoning.

**Jev 1.14 Choice** changed its top decision on **6/30** original vignettes when the same six options were reversed; Brier changed **0.824 → 0.734** (repeated result subject to stochastic variation). A semantically equivalent option presentation can move evaluation results considerably. This is a model robustness property to include going forward. Jev's Score remains separate from Choice because rubric wording also changes, so no isolated causal output-head claim.

## Result 3 — tiny specialists reveal the calibration versus escalation Pareto frontier

Fit six original-data specialists on **171 training case texts and 513 vet ratings**, with L2 regularization values (0.03,0.3,3) selected by three-vote Brier on **67 calibration cases**. Outcome is then scored on the **102 repeatedly reused exploratory test cases**, not a blind ranking.

| Original task-only CPU specialist | Test three-vet Brier ↓ | Escalated majority-Immediate on test using only-cal threshold | Total Immediate escalations |
|---|---:|---:|---:|
| Word TF-IDF multinomial | 0.305077 | 9/9 | 22/102 |
| Word TF-IDF ordinal cumulative | 0.305147 | 9/9 | 22/102 |
| Character TF-IDF multinomial | **0.292629** | 6/9 | 13/102 |
| Character TF-IDF ordinal cumulative | 0.297261 | 6/9 | 13/102 |
| Word+char multinomial | 0.301200 | 7/9 | 14/102 |
| Word+char ordinal cumulative | 0.308237 | 7/9 | 14/102 |

**Calibration-selected winner** was the **character-level cumulative ordinal** head (cal Brier 0.278762, even though the char multinomial scored best on *test* Brier). Its emergency recall was 6/9 with 13 Immediate escalations. The difference from 9/9 with 22 escalations demonstrates the danger of optimizing average proper scoring alone. Do **not** select the best test result for future publication, since it was not selected on calibration.

Reference comparison: train-only majority prior Brier 0.34064, Jev 1.14 Score 0.35970, Liquid Choice 0.43552, Julia-1 Choice 0.59230 against the same **Wong** three-vet target. Different inference modalities and deployment runtimes are not hardware-normalized.

## Result 4 — original Big Broiler visual dataset is no longer hypothetical

Original 32.5 GB ZIP and 102 file pairs verified. Downloaded original source MP4 files with upstream exact-stem action annotations by **HTTP byte-range reads**, no full archive. One actual sample from each camera setup:

| Camera setup | Original video source stem | Original video bytes | fps | Label-positive frame |
|---|---|---:|---:|---:|
| 1 | `037_1_10_1920_157_0_01` | 152,982,641 | 25 | 2,016 |
| 2 | `092_2_20_1920_011_1_01` | 31,647,025 | 25 | 1,103 |
| 3 | `090_3_32_1920_010_0_10` | 183,490,349 | 25 | 1,572 |

**Nine genuine 1920×1080 frames decoded** and frame/timestamp alignment checked; source bytes hashes and annotation SHA pinned in Actions run. There is **no video model performance measurement in this verification task**. Benchmark video under actual independent camera/generalization splits, not just hand-selected positive clips. Multimedia source rights remain task-specific.

## Decisions, blockers, and next falsifiable gates

**Completed in this sprint:** versioned task schema, tested typed metric implementation, authenticated real reference benchmarks, rater discrepancy analysis, native model prompt/order forensics, actual video-byte decoding and alignment, six train/cal/test small specialist comparisons.

**Not completed / outside evidentiary reach of present sources:**
- New **patient-disjoint, independent-hospital, untouched** veterinary cases with real adjudicated urgency/outcome evidence; current Wong/Farrow are retrospective and repeatedly evaluated.
- Proof that any model improves animal outcomes, care decisions or safety; veterinarian assessments are reference opinions, not observed prospective treatment effects.
- Original model-weight training beyond cheap small text heads; 171 cases is too little to support a domain foundation model claim.
- Full-video human and model evaluation of Big Broiler across camera setups and site-heldout farm conditions; sparse visual frames are not a true video encoder.
- GrimACE/Aarhus/Big Broiler PRs #6/#4/#7 remain independent **drafts**; source-level agreements, split limitations and rights must be reviewed before treating them as official admitted score tasks.
- Cow, meerkat and HotPig previously approved observation sources are listed and pinned, but their older legacy runs are not yet fully normalized into v0.1 model results.
- Task/school rights and deployment licensing remain separate gates; some sources are non-commercial or share-alike, and patient text access must be independently authorized.

**Next reliable evaluation:** build a *new blinded* research packet with independent clinics and raters (plus longitudinal or treatment outcomes when possible), before calling OpenFARM a clinical triage leaderboard or tuning more prompts on the same 30 Farrow cases.

Sources/code: [task registry](../../../benchmarks/decision-index/v0.1/README.md), [per-vet script](../../../scripts/index_v01_wong_jev_per_vet.py), [Liquid forensic script](../../../scripts/index_v01_farrow_liquid_forensics.py), [source MP4 probe](../../../scripts/index_v01_big_broiler_video_probe.py). **All risk analyses are descriptive research, not veterinary advice.**
