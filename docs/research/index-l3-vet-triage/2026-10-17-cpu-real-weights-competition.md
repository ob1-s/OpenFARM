# OpenFARM small-CPU decision-model competition — original vet decisions (17 Oct 2026)

## Purpose
Compare native tiny System-1 *decision models* and boring calibrated specialists on **the same original expert-supervised task**, without treating high mean accuracy or low Brier as deployment safety. Levels are independent task-complexity categories, **not a mandatory L1→L2→L3 pipeline**.

## Reproducibility and real-weight policy
- **SupersonicLabs/Julia-1**, 144.3M-parameter official Apache 2.0 weights, native named question API, PyTorch CPU in `scripts/index_l3_cpu_julia_evaluate.py`; actual full run: https://github.com/ob1-s/OpenFARM/actions/runs/39005324887. Source model card: https://huggingface.co/SupersonicLabs/Julia-1
- **LiquidAI/d1-omni-600M-GGUF:Q8_0**, official 587M-parameter open decision weights, 407 MB GGUF; genuine llama.cpp CPU /v1/systemone runner `scripts/index_l3_cpu_liquid_omni_eval.py` with separate CI workflow `.github/workflows/openfarm-cpu-liquid-vet.yml`. Model license is **lfm1.0** (not Apache). Media model is text/image/audio-capable, but this round tests **text only**, not chicken video. Native sequential images require a new modality-specific protocol. Do not substitute a generic autoregressive vLLM chat/completions call for the native decision head.
- **Jev 1.13 and 1.14**, native hosted proprietary decision APIs using user-provided GitHub Actions secret; previously benchmarked in 408 actual calls on Wong across Choice/Score. Full results in [Wong 102-case report](2026-10-17-wong-full-102-jev-choice-score-results.md).
- **TF-IDF(word unigrams/bigrams) + logistic regression** fitted *only on 171 original Wong training cases and their 513 clinician ratings*. Regularization grid `C={.01,.1,1,10}` selected by Brier on **67 calibration cases** (`C=1`), never on test. Source-pinned 340-case workbook and SHA verified by `wong_benchmark.py`. Readable local reproducible script and aggregate JSON delivered separately as `OpenFARM_Wong_Clinical_CPU_Baselines.zip`. Original clinical workbook is NOT in the code artifact or public GitHub.

## Full Julia-1 live CPU result (zero-shot Choice)

Actual genuine weights downloaded and loaded, successful on **all 132** original clinical cases (102 Wong + 30 Farrow), no invalid responses:

| Task | n cases | exact vs panel median/consensus | multiway Brier ↓ | reference-relative undertriage | reference-relative overtriage | most-urgent case recall | median CPU decision latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wong (three actual clinician votes) | 102 | 18/102 = 17.65% | 0.592296 | 0.375817 vote-weighted | 0.447712 vote-weighted | **0/9** | 299.24 ms |
| Farrow (single reconstructed expert-panel consensus) | 30 | 7/30 = 23.33% | 1.232058 | 22/30 | 1/30 | **0/5** | 158.57 ms |

Model load time ~6.5 seconds in each new process (not counted in medians); runner is GitHub-hosted x86 CPU. Julia's results under this exact zero-shot question/rubric are poor relative to Jev, not evidence its model is generally ineffective across tasks. No clinical text or raw responses in run logs.

## Trained Wong text specialist (only train/cal labels, test once)

For same frozen 102 cases:
| Model | Brier to 3-vet distribution ↓ | exact median | vote undertriage | vote overtriage | majority-immediate recall |
|---|---:|---:|---:|---:|---:|
| Train-only vote prior (no state) | 0.340640 | 36/102 | 27.45% | 41.83% | 0/9 |
| **TF-IDF + logistic (171 cases; C selected on cal=67)** | **0.305694** | 42/102 | **40.52%** | 27.45% | **0/9** |
| Jev 1.14 Choice | 0.380465 | 45/102 | **9.80%** | 52.61% | **8/9** |
| Jev 1.14 Score | 0.359701 | 51/102 | 11.44% | 47.39% | **8/9** |

**Interpretation:** the fitted specialist is a *better probabilistic forecaster by overall Brier* but an unsafe fixed argmax action classifier for this clinical reference: it doesn't pick Immediate as its modal class for any majority-immediate case. This illustrates calibration and decision utility as distinct objectives, *not* a claim of prospective model danger or actual patient adverse outcomes. Comparing models fairly would need an independently calibrated, train/cal-selected decision policy and fixed threshold/abstention cost on untouched test.

## Source and independence caveats
- Wong original data: 340 past hospital visits, three independently rated vet urgency bins for each, 102 case-exclusive test cases, only 9 majority-immediate. No patient ID linking and no external site.
- Farrow: 30 clinic-derived owner-facing vignettes with consensus inferred from the authors' original `Same urgency assessment` scoring. The 14 independent vet vote vectors are not verified available. Five urgent within one hour. No prospective outcomes.
- Under/overtriage measure agreement with **human-vote references**, not validated treatment harm/cost. A constant prior with good Brier is not action-aware.
- The earlier Jev Score-vs-Choice rubric also changes instructions, so observed gain does not isolate the backend output primitive.
- Run **actual weights** or record the lane as NOT EXECUTED; no inferences from model cards or API wrappers alone count as scores.

## Next fixed-scope experiments
1. Finish Liquid actual Q8 CPU inference, compare on Wong/Farrow with source-only inputs and annotate source-native decision-head behavior.
2. Tune candidate action thresholds / abstentions **on calibration cases only**, measuring emergency recall against escalation burden, then test *once* on new untouched cases for model selection claims. The existing 102 Wong test cases have already been investigated and should be considered exploratory dev benchmark from here forward.
3. For stronger small-model supervised decision learning, gather more independently labeled original veterinarian questions; do not overtrain 171 cases.
4. Extend to **video-native Big Broiler** using upstream MP4 frames, camera/video grouping and F1/false episodes; do not conflate 21-feature tracker-geometry baselines with pixel/video understanding.
5. Stage our already-approved L1 sources HotPig, cattle wearable and Meerkat as separate task protocols rather than pretend all have L3 clinical consequence labels.

**No training source records, secret payloads, or veterinarian case descriptions are committed to GitHub.** 

## Calibration-only escalation policy (one new, frozen test pass)

A *post-model action policy* was tuned on **67 calibration cases only**: choose the largest threshold on P(Immediate) meeting at least **5 of 6** majority-immediate calibration cases while minimizing the number of immediate escalations. The selected threshold was **0.1223801002**, flagging 24 of 67 calibration cases for immediate priority.

On **102 test cases**, unmodified 171-trained TF-IDF logistic probabilities + this frozen threshold had:
- **9/9** majority-immediate cases assigned Immediate (versus 0/9 using ordinary argmax)
- **22/102** cases assigned Immediate; original median reference contains only 9/102 immediate
- **47/102** exact median matches (46.1%) vs 42/102 with argmax
- Vote-weighted undertriage **24.51%** vs 40.52% with argmax
- Vote-weighted overtriage **36.93%** vs 27.45% with argmax
- Raw predictive Brier remains **0.30569**; the policy does not alter the underlying probability distribution, just converts that distribution to the proposed decision.

This is *not* a safety proof. Only six calibration and nine test immediate cases, and the Wong test set had previously been inspected in earlier exploratory comparisons. Nor does the rule eliminate mistakes in non-immediate urgency categories. This is a demonstration of **separating probabilistic estimation from an asymmetric-loss action policy**, with the burden of extra escalation measured explicitly. Reproducible source-only archive delivered to user as `OpenFARM_Wong_Clinical_CPU_Baselines_v2.zip` (contains scripts and aggregate-only JSON; original source workbook omitted).
