# OpenFARM — Animal Welfare Research and Monitoring

OpenFARM is an open research program investigating **small, fast, typed, calibrated decisions from animal observations**, with a long-term goal of narrowly scoped welfare-monitoring systems that can genuinely help human caretakers.

> Can we build a small, fast, calibrated system that continuously interprets animal observations, makes scientifically defensible decisions, and reliably identifies when human attention is warranted?

## October 2026 System-1 reboot

**Legacy datasets were forensically audited, and most of the old tasks have been retired from the executable suite.** Do **not** use the old task lineup to make animal-pain, stress, valence, or commercially deployable monitoring claims.

- [**Final product-fit verdict**](docs/system1/FINAL_PRODUCT_FIT_AUDIT.md) — source-paper recheck, welfare relevance and initial-framework promotion rules.
- [Per-source repair/retire decisions](docs/system1/SUITE_DECISIONS.md) — task-by-task verdicts and evidence.
- [Evaluation readiness](docs/system1/EVALUATION_READINESS.md) — source integrity tests, independently confirmed leakage, media QC, and remaining scientific blockers.
- [System-1 research charter](docs/system1/RESEARCH_CHARTER.md) — mission, modality ladder, commercial metrics and research gates.
- [Typed question and calibration contract](docs/system1/TYPED_QUESTION_CONTRACT.md) — observation vs inference vs action, distributions and proper scoring rules.

### Canonical decision suite v0.1 — 3 bounded evaluations + 2 exploratory welfare probes

[**Read the v0.1 scientific and task specification**](docs/research/OPENFARM_DECISION_SUITE_V0_1.md) · [Machine-readable task manifest](docs/research/OPENFARM_DECISION_SUITE_V0_1.json)

The research-branch decision suite includes **three already runnable, separately evaluated Level-1 observation tasks** (**HotPig**, **24-cow wearable behavior**, and **20-video Meerkat geometry/ethogram**) plus **two deliberately nonranking Level-2 welfare-adjacent probes** (**Horse Grimace** and **EquiFACS independent veterinarian judgments**). The suite supports an omni-modal / interleaved **input-state → typed-decision** design, but these five sources do **not** establish every proposed modality; model inputs and actual label origins are always reported separately.

**No Level-3 caretaker escalation benchmark or certified welfare product result exists yet.** Never collapse the five items into a single welfare score. Meerkat's source is noncommercial; horse/EquiFACS are extremely small. The historical integrity decisions remain authoritative for retired source tasks.

### v0.2 query compiler — multiple evidence-backed questions from one state

[**Read the five-source scientific and implementation report**](docs/research/OPENFARM_QUERY_COMPILER_V0_2.md) · [Compiler source](scripts/openfarm_query_compiler_v02.py) · [Real-source query builders](scripts/build_openfarm_decision_queries_v02.py) · [CI checks](.github/workflows/openfarm-query-compiler-v02.yml)

The v0.2 compiler produces typed `Noul`, `Choice`, `Score` and experimental numeric `Value` questions from *one verified observation state*, including **genuine future-measured outcomes** and observation-specific **missing-ground-truth masks**. It keeps the model-visible state separate from original answers and group/timestamp metadata, rejects duplicate group membership across splits, and counts **distinct independently measured evidence families** instead of inflating source labels by arbitrary one-vs-rest rewrites.

**Real-source v0.2 proof of execution:** 130,005 observation states / 623,174 valid reference question labels compiled across **HotPig, human-labeled cattle motion-index, Meerkat, PigWeb and EquiFACS** (five sources); 546 missing evidence slots masked; 248 actual pre-physiology + human-observed behavior **multichannel states**. The global family mean is **~1.002 independent measured reference families/state**; within PigWeb it is **~1.95**. The extra labels are mostly dependent Choice→Noul and continuous/binary rewrites—not newly measured animal welfare. These numbers are source *rows/windows*, **not animals or independent welfare outcomes**.

**Task gates unchanged:** HotPig's future feeder-bout question and PigWeb's cortisol/oxytocin historical outcomes are *source-audited candidates*, not approved animal distress, valence or caretaker decisions. EquiFACS remains a tiny nonranking clinical-interpretation pilot; Meerkat retains strict noncommercial rights. **Still no independently certified Level-3 intervention/alert labels.**

### Legacy task disposition — one welfare seed retained, no validated product benchmark

**No existing dataset qualifies as a validated real-world animal-welfare monitoring task.** Passing data integrity is not equivalent to observing affect, welfare state or deployment utility.

| Tier | Task | Decision |
|---|---|---|
| **Welfare-proximal seed (not certified)** | `openfarm-horse-grimace / region_score` | Keep: ordinal facial-region grimace cue score 0–2, with source-verified dedup training; *not* clinical pain diagnosis |
| **Optional audio perception diagnostic only** | `openfarm-bioacoustics / catmeows_context` | Demote: owner-heldout repaired elicitation-context task is **not** an emotion/welfare label; runs only with explicit `allow_auxiliary_context_task=True` |

CatMeows has been removed from `configs/eval/` and its optional old-style eval is in `configs/probes/catmeows-context.toml`. It is **excluded from welfare scorecards and product go/no-go decisions**. All other old environments and tasks (including SoundWel) have been removed from active evaluation. Their implementations and preparation notebooks remain for archaeological reproducibility; retired environment loaders refuse execution, and old eval/RL TOMLs were removed. The original Hugging Face repositories have **not** been deleted or edited.

### Reproducible checks

```bash
# Offline fixture and retirement checks
python -m unittest discover -s tests -p 'test_audit_openfarm_integrity.py' -v
python scripts/smoke_env_construction.py   # after installing legacy env packages

# Published HF source integrity checks require datasets + network:
python scripts/audit_openfarm_integrity.py --dataset cat --output /tmp/cat-audit.json
python scripts/triage_decision_sources.py --dataset horse --output /tmp/horse-triage.json
```

For training preparation, **do not use the original horse train rows unmodified**: run `python scripts/prepare_horse_unique.py --manifest horse-v2.json --output-dir horse-v2`. This removes 116 exact duplicate rows from balanced train (945→829) and 1,035 from natural train (3,931→2,896) and preserves the horse-heldout test groups. For CatMeows, the **optional** environment filters all test owners from training (balanced 201→126; natural 322→212), which was independently checked against published data. It is not an active product benchmark.

CI checks pinned published revisions; **even a clean split/hash test is not proof of biological label validity or commercial fitness**.

## Commercial hypothesis

The near-term product hypothesis is **a good-enough, cheap, zero-shot+ animal well-being/behavior monitor** that makes typed, calibrated, actionable observation-level decisions—not a universal veterinary diagnosis model or an omni-animal conversational assistant. Evaluate low-cost frozen models and specialist perception against larger generative models before expensive training. Successful deployment requires event recall, **false alerts per animal per day**, latency/compute cost per monitored animal-day, calibration under shift, expert review and real-world utility. A later System-2 agent can support unusual cases and deeper expert reasoning.

OpenFARM's prior multimodal and generative experiments are preserved in the historical notebooks, code and commit history; they are not the default research direction for the current suite.
