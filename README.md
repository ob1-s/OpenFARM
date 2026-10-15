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

### Initial framework: one seed, zero validated product benchmarks

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
