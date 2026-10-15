# OpenFARM — Animal Welfare Research and Monitoring

OpenFARM is an open research program investigating **small, fast, typed, calibrated decisions from animal observations**, with a long-term goal of narrowly scoped welfare-monitoring systems that can genuinely help human caretakers.

> Can we build a small, fast, calibrated system that continuously interprets animal observations, makes scientifically defensible decisions, and reliably identifies when human attention is warranted?

## October 2026 System-1 reboot

**Legacy datasets were forensically audited, and most of the old tasks have been retired from the executable suite.** Do **not** use the old task lineup to make animal-pain, stress, valence, or commercially deployable monitoring claims.

- [Per-source repair/retire decisions](docs/system1/SUITE_DECISIONS.md) — task-by-task verdicts and evidence.
- [Evaluation readiness](docs/system1/EVALUATION_READINESS.md) — source integrity tests, independently confirmed leakage, media QC, and remaining scientific blockers.
- [System-1 research charter](docs/system1/RESEARCH_CHARTER.md) — mission, modality ladder, commercial metrics and research gates.
- [Typed question and calibration contract](docs/system1/TYPED_QUESTION_CONTRACT.md) — observation vs inference vs action, distributions and proper scoring rules.

### Narrow research tasks retained (not certified)

| Task | Exact supervised target | What it **doesn't** establish |
|---|---|---|
| `openfarm-horse-grimace` / `region_score` | Ordinal facial-region grimace cue score 0–2 | Clinical pain, welfare diagnosis, unseen-farm generalization |
| `openfarm-bioacoustics` / `catmeows_context` | The recording's elicitation context | Animal distress, welfare valence or disease |
| `openfarm-bioacoustics` / `soundwel_call_type` | Source-defined acoustic HF/LF class (provisional, grouping audit incomplete) | Calibrated welfare detection or pig-level independent transfer |

Other old environments and tasks have been removed from active evaluation. Their implementations and preparation notebooks remain for archaeological reproducibility; retired environment loaders refuse execution, and old eval/RL TOMLs were removed. The original Hugging Face repositories have **not** been deleted or edited.

### Reproducible checks

```bash
# Offline fixture and retirement checks
python -m unittest discover -s tests -p 'test_audit_openfarm_integrity.py' -v
python scripts/smoke_env_construction.py   # after installing legacy env packages

# Published HF source integrity checks require datasets + network:
python scripts/audit_openfarm_integrity.py --dataset cat --output /tmp/cat-audit.json
python scripts/triage_decision_sources.py --dataset horse --output /tmp/horse-triage.json
```

CI checks pinned published revisions; **even a clean split/hash test is not proof of biological label validity or commercial fitness**.

## Commercial hypothesis

The near-term product hypothesis is **narrow, actionable animal monitoring**, not universal veterinary diagnosis or an omni-animal conversational assistant. Successful deployment requires event recall, **false alerts per animal per day**, latency/compute cost per monitored animal-day, calibration under shift, expert review and real-world utility. A later System-2 agent can support unusual cases and deeper expert reasoning.

OpenFARM's prior multimodal and generative experiments are preserved in the historical notebooks, code and commit history; they are not the default research direction for the current suite.
