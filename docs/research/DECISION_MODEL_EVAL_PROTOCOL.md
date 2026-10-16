# OpenFARM — Decision model evaluation gate (Jev and open models)

**2026-10-16. This is a future evaluation contract, not benchmark scores.**

## Sequence

1. Finalize an immutable **scientific decision index**. Every question has a source revision, independently recorded reference, animal/site/episode group, observation-end timestamp, evidence-quality mask, permitted input representation, exact output primitive and predetermined metric.
2. Freeze train, development/calibration and source/animal/site-heldout test manifests **before any model sees their references**. Model selection, prompt wording, thresholds and calibration may use only development data. Do not leak outcomes via state metadata.
3. Establish CPU-cheap rules, logistic/boosted trees, frozen encoders and modality-specialist baselines on the *same permitted state*, and a strong generative reference. Mark missing modality support as **not supported**, not zero accuracy.
4. Only then use the repo-level `TYPESAFE_API_KEY` to evaluate Jev via the **official** `https://api.typesafe.ai/v1/systemone` endpoint. Secrets must exist only in GitHub Actions environment, never logs/artifacts, never transmitted to third-party Jev proxy sites.
5. Make an **opt-in**, manually triggered CI workflow. First stage is a 1–3 state smoke test and strict call/input-token cap; a larger run requires an explicitly set budget. Record model version, API shape, usage, timestamps, questions, preprocessing/serialization and source hashes. Never run automatic paid inference on every push.
6. For open Jev-inspired replications, check released weights, architecture, available modalities, actual licenses, provenance of training data and a working inference script. **One independent reproduction does not imply dozens of well-validated alternatives.** If comparing text-only or structured-state models to vision models, stratify results by modality; no silent use of separately labeled perception features.
7. Report per-task and per-evidence-level: Brier/log loss, ordinal proper scores, group-bootstrap uncertainty, base rates, risk-coverage/abstention, label-quality masks and long-horizon false alerts/animal-day **only** where uninterrupted independently labeled monitoring data exist.
8. Where model native output doesn't express `Point`/`Box` or native image/audio, don't invent comparable predictions. Typed binary/choice/score are still valuable for numeric sensor and structured ethogram tasks.

## Jev compatibility snapshot

Checked 2026-10-16 against [official TypeSafe API reference](https://docs.typesafe.ai/api): `POST /v1/systemone`, a single `state` and named map of typed questions `noul`, `choice`, `score`, with corresponding typed probabilities and usage. **Do not assume** a new vision/box release follows the older model's request schema; independently confirm the official schema and required model version before submitting any image. Third-party Jev proxy sites are not the official service and MUST NOT receive our TypeSafe key.

## Evaluation hygiene

- Questions sharing the same underlying physical observation must be reported both per-question and per-observation; their labels are correlated, not independent scientific discoveries.
- Source-derived decisions and their deterministic transformations are grouped as one label family. Clearly separate human-observed behavior, externally measured clinical outcome and synthetic temporal target.
- Report unanswered/uncodeable intervals as *masked*, not negative. Preserve expert disagreement and rater-dependence exclusion.
- Prioritize prospecting and admission of meaningful welfare decisions. Fast inference on poor labels isn't a product win.

**Status:** protocol prepared; **no TypeSafe API request made and no billable evaluation authorized**. The secret is user-supplied repository configuration, not something this document verifies is present at runtime.
