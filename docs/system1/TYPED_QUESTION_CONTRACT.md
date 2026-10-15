# OpenFARM — Typed Questions, Evidence and Calibration Contract

**Draft / 2026-10-15**. This is an *evaluation design*, not a claim that legacy OpenFARM labels meet the standard.

## Reference: what TypeSafe actually means by a question

Source: https://docs.typesafe.ai/introduction  
Primitives: https://docs.typesafe.ai/primitives/noul , https://docs.typesafe.ai/primitives/choice , https://docs.typesafe.ai/primitives/score  
Confidence: https://docs.typesafe.ai/confidence  
System One overview: https://typesafe.ai/blog/introducing-system-one-models-and-jev

Their System-One request evaluates multiple atomic, independently specified questions in parallel against a shared state.

- **Noul:** P(yes | state, proposition); a binary proposition, **not** the amount/degree of a condition. No extra confidence field is needed.
- **Choice:** one selected option with a probability distribution over mutually exclusive options and derived confidence.
- **Score:** an ordered rubric, distribution over integer severity levels, derived score and order-aware confidence.
- **Confidence is not proof of calibration.** Their `confidence` for Choice/Score is a statistic of predicted distribution shape; scientifically defensible probabilities require tests against trustworthy labels from the target deployment distribution.
- **Abstention is a workflow decision**, e.g., check input sufficiency/OOD/uncertainty, then route or request better input. Never pretend the mere existence of `confidence` validates a decision.

Do not conflate TypeSafe's full commercially trained Jev stack or RLCD with any off-the-shelf Gemma/Qwen classifier. These are interface inspirations.

## Example proposed questions — subject to source-label audit

Given *one validated horse facial-region observation* with crop, time, region and quality:

| ID | Type | Exact semantics | Ground truth needed |
|---|---|---|---|
| `usable_region_crop` | Noul | Does this image provide adequate visibility of the labeled facial region for assessment? | Independent usable/unusable annotations |
| `grimace_region_score` | Score(0,1,2) | Strength of source-defined grimace-associated region cue, **not clinical pain** | Animal-independent rater score; ideally multiple raters |
| `region_identity` | Choice(ears, eyes, muzzle/nostrils, other) | Which anatomical region is shown? | Annotated region labels, if not already supplied to the model |
| `flag_for_review` | Downstream programmatic action; not initially a supervised label | Escalate under pre-defined risk and confidence thresholds | Action policy, veterinary risk tolerance and real event outcomes |

**Do not claim three learnable tasks simply because we wrote three questions.** The source may only annotate the ordinal score. Quality and region labels might need separate evidence; review decision requires downstream ground truth or explicit policy, not invented labels.

Given *one validated animal vocalization segment* and only when original annotations justify it:

| ID | Type | Exact semantics | Evidence requirement |
|---|---|---|---|
| `has_call` | Noul | Does this window contain a relevant animal vocalization? | Time-aligned call/no-call annotations, including negatives |
| `call_type` | Choice | Source-defined acoustic class, e.g. SoundWel high-vs-low-frequency call | Original validated acoustic classification |
| `valence` | Choice | Predict the original study's valence annotation, **not diagnose general distress** | Clear protocol; held-out individuals/sites and context controls |
| `abnormal_relative_to_baseline` | Noul | Did a pre-defined acoustic feature deviate from this animal's longitudinal baseline? | Repeated measurements per animal; change-definition protocol |

## Dataset-to-decision registry: required fields

For each source and question, record:

- **Raw provenance:** dataset ID, revision, source DOI/file, upstream license/usage limits, hash, annotator instructions, source animal/farm/session, collection design.
- **State:** precisely which information is exposed to the model, sample rate/frame rate, preprocessing, time span and resolution, exclusions, modality quality gates.
- **Question:** stable ID, primitive, instructions, exact criteria/rubric/option boundaries and semantic scope.
- **Target:** original vs derived label, independent evidence, inter-rater disagreement, unknown/missing policy, target prevalence.
- **Split:** subject/session/site-heldout regime, no-overlap proof, train/eval hash manifests.
- **Metrics:** precision/recall, Brier/log loss, reliability curves, classwise calibration, performance under domain shift, latency, and cost.
- **Use constraints:** which decisions may be automated, which require human/veterinary intervention, supported species/hardware/sites, abstention criteria.

## Calibrated outputs and epistemic honesty

For a Noul, `p=0.8` is *a prediction* of an event, not certification that 80% of all analogous cases are positive. Evaluate that correspondence on held-out annotated examples. A high entropy/low confidence distribution indicates predicted uncertainty, not necessarily OOD; a wrong but highly confident answer can still occur. Shift detection and abstention require explicit tests.

For Choice, collect all option probabilities and evaluate proper scoring rules, rather than only top-choice exact match. For Score, preserve ordinal structure and evaluate its calibration and ordinal error. For sparse or unobserved labels, distinguish genuine partial evidence and missing labels from artificial `unknown` categories.

**Do not post-hoc learn operating thresholds on the test set.** Fit calibration/thresholds on dedicated validation groups; report test estimates and confidence intervals by animal/site.

## A deployment-aware evaluation contract

For a monitoring alert model, static curated clips are insufficient to measure false alarms per animal-day. Obtain continuous or sampled representative streams with timestamps, verified event episodes, naturally occurring negatives, and target prevalence. Split by animal and collection site.

Report alert events (with suppression/deduplication windows), not just independent window classifications:

- **False alerts per animal per day** at one or more fixed event-recall operating points.
- Time-to-detection; event-level recall; alert precision.
- Review minutes per useful alert; abstention/escalation coverage.
- Model/perception cost per observed animal-day, including preprocessing and camera/mic duty cycle.
- Drift and performance on unseen animals, microphones, lighting and facilities.

For early-stage academic datasets lacking continuous streams, mark this as **not measurable**, rather than fabricating commercial-readiness metrics.

## Modality ladder / long states

Cheap source-level detectors (voice activity, animal detection, tracking), spectrograms, specialist embeddings, short-frame crops, and pose events feed a **bounded state**. State over an hour is a sparse sequence/aggregate with source timestamps and raw-media pointers. Long footage itself requires distinct streaming/summarizing infrastructure; do not hand a giant video to a 740M model and assume cheap reasoning.

Avoid claiming native audio beats spectrogram or vice versa until exactly matched sample preprocessing and cost measurement. Preserve the older OpenFARM spectrogram-mode implementation and reproduce the earlier observations.

## Gates

1. Restore/verifiably fix critical data/eval defects.
2. Verify real examples, ground-truth ontology and held-out groups.
3. Materialize only scientifically justified question contracts; reject dubious derivatives.
4. Evaluate trivial/simple controls and frozen-backbone probes.
5. Train the smallest viable heads/adapters, then compare to matched generative baselines.
6. Prospectively test one narrow use case before calling any checkpoint commercially viable.
