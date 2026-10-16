# OpenFARM System-1 Research Charter (draft)

**Status:** working hypothesis / research direction, not a validated capability claim  
**Initiated:** 2026-10-15  
**Scope:** research-first animal-welfare decision models, with an eventual narrowly scoped monitoring product

## Mission

> Can we build a small, fast, calibrated system that continuously interprets animal observations, makes scientifically defensible decisions, and reliably identifies when human attention is warranted?

**Operational north star for the fresh data sweep (2026-10-15):**

> OpenFARM doesn't necessarily need to infer how an animal feels from arbitrary images. It can learn **what deserves attention, based on observable behavior and defensible evidence**.

This is **System-1 empathy as decision intelligence** across supported modalities—not emotion classification by conjecture. Keep three separately supervised levels: (1) direct observations of behavior and physiology; (2) independently validated welfare indicators, including graded and subjective expert disagreement; (3) escalation decisions supported by real consequences, uncertainty, lead time and false alerts per animal-day. Accept multi-label, Noul, Choice, ordinal Score, continuous change and abstention only when *independently supported by evidence*. Default product anchor: **cheap, good-enough zero-shot+ monitoring**, possibly using specialized low-cost modality adapters and compact temporal state.

This is both a **scientific** and **product** hypothesis. A win requires more than beating an autoregressive model on a benchmark. We seek decisions that are trainable with constrained compute, generalize outside the source experiment, retain useful calibration, and support a commercially viable workflow.

## New orientation

OpenFARM started as a predominantly generative/System-2 animal-welfare reasoning effort. The new priority is System-1: typed, probabilistic decisions about bounded states, with explicit scope and calibrated uncertainty. This **does not** mean all veterinary diagnosis or animal welfare is reducible to System-1 classification. Complex diagnoses, explanation, and difficult escalations may still require veterinarians, deterministic software, or System-2 agents. Successful System-1 infrastructure may eventually provide both funding and supervision for System-2.

**Assume legacy task definitions are wrong until re-audited.** Do not inherit old prompt templates, labels, splits, or dataset-card claims uncritically.

**2026-10-15 legacy audit closed:** See [final product-fit verdict](FINAL_PRODUCT_FIT_AUDIT.md). Horse grimace-region scoring is an exploratory welfare-proximal seed; CatMeows is an *optional* acoustic transfer diagnostic, not welfare. **No legacy source establishes a deployable animal-welfare monitor.** The next phase is prospecting better observed behavior/welfare data, using cheap pretrained/zero-shot+ baselines before specialist training.

## State → question → decision → action contract

Use TypeSafe's public System-One/Jev framing as an interface inspiration, not a claim that we have reproduced its RLCD training method or calibration. A shared state may support many independent questions, with outputs such as:

- **Noul**: probability of a specific yes/no proposition, e.g. “Is the observed gait asymmetric?”
- **Choice**: distribution over mutually exclusive classes, e.g. vocalization type.
- **Score**: distribution over ordered levels with a rubric, e.g. facial action-unit severity.
- **Abstention / quality gate**: explicit no-decision routing when the input is insufficient, out of distribution, or prediction risk exceeds tolerance.
- **Other research targets**: regression, multi-label, pairwise comparison, temporal change, and event detection, when data justify them. Do not flatten fundamentally different tasks into one-hot labels for convenience.

**A typed output is not proof of a true prediction.** Model confidence, distribution concentration, and empirical calibration are different quantities; measure calibration on data held out at the right group/domain unit. Do not equate “low confidence” with a diagnosis or “high confidence” with a guarantee.

Key primary docs: https://typesafe.ai/blog/introducing-system-one-models-and-jev ; https://docs.typesafe.ai/introduction ; https://docs.typesafe.ai/confidence ; https://docs.typesafe.ai/primitives/score

## Modality ladder and deployment constraints

Start with the cheapest *sufficient* observation, and increase complexity only where justified:

1. Metadata / deterministic signal processing / quality gate.
2. Single image, crop, short sound segment, spectrogram, or precomputed specialist embedding.
3. Short temporal windows, sparse video frames, pose/keypoint streams, event counts, or aggregated embeddings.
4. Long-horizon states assembled by cheap event detectors, streaming summary statistics, or memory.
5. Expensive specialized agents or System-2 review for difficult, atypical cases.

**Do not assume raw 10-minute/hour-long videos can be processed affordably.** A state spanning hours should generally be a compact, provenance-linked temporal representation, not the original footage.

## Commercial test is part of the research

Initial hypothesis: narrowly scoped animal monitoring / triage, not an omni-animal diagnostic model. Measure:

- useful event sensitivity / recall on truly held-out animals and sites;
- **false alerts per animal per day** (and per site/day), at specified recall or operating threshold;
- alert precision at natural prevalence; missed events and time-to-detection;
- probability calibration (Brier, proper scoring, reliability diagrams, classwise calibration);
- selective risk vs review/escalation coverage; shift and OOD behavior;
- latency, throughput, peak memory, joules/cost per monitored animal-day, deployment costs;
- expert review time and whether the alert leads to actionable intervention.

Require real-world labels and domain expert review before making medical or veterinary claims. Include legal/rights checks before commercializing any dataset or checkpoint.

## Research sequence (gated)

**Gate 0 — Forensic asset recovery.** Inspect the actual GitHub repo, HF datasets, source papers/archives, transforms and notebooks; identify untracked or lost artifacts. Record exactly what has been inspected. Check upstream terms, data integrity, labels, leakage and split boundaries. Fix hard evaluation/data bugs before model comparisons.

**Gate 1 — Task reconstruction.** For each source observation, inventory all *scientifically defensible* decision questions. Distinguish directly observed behavior from experimental context proxies, interpretations, and clinical decisions. Define null/unknown states and annotation disagreement where applicable.

**Gate 2 — Targeted fresh discovery.** Broad sweep of veterinary scoring, biomechanics, pose, acoustic ecology, ethology, physiological and commercial monitoring data. Prioritize accessible, rights-clear, quality supervision, subject/site identifiers, and real deployment relevance. Mine broadly, select narrowly.

**Gate 3 — Sanity baselines.** Once data pass the audit: majority/class-prior, metadata-only, simple acoustic/CV features, frozen specialist encoders, frozen general multimodal embeddings, generative models with constrained answers, and open decision-tuned models. Use exactly matched question semantics and audited splits.

**Gate 4 — Minimal training.** Probe heads first, then adapters or end-to-end if justified. Compare untuned backbone vs tuned decision head, generative vs non-generative cost-quality frontiers, calibration, and transfer.

**Gate 5 — Product viability.** Select one narrow use case and obtain out-of-source, prospective, representative examples. Measure false alerts/animal-day and operator utility before developing a monitoring MVP.

## Parallel, secondary literature survey

Survey *existing* evidence on synthetic animal vocalization, audio augmentation, behavior/movement synthesis and synthetic embeddings. Separate source-preserving augmentation from synthesized biological behavior. **Never evaluate on synthetic test data**, or allow generator training to see held-out subjects/sessions. Survey first; no commitment to build a foundation generator.

## Non-goals at launch

- Claiming general animal emotions or clinical pain from labels that only encode experimental conditions.
- Leaderboards on duplicated, balanced or label-leaking data presented as commercial performance.
- Full System-2 agents, live monitoring infrastructure, or lengthy video ingestion before a usable, narrow decision task exists.
- Presuming today's small omni models work out of distribution just because they support images/audio.

See [Initial asset audit](INITIAL_ASSET_AUDIT.md) for *preliminary, source-backed findings*, open questions and the next verification passes.
