# OpenFARM data approval protocol — source to System-1 decision

**2026-10-15.** Approval attaches to a **specific source revision + observation state + labeled question + split + permitted use**. It does not attach to a paper's reputation or raw dataset title.

> OpenFARM can learn **what deserves attention, based on observable behavior and defensible evidence**.

## Evidence ladder

| Gate | What must be demonstrated | What it permits |
|---|---|---|
| DISCOVERED | Primary paper/DOI and candidate decision | Prospecting only |
| ACCESS_VERIFIED | Actual downloadable file, revision, parsed examples, recorded license | Source inspection; no score claims |
| TASK_VALIDATED | Source-grounded target and observation; annotation method; inspected examples; group IDs; noncircular task; media-label integrity | Candidate task design |
| APPROVED_RESEARCH_TRAIN | Lawful research use, manifest, disjoint training groups, defined target, clear missing/unknown policy | Training for that scope only |
| APPROVED_RESEARCH_EVAL | Independent group-heldout test, source/media duplicate tests, pinned annotations, clean evaluation/verifier, research evaluation and redistribution rights | Noncommercial benchmark for that scope only |
| APPROVED_COMMERCIAL_TRAIN | Proven commercial rights for media/annotations and contemplated ML use; validated construct; consent/privacy review as needed | Commercial training, not field validation |
| APPROVED_DEPLOYMENT_VALIDATION | Representative continuous streams, true event episodes and negatives, realistic prevalence, shift evaluation, alert costs and expert/caretaker review | Test a real welfare monitor |
| RETIRED | Unrepairable target validity/leakage/rights issue | No active benchmark |

**Training and evaluation are separately approved.** A high-quality supervised source may be valuable for research training but unsuitable as an independent test. A noncommercial benchmark is not automatically permissible in product model training or commercial model comparisons.

## Multi-modal System-1 decision contracts

Consider image/video, spectrogram/native audio, keypoints/pose, accelerometer, temperature, physiological measures, acoustic event counters, clinical tabular indicators, expert behavior logs, and compact longitudinal states. Do not require one giant multimodal encoder where specialized processing or low-cost sensor fusion suffices.

The task must specify: animal/pen/site/bout IDs; source revision/rights; state inputs and time window; question wording and output type (binary probability, mutually exclusive choice, ordinal score, continuous value, multi-label or abstain); exact annotation protocol (expert, multi-rater, instrumental, experimental proxy, model output); missingness and inter-rater disagreement; no shortcut predictor; train/calibration/test splits; observable uncertainty; candidate monitoring decision and unit cost.

The three supervision levels are separate:

1. **Observed behavior:** eating, grazing, grooming, social contact, biting, posture, gait events. Verifier: independent labels/time boundaries.
2. **Welfare interpretation:** a validated clinical scale, independently observed injury, pain-associated action-unit rubric, intervention and longitudinal animal-relative deviation. Verifier: expert/instrument data; do not infer welfare valence merely from behavioral context.
3. **Human attention decision:** escalation thresholds, prospective event risk, expected missed events, false alerts per animal or pen per day. Verifier: naturally sampled, temporally grounded event/negative timelines with expert-defined action policy. **No alert metric from balanced clips.**

## Highest priority micro-approval candidates

| Source | Potential question | Required independent reference | Specific blocker to investigate |
|---|---|---|---|
| Pig feeding, two farms | Will a damaging tail-biting episode begin in the *next seven days* given only the previous feeding history? | Real TBSTART dates in outbreak pens | Controls have **arbitrary pseudo-onset**; exclude PenType and TBSTART from state; farm and pen heldout, event count |
| Pig social observation+lesions | Was damaging social interaction observed in this session? Is a later lesion grade elevated? (two independent tasks) | Manually coded events and lesion examinations | Check timestamps, pig/session linking, repeated samples, treatment effects |
| HotPig | Has a pig's feeding/lying time changed relative to its own baseline? | Manually labeled observations and independently recorded feeder outcomes | Minute-level posture/activity series were machine-generated, not all expert GT |
| EquiFACS | Is a particular facial action active in a given horse/interval? | Certified coder onset/offset annotations; independent clinical ratings for a separate task | Some clinical videos unavailable; six experimental horses; no clip leakage; single AU coder |
| Cow gait ten-rater scores | What is the human-rater Score(0–3) distribution on a short locomotion video? | Ten distinct locomotion scores per paired recording | Download rights and horse/cow identities; animal/site holdout |
| Cow grazing/rumination | Is a true chewing/rumination bout present in this interval? | Expert time-coded acoustic reference | Sensor drift, animal-group holdout, negative-window coverage |
| GrimACE | How severe is an independently scored facial region cue? | Mouse/rater/recording IDs; expert ratings | 31 mice, image quality, intervention-vs-observation confusion |

## Noncommercial open benchmark rules

CC BY-NC or CC BY-NC-SA may permit **genuinely noncommercial** research benchmarking, including creating and publishing adaptations with the required attribution, noncommercial conditions and ShareAlike terms where applicable. They do **not** grant all uses merely because a benchmark is open or nonprofit-hosted. NC turns on the actual use and benefit. In particular:
- Keep research training, independent evaluation, and possible future commercial licensing as separate permissions.
- A public code/verifier repository can link to a source without redistributing the source's video/audio, images or annotations.
- Where republishing annotations, transformed clips or screenshots, verify rights to **every underlying source**; a paper license need not cover all incorporated media.
- No-derivatives sources require special care: do not publicly redistribute adapted protected material without separate rights.
- Do not expose entire heldout labeled cases in publicly indexed benchmark prompts when their integrity depends on remaining unseen.
- Commercial model assessment in direct support of a paid product may raise NC questions. Obtain permission rather than asserting certainty.

## Automatic rejection / fail-closed checks

Stop an approval when a dataset lacks stable subject/pen/session metadata needed for its claimed generalization, duplicates source clips across evaluation groups, contains contradictory labels without a disagreement protocol, predicts the intended outcome directly from target-encoding metadata, lacks non-event examples required by its proposed detector, uses pseudo-labels as expert GT, or has unresolved redistribution/training rights for the intended use.

Before deployment: compare prior/rules/trees, frozen/specialist cheap encoders, generative zero-shot+ baselines, trained typed probabilistic heads; report proper scores, classwise calibration, selective risk, out-of-domain/animal/site shifts, event-level recall and **false alerts per animal-day or pen-day**, with genuinely representative observation streams.

No named source is approved by merely appearing in the discovery register. Approval changes only after concrete source-level verification and evidence records.
