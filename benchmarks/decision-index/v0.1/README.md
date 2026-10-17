# OpenFARM Decision Index v0.1

**Status:** versioned internal research evaluation suite, **not** a public blinded leaderboard or approved veterinary application.

The index collects **eight original-source animal decisions across six species**, grouped as independent task complexities:
- **L1 observation:** HotPig feeding, cattle wearable behavior, meerkat ethogram, Big Broiler feather pecking.
- **L2 interpretation:** GrimACE mouse facial action units.
- **L2–L3 longitudinal risk:** Aarhus first new pig tail injury within 7 days.
- **L3 attention/triage:** Wong hospital urgency (3 veterinary raters) and Farrow owner seek-care (one published vet-panel consensus).

These are not sequential mandatory model stages. The goal is a common experimental contract over independent decisions; welfare interventions, prospective benefit, cross-farm generalization, and clinical care safety are **not yet measured**.

## What is concretely implemented here

- [tasks.json](tasks.json) — machine-readable source, species, allowed input modality, primitive, labels, relevant grouping unit, source license, approval status, split limitations and predeclared metrics.
- [observed-results.json](observed-results.json) — 15 actually completed, source-grounded model metrics on **Wong and Farrow only**. This is *not* a fair aggregate model ranking across species.
- [Index contract validator](../../../scripts/index_v01_validate.py) and [result registry validator](../../../scripts/index_v01_validate_results.py).
- [Generic typed scorer](../../../scripts/index_v01_score.py): **Noul** binary Brier, **Choice/Score** proper multiclass Brier against empirical original expert vote distributions, masking of unscorable and abstained predictions, optionally priority under/overtriage, majority-Immediate recall, and group-bootstrap Brier interval. It refuses raw clinical states in the score file.
- [CI scientific checks](../../../.github/workflows/openfarm-decision-index-v01.yml): source/split/metric schema, result denominators, synthetic label regression tests, private Wong test expert-disagreement audit.
- [Full original Wong expert reference analysis](../../../docs/research/decision-index-v01/2026-10-17-wong-human-disagreement.md).
- [Jev and Liquid Farrow forensic probes](../../../scripts/index_v01_farrow_liquid_forensics.py), using real original source data with private TypeSafe API auth or native GGUF CPU inference.
- [Big Broiler actual original MP4 verification](../../../scripts/index_v01_big_broiler_video_probe.py) plus a tightly bounded [real multimodal source pilot](../../../scripts/index_v01_big_broiler_multiframe_pilot.py).

Source task builders and notebooks remain on **independent unmerged research branches** for #4 Aarhus, #5 calf, #6 GrimACE, #7 Big Broiler, #8 Wong/Farrow. The manifest references source branches; it deliberately does not claim to integrate their datasets or silently approve them.

## Running the standardized score verifier

A JSONL file of **predictions and references only** (no patient descriptions/images) with one record per (task_id, unit_id). Each record includes a scientifically meaningful `group_id`, `kind` (`noul`, `choice` or `score`), `prediction`, `reference`, and source-specific option order.

Example clinical record with synthetic values:

```json
{"task_id":"wong_er_priority_v01","unit_id":"synthetic-01","group_id":"clinical-case-synthetic-01","kind":"score","options":["Immediate","15 mins","30-60 mins","120 mins","240 mins"],"priority_order":"most_urgent_first","reference":{"votes":["Immediate","Immediate","15 mins"]},"prediction":{"probabilities":{"Immediate":0.7,"15 mins":0.2,"30-60 mins":0.05,"120 mins":0.03,"240 mins":0.02}}}
```

Then:

```bash
python scripts/index_v01_score.py --input local_predictions.jsonl --output local_scores.json --bootstrap 1000 --model-id model-checkpoint@sha256 --protocol-id wong-v01-canonical
```

**Critical detail:** for Wong, the probability target is the **three-vet empirical distribution**, not a fabricated absolute clinical truth. If model probabilities `p` and three-rater relative frequencies `q` are defined on five classes, report `Σ(p−q)²`; `1−Σq²` is the additive variability term in expected single-vet one-hot Brier. That variance is measurable, not necessarily irreducible across new clinicians.

For physical animal data, `group_id` must be the confirmed animal/site/cohort/whole-video unit, **not one independent 1-second frame or overlapping 10-second window**. Absent labels or video corruption must be `status=unscorable`; explicit model nonanswers must be `status=abstain`, never negative examples.

## Live model results are *developmental*

Reusing Wong's 102 test cases and Farrow's 30 cases for many prompt/rubric comparisons has turned them into **exploratory benchmark sets**. They are useful for debugging but not an untouched final evaluation. The correct remedy is genuinely **new, external expert-rated cases/sites/animals**, not only a shuffled partition of the same 340 cases.

Do not select models using cross-task average Brier: the class count, gold reference and loss interpretation differ between Wong/Farrow/animal observation tasks.

## Running the task specific original-data CI workflows

- [Index validation Actions](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-decision-index-v01.yml)
- [Farrow model forensics Actions](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-v01-farrow-forensics.yml)
- [Original broiler video integrity Actions](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-v01-broiler-video-probe.yml)
- [Broiler sampled multi-frame Liquid pilot](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-v01-broiler-multiframe.yml)

**Security:** Wong's original patient descriptions are not committed. A GitHub Actions repository secret holds only the 102 case test states for on-run inference. Clinical-source licenses and research permissions must be checked per task before redistribution or commercial use.
