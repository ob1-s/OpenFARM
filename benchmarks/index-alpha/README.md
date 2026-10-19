# OpenFARM Index alpha — executable developmental index (2026-10-19)

**This is an active research branch, not a public blind leaderboard or a welfare-monitoring release.** It branches from the immutable v0.1 checkpoint rather than rewriting it.

## Questions and sources

See [tasks.json](tasks.json) for three source-authenticated original data protocols.

| Task | Independent reference | Allowed model-visible state | Evaluation grouping |
|---|---|---|---|
| HotPig | Electronic feeder readout | 12 upstream machine-camera counts | 16/4/4 completely distinct pigs, one experimental site |
| Cattle MI | Human video-coded activity | Proprietary wearable motion-index scalar | 16/4/4 cows + source site transfer |
| Brazilian dairy cows | Human-rated gait/BCS | Breed, lactation, days in milk | 16 separate held-out farms |

No raw source records, videos, sensitive data or pretrained model weights are committed; source checksums are mandatory. The third source is **context-to-clinical risk**, not direct welfare perception. Gait thresholds reuse one underlying expert score.

## Run original source-authenticated baseline tasks

```bash
python -m pip install pandas numpy scikit-learn requests scipy

python scripts/index_alpha_runner.py --task hotpig.feeder-current.noul
python scripts/index_alpha_runner.py --task cattle.mi-behavior.choice
python scripts/index_alpha_runner.py --task brazil.cow-locomotion-risk.noul
```

Each generates a source-validated `*.scorecard.json` and an original aggregate report under `reports/index-alpha`. These are separate scoring domains; never average their Brier scores into a universal animal intelligence score.

**Current provider blocker:** original cattle Figshare download `cattle_behavior_data_time_V2.csv` intermittently gives server HTTP 500 on ndownloader and Figshare's official alternate API routes. A SHA-256/MD5-verified local cache is supported in `.cache/cattle/`; the runner fails closed if source bytes are not independently authenticated. [Actual original source CI](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-alpha.yml).

## Run the primary Stage B model decision experiment

The original wearable cattle site currently blocks download, so the reproducible fallback uses **two separately authenticated biological-study sources**: HotPig current feeder intake and Brazilian dairy-cow contemporaneous human gait score ≥3.

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/index_alpha_stage_b_auth_sources.py \
  --out reports/index-alpha/stage-b-authenticated-two-source.json
```

Source and group conditions:

- HotPig original source SHA `1ba5125167224b969281509b88ef7b3cba121114aeb0532f3903894c796df7e8`: training pigs 1–16, calibration pigs 17–20, evaluation pigs 21–24; their time-correlated minute readings are never treated as separate animal groups.
- Bran 2019 clinical cow original `Cowlevel.csv` SHA `3b345fa0a0c2495aab40e313b5703c7e7c5d0688393a1759684b801179e6b529`: 16 distinct farms, deterministic `20261019`-seed permutation to 10 train / three calibration / three test farms; no class-outcome-informed split modification. **This entire original cohort was previously explored using 16-fold cross-validation**, so the fixed split is developmental.
- Train-only source-specific numerical encoding and feature scaling; no source identity or labels passed as model features. Each individual specialist and shared-trunk model sees identical task input channels and targets, with separate source adapters and output heads for the shared net.
- Models: train-only label prior, canonical deterministic rule where applicable, L2-regularized logistic, tiny boosted tree, two independent tiny nonlinear networks, and one network sharing an intermediate trunk across source tasks. All train/validation policies declared before inspecting this run's test scores.
- Neural checkpoint selection uses equally weighted prior-normalized calibration Brier across sources. No held-out group used to select hyperparameters. Never conflate the already-run vision/video multimodal pretrained Liquid results with matched *tabular feature* baselines; a **fair same-input pretrained general foundation model arm is not yet available**.
- Primary **PASS** gate: shared tiny model reduces heldout Brier by ≥2% versus best individual trained specialist on **each** source AND wins on ≥3/4 pigs and ≥2/3 farms. Otherwise **FAIL for this particular tiny shared-trunk pilot only**. This is not a transfer-learning or pretrained-model gate. Passing this gate would still only justify more research, **not** generalization, clinical safety, publication or deployment.

No inference-time clinical decision, caretaker action, farm-hour false-alert rate, or image/audio interpretation is established by this experiment. Statistical confidence and model-selection protocols need additional independent groups for external science.

## GitHub Actions

- [Index alpha source checks](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-alpha.yml)
- [Authenticated two-source Stage B training gate](https://github.com/ob1-s/OpenFARM/actions/workflows/openfarm-index-alpha-auth-models.yml)

The exploratory cross-source wearable MI neural run is retained under its own workflow as evidence of an unavailable original source dependency, not substituted or silently relabeled.

## Why this matters

The completed experiment tests **whether a tiny randomly initialized shared trunk trained jointly on two known numerical tasks satisfies one internal threshold**. It does **not** test pretrained model capability, LoRA fine-tuning, question-conditioned decisions, unseen-task transfer or scaling. The negative pilot does not establish a stop signal for those hypotheses. For an actual Stage B foundation-model test, compare the same pretrained backbone without versus with source-task adaptation on an unseen heldout task across zero-/few-shot target-label budgets, including specialist, frozen-feature and LoRA baselines with animal/site-disjoint evaluation. See [corrected Stage B decision](STAGE_B_DECISION_2026_10_19.md).
