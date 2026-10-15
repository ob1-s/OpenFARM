# openfarm-sheep-apr

> **RETIRED (2026-10-15):** Source split views overlap, matched partitions have high-similarity near-duplicate image/label conflicts, and upstream commercial photo rights need verification. The old forensic override no longer enables evaluation; the loader raises a retirement error.

See [repair-or-retire decisions](../../docs/system1/SUITE_DECISIONS.md).

<p>
  <a href="https://github.com/ob1-s/happy-farm/tree/main/environments/openfarm_sheep_apr">
    <img align="left" hspace="4" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://app.primeintellect.ai/dashboard/environments/openfarm/openfarm-sheep-apr">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717&logoColor=white" alt="Prime Intellect Environments Hub">
  </a>
  <a href="https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717" alt="Hugging Face Dataset">
  </a>
</p>
<br clear="all" />

> **Research integrity notice (2026-10-15):** Sheep APR evaluation is **quarantined by default**. Balanced/raw splits were created independently and are contaminated if mixed (32 and 60 identical images), and perceptual near-duplicate candidates with conflicting labels exist even across the matched balanced and raw partitions. `allow_uncertified_sheep_eval=True` is a forensic-only override, not validation. See [System-1 evaluation readiness](../../docs/system1/EVALUATION_READINESS.md).

### Overview
- **Environment ID**: `openfarm-sheep-apr`
- **Short description**: Sheep facial pain-expression benchmark using SPFES-style proxy labels.
- **Tags**: animal-behavior, animal-welfare, livestock, automated-pain-recognition, facial-expression, pain-proxy-labels, spfes, vision, multimodal, eval

### Datasets
- **Primary dataset(s)**: `oliveirabruno01/sheep-facial-expression-benchmark`
- **Source links**:
  - https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark
  - Original paper: Noor et al. (2020), "Automated sheep facial expression classification using deep transfer learning" ([`10.1016/j.compag.2020.105528`](https://doi.org/10.1016/j.compag.2020.105528))
  - Original data: Mendeley "Sheep Facial Expression Primary Data-set" ([`10.17632/y5sm4smnfr.5`](https://doi.org/10.17632/y5sm4smnfr.5)), CC-BY-4.0.
- **Split sizes**: `train` 172 / `test` 74 balanced examples; `train_raw` 898 / `test_raw` 225 natural-distribution examples.

### Source Caveat

This environment evaluates image-only facial-expression classification. The source paper frames the task as normal/no-pain vs abnormal/pain sheep face classification under Sheep Pain Facial Expression Scale (SPFES) rules, but the original images were gathered from web sources and labeled from visible facial cues. Treat labels as SPFES-style pain-expression proxy labels rather than full clinical pain diagnoses.

### Task
- **Type**: single-turn
- **Target labels**: `0` = normal/no-pain facial expression, `1` = pain-associated/abnormal facial expression.
- **Output format expectations**: XML tags. Depending on `require_explanation`, models must output just `<answer>...</answer>` or both `<explanation>...</explanation>` and `<answer>...</answer>`.
- **Rubric overview**: Evaluates exact-match classification accuracy (0 or 1), XML format adherence, and optionally uses an LLM-as-a-judge to score whether the model's reasoning is biologically consistent with ovine facial pain-expression scales without overclaiming a clinical diagnosis.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run openfarm-sheep-apr
```

Configure model and sampling:

```bash
prime eval run openfarm-sheep-apr \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"require_explanation": true, "judge_mode": "external"}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- `judge_mode` can be one of these: `external` (requires an external model like GPT-4o), `self` (the model itself acts as the judge), or `none` (no judge is used).

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| dataset_id | str | "oliveirabruno01/sheep-facial-expression-benchmark" | The ID of the dataset to use. |
| train_split | str | "train" | The train split to use for training/rollout datasets. |
| test_split | str | "test" | The test split to use. |
| max_examples | int | -1 | The maximum number of examples to load from the dataset. |
| seed | int | 42 | The seed to use for dataset shuffling. |
| judge_mode | str | "external" | The judging strategy: "external", "self", or "none". |
| judge_model | str | "gpt-4o-mini" | The external judge model. |
| judge_base_url | str | "https://api.openai.com/v1" | The base url for the judge OpenAI client. |
| judge_api_key_var | str | "OPENAI_API_KEY" | The environment variable containing the judge's API key. |
| require_explanation | bool | True | Whether to require step-by-step reasoning in the output. |
| accuracy_reward_weight | float | 1.0 | The reward weight for matching the exact label. |
| judge_reward_weight | float | 1.0 | The reward weight for the grounded judge analysis. |
| format_reward_weight | float | 0.0 | The reward weight for correct XML formatting. |
| cheatsheet_version | str | "short" | The version of the cheatsheet to use. Can be "short" or "full". |
| cheatsheet_override | str | None | An override for the cheatsheet. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy_reward` | (0.0 or 1.0) Exact match on target answer (0 or 1) |
| `format_reward` | (0.0 or 1.0) Reward for correct XML formatting |
| `judge_reward` | (0.0 to 1.0) Reward for biologically sound reasoning against the ovine facial-expression cheatsheet |
