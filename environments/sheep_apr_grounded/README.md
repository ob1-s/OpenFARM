# sheep-apr-grounded

<a href="https://github.com/ob1-s/happy-farm/tree/main/environments/sheep_apr_grounded">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `sheep-apr-grounded`
- **Short description**: Sheep Pain Recognition Benchmark.
- **Tags**: animal-behavior, animal-welfare, livestock, automated-pain-recognition, vision, multimodal, eval

### Datasets
- **Primary dataset(s)**: `oliveirabruno01/sheep-facial-expression-benchmark`
- **Source links**: https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark
- **Split sizes**: Evaluates on the `train` split by  (246 multimodal examples).

### Task
- **Type**: single-turn
- **Output format expectations**: XML tags. Depending on `require_explanation`, models must output just `<answer>...</answer>` or both `<explanation>...</explanation>` and `<answer>...</answer>`.
- **Rubric overview**: Evaluates exact-match classification accuracy (0 or 1), XML format adherence, and optionally uses an LLM-as-a-judge to score the model's reasoning against a veterinary clinical cheatsheet.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run sheep-apr-grounded
```

Configure model and sampling:

```bash
prime eval run sheep-apr-grounded \
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
| test_split | str | "train" | The test split to use. |
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
| `judge_reward` | (0.0 to 1.0) Reward for correct reasoning and recall via the clinical cheatsheet |