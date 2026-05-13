# aww-bench

<a href="https://github.com/ob1-s/happy-farm/tree/main/environments/aww_bench">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview
- **Environment ID**: `aww-bench`
- **Short description**: Animal Welfare Watcher: Single-turn evaluation environment testing a model's ability to classify animal welfare states while reasoning through biological confounders and adversarial distractors.
- **Tags**: animal-welfare, classification, llm-judge, self-judge, single-turn, reasoning, eval, train

### Datasets
- **Primary dataset(s)**: `oliveirabruno01/AWW-bench` (Dataset containing visual cue descriptions, context metadata, and adversarial biological confounders)
- **Source links**: [Hugging Face Datasets](https://huggingface.co/datasets/oliveirabruno01/AWW-bench)
- **Split sizes**: 232 `train` tasks and 40 `test` tasks.

### Task
- **Type**: single-turn
- **Output format expectations**: XML tags. Depending on `require_reasoning`, models must output just `<answer>...</answer>` or both `<reasoning>...</reasoning>` and `<answer>...</answer>`.
- **Rubric overview**: `JudgeRubric` that evaluates exact-match classification accuracy, XML format adherence, and an LLM-as-a-judge score assessing the model's reasoning against adversarial distractors.

### Quickstart
Run an evaluation with default settings:

```bash
prime eval run aww-bench
```

Configure model and sampling:

```bash
prime eval run aww-bench \
  -m openai/gpt-4.1-mini \
  -n 20 -r 3 -t 1024 -T 0.7 \
  -a '{"require_reasoning": true, "include_adversarial_context": false}'  # env-specific args as JSON
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- You can save detailed rollout predictions and correctness flags by appending `-s -C "aww_pred,aww_correct,aww_outcome"` to your CLI command.
- `judge_mode` can be one of these: `external` (requires an external model like GPT-4), `self` (the model itself acts as the judge), or `none` (no judge is used).

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_id` | str | `"oliveirabruno01/AWW-bench"` | HuggingFace dataset ID |
| `train_split` | str | `"train"` | Split used for training dataset |
| `test_split` | str | `"test"` | Split used for evaluation dataset |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |
| `seed` | int | `42` | Random seed for dataset shuffling |
| `include_adversarial_context` | bool | `True` | Whether to inject adversarial distractor/confounder data into the prompt |
| `require_reasoning` | bool | `False` | If True, forces `<reasoning>` tags. Set to `False` for native reasoning models (e.g., DeepSeek-R1) |
| `accuracy_reward_weight` | float | `1.0` | Weight for the exact-match category classification reward |
| `judge_reward_weight` | float | `1.0` | Weight for the LLM-as-a-judge reasoning evaluation reward |
| `format_reward_weight` | float | `0.0` | Weight for XML format adherence |
| `judge_mode` | str | `"external"` | Judge strategy for soft reward. |
| `judge_model` | str | `"gpt-4.1-mini"` | OpenAI-compatible model used for the LLM judge |
| `judge_api_key_var` | str | `"OPENAI_API_KEY"` | Environment variable containing the judge API key |
| `judge_base_url` | str | `"https://api.openai.com/v1"` | Base URL for the judge model provider |
| `system_prompt_override` | str | `None` | Optional custom system prompt string |
| `user_prompt_override` | str | `None` | Optional custom user prompt template string |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of accuracy, judge, and format rewards) |
| `accuracy_reward` | (0.0 or 1.0) Exact or substring match on the correct Target Category |
| `judge_reward_func` | (0.0 to 1.0) LLM judge score evaluating how well the model reasoned through the adversarial confounder |
| `format_reward` | (0.0 to 1.0) Adherence to the requested XML tag format |
| `valid_attempts` | (0.0 or 1.0) 1.0 if the model successfully outputted a parseable answer |
| `format_failures` | (0.0 or 1.0) 1.0 if the model failed to output parseable XML tags |