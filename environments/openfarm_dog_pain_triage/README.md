# openfarm-dog-pain-triage

<p>
  <a href="https://github.com/ob1-s/happy-farm/tree/main/environments/openfarm_dog_pain_triage">
    <img align="left" hspace="4" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://app.primeintellect.ai/dashboard/environments/openfarm/openfarm-dog-pain-triage">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717&logoColor=white" alt="Prime Intellect Environments Hub">
  </a>
  <a href="https://huggingface.co/datasets/oliveirabruno01/openfarm-dog-pain">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717" alt="Hugging Face Dataset">
  </a>
</p>
<br clear="all" />

### Overview
- **Environment ID**: `openfarm-dog-pain-triage`
- **Short description**: A canine clinical text triage environment evaluating whether models can accurately deduce OpenFARM/AWW affective pain states from leakage-safe tabular veterinary metadata.
- **Tags**: animal-welfare, veterinary, dog, pain, clinical-text, openfarm, llm-judge, single-turn, eval, train

### Datasets
- **Primary dataset(s)**: `oliveirabruno01/openfarm-dog-pain`
- **Source links**:
  - [Hugging Face Datasets](https://huggingface.co/datasets/oliveirabruno01/openfarm-dog-pain)
  - [Zenodo Record 15303646](https://zenodo.org/records/15303646)
- **Split sizes**:
  - `train` (68 rows) / `test` (18 rows): Strictly class-balanced, dog-heldout splits to prevent majority-class memorization during RL training.
  - `train_raw` (441 rows) / `test_raw` (116 rows): The natural, highly imbalanced dog-heldout distribution.

*Note: Use `test` for baseline reasoning checks. Use `test_raw` with Macro-F1 scoring to evaluate real-world prevalence handling.*

### Task
- **Type**: single-turn
- **Target labels**: `Negative_Nociceptive` vs `Neutral_Resting`.
- **Output format expectations**: XML tags. Depending on `require_explanation`, models must output just `<answer>...</answer>` or both `<explanation>...</explanation>` and `<answer>...</answer>`.
- **Rubric overview**:
  - Exact-match classification accuracy (0.0 or 1.0).
  - XML format adherence.
  - Optional `AdaptiveJudgeRubric` to score the medical soundness of the model's `<explanation>`.

### Quickstart
Run an evaluation with default settings (uses the balanced `test` split):

```bash
prime eval run openfarm-dog-pain-triage
```

Evaluate on the natural, imbalanced raw distribution to check for mode collapse:

```bash
prime eval run openfarm-dog-pain-triage \
  -m openai/gpt-4.1-mini \
  -n 100 -r 1 \
  -a '{"test_split": "test_raw", "require_explanation": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- The public dataset rows may include rich clinical/source metadata for audits and ablations. The env prompt intentionally uses only a selected leakage-safe subset by default.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_id` | str | `"oliveirabruno01/openfarm-dog-pain"` | HuggingFace dataset ID |
| `dataset_revision` | Optional[str] | `None` | HuggingFace dataset revision |
| `train_split` | str | `"train"` | Split used for training dataset (`train` or `train_raw`) |
| `test_split` | str | `"test"` | Split used for evaluation dataset (`test` or `test_raw`) |
| `max_examples` | int | `-1` | Limit on dataset size (use -1 for all) |
| `seed` | int | `42` | Random seed for dataset shuffling |
| `include_extended_surgery_fields` | bool | `False` | Include `time_since_surgery` and `recovered_anesthesia` if present. Disabled by default because missingness is label-correlated. |
| `require_explanation` | bool | `True` | If True, forces `<explanation>` tags for Chain-of-Thought reasoning. |
| `accuracy_reward_weight` | float | `1.0` | Weight for the exact-match category classification reward. |
| `judge_reward_weight` | float | `1.0` | Weight for the LLM-as-a-judge medical reasoning reward. |
| `format_reward_weight` | float | `0.0` | Weight for XML format adherence. |
| `judge_mode` | str | `"none"` | Judge strategy: `"external"`, `"self"`, or `"none"`. |
| `judge_model` | str | `"gpt-4o-mini"` | Model used for the LLM judge. |
| `judge_api_key_var` | str | `"PRIME_API_KEY"` | Environment variable containing the judge API key. |
| `judge_base_url` | str | `"https://api.pinference.ai/api/v1"` | Base URL for the judge model provider. |
| `system_prompt_override` | str | `None` | Optional custom system prompt string. |
| `user_prompt_override` | str | `None` | Optional custom user prompt template string. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of accuracy, judge, and format rewards) |
| `accuracy_reward` | (0.0 or 1.0) Exact match on the correct Target Category |
| `format_reward` | (0.0 or 1.0) Adherence to the requested XML tag format |
| `hybrid_judge_reward` | (0.0 to 1.0) LLM judge score evaluating the biological soundness of the reasoning |
