# soundwel-pvwb

<a href="https://github.com/ob1-s/happy-farm/tree/main/environments/soundwel_pvwb">
<img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="Source Code">
</a>

### Overview

- **Environment ID**: `soundwel-pvwb`
- **Short description**: Porcine Vocal Welfare Benchmark (PVWB) — a multi-task audio and multimodal evaluation suite for pig vocalization understanding.
- **Tags**: `animal-welfare`, `audio`, `multimodal`, `eval`, `benchmark`, `livestock`

### Datasets

- **Primary dataset**: [`oliveirabruno01/soundwel-pig-vocalizations`](https://huggingface.co/datasets/oliveirabruno01/soundwel-pig-vocalizations)
- **Source links**: 
  - [Original Zenodo Dataset](https://zenodo.org/records/8252482) (Raw Audio, Spectrograms, and Labels)
  - [Paper: Classification of pig calls produced from birth to slaughter according to their emotional valence and context of production (Briefer et al., 2022)](https://www.nature.com/articles/s41598-022-07174-8)

### Task

- **Type**: Single-turn classification (binary).
- **Output format expectations**: Strict XML tags parsed via `vf.XMLParser`. 
  - By default (`require_reasoning=False`), it expects:
      ```xml
      <answer>[Prediction]</answer>
      ```
  - If `require_reasoning=True`, it expects:
      ```xml
      <reasoning>[Step-by-step analysis]</reasoning>
      <answer>[Prediction]</answer>
      ```
  - **Note**: use `require_reasoning=False` when evaluating or training a reasoning-enabled model.
 
### Quickstart

Run a quick smoke-test evaluation with default settings (audio modality, valence classification):

```bash
prime eval run soundwel-pvwb -m openai/gpt-4.1-mini -n 5
```

Run a benchmark sweep with specific environment arguments (e.g., evaluating both audio and vision simultaneously):

```bash
prime eval run soundwel-pvwb \
  -m openai/gpt-4.1-mini \
  -n 50 -r 3 \
  -a '{"task": "all", "modality": "multimodal"}' 
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | `str` | `list` | `"valence_classification"` | Which task(s) to run. Pass `"all"` for the full suite, or a list of task names (`call_type_classification`, `valence_classification`). |
| `modality` | `str` | `"audio"` | Data modality to present: `"audio"` (wav), `"vision"` (spectrogram), or `"multimodal"` (both). |
| `use_escape_hatch` | `bool` | `false` | If `true`, allows the model to output `UNINTELLIGIBLE` for missing or completely silent samples without penalty. |
| `max_examples_per_task`| `int` | `-1` | Limit on dataset size per task (use `-1` for all available rows). |
| `dataset_name` | `str` | `"oliveirabruno01/...`" | HuggingFace dataset ID or local path. |
| `train_split` | `str` | `"train"` | Split used to populate the RL training dataset. |
| `test_split` | `str` | `"test"` | Split used to populate the evaluation dataset. |
| `seed` | `int` | `42` | Random seed for dataset shuffling and balancing. |
| `target_pad_seconds` | `float` | `3.595` | Seconds to centrally zero-pad audio arrays (paper alignment). |
| `system_prompt_override`| `str` | `null` | Optional custom string to replace the default ethologist system prompt. |
| `user_prompt_override` | `str` | `null` | Optional custom template string for the user task prompt. |
| `require_reasoning` | `bool` | `false` | Whether to require a `<reasoning>` block before the `<answer>` block. |
| `format_reward_weight` | `float` | `0.0` | Weight for the format reward. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Weighted sum of accuracy and format rewards. |
| `accuracy_reward` | `1.0` if the parsed answer matches ground truth, `0.0` otherwise. |
| `format_reward` | `1.0` if the required XML schema was successfully generated, `0.0` otherwise. |
| `valid_attempts` | `1.0` if the model successfully provided a parsable, non-refusal prediction. |
| `clean_refusals` | `1.0` if the model explicitly used the escape hatch (`UNINTELLIGIBLE`). |
| `messy_refusals` | `1.0` if the model output conversational refusal keywords (e.g. "missing audio file"). |
| `format_failures` | `1.0` if the model failed to use the requested XML tags entirely. |