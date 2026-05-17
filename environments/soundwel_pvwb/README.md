# soundwel-pvwb

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ob1-s/happy-farm/tree/main/environments/soundwel_pvwb)&nbsp;&nbsp;[![Prime Intellect](https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717)](https://app.primeintellect.ai/dashboard/environments/hfarm/soundwel-pvwb)&nbsp;&nbsp;[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717)](https://huggingface.co/datasets/oliveirabruno01/soundwel-pig-vocalizations)

### Overview

- **Environment ID**: `soundwel-pvwb`
- **Short description**: Porcine Vocal Welfare Benchmark (PVWB) — a multi-task audio and multimodal evaluation suite for pig vocalization understanding.
- **Tags**: `animal-behavior`, `animal-welfare`, `automated-pain-recognition`, `livestock`, `audio`, `multimodal`, `eval`, `train`

### Datasets

- **Primary dataset**:[`oliveirabruno01/soundwel-pig-vocalizations`](https://huggingface.co/datasets/oliveirabruno01/soundwel-pig-vocalizations)
- **Source links**: 
  - [Original Zenodo Dataset](https://zenodo.org/records/8252482) (Raw Audio, Spectrograms, and Labels)
  -[Paper: Classification of pig calls produced from birth to slaughter according to their emotional valence and context of production (Briefer et al., 2022)](https://www.nature.com/articles/s41598-022-07174-8)

### Task

- **Type**: Single-turn classification (binary & multi-class).
- **Available Tasks**: `call_type_classification`, `valence_classification`, `context_classification`, `age_classification`.
- **Output format expectations**: Strict XML tags parsed via `vf.XMLParser`. 
  - By default (`require_explanation=False`), it expects:
      ```xml
      <answer>[Prediction]</answer>
      ```
  - If `require_explanation=True`, it expects:
      ```xml
      <explanation>[Step-by-step analysis]</explanation>
      <answer>[Prediction]</answer>
      ```
 
### Quickstart

Run a quick smoke-test evaluation with default settings (audio modality, valence classification):

```bash
prime eval run soundwel-pvwb -m openai/gpt-4.1-mini -n 5
```

Run a benchmark sweep with specific environment arguments (e.g., evaluating both audio and vision cross-modality across all tasks):

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
| `task` | `str` \| `list` | `"valence_classification"` | Which task(s) to run. Pass `"all"` for the full suite, or a list of task names (`call_type_classification`, `valence_classification`, `context_classification`, `age_classification`). |
| `modality` | `str` | `"audio"` | Data modality to present: `"audio"` (wav), `"vision"` (spectrogram), `"multimodal"` (both), or `"text"` (tabular only). |
| `include_tabular_data` | `bool` | `false` | If `true`, injects programmatic acoustic and demographic features as a markdown list in the prompt. (Forced `true` if `modality="text"`). |
| `use_escape_hatch` | `bool` | `false` | If `true`, adds `UNINTELLIGIBLE` to the prompt options for missing or completely silent samples. Normal labeled samples still score against the gold label. |
| `max_examples_per_task`| `int` | `-1` | Limit on dataset size per task (use `-1` for all available rows). |
| `dataset_name` | `str` | `"oliveirabruno01/...`" | HuggingFace dataset ID or local path. |
| `train_split` | `str` | `"train"` | Split used to populate the RL training dataset. |
| `test_split` | `str` | `"test"` | Split used to populate the evaluation dataset. |
| `seed` | `int` | `42` | Random seed for dataset shuffling and balancing. |
| `target_pad_seconds` | `float` | `3.592` | Seconds to centrally zero-pad audio arrays. Set to `0.0` or `-1.0` to disable padding. |
| `system_prompt_override`| `str` | `null` | Optional custom string to replace the default ethologist system prompt. |
| `user_prompt_override` | `str` | `null` | Optional custom template string for the user task prompt. |
| `require_explanation` | `bool` | `false` | Whether to require a `<explanation>` block before the `<answer>` block. |
| `format_reward_weight` | `float` | `0.0` | Weight for the format reward. |
| `balancing_strategy` | `str` | `"proportional"` | How to balance classes. `"proportional"` to class balance, `"balanced"` to force equal splits across all N classes. |

### Metrics

| Metric | Meaning |
| --- | --- |
| `reward` | Weighted sum of accuracy and format rewards. |
| `accuracy_reward` | `1.0` if the parsed answer correctly identifies the true label, `0.0` otherwise. |
| `format_reward` | `1.0` if the required XML schema was successfully generated, `0.0` otherwise. |
