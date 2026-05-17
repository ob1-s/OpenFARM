# catmeows-audio-context

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ob1-s/happy-farm/tree/main/environments/catmeows_audio_context)&nbsp;&nbsp;[![Prime Intellect](https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717)](https://app.primeintellect.ai/dashboard/environments/hfarm/catmeows-audio-context)&nbsp;&nbsp;[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717)](https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows)

### Overview
- **Environment ID**: `catmeows-audio-context`
- **Short description**: Feline audio context classification evaluating a model's ability to identify the eliciting context (`brushing`, `isolation`, `waiting for food`) from cat vocalizations.
- **Tags**: `animal-behavior`, `animal-welfare`, `bioacoustics`, `feline`, `audio`, `multimodal`, `single-turn`, `eval`, `train`

### Datasets
- **Primary dataset(s)**: `oliveirabruno01/openfarm-catmeows` (Prepared audio with obfuscated filenames to prevent leakage).
- **Source links**: 
  - [Hugging Face Dataset](https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows)
  - [Original Zenodo Dataset](https://zenodo.org/records/4008297) (CatMeows: A Publicly-Available Dataset of Cat Vocalizations)
- **Split sizes**: Evaluates on `train` and `test` splits, which are strictly cat-heldout and class-balanced. Natural distributions are available via `train_raw` and `test_raw`.

### Task
- **Type**: single-turn classification (3-class forced choice).
- **Target labels**: `brushing`, `isolation_unfamiliar_environment`, `waiting_for_food`.
- **Output format expectations**: Strict XML tags parsed via `vf.XMLParser`.
  - By default (`require_explanation=False`), it expects:
      ```xml
      <answer>[Prediction]</answer>
      ```
  - If `require_explanation=True`, it expects:
      ```xml
      <explanation>[Acoustic analysis]</explanation>
      <answer>[Prediction]</answer>
      ```
- **Rubric overview**: Evaluates classification accuracy (exact match with robust normalization) and XML format adherence. Includes an optional "escape hatch" to penalize or ignore completely silent/unintelligible audio.

### Quickstart
Run an evaluation with default settings (audio modality, no explanation required):

```bash
prime eval run catmeows-audio-context
```

Run an evaluation forcing reasoning and acoustic metadata injection:

```bash
prime eval run catmeows-audio-context \
  -m openai/gpt-4o-audio-preview \
  -n 20 -r 3 -T 0.7 \
  -a '{"require_explanation": true, "include_tabular_data": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `modality` | `str` | `"audio"` | Data modality to present: `"audio"` or `"text"` (tabular features only). |
| `include_tabular_data` | `bool` | `False` | If `true`, injects acoustic and demographic features (e.g. `rms_energy`, `duration_sec`) as a markdown list in the prompt. Forced to `true` if `modality="text"`. |
| `use_escape_hatch` | `bool` | `False` | If `true`, allows the model to output `UNINTELLIGIBLE` for missing or completely silent samples. |
| `max_examples` | `int` | `-1` | Limit on dataset size (use `-1` for all available rows). |
| `dataset_name` | `str` | `"oliveirabruno01/..."` | HuggingFace dataset ID or local path. |
| `dataset_revision` | `str` | `None` | Optional HuggingFace dataset revision/commit hash. |
| `train_split` | `str` | `"train"` | Split used to populate the RL training dataset. |
| `test_split` | `str` | `"test"` | Split used to populate the evaluation dataset. |
| `seed` | `int` | `42` | Random seed for dataset shuffling. |
| `target_pad_seconds` | `float` | `0.0` | Seconds to centrally zero-pad audio arrays. Defaults to `0.0` (disabled, uses raw variable length). |
| `system_prompt_override` | `str` | `None` | Optional custom string to replace the default ethologist system prompt. |
| `user_prompt_override` | `str` | `None` | Optional custom template string for the user task prompt. |
| `require_explanation` | `bool` | `False` | Whether to require an `<explanation>` block before the `<answer>` block. |
| `format_reward_weight` | `float` | `0.0` | Weight for the XML format reward. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of accuracy and format adherence). |
| `accuracy_reward` | `1.0` if the parsed answer correctly identifies the true context label, `0.0` otherwise. |
| `format_reward` | `1.0` if the required XML schema was successfully generated, `0.0` otherwise. |