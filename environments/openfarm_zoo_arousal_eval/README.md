# openfarm-zoo-arousal-eval

<p>
  <a href="https://github.com/ob1-s/happy-farm/tree/main/environments/openfarm_zoo_arousal_eval">
    <img align="left" hspace="4" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://app.primeintellect.ai/dashboard/environments/openfarm/openfarm-zoo-arousal-eval">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717&logoColor=white" alt="Prime Intellect Environments Hub">
  </a>
  <a href="https://huggingface.co/datasets/oliveirabruno01/openfarm-zoo-valence-arousal">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717" alt="Hugging Face Dataset">
  </a>
</p>
<br clear="all" />

## Overview

OpenFARM Zoo Arousal is a tiny visual affect eval built from expert-labeled
zoo-animal video stimuli. It tests whether multimodal models can classify
expert-coded arousal and/or valence from visible behavior, posture, movement,
and facial/body cues.

- **Environment ID**: `openfarm-zoo-arousal-eval`
- **Type**: single-turn classification / EnvGroup when multiple tasks are selected
- **Default modality**: `filmstrip` image
- **Other modalities**: `video`, `frames`, `text`
- **Output format**: XML answer, with optional explanation
- **Primary metric**: exact normalized answer reward

The headline benchmark is visual-only. The prepared dataset uses muted clips
because the source study frames the task as visual recognition from mute video
clips. Source audio was audited during prep and should not be treated as the
scientific signal for this env.

## Dataset

- **Primary dataset**: `oliveirabruno01/openfarm-zoo-valence-arousal`
- **Source**: Figshare collection `10.6084/m9.figshare.c.7807931`
- **Article**: Hiisivuori et al. (2025), *Human recognition of emotional valence and arousal of zoo animals*, `10.1038/s41598-025-28646-7`
- **Split sizes**: 15 `test` examples
- **Species**: Barbary macaque, Siberian tiger, Turkmenian markhor

## Tasks

| Task | Label |
| --- | --- |
| `arousal` | `low` / `high` |
| `valence` | `negative` / `neutral` / `positive` |
| `valence_arousal` | `negative_high` / `neutral_low` / `positive_low` / `positive_high` |

## Quickstart

```bash
prime eval run openfarm-zoo-arousal-eval \
  -a '{"task": "arousal", "modality": "filmstrip", "max_examples": 5}'
```

Run valence and arousal as an EnvGroup:

```bash
prime eval run openfarm-zoo-arousal-eval \
  -a '{"task": ["arousal", "valence"], "modality": "filmstrip"}'
```

Give the model nine separate sampled image inputs instead of one montage:

```bash
prime eval run openfarm-zoo-arousal-eval \
  -a '{"task": "valence_arousal", "modality": "frames"}'
```

Use the muted prepared video directly on endpoints that support video input:

```bash
prime eval run openfarm-zoo-arousal-eval \
  -a '{"task": "valence_arousal", "modality": "video"}'
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | str/list | `"arousal"` | `arousal`, `valence`, `valence_arousal`, a list, or `"all"`. |
| `dataset_id` | str | `"oliveirabruno01/openfarm-zoo-valence-arousal"` | Hugging Face dataset ID. |
| `dataset_revision` | str/null | `null` | Optional dataset revision. |
| `test_split` | str | `"test"` | Eval split. This dataset is eval-only. |
| `max_examples` | int | `-1` | Optional subsampling budget. |
| `seed` | int | `42` | Shuffle seed before subsampling. |
| `modality` | str | `"filmstrip"` | `filmstrip`, `video`, `frames`, or `text`. |
| `include_species_context` | bool | `false` | Adds species as prompt text. Kept off by default for a purer visual task. |
| `require_explanation` | bool | `false` | Requires an `<explanation>` field before `<answer>`. |
| `format_reward_weight` | float | `0.0` | Optional XML format reward weight. |

## Dataset Notes

- The dataset is intentionally eval-only; there is no meaningful train split.
- The public rows use opaque media filenames and omit source filenames, clip
  codes, segment timestamps, expert notes, pre-rendered messages, task ids, and
  OpenFARM-specific row IDs.
- `video` mode sends the muted prepared MP4 clip from embedded HF `Video` bytes.
  It is endpoint-dependent and intentionally has no local file fallback.
- `filmstrip` mode sends the 3x3 montage as one image. It is the most portable
  vision path across current multimodal endpoints.
- `frames` mode splits that same 3x3 filmstrip into nine separate image inputs,
  ordered left-to-right and top-to-bottom. This is useful for models such as
  Gemma 4 that can attend to multiple input images.

## Metrics

| Metric | Meaning |
| --- | --- |
| `accuracy_reward` | 1.0 when the parsed answer matches the target label after normalization. |
| `format_reward` | Optional XML-format reward when `format_reward_weight > 0`. |
