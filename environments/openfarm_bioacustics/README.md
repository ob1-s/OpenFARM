# openfarm-bioacustics

<p>
  <a href="https://github.com/ob1-s/happy-farm/tree/main/environments/openfarm_bioacustics">
    <img align="left" hspace="4" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://app.primeintellect.ai/dashboard/environments/openfarm/openfarm-bioacustics">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717&logoColor=white" alt="Prime Intellect Environments Hub">
  </a>
</p>
<br clear="all" />

## Overview

OpenFARM Bioacoustics is a task-registry environment for animal vocalization
benchmarks. It keeps each source dataset/task explicit, while exposing a single
SoundWel-like interface for audio, spectrogram, multimodal, and text/acoustic
metadata ablations.

- **Environment ID**: `openfarm-bioacustics`
- **Type**: single-turn classification / EnvGroup when multiple tasks are selected
- **Modalities**: `audio`, `vision`, `multimodal`, `text`
- **Output format**: XML answer, with optional explanation
- **Primary metric**: exact normalized answer reward

Vision mode uses a source spectrogram column when one exists. For audio-only
datasets, the environment generates a compact spectrogram from the audio at load
time.

## Tasks

| Task | Dataset | Label |
| --- | --- | --- |
| `soundwel_valence` | `oliveirabruno01/soundwel-pig-vocalizations` | `Positive` / `Negative` |
| `soundwel_context` | `oliveirabruno01/soundwel-pig-vocalizations` | pig vocalization context |
| `soundwel_call_type` | `oliveirabruno01/soundwel-pig-vocalizations` | `HF` / `LF` |
| `soundwel_age` | `oliveirabruno01/soundwel-pig-vocalizations` | pig age category |
| `catmeows_context` | `oliveirabruno01/openfarm-catmeows` | feline source context |
| `laying_hen_stress_context` | `oliveirabruno01/openfarm-laying-hen-stress` | experimental stress context |
| `laying_hen_stress_binary` | `oliveirabruno01/openfarm-laying-hen-stress` | stress-response binary |
| `ungulate_valence` | `oliveirabruno01/openfarm-ungulate-valence` | `Positive` / `Negative` |

## Source Attribution

| Dataset | Prepared HF Dataset | Source Dataset / Paper |
| --- | --- | --- |
| SoundWel pig vocalizations | [`oliveirabruno01/soundwel-pig-vocalizations`](https://huggingface.co/datasets/oliveirabruno01/soundwel-pig-vocalizations) | The Soundwel Database: a labeled pig vocalization repository, Zenodo record [`8252482`](https://zenodo.org/records/8252482), DOI [`10.1038/s41598-022-07174-8`](https://doi.org/10.1038/s41598-022-07174-8), CC-BY-4.0 |
| CatMeows | [`oliveirabruno01/openfarm-catmeows`](https://huggingface.co/datasets/oliveirabruno01/openfarm-catmeows) | CatMeows: A Publicly-Available Dataset of Cat Vocalizations, Zenodo record [`4008297`](https://zenodo.org/records/4008297), DOI [`10.5281/zenodo.4008297`](https://doi.org/10.5281/zenodo.4008297), CC-BY-4.0 |
| Laying hen stress vocalizations | [`oliveirabruno01/openfarm-laying-hen-stress`](https://huggingface.co/datasets/oliveirabruno01/openfarm-laying-hen-stress) | Vocalization Patterns in Laying Hens - An Analysis of Stress-Induced Audio Responses, Zenodo record [`10433023`](https://zenodo.org/records/10433023), DOI [`10.5281/zenodo.10433023`](https://doi.org/10.5281/zenodo.10433023), CC-BY-4.0 |
| Ungulate vocalization valence | [`oliveirabruno01/openfarm-ungulate-valence`](https://huggingface.co/datasets/oliveirabruno01/openfarm-ungulate-valence) | Machine Learning Algorithms Can Predict Emotional Valence Across Ungulate Vocalizations, Zenodo record [`14636641`](https://zenodo.org/records/14636641), DOI [`10.5281/zenodo.14636641`](https://doi.org/10.5281/zenodo.14636641), CC-BY-4.0 |

## Quickstart

```bash
prime eval run openfarm-bioacustics \
  -a '{"task": "soundwel_valence", "modality": "audio", "max_examples_per_task": 20}'
```

Run several tasks as one EnvGroup:

```bash
prime eval run openfarm-bioacustics \
  -a '{"task": ["soundwel_valence", "catmeows_context"], "modality": "multimodal", "max_examples_per_task": 20}'
```

Use generated spectrograms for an audio-only dataset:

```bash
prime eval run openfarm-bioacustics \
  -a '{"task": "catmeows_context", "modality": "vision", "max_examples_per_task": 20}'
```

Run the ungulate valence benchmark on its balanced, non-pig animal-heldout test
split:

```bash
prime eval run openfarm-bioacustics \
  -a '{"task": "ungulate_valence", "modality": "audio", "max_examples_per_task": 20}'
```

## Environment Arguments

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `task` | str/list | `"soundwel_valence"` | Task name, list of task names, or `"all"`. |
| `modality` | str | `"audio"` | `audio`, `vision`, `multimodal`, or `text`. |
| `include_tabular_data` | bool | `false` | Adds task-specific leakage-safe metadata. Forced on for `text`. |
| `use_escape_hatch` | bool | `false` | Adds `UNINTELLIGIBLE` as an allowed answer. |
| `max_examples_per_task` | int | `-1` | Optional per-task subsampling budget. |
| `balancing_strategy` | str | `"proportional"` | `proportional` or `balanced` when subsampling. |
| `target_pad_seconds` | float | `3.592` | Center-pads short audio before encoding. |
| `max_audio_seconds` | float/null | `null` | Clips long audio before encoding or spectrogram generation. |
| `audio_clip_strategy` | str | `"start"` | `start`, `center`, or `even_windows` for clipping long recordings. |
| `require_explanation` | bool | `false` | Requires an `<explanation>` field before `<answer>`. |
| `format_reward_weight` | float | `0.0` | Optional XML format reward weight. |

## Dataset Notes

### Laying Hen Stress

- Public audio artifacts are bounded 15-second mono 16 kHz excerpts.
- Excerpts use `even_windows` over the longer Zenodo recordings because the
  source data does not provide stress-event timestamps.
- The task still defaults to `max_audio_seconds=15.0` and
  `audio_clip_strategy="even_windows"` as a safety net for older dataset
  revisions or local experiments pointed at full source recordings.

### Ungulate Valence

- Headline evals should use the cleaned, balanced `train`/`test` split pair.
- `train_raw` and `test_raw` preserve the natural non-pig class distribution for
  explicit diagnostics.
- `pig_family_heldout` isolates domestic pig and wild boar rows for optional
  transfer checks against SoundWel/PVWB-adjacent work.
- Default tabular prompts use only acoustic features. Context, source reference,
  and animal ID can be predictive of the valence label.

## Metrics

The live rubric is intentionally small:

| Metric | Meaning |
| --- | --- |
| `accuracy_reward` | 1.0 when the parsed answer matches the task label after normalization. |
| `format_reward` | Optional XML-format reward when `format_reward_weight > 0`. |

Use post-eval reporting for macro accuracy, balanced accuracy, macro F1, and
per-class recall.
