# openfarm-bioacustics

OpenFARM Bioacoustics is a task-registry environment for animal vocalization
benchmarks. It keeps each source dataset/task explicit, while exposing a single
SoundWel-like interface for audio, spectrogram, multimodal, and text/acoustic
metadata ablations.

## Overview

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

Legacy SoundWel task names are accepted as aliases:
`valence_classification`, `context_classification`, `call_type_classification`,
and `age_classification`.

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

The laying-hen dataset prep now writes bounded public audio artifacts: 15-second
mono 16 kHz excerpts built with `even_windows` from the longer Zenodo source
recordings. The task still defaults to `max_audio_seconds=15.0` and
`audio_clip_strategy="even_windows"` as a safety net for older dataset revisions
or local experiments that point the env at full source recordings. The source
data does not provide stress-event timestamps, so the windowing policy samples
short positions across the recording instead of assuming the first or middle
seconds contain the relevant vocal response.

## Metrics

The live rubric is intentionally small:

| Metric | Meaning |
| --- | --- |
| `accuracy_reward` | 1.0 when the parsed answer matches the task label after normalization. |
| `format_reward` | Optional XML-format reward when `format_reward_weight > 0`. |

Use post-eval reporting for macro accuracy, balanced accuracy, macro F1, and
per-class recall.
