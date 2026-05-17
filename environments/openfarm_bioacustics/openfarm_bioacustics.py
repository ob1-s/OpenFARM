"""OpenFARM Bioacoustics.

A task-registry environment for animal vocalization benchmarks. The atomic unit
is still the source dataset/task pair, but the interface lets us run related
audio benchmarks through one Verifiers environment.
"""

import base64
import io
import logging
import re
from typing import Any

import numpy as np
import verifiers as vf
from PIL import Image as PILImage

logger = logging.getLogger("verifiers.openfarm_bioacustics")


# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = (
    "You are an animal bioacoustics evaluator. Classify vocalization recordings "
    "according to the requested source-dataset task. Do not infer pain, disease, "
    "stress, or welfare status unless that is exactly what the task label means."
)

MODALITY_INSTRUCTIONS = {
    "audio": "<audio>Listen to {subject}.",
    "vision": "Look at the provided spectrogram of {subject}.",
    "multimodal": (
        "<audio>You are provided with both an audio recording and its "
        "corresponding spectrogram image for {subject}."
    ),
    "text": "Review the tabular acoustic/source metadata for {subject}.",
}

ESCAPE_HATCH_NOTES = {
    "audio": (
        "NOTE: If the audio is completely missing, entirely silent, or too short "
        "to hear any animal vocalization, classify it as UNINTELLIGIBLE.\n\n"
    ),
    "vision": (
        "NOTE: If the spectrogram is missing or visually empty, classify it as "
        "UNINTELLIGIBLE.\n\n"
    ),
    "multimodal": (
        "NOTE: If both the audio and spectrogram are missing or unintelligible, "
        "classify it as UNINTELLIGIBLE.\n\n"
    ),
    "text": (
        "NOTE: If the tabular metadata is completely missing or empty, classify "
        "it as UNINTELLIGIBLE.\n\n"
    ),
}

TASK_TEMPLATE = """{modality_instruction}
{question}
{tabular_data}
{escape_hatch_note}{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[reasoning]
</explanation>
<answer>
[{options}]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[{options}]
</answer>"""


# ============================================================================
# Task Registry
# ============================================================================

SOUNDWEL_DATASET = "oliveirabruno01/soundwel-pig-vocalizations"
CATMEOWS_DATASET = "oliveirabruno01/openfarm-catmeows"
LAYING_HEN_DATASET = "oliveirabruno01/openfarm-laying-hen-stress"

SOUNDWEL_TABULAR_KEYS = [
    "Age Category",
    "Sex",
    "Dur",
    "F0Mean",
    "Fpeak",
    "WienEntrMean",
    "Harmonicity",
    "Q25",
    "Q50",
    "Q75",
    "AMVar",
    "AMRate",
    "AMExtent",
    "DFMin",
]

CATMEOWS_TABULAR_KEYS = [
    "duration_sec",
    "rms_energy",
    "peak_abs_amplitude",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "breed",
    "sex_status",
]

LAYING_HEN_TABULAR_KEYS = [
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "mfcc_segment_count",
]

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "soundwel_call_type": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Type",
        "subject": "a pig vocalization",
        "question": "Is this a High Frequency (HF) call or a Low Frequency (LF) call?",
        "options": ["HF", "LF"],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "exclude_columns": {"Q25", "Q50", "Q75", "F0Mean", "Fpeak", "DFMin"},
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_valence": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Valence",
        "label_map": {"Pos": "Positive", "Neg": "Negative"},
        "subject": "a pig vocalization",
        "question": "Is the emotional valence of this animal Positive or Negative?",
        "options": ["Positive", "Negative"],
        "answer_aliases": {
            "Positive": ["positive", "pos"],
            "Negative": ["negative", "neg"],
        },
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_context": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Context Category",
        "subject": "a pig vocalization",
        "question": "What is the source context category of this vocalization?",
        "options": [
            "AfterNursing",
            "Barren",
            "BeforeNursing",
            "Castration",
            "Crushing",
            "Enriched",
            "Fighting",
            "Huddling",
            "Isolation",
            "MissedNursing",
            "NegativeConditioning",
            "NovelObject",
            "PositiveConditioning",
            "Restrain",
            "Reunion",
            "Run",
            "Surprise",
        ],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_age": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Age Category",
        "subject": "a pig vocalization",
        "question": "What is the age category of the pig?",
        "options": ["Piglet", "Weaner", "GrFinishing"],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Dur", "AMRate"],
    },
    "catmeows_context": {
        "dataset_name": CATMEOWS_DATASET,
        "label_column": "context",
        "subject": "a cat vocalization",
        "question": (
            "Choose the most likely source dataset context for this recording. "
            "This is not a clinical pain, stress, or welfare diagnosis."
        ),
        "options": [
            "brushing",
            "isolation_unfamiliar_environment",
            "waiting_for_food",
        ],
        "answer_aliases": {
            "brushing": ["brushing"],
            "isolation_unfamiliar_environment": [
                "isolation_unfamiliar_environment",
                "isolation",
                "unfamiliar environment",
            ],
            "waiting_for_food": ["waiting_for_food", "waiting", "food"],
        },
        "tabular_keys": CATMEOWS_TABULAR_KEYS,
        "info_keys": ["cat_id", "duration_sec"],
        "target_pad_seconds": 0.0,
    },
    "laying_hen_stress_context": {
        "dataset_name": LAYING_HEN_DATASET,
        "label_column": "stress_context",
        "subject": "a laying-hen vocalization recording",
        "question": "Choose the source experimental stress-context label for this recording.",
        "options": [
            "poststress_control",
            "prestress_control",
            "poststress_treatment",
            "prestress_treatment",
        ],
        "tabular_keys": LAYING_HEN_TABULAR_KEYS,
        "info_keys": ["source_audio_id", "experimental_day_group", "duration_sec"],
        "max_audio_seconds": 15.0,
    },
    "laying_hen_stress_binary": {
        "dataset_name": LAYING_HEN_DATASET,
        "label_column": "stress_binary",
        "subject": "a laying-hen vocalization recording",
        "question": "Classify whether this recording represents a stress response.",
        "options": ["stress_response", "not_stress_response"],
        "tabular_keys": LAYING_HEN_TABULAR_KEYS,
        "info_keys": ["source_audio_id", "experimental_day_group", "duration_sec"],
        "max_audio_seconds": 15.0,
    },
}

TASK_ALIASES = {
    "call_type_classification": "soundwel_call_type",
    "valence_classification": "soundwel_valence",
    "context_classification": "soundwel_context",
    "age_classification": "soundwel_age",
    "catmeows_audio_context": "catmeows_context",
    "cat_context": "catmeows_context",
    "hen_stress_context": "laying_hen_stress_context",
    "hen_stress_binary": "laying_hen_stress_binary",
}


# ============================================================================
# Media Utilities
# ============================================================================


def pad_audio_array(audio_array: np.ndarray, sr: int, target_seconds: float) -> np.ndarray:
    target_length = int(target_seconds * sr)
    current_length = len(audio_array)
    if current_length >= target_length:
        return audio_array

    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio_array, (pad_left, pad_right), mode="constant")


def clip_audio_array(
    audio_array: np.ndarray,
    sr: int,
    max_seconds: float | None,
    strategy: str = "start",
) -> np.ndarray:
    if max_seconds is None or max_seconds <= 0:
        return audio_array

    max_samples = int(max_seconds * sr)
    if len(audio_array) <= max_samples:
        return audio_array

    if strategy == "center":
        start = max(0, (len(audio_array) - max_samples) // 2)
    else:
        start = 0
    return audio_array[start : start + max_samples]


def audio_dict_to_array(audio_dict: dict) -> tuple[np.ndarray, int]:
    """Decode a HF Audio row dict into an audio array and sampling rate."""
    import soundfile as sf

    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
        array, sr = sf.read(io.BytesIO(audio_bytes))
    elif "path" in audio_dict and audio_dict["path"] is not None:
        array, sr = sf.read(audio_dict["path"])
    else:
        array = audio_dict.get("array")
        sr = audio_dict.get("sampling_rate")

    if array is None or sr is None:
        raise ValueError("Audio row does not contain bytes, path, or decoded array.")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, int(sr)


def prepare_audio_array(
    audio_dict: dict,
    target_pad_seconds: float = 0.0,
    max_audio_seconds: float | None = None,
    audio_clip_strategy: str = "start",
) -> tuple[np.ndarray, int]:
    array, sr = audio_dict_to_array(audio_dict)
    array = clip_audio_array(array, sr, max_audio_seconds, audio_clip_strategy)
    if target_pad_seconds > 0:
        array = pad_audio_array(array, sr, target_pad_seconds)
    return array, sr


def encode_audio_to_base64(
    audio_dict: dict,
    target_pad_seconds: float = 0.0,
    max_audio_seconds: float | None = None,
    audio_clip_strategy: str = "start",
) -> str:
    import soundfile as sf

    array, sr = prepare_audio_array(
        audio_dict,
        target_pad_seconds=target_pad_seconds,
        max_audio_seconds=max_audio_seconds,
        audio_clip_strategy=audio_clip_strategy,
    )
    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def spectrogram_from_audio(audio_array: np.ndarray, sr: int) -> PILImage.Image:
    """Generate a compact spectrogram image using only numpy and Pillow."""
    del sr  # The visual benchmark needs relative time/frequency energy, not axes.

    if audio_array.size == 0:
        audio_array = np.zeros(2048, dtype=np.float32)

    frame_size = 1024
    hop = 256
    if len(audio_array) < frame_size:
        audio_array = pad_audio_array(audio_array, 1, frame_size)

    n_frames = 1 + max(0, (len(audio_array) - frame_size) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(audio_array, frame_size)[::hop]
    frames = frames[:n_frames]
    window = np.hanning(frame_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).T
    spec = np.log1p(spec)

    low, high = np.percentile(spec, [5, 99])
    if high <= low:
        high = low + 1e-6
    spec = np.clip((spec - low) / (high - low), 0.0, 1.0)
    spec = np.flipud(spec)

    gray = (spec * 255).astype(np.uint8)
    image = PILImage.fromarray(gray, mode="L").convert("RGB")
    return image.resize((512, 384), resample=PILImage.BILINEAR)


def coerce_image(value: Any) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return PILImage.open(value["path"])
    raise ValueError("Unsupported spectrogram image value.")


def encode_image_to_base64(image: PILImage.Image) -> str:
    buf = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def build_media_parts(
    row: dict,
    modality: str,
    target_pad_seconds: float,
    max_audio_seconds: float | None,
    audio_clip_strategy: str,
) -> list[dict]:
    parts = []
    if modality == "text":
        return parts

    audio_array = None
    sr = None

    if modality in {"audio", "multimodal"}:
        audio_b64 = encode_audio_to_base64(
            row["audio"],
            target_pad_seconds=target_pad_seconds,
            max_audio_seconds=max_audio_seconds,
            audio_clip_strategy=audio_clip_strategy,
        )
        parts.append(
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        )

    if modality in {"vision", "multimodal"}:
        if "spectrogram" in row and row["spectrogram"] is not None:
            image = coerce_image(row["spectrogram"])
        else:
            audio_array, sr = prepare_audio_array(
                row["audio"],
                target_pad_seconds=target_pad_seconds,
                max_audio_seconds=max_audio_seconds,
                audio_clip_strategy=audio_clip_strategy,
            )
            image = spectrogram_from_audio(audio_array, sr)

        parts.append({"type": "image_url", "image_url": {"url": encode_image_to_base64(image)}})

    return parts


# ============================================================================
# Dataset/Text Utilities
# ============================================================================


def normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def option_variants(option: str) -> set[str]:
    normalized = normalized_text(option)
    variants = {normalized}
    if "_" in option:
        variants.add(normalized_text(option.replace("_", " ")))
    camel = re.sub(r"(?<!^)(?=[A-Z])", " ", option)
    variants.add(normalized_text(camel))
    return {v for v in variants if v}


def build_answer_aliases(task_config: dict) -> dict[str, set[str]]:
    aliases = {}
    for option in task_config["options"]:
        values = set(task_config.get("answer_aliases", {}).get(option, []))
        values.add(option)
        expanded = set()
        for value in values:
            expanded.update(option_variants(value))
        aliases[option] = expanded
    return aliases


def normalize_answer(value: Any, task_config: dict) -> str:
    label_map = task_config.get("label_map", {})
    mapped = label_map.get(value, value)
    text = normalized_text(mapped)

    aliases = build_answer_aliases(task_config)
    for canonical, values in aliases.items():
        if text in values:
            return canonical

    for canonical, values in aliases.items():
        for alias in values:
            if re.search(rf"\b{re.escape(alias)}\b", text):
                return canonical

    return str(mapped).strip()


def format_options(options: list[str], use_escape_hatch: bool) -> str:
    final_options = list(options)
    if use_escape_hatch:
        final_options.append("UNINTELLIGIBLE")
    if len(final_options) == 1:
        return final_options[0]
    if len(final_options) == 2:
        return f"{final_options[0]} or {final_options[1]}"
    return ", ".join(final_options[:-1]) + f", or {final_options[-1]}"


def format_tabular_data(row: dict, task_config: dict, label_column: str) -> str:
    exclude = {
        label_column,
        "answer",
        "messages",
        "spectrogram",
        "audio",
        "audio_filename",
        "source_filename",
        "source_path",
        "openfarm_id",
    }
    exclude.update(task_config.get("exclude_columns", set()))

    lines = []
    for key in task_config.get("tabular_keys", []):
        if key in exclude or key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        lines.append(f"- {key}: {text}")

    if not lines:
        return ""
    return "\n[Tabular Data]\n" + "\n".join(lines) + "\n"


def load_source_dataset(dataset_name: str, split: str, dataset_revision: str | None):
    from datasets import Audio, Image, load_dataset

    logger.info("Loading dataset %s split=%s", dataset_name, split)
    try:
        ds = load_dataset(dataset_name, split=split, revision=dataset_revision)
    except Exception as exc:
        logger.warning(
            "Failed to load split %r from %s; falling back to train. Error: %s",
            split,
            dataset_name,
            exc,
        )
        ds = load_dataset(dataset_name, split="train", revision=dataset_revision)

    if "audio" in ds.column_names:
        ds = ds.cast_column("audio", Audio(decode=False))
    if "spectrogram" in ds.column_names:
        ds = ds.cast_column("spectrogram", Image(decode=True))
    return ds


def balance_and_shuffle(
    ds,
    label_column: str,
    max_examples: int,
    seed: int,
    strategy: str = "proportional",
):
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if max_examples <= 0:
        return ds

    from datasets import concatenate_datasets

    labels = [label for label in sorted(set(ds[label_column])) if label is not None]
    if not labels:
        return ds.select(range(0))

    class_datasets = {
        label: ds.filter(lambda x, label=label: x[label_column] == label)
        for label in labels
    }
    if strategy == "balanced":
        if max_examples < len(labels):
            selected_labels = labels[:max_examples]
            selected = [
                class_datasets[label].select(range(1))
                for label in selected_labels
                if len(class_datasets[label]) > 0
            ]
            return concatenate_datasets(selected).shuffle(seed=seed)

        take_per_class = max_examples // len(labels)
        remainder = max_examples % len(labels)
        selected = [
            class_ds.select(range(min(take_per_class + (1 if i < remainder else 0), len(class_ds))))
            for i, class_ds in enumerate(class_datasets.values())
            if len(class_ds) > 0
        ]
    else:
        return ds.select(range(min(max_examples, len(ds)))).shuffle(seed=seed)

    if not selected:
        return ds.select(range(0))
    return concatenate_datasets(selected).shuffle(seed=seed)


# ============================================================================
# Rubric
# ============================================================================


class BioacousticsRubric(vf.Rubric):
    """Exact-answer rubric for registry-backed bioacoustic classification tasks."""

    def __init__(
        self,
        task_config: dict,
        require_explanation: bool = False,
        format_reward_weight: float = 0.0,
    ):
        self.task_config = task_config
        fields = ["explanation", "answer"] if require_explanation else ["answer"]
        self.parser = vf.XMLParser(fields=fields, answer_field="answer")
        super().__init__(parser=self.parser)

        self.add_reward_func(self.accuracy_reward, weight=1.0)
        format_reward = self.parser.get_format_reward_func()
        format_reward.__name__ = "format_reward"
        self.add_reward_func(format_reward, weight=format_reward_weight)

    async def accuracy_reward(self, completion, answer) -> float:
        parsed = self.parser.parse_answer(completion) or ""
        if not parsed.strip():
            return 0.0

        pred = normalize_answer(parsed, self.task_config)
        target = normalize_answer(answer, self.task_config)
        return 1.0 if normalized_text(pred) == normalized_text(target) else 0.0


# ============================================================================
# Environment Builder
# ============================================================================


def canonical_task_name(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def build_task_env(
    task_name: str,
    task_config: dict,
    modality: str,
    include_tabular_data: bool,
    use_escape_hatch: bool,
    max_examples: int,
    dataset_revision: str | None,
    dataset_name_override: str | None,
    train_split: str,
    test_split: str,
    seed: int,
    target_pad_seconds: float,
    max_audio_seconds: float | None,
    audio_clip_strategy: str,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    require_explanation: bool,
    format_reward_weight: float,
    balancing_strategy: str,
) -> vf.SingleTurnEnv:
    label_column = task_config["label_column"]
    dataset_name = dataset_name_override or task_config["dataset_name"]
    task_pad_seconds = task_config.get("target_pad_seconds", target_pad_seconds)
    task_max_seconds = task_config.get("max_audio_seconds", max_audio_seconds)

    def get_dataset_builder(split: str):
        def build():
            ds = load_source_dataset(dataset_name, split, dataset_revision)
            ds = ds.filter(lambda row: row.get(label_column) is not None)
            ds = balance_and_shuffle(
                ds,
                label_column=label_column,
                max_examples=max_examples,
                seed=seed,
                strategy=balancing_strategy,
            )

            def transform(row):
                system_prompt = system_prompt_override or task_config.get(
                    "system_prompt", SYSTEM_PROMPT
                )
                user_template = user_prompt_override or TASK_TEMPLATE
                format_template = (
                    FORMAT_WITH_REASONING
                    if require_explanation
                    else FORMAT_WITHOUT_REASONING
                )
                options = format_options(task_config["options"], use_escape_hatch)
                modality_instruction = MODALITY_INSTRUCTIONS[modality].format(
                    subject=task_config["subject"]
                )
                tabular = (
                    format_tabular_data(row, task_config, label_column)
                    if include_tabular_data
                    else ""
                )
                prompt_text = user_template.format(
                    modality_instruction=modality_instruction,
                    question=task_config["question"],
                    tabular_data=tabular,
                    escape_hatch_note=ESCAPE_HATCH_NOTES[modality]
                    if use_escape_hatch
                    else "",
                    format_instructions=format_template.format(options=options),
                )

                media_parts = build_media_parts(
                    row,
                    modality=modality,
                    target_pad_seconds=task_pad_seconds,
                    max_audio_seconds=task_max_seconds,
                    audio_clip_strategy=audio_clip_strategy,
                )
                prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt}],
                    },
                    {
                        "role": "user",
                        "content": media_parts
                        + [{"type": "text", "text": prompt_text}],
                    },
                ]

                answer = normalize_answer(row[label_column], task_config)
                info = {
                    "env_id": task_name,
                    "openfarm_bioacoustic_task": task_name,
                    "dataset_name": dataset_name,
                    "label_column": label_column,
                }
                for key in task_config.get("info_keys", []):
                    if key in row:
                        info[key] = row.get(key)

                return {"prompt": prompt, "answer": answer, "info": info}

            return ds.map(transform, remove_columns=ds.column_names, num_proc=1)

        return build

    rubric = BioacousticsRubric(
        task_config=task_config,
        require_explanation=require_explanation,
        format_reward_weight=format_reward_weight,
    )

    return vf.SingleTurnEnv(
        dataset=get_dataset_builder(train_split),
        eval_dataset=get_dataset_builder(test_split),
        parser=rubric.parser,
        rubric=rubric,
    )


def load_environment(
    task: str | list[str] = "soundwel_valence",
    modality: str = "audio",
    include_tabular_data: bool = False,
    use_escape_hatch: bool = False,
    max_examples_per_task: int = -1,
    dataset_name: str | None = None,
    dataset_revision: str | None = None,
    train_split: str = "train",
    test_split: str = "test",
    seed: int = 42,
    target_pad_seconds: float = 3.592,
    max_audio_seconds: float | None = None,
    audio_clip_strategy: str = "start",
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    require_explanation: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
    **kwargs,
) -> vf.Environment:
    """Load the OpenFARM Bioacoustics environment.

    Args:
        task: A task name, list of task names, or "all". Legacy SoundWel names
            such as "valence_classification" are accepted as aliases.
        modality: "audio", "vision", "multimodal", or "text". Vision uses an
            existing spectrogram column when available and otherwise generates a
            spectrogram from the audio at load time.
        include_tabular_data: Include leakage-safe task-specific metadata.
            Forced on for text-only mode.
        max_audio_seconds: Optional audio clipping budget before base64 encoding
            or spectrogram generation. Laying-hen tasks default to 15 seconds.
        balancing_strategy: "proportional" or "balanced" when subsampling with
            max_examples_per_task.
    """
    valid_modalities = {"audio", "vision", "multimodal", "text"}
    if modality not in valid_modalities:
        raise ValueError(f"Unknown modality {modality!r}. Valid: {sorted(valid_modalities)}")
    if modality == "text":
        include_tabular_data = True

    if task == "all":
        selected_tasks = list(TASK_CONFIGS)
    elif isinstance(task, str):
        selected_tasks = [canonical_task_name(task)]
    else:
        selected_tasks = [canonical_task_name(name) for name in task]

    unknown = [name for name in selected_tasks if name not in TASK_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. Valid tasks: {list(TASK_CONFIGS)}; "
            f"aliases: {list(TASK_ALIASES)}"
        )

    envs = []
    for task_name in selected_tasks:
        logger.info("Building OpenFARM Bioacoustics task: %s", task_name)
        envs.append(
            build_task_env(
                task_name=task_name,
                task_config=TASK_CONFIGS[task_name],
                modality=modality,
                include_tabular_data=include_tabular_data,
                use_escape_hatch=use_escape_hatch,
                max_examples=max_examples_per_task,
                dataset_revision=dataset_revision,
                dataset_name_override=dataset_name,
                train_split=train_split,
                test_split=test_split,
                seed=seed,
                target_pad_seconds=target_pad_seconds,
                max_audio_seconds=max_audio_seconds,
                audio_clip_strategy=audio_clip_strategy,
                system_prompt_override=system_prompt_override,
                user_prompt_override=user_prompt_override,
                require_explanation=require_explanation,
                format_reward_weight=format_reward_weight,
                balancing_strategy=balancing_strategy,
            )
        )

    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=selected_tasks)
