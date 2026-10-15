"""OpenFARM Bioacoustics.

A task-registry environment for animal vocalization benchmarks. The atomic unit
is still the source dataset/task pair, but the interface lets us run related
audio benchmarks through one Verifiers environment.
"""

import logging
import re
from typing import Any

import verifiers as vf

from .task_registry import (
    ESCAPE_HATCH_NOTES,
    FORMAT_WITH_REASONING,
    FORMAT_WITHOUT_REASONING,
    MODALITY_INSTRUCTIONS,
    SYSTEM_PROMPT,
    TASK_ALIASES,
    TASK_CONFIGS,
    TASK_TEMPLATE,
)
from .media_utils import build_media_parts

logger = logging.getLogger("verifiers.openfarm_bioacustics")


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

    # Do not accept an answer merely because it contains a valid label.
    # "Positive or Negative" is not an unambiguous decision.
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
        # A missing or inaccessible test split MUST NOT silently become 'train'.
        # Fail closed for every split, including transient network/revision errors.
        raise RuntimeError(
            f"Could not load requested split {split!r} from {dataset_name!r} "
            f"(revision={dataset_revision!r}); no fallback dataset was used."
        ) from exc

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
            class_ds.select(
                range(min(take_per_class + (1 if i < remainder else 0), len(class_ds)))
            )
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
    audio_clip_strategy: str | None,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    require_explanation: bool,
    format_reward_weight: float,
    balancing_strategy: str,
) -> vf.SingleTurnEnv:
    label_column = task_config["label_column"]
    dataset_name = dataset_name_override or task_config["dataset_name"]
    task_pad_seconds = task_config.get("target_pad_seconds", target_pad_seconds)
    task_max_seconds = (
        max_audio_seconds
        if max_audio_seconds is not None
        else task_config.get("max_audio_seconds")
    )
    task_clip_strategy = audio_clip_strategy or task_config.get(
        "audio_clip_strategy", "start"
    )

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
                    audio_clip_strategy=task_clip_strategy,
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
                    "audio_clip_strategy": task_clip_strategy,
                    "max_audio_seconds": task_max_seconds,
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
    task: str | list[str] = "catmeows_context",
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
    audio_clip_strategy: str | None = None,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    require_explanation: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
    allow_uncertified_ungulate_eval: bool = False,
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
        allow_uncertified_ungulate_eval: Forensic opt-in ONLY; published
            ungulate splits have verified byte-identical cross-split media.
    """
    if train_split == test_split:
        raise ValueError(
            f"Invalid evaluation setup: train_split and test_split both use "
            f"{train_split!r}. Hold out an independent evaluation split."
        )

    valid_modalities = {"audio", "vision", "multimodal", "text"}
    if modality not in valid_modalities:
        raise ValueError(
            f"Unknown modality {modality!r}. Valid: {sorted(valid_modalities)}"
        )
    # Our *retained* tasks test observable audio, not metadata reconstruction.
    # Metadata-only features can reveal species/condition proxies and are
    # not interchangeable with a recording-based decision.
    if modality == "text" or include_tabular_data:
        raise ValueError(
            "Only recording/spectrogram observations are supported in the "
            "retained OpenFARM suite. Metadata-only evaluation is retired."
        )
    if dataset_name is not None:
        raise ValueError(
            "Unreviewed dataset_name override is not an active-suite dataset. "
            "Audit and register the new source revision before using it."
        )
    if (train_split,test_split) not in {("train","test"),("train_raw","test_raw")}:
        raise ValueError(
            "Unsupported split combination: use matched published train/test "
            "or train_raw/test_raw only."
        )

    if task == "all":
        selected_tasks = list(TASK_CONFIGS)
    elif isinstance(task, str):
        selected_tasks = [canonical_task_name(task)]
    else:
        selected_tasks = [canonical_task_name(name) for name in task]

    # Retired data cannot be reintroduced through the old forensic override.
    # The commit history retains the original environment for reproduction.
    retired = [name for name in selected_tasks if name not in TASK_CONFIGS]
    if retired:
        from .task_registry import RETIRED_TASK_REASONS
        details = {name: RETIRED_TASK_REASONS.get(name, "not in active suite")
                   for name in retired}
        raise ValueError(f"Task retired from active OpenFARM suite: {details}")

    unknown = [name for name in selected_tasks if name not in TASK_CONFIGS]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. Valid tasks: {list(TASK_CONFIGS)}; "
            f"aliases: {list(TASK_ALIASES)}"
        )

    pinned_revisions = {
        "soundwel_call_type": "4164e39798c2e65bce5ef8d56159d98e1267f577",
        "catmeows_context": "57fe220b22eae46eb555af9f4169bd4677584cea",
    }
    if dataset_revision is not None and any(
        dataset_revision != pinned_revisions[name] for name in selected_tasks
    ):
        raise ValueError("Active suite requires the audited pinned Hugging Face revision.")
    # Source HF revisions are immutable; never silently drift to latest.
    if len(selected_tasks)>1:
        # The tasks have different source revisions and cannot share one
        # dataset_revision argument without breaking provenance.
        raise ValueError("Run the two retained task datasets separately.")
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
                dataset_revision=pinned_revisions[task_name],
                dataset_name_override=None,
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
