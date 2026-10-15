"""OpenFARM Zoo Arousal.

Small visual affect benchmark built from expert-labeled zoo-animal stimuli.
The standalone paths are a muted video, a 3x3 filmstrip, the same filmstrip
split into separate image inputs, or metadata-only text.
"""

import base64
import io
import logging
import re
from typing import Any

from PIL import Image as PILImage
import verifiers as vf

logger = logging.getLogger("verifiers.openfarm_zoo_arousal")


DATASET_ID = "oliveirabruno01/openfarm-zoo-valence-arousal"

SYSTEM_PROMPT = (
    "You are evaluating visual affect cues in zoo animals. Predict the requested "
    "expert-coded label from visible behavior, posture, facial/body cues, and "
    "movement. Do not treat the label as a clinical pain, disease, or complete "
    "welfare diagnosis."
)

MODALITY_INSTRUCTIONS = {
    "filmstrip": "You are given a 3x3 filmstrip summarizing the visual stimulus.",
    "frames": (
        "You are given nine separate sampled frames from the visual stimulus, "
        "ordered left-to-right and top-to-bottom from the original filmstrip."
    ),
    "video": "You are given the muted short video stimulus.",
    "text": "You are given non-target source metadata only.",
}

TASKS: dict[str, dict[str, Any]] = {
    "arousal": {
        "label_column": "arousal",
        "question": "Is the animal's expert-coded arousal low or high?",
        "options": ["low", "high"],
    },
    "valence": {
        "label_column": "valence",
        "question": "Is the animal's expert-coded emotional valence negative, neutral, or positive?",
        "options": ["negative", "neutral", "positive"],
    },
    "valence_arousal": {
        "label_column": "valence_arousal",
        "question": (
            "Choose the combined expert-coded valence/arousal label for this stimulus."
        ),
        "options": ["negative_high", "neutral_low", "positive_low", "positive_high"],
    },
}

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

USER_PROMPT_TEMPLATE = """{modality_instruction}
{metadata_context}
[TASK]
{question}

These labels come from expert interpretation in a zoo-animal visual emotion-recognition study. They are not clinical diagnoses.

{format_instructions}"""


def normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def option_variants(option: str) -> set[str]:
    return {
        normalize_text(option),
        normalize_text(option.replace("_", " ")),
        normalize_text(option.replace("_", "/")),
    }


def normalize_answer(value: Any, task_config: dict) -> str:
    text = normalize_text(value)
    for option in task_config["options"]:
        if text in option_variants(option):
            return option
    for option in task_config["options"]:
        if any(
            re.search(rf"\b{re.escape(variant)}\b", text)
            for variant in option_variants(option)
        ):
            return option
    return str(value).strip()


def format_options(options: list[str]) -> str:
    if len(options) == 1:
        return options[0]
    if len(options) == 2:
        return f"{options[0]} or {options[1]}"
    return ", ".join(options[:-1]) + f", or {options[-1]}"


def coerce_image(value: Any) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return PILImage.open(value["path"])
    raise ValueError("Unsupported image value.")


def encode_image_to_base64(value: Any) -> str:
    image = coerce_image(value)
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def read_video_bytes(value: Any) -> bytes:
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, dict) and value.get("bytes") is not None:
        return value["bytes"]
    raise ValueError(
        "Video media is missing embedded bytes. Republish the dataset with "
        "portable HF Video bytes before using modality='video'."
    )


def encode_video_to_base64(value: Any) -> str:
    raw = read_video_bytes(value)
    return "data:video/mp4;base64," + base64.b64encode(raw).decode("utf-8")


def split_filmstrip(value: Any, grid_size: int = 3) -> list[PILImage.Image]:
    image = coerce_image(value)
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    xs = [round(width * i / grid_size) for i in range(grid_size + 1)]
    ys = [round(height * i / grid_size) for i in range(grid_size + 1)]
    frames = []
    for row in range(grid_size):
        for col in range(grid_size):
            frames.append(image.crop((xs[col], ys[row], xs[col + 1], ys[row + 1])))
    return frames


def build_media_parts(row: dict, modality: str) -> list[dict]:
    if modality == "text":
        return []

    if modality == "filmstrip":
        return [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_base64(row["filmstrip"])},
            }
        ]

    if modality == "frames":
        return [
            {
                "type": "image_url",
                "image_url": {"url": encode_image_to_base64(frame)},
            }
            for frame in split_filmstrip(row["filmstrip"])
        ]

    if modality == "video":
        return [
            {
                "type": "video_url",
                "video_url": {"url": encode_video_to_base64(row["video"])},
            }
        ]

    raise ValueError(
        f"Unknown modality {modality!r}. Valid: {list(MODALITY_INSTRUCTIONS)}"
    )


def format_metadata_context(
    row: dict,
    modality: str,
    include_species_context: bool,
) -> str:
    keys = []
    if modality == "text" or include_species_context:
        keys.append("species")
    if modality == "text":
        keys.append("duration_sec")

    lines = []
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        lines.append(f"- {key}: {text}")

    if not lines:
        return ""
    return "\n[Metadata]\n" + "\n".join(lines) + "\n"


def load_source_dataset(dataset_id: str, split: str, dataset_revision: str | None):
    from datasets import Image, Video, load_dataset

    logger.info("Loading dataset %s split=%s", dataset_id, split)
    ds = load_dataset(dataset_id, split=split, revision=dataset_revision)
    if "filmstrip" in ds.column_names:
        ds = ds.cast_column("filmstrip", Image(decode=True))
    if "video" in ds.column_names:
        ds = ds.cast_column("video", Video(decode=False))
    return ds


def shuffle_and_limit(ds, max_examples: int, seed: int):
    if seed is not None:
        ds = ds.shuffle(seed=seed)
    if max_examples <= 0:
        return ds
    return ds.select(range(min(max_examples, len(ds))))


class ZooArousalRubric(vf.Rubric):
    """Exact XML-answer rubric for zoo affect labels."""

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
        pred = normalize_answer(parsed, self.task_config)
        target = normalize_answer(answer, self.task_config)
        return 1.0 if normalize_text(pred) == normalize_text(target) else 0.0


def build_task_env(
    task_name: str,
    task_config: dict,
    dataset_id: str,
    dataset_revision: str | None,
    test_split: str,
    max_examples: int,
    seed: int,
    modality: str,
    include_species_context: bool,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    require_explanation: bool,
    format_reward_weight: float,
) -> vf.SingleTurnEnv:
    label_column = task_config["label_column"]

    def build_dataset():
        ds = load_source_dataset(dataset_id, test_split, dataset_revision)
        ds = ds.filter(lambda row: row.get(label_column) is not None)
        ds = shuffle_and_limit(ds, max_examples, seed)

        def transform(row):
            sys_prompt = system_prompt_override or SYSTEM_PROMPT
            user_template = user_prompt_override or USER_PROMPT_TEMPLATE
            format_template = (
                FORMAT_WITH_REASONING
                if require_explanation
                else FORMAT_WITHOUT_REASONING
            )
            prompt_text = user_template.format(
                modality_instruction=MODALITY_INSTRUCTIONS[modality],
                metadata_context=format_metadata_context(
                    row,
                    modality=modality,
                    include_species_context=include_species_context,
                ),
                question=task_config["question"],
                format_instructions=format_template.format(
                    options=format_options(task_config["options"])
                ),
            )

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": sys_prompt}],
                },
                {
                    "role": "user",
                    "content": build_media_parts(row, modality)
                    + [{"type": "text", "text": prompt_text}],
                },
            ]

            answer = normalize_answer(row[label_column], task_config)
            return {
                "prompt": prompt,
                "answer": answer,
                "info": {
                    "env_id": "openfarm-zoo-arousal",
                    "openfarm_zoo_task": task_name,
                    "dataset_id": dataset_id,
                    "label_column": label_column,
                    "modality": modality,
                    "target_label": answer,
                    "species": row.get("species"),
                    "duration_sec": row.get("duration_sec"),
                    "video_filename": row.get("video_filename"),
                    "filmstrip_filename": row.get("filmstrip_filename"),
                    "source_url": row.get("source_url"),
                    "source_article_url": row.get("source_article_url"),
                },
            }

        return ds.map(transform, remove_columns=ds.column_names, num_proc=1)

    rubric = ZooArousalRubric(
        task_config=task_config,
        require_explanation=require_explanation,
        format_reward_weight=format_reward_weight,
    )
    return vf.SingleTurnEnv(
        eval_dataset=build_dataset,
        parser=rubric.parser,
        rubric=rubric,
    )


def load_environment(
    task: str | list[str] = "arousal",
    dataset_id: str = DATASET_ID,
    dataset_revision: str | None = None,
    test_split: str = "test",
    max_examples: int = -1,
    seed: int = 42,
    modality: str = "filmstrip",
    include_species_context: bool = False,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    require_explanation: bool = False,
    format_reward_weight: float = 0.0,
    **kwargs,
) -> vf.Environment:
    raise RuntimeError(
        "RETIRED FROM ACTIVE OPENFARM SUITE: Zoo affect task has 15 clips, subjective interpretations and no meaningful independent generalization split. "
        "See docs/system1/SUITE_DECISIONS.md; use Git history for forensic "
        "reproduction, not current benchmark runs."
    )

    """Load the OpenFARM Zoo Arousal environment."""
    if modality not in MODALITY_INSTRUCTIONS:
        raise ValueError(
            f"Unknown modality {modality!r}. Valid: {list(MODALITY_INSTRUCTIONS)}"
        )

    if task == "all":
        selected_tasks = list(TASKS)
    elif isinstance(task, str):
        selected_tasks = [task]
    else:
        selected_tasks = list(task)

    unknown = [name for name in selected_tasks if name not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Valid tasks: {list(TASKS)}")

    envs = [
        build_task_env(
            task_name=name,
            task_config=TASKS[name],
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            test_split=test_split,
            max_examples=max_examples,
            seed=seed,
            modality=modality,
            include_species_context=include_species_context,
            system_prompt_override=system_prompt_override,
            user_prompt_override=user_prompt_override,
            require_explanation=require_explanation,
            format_reward_weight=format_reward_weight,
        )
        for name in selected_tasks
    ]

    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=selected_tasks)
