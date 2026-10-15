import base64
import io
import logging
import os
import re
from typing import Any, Literal

from PIL import Image
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

from .adaptive_judge_rubric import AdaptiveJudgeRubric
from .cheatsheet import CHEATSHEET

logger = logging.getLogger("verifiers.openfarm_horse_grimace")


SYSTEM_PROMPT = (
    "You are evaluating horse facial-region images for Horse Grimace Scale "
    "style pain-associated cues. This is an image-only facial-region scoring "
    "task, not a full clinical diagnosis."
)

REGION_LABELS = {
    "ears": "ears",
    "eyes": "eyes / orbital area",
    "mouth_nostrils": "mouth, nostrils, muzzle, and chin profile",
}

SCORE_LABELS = {
    "0": "not present",
    "1": "moderately present",
    "2": "obviously present",
}

SHORT_CHEATSHEET = """
# Horse Grimace Scale Quick Reference

HGS-style facial-region scores use 0 = not present, 1 = moderately present,
and 2 = obviously present.

- Ears: pain-associated cues include stiff backward rotation, adduction toward
  the midline, or asymmetry. Relaxed lowered ears can reflect rest/sleep.
- Eyes: pain-associated cues include orbital tightening, partial closure,
  angled/tensed eye appearance, and tension above the eye. Fear often widens
  the eyes rather than tightening them.
- Mouth/nostrils: pain-associated cues include strained nostrils, flattened
  facial profile, mouth/chin tension, and visible chewing-muscle strain.
- Confounders include sleep/fatigue, sedation/anesthesia, fear/stress, hard
  shadows, coat color, and the limits of judging a whole clinical state from a
  single cropped region.
"""

TASK_CONFIGS = {
    "region_score": {
        "target": "pain_score",
        "options": ["0", "1", "2"],
        "option_text": (
            "0 = cue not present, 1 = cue moderately present, 2 = cue obviously present"
        ),
        "instruction": (
            "Score the visible horse facial region using the HGS-style "
            "three-point scale."
        ),
        "answer_hint": "0, 1, or 2",
        "answer_aliases": {
            "0": {"0", "absent", "not present", "no cue", "normal"},
            "1": {"1", "moderate", "moderately present", "equivocal"},
            "2": {"2", "obvious", "obviously present", "marked", "severe"},
        },
    },
    "binary_pain": {
        "target": "binary_pain_label",
        "options": ["0", "1"],
        "option_text": "0 = no pain-associated cue present, 1 = pain-associated cue present",
        "instruction": (
            "Classify whether the visible horse facial region shows a "
            "pain-associated grimace cue."
        ),
        "answer_hint": "0 or 1",
        "answer_aliases": {
            "0": {"0", "no", "no pain", "no_pain", "not present", "absent"},
            "1": {"1", "yes", "pain", "pain cue", "present", "pain-associated"},
        },
    },
}

USER_PROMPT_TEMPLATE = """Recall what you know about the Horse Grimace Scale and apply it to this image.

[TASK]
{task_instruction}

{region_context}
Answer options: {option_text}.

Do not overclaim a whole clinical diagnosis from this crop. Judge only the visible facial-region cue.

{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[brief reasoning grounded in horse grimace-scale cues]
</explanation>
<answer>
[{answer_hint}]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[{answer_hint}]
</answer>"""

JUDGE_PROMPT = """Evaluate this multimodal reasoning task about horse facial-region grimace scoring.

[BACKGROUND KNOWLEDGE & GROUND TRUTH]
{answer}

[STUDENT'S REASONING]
{response}

Score the student's reasoning on a scale from 0 to 10 based purely on biological and factual soundness according to the background knowledge.
- Does the student reason from the relevant horse facial region rather than generic animal pain tropes?
- For ears, do they discuss stiff backward rotation/adduction or asymmetry without treating relaxed lowered ears as automatically painful?
- For eyes, do they discuss orbital tightening, angled eye appearance, or tension around the eye rather than fear-wide eyes?
- For mouth/nostrils, do they discuss strained nostrils, flattened profile, mouth/chin tension, or chewing-muscle tension?
- Do they respect the target score or binary target without hallucinating a full clinical diagnosis from a crop?
- Do they mention uncertainty/confounders such as sleep, sedation, fear, lighting, or coat color when appropriate?
(Note: You do not see the actual image; you are grading whether the written medical logic is internally consistent with the target and cheatsheet.)

Respond in the following format:
<assessment>
Briefly explain your reasoning for the score.
</assessment>
<score>
A single number from 0 to 10
</score>"""


def encode_image_to_base64(image: Image.Image) -> str:
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def normalized_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def canonical_task_name(task: str) -> str:
    aliases = {
        "score": "region_score",
        "hgs_score": "region_score",
        "region": "region_score",
        "binary": "binary_pain",
        "pain_binary": "binary_pain",
    }
    return aliases.get(task, task)


def normalize_answer(value: Any, task_config: dict) -> str:
    text = normalized_text(value)
    if not text:
        return ""

    for canonical, aliases in task_config["answer_aliases"].items():
        values = {normalized_text(alias) for alias in aliases}
        values.add(canonical)
        if text in values:
            return canonical

    # Require exactly one admissible response. Do not extract a score from
    # "0 or 2" or accept a class mentioned in additional free text.
    return text


def target_for_example(example: dict, task_name: str) -> str:
    if task_name == "region_score":
        value = example.get("pain_score")
        if isinstance(value, str) and value in SCORE_LABELS.values():
            reverse = {v: k for k, v in SCORE_LABELS.items()}
            return reverse[value]
        return (
            str(int(value)) if isinstance(value, (int, float)) else str(value).strip()
        )

    label = normalized_text(example.get("binary_pain_label", ""))
    if label in {"pain", "present", "yes", "1"}:
        return "1"
    if label in {"no pain", "no_pain", "absent", "not present", "0"}:
        return "0"

    score = example.get("pain_score")
    if score is not None:
        return "1" if int(score) > 0 else "0"
    raise ValueError("Could not derive binary_pain target from dataset row.")


def build_region_context(example: dict, include_region_context: bool) -> str:
    if not include_region_context:
        return ""
    face_region = str(example.get("face_region", "")).strip()
    readable = REGION_LABELS.get(face_region, face_region.replace("_", " "))
    if not readable:
        return ""
    return (
        f"The dataset identifies the visible crop as the horse's {readable} region.\n"
    )


def label_description(target: str, task_name: str) -> str:
    if task_name == "region_score":
        return f"{target} ({SCORE_LABELS.get(target, 'unknown score')})"
    return (
        "1 (pain-associated cue present)"
        if target == "1"
        else "0 (no pain-associated cue present)"
    )


def load_environment(
    dataset_id: str = "oliveirabruno01/openfarm-horse-grimace-region",
    dataset_revision: str | None = None,
    test_split: str = "test",
    task: Literal["region_score", "binary_pain"] = "region_score",
    max_examples: int = -1,
    max_examples_per_task: int | None = None,
    seed: int = 42,
    include_region_context: bool = True,
    require_explanation: bool = True,
    require_reasoning: bool | None = None,
    accuracy_reward_weight: float = 1.0,
    judge_reward_weight: float = 1.0,
    format_reward_weight: float = 0.0,
    reward_weights: dict[str, float] | None = None,
    judge_mode: Literal["external", "self", "none"] = "none",
    judge_model: str = "gpt-4o-mini",
    judge_api_key_var: str = "PRIME_API_KEY",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    cheatsheet_version: Literal["full", "short"] = "full",
    cheatsheet_override: str | None = None,
    **kwargs,
) -> vf.Environment:
    if max_examples_per_task is not None:
        max_examples = max_examples_per_task
    if require_reasoning is not None:
        require_explanation = require_reasoning
    if reward_weights:
        accuracy_reward_weight = reward_weights.get(
            "accuracy_reward", reward_weights.get("accuracy", accuracy_reward_weight)
        )
        judge_reward_weight = reward_weights.get(
            "judge_reward", reward_weights.get("judge", judge_reward_weight)
        )
        format_reward_weight = reward_weights.get(
            "format_reward", reward_weights.get("format", format_reward_weight)
        )

    task_name = canonical_task_name(task)
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unknown task {task!r}. Supported tasks: {sorted(TASK_CONFIGS)}"
        )
    task_config = TASK_CONFIGS[task_name]

    def build_dataset():
        ds = load_dataset(dataset_id, split=test_split, revision=dataset_revision)
        if seed is not None:
            ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))

        def transform(example):
            base64_image = encode_image_to_base64(example["image"])
            target = target_for_example(example, task_name)
            region_context = build_region_context(example, include_region_context)
            format_instructions = (
                FORMAT_WITH_REASONING
                if require_explanation
                else FORMAT_WITHOUT_REASONING
            ).format(answer_hint=task_config["answer_hint"])

            user_template = user_prompt_override or USER_PROMPT_TEMPLATE
            prompt_text = user_template.format(
                task_instruction=task_config["instruction"],
                region_context=region_context,
                option_text=task_config["option_text"],
                format_instructions=format_instructions,
            )
            system_prompt = system_prompt_override or SYSTEM_PROMPT

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_prompt}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                },
            ]

            target_description = label_description(target, task_name)
            return {
                "prompt": prompt,
                "answer": target,
                "info": {
                    "task_name": task_name,
                    "target_label": target,
                    "target_description": target_description,
                    "face_region": example.get("face_region", ""),
                    "pain_score": example.get("pain_score", None),
                    "pain_label": example.get("pain_label", ""),
                    "binary_pain_label": example.get("binary_pain_label", ""),
                    "horse_id": example.get("horse_id", ""),
                    "source_filename": example.get("source_filename", ""),
                },
            }

        return ds.map(transform, remove_columns=ds.column_names)

    fields = ["explanation", "answer"] if require_explanation else ["answer"]
    assistant_parser = vf.XMLParser(fields=fields, answer_field="answer")
    judge_parser = vf.XMLParser(fields=["assessment", "score"])

    if judge_mode == "external":
        vf.ensure_keys([judge_api_key_var])
        judge_client = AsyncOpenAI(
            api_key=os.getenv(judge_api_key_var),
            base_url=judge_base_url,
        )
    else:
        judge_client = None

    rubric = AdaptiveJudgeRubric(
        parser=assistant_parser,
        judge_mode=judge_mode,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
    )
    rubric.add_class_object("judge_parser", judge_parser)

    final_cheatsheet = CHEATSHEET if cheatsheet_version == "full" else SHORT_CHEATSHEET
    if cheatsheet_override is not None:
        final_cheatsheet = cheatsheet_override

    async def accuracy_reward(completion, info, parser, **kwargs) -> float:
        parsed = parser.parse_answer(completion)
        pred = normalize_answer(parsed or "", task_config)
        target = normalize_answer(info.get("target_label", ""), task_config)
        return 1.0 if pred == target else 0.0

    async def hybrid_judge_reward(
        prompt, completion, state, info, parser, judge, judge_parser, **kwargs
    ) -> float:
        if judge_mode == "none":
            return 0.0

        response_text = (
            str(completion[-1].get("content", ""))
            if isinstance(completion, list)
            else str(completion)
        )
        parsed_assistant = parser.parse(response_text)
        reasoning_text = getattr(parsed_assistant, "explanation", "")
        if not reasoning_text or not reasoning_text.strip():
            state["judge_assessment"] = "No reasoning provided to evaluate."
            return 0.0

        compound_answer = (
            f"Task: {info.get('task_name', task_name)}\n"
            f"Face region: {info.get('face_region', '')}\n"
            f"Ground truth answer: {info.get('target_description', info.get('target_label', ''))}\n"
            f"Source pain score: {info.get('pain_score', '')}\n"
            f"Source pain label: {info.get('pain_label', '')}\n\n"
            f"Cheatsheet:\n{final_cheatsheet}"
        )

        fake_completion = f"<answer>{reasoning_text}</answer>"

        try:
            verdict = await judge(
                prompt=prompt,
                completion=fake_completion,
                answer=compound_answer,
                state=state,
            )
            state["judge_assessment"] = verdict or ""
            parsed = judge_parser.parse(verdict or "")
            score_str = getattr(parsed, "score", None)
            if score_str:
                numbers = re.findall(
                    r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", str(score_str)
                )
                if numbers:
                    score = float(numbers[-1])
                    return max(0.0, min(1.0, score / 10.0))
        except Exception as exc:
            state["judge_error"] = str(exc)
            logger.error("Judging failed: %s", exc)
        return 0.0

    rubric.add_reward_func(accuracy_reward, weight=accuracy_reward_weight)
    if judge_mode != "none" and require_explanation:
        rubric.add_reward_func(hybrid_judge_reward, weight=judge_reward_weight)

    format_reward = assistant_parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=format_reward_weight)

    return vf.SingleTurnEnv(
        eval_dataset=build_dataset,
        parser=assistant_parser,
        rubric=rubric,
        **kwargs,
    )
