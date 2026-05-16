import io
import os
import re
import base64
import logging
from typing import Literal
from PIL import Image
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

from .adaptive_judge_rubric import AdaptiveJudgeRubric
from .cheatsheet import CHEATSHEET as FULL_CHEATSHEET
from .cheatsheet_short import CHEATSHEET as SHORT_CHEATSHEET

logger = logging.getLogger("verifiers.sheep_apr_grounded")

SYSTEM_PROMPT = (
    "You are evaluating sheep facial-expression cues associated with pain. "
    "Estimate whether the image matches SPFES-style pain-associated facial features; "
    "this is an image-only facial-expression classification task, not a full clinical diagnosis."
)

USER_PROMPT_TEMPLATE = """Recall what you know about established ovine facial pain-expression scales and apply it to this image.

[TASK]
Provide your reasoning, then conclude with a final binary classification score (0 = normal/no-pain facial expression, 1 = pain-associated/abnormal facial expression).

{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[your step-by-step analysis of facial features]
</explanation>
<answer>
[0 or 1]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[0 or 1]
</answer>"""

JUDGE_PROMPT = """Evaluate this multimodal reasoning task about sheep facial pain-expression classification.

[BACKGROUND KNOWLEDGE & GROUND TRUTH]
{answer}

[STUDENT'S REASONING]
{response}

Score the student's reasoning on a scale from 0 to 10 based purely on biological and factual soundness according to the background knowledge. 
- Does the student correctly recall and define ovine Facial Action Units (FAUs) and recognized grimace scales?
- Do they avoid hallucinating fake signals or misrepresenting how the scales work?
- Is their medical logic internally consistent and matched to the facts?
- Do they avoid overclaiming a full clinical diagnosis from an image-only facial-expression proxy label?
(Note: You do not see the actual image; you are purely grading if their written medical logic is internally consistent).

Respond in the following format:
<assessment>
Briefly explain your reasoning for the score.
</assessment>
<score>
A single number from 0 to 10
</score>"""


def encode_image_to_base64(image: Image.Image) -> str:
    """Helper to convert PIL images to base64 data URIs for multimodal LLMs."""
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def load_environment(
    dataset_id: str = "oliveirabruno01/sheep-facial-expression-benchmark",
    dataset_revision: str | None = None,
    test_split: str = "train",
    max_examples: int = -1,
    seed: int = 42,
    require_explanation: bool = True,
    accuracy_reward_weight: float = 1.0,
    judge_reward_weight: float = 1.0,
    format_reward_weight: float = 0.0,
    judge_mode: Literal["external", "self", "none"] = "none",
    judge_model: str = "gpt-4o-mini",
    judge_api_key_var: str = "PRIME_API_KEY",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    cheatsheet_version: Literal["full", "short"] = "short",
    cheatsheet_override: str | None = None,
    **kwargs,
) -> vf.Environment:

    def build_dataset():
        ds = load_dataset(dataset_id,split=test_split, revision=dataset_revision)
        if seed:
            ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))

        def map_to_standard(example):
            base64_image = encode_image_to_base64(example["image"])
            ground_truth = str(example["label"]).strip()

            sys_prompt = (
                system_prompt_override
                if system_prompt_override is not None
                else SYSTEM_PROMPT
            )
            user_template = (
                user_prompt_override
                if user_prompt_override is not None
                else USER_PROMPT_TEMPLATE
            )
            format_instr = (
                FORMAT_WITH_REASONING if require_explanation else FORMAT_WITHOUT_REASONING
            )

            prompt_text = user_template.format(
                format_instructions=format_instr,
            )

            # Wrap the system prompt content in a list so it matches the multimodal user message schema
            prompt = [
                {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                },
            ]

            return {
                "prompt": prompt,
                "answer": ground_truth,
                "info": {
                    "target_label": ground_truth,
                },
            }

        return ds.map(map_to_standard, remove_columns=ds.column_names)

    # 1. Define Dual Parsers
    fields = ["explanation", "answer"] if require_explanation else ["answer"]
    assistant_parser = vf.XMLParser(fields=fields, answer_field="answer")
    judge_parser = vf.XMLParser(fields=["assessment", "score"])

    # 2. Setup the Adaptive Judge Rubric
    if judge_mode == "external":
        vf.ensure_keys([judge_api_key_var])
        judge_client = AsyncOpenAI(
            api_key=os.getenv(judge_api_key_var), base_url=judge_base_url
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

    FINAL_CHEATSHEET = SHORT_CHEATSHEET
    if cheatsheet_version == "full":
        FINAL_CHEATSHEET = FULL_CHEATSHEET

    if cheatsheet_override is not None:
        FINAL_CHEATSHEET = cheatsheet_override

    # 3. Declarative Reward & Metric Functions
    async def accuracy_reward(completion, info, parser, **kwargs) -> float:
        parsed = parser.parse_answer(completion)
        pred = str(parsed).strip() if parsed else None
        ans_clean = str(info.get("target_label", "")).strip()
        return 1.0 if pred == ans_clean else 0.0

    async def hybrid_judge_reward(
        prompt, completion, state, info, parser, judge, judge_parser, **kwargs
    ) -> float:
        if judge_mode == "none":
            return 0.0

        if isinstance(completion, list):
            response_text = str(completion[-1].get("content", ""))
        else:
            response_text = str(completion)

        parsed_assistant = parser.parse(response_text)
        reasoning_text = getattr(parsed_assistant, "explanation", "")

        if not reasoning_text or not reasoning_text.strip():
            state["judge_assessment"] = "No reasoning provided to evaluate."
            return 0.0

        # Combine cheatsheet and ground truth for the canonical '{answer}' placeholder
        ground_truth = str(info.get("target_label", "")).strip()
        compound_answer = (
            f"Ground Truth Label: {ground_truth}\n\nCheatsheet:\n{FINAL_CHEATSHEET}"
        )

        # To use the canonical judge() method for evaluating reasoning specifically,
        # we wrap the reasoning text so assistant_parser extracts it as the '{response}'.
        fake_completion = f"<answer>{reasoning_text}</answer>"

        try:
            verdict = await judge(
                prompt=prompt,
                completion=fake_completion,
                answer=compound_answer,
                state=state,
            )

            if verdict:
                state["judge_assessment"] = verdict
                logger.debug(f"Judge verdict: {verdict}")
            else:
                state["judge_assessment"] = ""

            parsed = judge_parser.parse(verdict)
            score_str = getattr(parsed, "score", None)

            if score_str:
                try:
                    numbers = re.findall(
                        r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", str(score_str)
                    )
                    if numbers:
                        score = float(numbers[-1])
                        return max(0.0, min(1.0, score / 10.0))
                except ValueError:
                    pass
        except Exception as e:
            state["judge_error"] = str(e)
            logger.error(f"Judging failed: {str(e)}")

        return 0.0

    # 4. Attach Functions to the Single Rubric
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
