import io
import os
import json
import base64
import logging
from PIL import Image
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

from .cheatsheet import CHEATSHEET

logger = logging.getLogger("verifiers.sheep_apr_grounded")


def encode_image_to_base64(image: Image.Image) -> str:
    """Helper to convert PIL images to base64 data URIs for multimodal LLMs."""
    buffered = io.BytesIO()
    # Convert to RGB if necessary (to handle PNG/RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def load_environment(
    dataset_id: str = "oliveirabruno01/sheep-facial-expression-benchmark",
    seed: int = 42,
    use_judge: bool = True,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    format_reward_weight: float = 0.0,
    reasoning_reward_weight: float = 1.0,
    **kwargs,
) -> vf.Environment:
    """
    Multimodal environment for sheep pain assessment.
    Labels: 0 = No Pain, 1 = Pain.
    """
    # 1. Fail fast if the judge API key is not configured
    vf.ensure_keys([judge_api_key_var])

    def build_dataset():
        ds = load_dataset(dataset_id, split="train")

        def transform(example):
            # 1. Convert PIL image to base64
            base64_image = encode_image_to_base64(example["image"])

            # 2. Create the multimodal prompt structure
            # verifiers handles message lists with 'type': 'image_url' for multimodal models
            instructions = (
                "You are a veterinary clinical evaluator. Assess this sheep for pain "
                "based on its facial expressions. Recall what you know about established "
                "ovine pain assessment methods and apply it to this image.\n\n"
                "Provide your reasoning, then conclude with a final binary classification score "
                "(0 = No Pain, 1 = Pain).\n\n"
                "Respond using this format:\n"
                "<thought>\n[your analysis and method application]\n</thought>\n"
                "<score>\n[0 or 1]\n</score>"
            )
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }
            ]

            return {"prompt": prompt, "answer": str(example["label"])}

        return ds.map(transform, remove_columns=ds.column_names).shuffle(seed=seed)

    # Define the Parser to extract values from the XML tags
    parser = vf.XMLParser(fields=["thought", "score"], answer_field="score")

    # Reward function: Check if the parsed score matches the ground truth
    async def accuracy_reward(completion, answer, parser) -> float:
        parsed_answer = parser.parse_answer(completion)
        if parsed_answer is None:
            return 0.0

        # Clean string comparison (handles potential whitespace)
        return 1.0 if str(parsed_answer).strip() == str(answer).strip() else 0.0

    # 2. Setup the Judge Client & Prompt
    judge_client = AsyncOpenAI(
        api_key=os.getenv(judge_api_key_var, ""), base_url=judge_base_url
    )

    JUDGE_PROMPT_TEMPLATE = f"""Evaluate this multimodal reasoning task about sheep pain detection.

BACKGROUND KNOWLEDGE (CHEATSHEET):
{CHEATSHEET}

Student's Reasoning: 
{{reasoning}}

Score the student's reasoning on a scale from 0 to 10 based purely on biological and factual soundness according to the cheatsheet. 
- Does the student correctly recall and define ovine Facial Action Units (FAUs) and recognized grimace scales?
- Do they avoid hallucinating fake signals or misrepresenting how the scales work (e.g., claiming a recognized grimace scale is binary when it isn't, even though their final task output is binary)?
- Is their medical logic internally consistent and matched to the cheatsheet facts?
(Note: You do not know the actual image, you are purely grading if their written medical logic is internally consistent).

Provide a structured JSON evaluation with two fields:
- "assessment": A brief justification of the score.
- "score": An integer from 0 to 10."""

    # 3. Custom Reasoning Reward
    async def reasoning_reward(
        completion, parser, judge_client, judge_model, judge_prompt
    ) -> float:
        assistant_messages = parser.get_assistant_messages(completion)
        if not assistant_messages:
            return 0.0

        # Safely extract text from the multimodal completion
        msg = assistant_messages[-1]
        content = msg.get("content", "") if isinstance(msg, dict) else msg.content
        text_content = parser._content_to_text(content)

        # Parse the reasoning block
        parsed = parser.parse(text_content)
        reasoning = getattr(parsed, "thought", None)

        if not reasoning or not reasoning.strip():
            return 0.0

        filled_prompt = judge_prompt.format(reasoning=reasoning.strip())

        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": filled_prompt}],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "judge_evaluation",
                        "schema": {
                            "type": "object",
                            "properties": {
                                "assessment": {"type": "string"},
                                "score": {"type": "integer"},
                            },
                            "required": ["assessment", "score"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
            )
            verdict = response.choices[0].message.content or "{}"

            # Log the full structured assessment and score for debugging
            logger.info(f"Judge Verdict: {verdict}")

            # Parse JSON to extract the score
            parsed_verdict = json.loads(verdict)
            score_raw = parsed_verdict.get("score", 0)

            # Normalize to 0.0 - 1.0
            score = float(score_raw) / 10.0
            return max(0.0, min(1.0, score))  # Clamp to [0, 1]

        except Exception as e:
            logger.info(f"Judge reward error: {e}")
            pass

        return 0.0

    # 4. Transition to JudgeRubric
    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT_TEMPLATE,
        parser=parser,
    )

    # 5. Attach Rewards
    rubric.add_reward_func(accuracy_reward, weight=1.0)
    if use_judge:
        rubric.add_reward_func(reasoning_reward, weight=reasoning_reward_weight)

    # Add a format reward to encourage the model to follow the XML schema
    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=format_reward_weight)

    return vf.SingleTurnEnv(
        dataset=build_dataset, rubric=rubric, parser=parser, **kwargs
    )
