import os
import re
import logging
from typing import Literal

import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

from .adaptive_judge_rubric import AdaptiveJudgeRubric

logger = logging.getLogger("verifiers.dog_pain_triage")

# ============================================================================
# Declarative Prompt Templates
# ============================================================================

SYSTEM_PROMPT = (
    "You are an expert veterinary AI analyzing canine clinical metadata. "
    "Based on the provided demographic and clinical history, assess the dog's affective pain state."
)

USER_PROMPT_TEMPLATE = """Review the following clinical metadata for a canine patient.

[CLINICAL CONTEXT]
{case_text}

[TASK]
Classify the animal's affective state into exactly one of the following categories:
- Negative_Nociceptive
- Neutral_Resting

{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[your step-by-step clinical reasoning]
</explanation>
<answer>
[your final classification]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[your final classification]
</answer>"""

JUDGE_PROMPT = """Evaluate the following veterinary triage assessment.

[CLINICAL CONTEXT (The exact data provided to the model)]
{question}

[GOLD STANDARD REFERENCE]
{answer}

[MODEL RESPONSE]
{response}

Compare the Model's analysis to the Gold Standard. 
Score the model from 0 to 10 based on:
1. Did it select the correct Target Category? (Heavy penalty if wrong)
2. Is the medical reasoning biologically sound and consistent with the provided clinical metadata?

Respond in the following format:
<explanation>
Briefly explain your reasoning for the score, comparing the model response to the gold standard.
</explanation>
<score>
A single number from 0 to 10
</score>"""

# ============================================================================
# Environment Loader
# ============================================================================


def load_environment(
    dataset_id: str = "oliveirabruno01/openfarm-dog-pain",
    dataset_revision: str | None = None,
    train_split: str = "train",
    test_split: str = "test",
    max_examples: int = -1,
    seed: int = 42,
    include_extended_surgery_fields: bool = False,
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
    **kwargs,
) -> vf.Environment:
    """Load the Dog Pain Triage Environment."""

    def get_dataset_builder(split: str) -> callable:
        def build_dataset():
            ds = load_dataset(dataset_id, split=split, revision=dataset_revision)
            if seed:
                ds = ds.shuffle(seed=seed)
            if max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))

            def map_to_standard(row):
                target = str(row.get("aww_category", "Unknown"))

                # 1. Build the safe case text, explicitly excluding target/leaky fields
                safe_fields = [
                    ("Species", "species"),
                    ("Breed", "breed"),
                    ("Sex", "sex"),
                    ("Neuter status", "neuter_status"),
                    ("Primary diagnosis", "primary_diagnosis"),
                    ("Medical history", "medical_history"),
                    ("Clinical situation", "situation"),
                ]

                if include_extended_surgery_fields:
                    safe_fields.extend(
                        [
                            ("Time since end of surgery", "time_since_surgery"),
                            ("Fully recovered from anesthesia", "recovered_anesthesia"),
                        ]
                    )

                lines = []
                for label, col in safe_fields:
                    raw_val = row.get(col)
                    val = str(raw_val).strip() if raw_val is not None else ""
                    if val and val.lower() != "nan":
                        lines.append(f"- {label}: {val}")
                case_text = "\n".join(lines)

                # The rich text for the LLM Judge
                gold_standard = f"Target Category: {target}"

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
                    FORMAT_WITH_REASONING
                    if require_explanation
                    else FORMAT_WITHOUT_REASONING
                )

                prompt_text = user_template.format(
                    case_text=case_text,
                    format_instructions=format_instr,
                )

                prompt = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt_text},
                ]

                return {
                    "prompt": prompt,
                    "answer": gold_standard,  # Feeds into the LLM Judge
                    "info": {
                        **dict(row),
                        "target_label": target,  # Feeds into our accuracy reward cleanly
                    },
                }

            return ds.map(map_to_standard, remove_columns=ds.column_names)

        return build_dataset

    # 1. Define Dual Parsers
    fields = ["explanation", "answer"] if require_explanation else ["answer"]
    assistant_parser = vf.XMLParser(fields=fields, answer_field="answer")
    judge_parser = vf.XMLParser(fields=["explanation", "score"])

    # 2. Setup the Judge Client & Adaptive Judge Rubric
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

    # Inject the second parser so reward functions can request it via kwargs
    rubric.add_class_object("judge_parser", judge_parser)

    # 3. Declarative Reward & Metric Functions
    async def accuracy_reward(parser, completion, info, **kwargs) -> float:
        parsed = parser.parse_answer(completion) or ""
        pred = parsed.strip().lower()
        ans_clean = str(info.get("target_label", "")).strip().lower()
        is_correct = bool(pred) and ((ans_clean == pred) or (ans_clean in pred))
        return 1.0 if is_correct else 0.0

    async def hybrid_judge_reward(
        prompt, completion, answer, state, judge, judge_parser, **kwargs
    ) -> float:
        """LLM-as-a-judge score using self-judge, external-judge, or none."""
        if judge_mode == "none":
            return 0.0

        try:
            verdict = await judge(
                prompt=prompt,
                completion=completion,
                answer=answer,
                state=state,
            )

            if verdict:
                state["judge_reasoning"] = verdict
                logger.debug(f"Judge reasoning: {verdict}")
            else:
                state["judge_reasoning"] = ""

            parsed = judge_parser.parse(verdict)
            score_str = parsed.score if parsed and hasattr(parsed, "score") else None

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
            logger.error(f"Judging failed: {str(e)}")

        return 0.0

    # 4. Attach Functions to Rubric
    rubric.add_reward_func(accuracy_reward, weight=accuracy_reward_weight)

    if judge_mode != "none":
        rubric.add_reward_func(hybrid_judge_reward, weight=judge_reward_weight)

    format_reward = assistant_parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=format_reward_weight)

    return vf.SingleTurnEnv(
        dataset=get_dataset_builder(train_split),
        eval_dataset=get_dataset_builder(test_split),
        parser=assistant_parser,
        rubric=rubric,
        **kwargs,
    )
