import os
import re
import logging
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

logger = logging.getLogger("verifiers.aww_bench")


SYSTEM_PROMPT = "You are an expert veterinary AI."

USER_PROMPT_TEMPLATE = """[SPECIES CLASS]
{species_class}

[SUBJECT]
{subject_description}

[CONTEXT]
{context_metadata}

[VISUAL CUES]
{visual_cues}
{adversarial_section}
[TASK]
Based on the visual cues and context provided, analyze the animal's affective state. Reason through any potential biological confounders or distractor information. Provide your final classification as exactly one of the following:
- Negative_Nociceptive
- Negative_Affective
- Positive_Welfare
- Neutral_Resting
- False_Positive_Morphological
- False_Positive_Physiological
- False_Negative_Masking

{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<reasoning>
[your step-by-step analysis]
</reasoning>
<answer>
[your final classification]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[your final classification]
</answer>"""

JUDGE_PROMPT = """Evaluate the following animal welfare assessment.

[QUESTION (the exact words that prompted the assessment)]
{question}

[GOLD STANDARD REFERENCE]
{answer}

[MODEL RESPONSE]
{response}

Compare the Model's analysis to the Gold Standard. 
Score the model from 0 to 10 based on:
1. Did it select the correct Target Category? (Heavy penalty if wrong)
2. Did it correctly identify and reason through the adversarial distractor/confounder?

Respond in the following format:
<reasoning>
Briefly explain your reasoning for the score, comparing the model response to the gold standard.
</reasoning>
<score>
A single number from 0 to 10
</score>"""

# ============================================================================
# Environment Loader
# ============================================================================


def load_environment(
    dataset_id: str = "oliveirabruno01/AWW-bench",
    train_split: str = "train",
    test_split: str = "test",
    max_examples: int = -1,
    seed: int = 42,
    include_adversarial_context: bool = True,
    require_reasoning: bool = False,
    accuracy_reward_weight: float = 1.0,
    judge_reward_weight: float = 1.0,
    format_reward_weight: float = 0.0,
    judge_mode: str = "external",  # "external", "self", or "none"
    judge_model: str = "gpt-4.1-mini",
    judge_api_key_var: str = "PRIME_API_KEY",
    judge_base_url: str = "https://api.pinference.ai/api/v1",
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    **kwargs,
) -> vf.Environment:
    """Load the Animal Welfare Assessment Benchmark."""

    def get_dataset_builder(split: str) -> callable:
        def build_dataset():
            ds = load_dataset(dataset_id, split=split)
            if seed:
                ds = ds.shuffle(seed=seed)
            if max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))

            def map_to_standard(row):
                target = row.get("target_category", "Unknown")
                reasoning = row.get("sota_judge_summary", "No reasoning provided")

                # The rich text for the LLM Judge
                gold_standard = f"Target Category: {target}\nReasoning: {reasoning}"

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
                    if require_reasoning
                    else FORMAT_WITHOUT_REASONING
                )

                adversarial_section = ""
                if include_adversarial_context and row.get("adversarial_distractor"):
                    adversarial_section = f"\n[ADDITIONAL INFO]\n{row.get('adversarial_distractor', '')}\n"

                prompt_text = user_template.format(
                    species_class=str(row.get("species_class", "")).capitalize(),
                    subject_description=row.get("subject_description", ""),
                    context_metadata=row.get("context_metadata", ""),
                    visual_cues=row.get("visual_cues", ""),
                    adversarial_section=adversarial_section,
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
    fields = ["reasoning", "answer"] if require_reasoning else ["answer"]
    assistant_parser = vf.XMLParser(fields=fields, answer_field="answer")
    judge_parser = vf.XMLParser(fields=["reasoning", "score"])

    # 2. Setup the Judge Client & Standard Rubric
    if judge_mode == "external":
        vf.ensure_keys([judge_api_key_var])
        judge_client = AsyncOpenAI(
            api_key=os.getenv(judge_api_key_var), base_url=judge_base_url
        )
    else:
        judge_client = None

    rubric = vf.Rubric(parser=assistant_parser)

    # Inject the second parser so reward functions can request it via kwargs
    rubric.add_class_object("judge_parser", judge_parser)

    # 3. Declarative Reward & Metric Functions
    async def parse_and_categorize(completion, info, state, parser, **kwargs) -> float:
        """Single-pass parsing to set categorization in the shared rollout state."""
        parsed = parser.parse_answer(completion)
        pred = parsed.strip() if parsed else None

        if pred:
            state["aww_outcome"] = "valid_attempt"
            state["aww_pred"] = pred
            ans_clean = str(info.get("target_label", "")).strip().lower()
            pred_clean = pred.lower()

            state["aww_correct"] = (ans_clean == pred_clean) or (
                ans_clean in pred_clean
            )
        else:
            state["aww_outcome"] = "format_failure"
            state["aww_pred"] = None
            state["aww_correct"] = False
        return 0.0

    async def valid_attempts(state, **kwargs) -> float:
        return 1.0 if state.get("aww_outcome") == "valid_attempt" else 0.0

    async def format_failures(state, **kwargs) -> float:
        return 1.0 if state.get("aww_outcome") == "format_failure" else 0.0

    async def accuracy_reward(state, **kwargs) -> float:
        return 1.0 if state.get("aww_correct", False) else 0.0

    async def hybrid_judge_reward(
        prompt, completion, answer, state, judge_parser, **kwargs
    ) -> float:
        """LLM-as-a-judge score using self-judge, external-judge, or none."""
        if judge_mode == "none":
            return 0.0

        question_text = (
            prompt[-1]["content"] if isinstance(prompt, list) else str(prompt)
        )

        if isinstance(completion, list):
            response_text = str(completion[-1].get("content", ""))
        else:
            response_text = str(completion)

        judge_text = JUDGE_PROMPT.format(
            question=question_text, answer=answer, response=response_text
        )

        verdict = ""
        try:
            if judge_mode == "self":
                policy_client = state["client"]
                policy_model = state["model"]

                judge_response = await policy_client.get_response(
                    prompt=[vf.UserMessage(content=judge_text)],
                    model=policy_model,
                    sampling_args={"temperature": 0.0, "max_tokens": 512},
                    state=state,
                )
                from verifiers.utils.response_utils import parse_response_message

                verdict_msgs = await parse_response_message(judge_response)
                verdict = str(verdict_msgs[-1].content or "")

            elif judge_mode == "external":
                judge_response = await judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_text}],
                    temperature=0.0,
                    max_tokens=512,
                )
                verdict = judge_response.choices[0].message.content

            if verdict:
                state["judge_reasoning"] = verdict
                logger.info(f"Judge reasoning: {verdict}")
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
    rubric.add_reward_func(parse_and_categorize, weight=0.0)  # Runs first
    rubric.add_metric(valid_attempts)
    rubric.add_metric(format_failures)
    rubric.add_reward_func(accuracy_reward, weight=accuracy_reward_weight)

    # Conditionally add the judge reward so it doesn't skew overall scale when ablated
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
