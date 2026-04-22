import os
import re
import json
import logging
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI

logger = logging.getLogger("verifiers.farm_v0_confounders")

# 1. Structured Judge Prompt
JUDGE_PROMPT = """Evaluate the following animal welfare assessment.

[QUESTION (the exact words that prompted the assessment; already in context.)]
{question}

[GOLD STANDARD REFERENCE]
{answer}

[MODEL RESPONSE]
{response}

Compare the Model's analysis to the Gold Standard. 
Score the model from 0 to 10 based on accuracy and mechanism identification.
Penalize mistakes by severity.

Respond in the following format:
<thought>
Briefly explain your reasoning for the score, comparing the model response to the gold standard.
</thought>
<score>
A single number from 0 to 10
</score>"""


def load_environment(
    dataset_id: str = "oliveirabruno01/animal-welfare-confounders",
    split: str = "train",
    max_examples: int = -1,
    judge_model: str = "gpt-4o",
    judge_api_key_var: str = "OPENAI_API_KEY",
    judge_base_url: str = "https://api.openai.com/v1",
    seed: int = 42,
    **kwargs,
) -> vf.Environment:
    vf.ensure_keys([judge_api_key_var])
    
    def build_dataset():
        ds = load_dataset(dataset_id, split=split)
        if seed:
            ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))

        def map_to_standard(row):
            # 2. Robust Extraction: Handle nested metadata in the 'info' column
            info_data = row.get("info", {})
            if isinstance(info_data, str):
                try:
                    info_data = json.loads(info_data)
                except:
                    info_data = {}
            
            target = row.get("target_category") or info_data.get("target_category", "Unknown")
            reasoning = row.get("sota_judge_summary") or info_data.get("sota_judge_summary", "No reasoning provided")
            
            gold_standard = f"Target Category: {target}\nReasoning: {reasoning}"

            question = (
                f'[SPECIES]\n{row.get("species", "")}\n\n'
                f'[CONTEXT]\n{row.get("context", "")}\n\n'
                f'[VISUAL]\n{row.get("visual_cues", "")}\n\n'
                f'[TASK]\nAssess pain level.'
            )
            return {
                "question": question,
                "answer": gold_standard,
                "info": info_data
            }
        
        # remove_columns ensures we don't have schema conflicts
        return ds.map(map_to_standard, remove_columns=ds.column_names)

    parser = vf.Parser()
    judge_parser = vf.XMLParser(fields=["thought", "score"])

    judge_client = AsyncOpenAI(
        api_key=os.getenv(judge_api_key_var),
        base_url=judge_base_url
    )

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT,
        parser=parser,
    )

    async def ethology_reward_func(judge, prompt, completion, answer, state, judge_parser) -> float:
        """
        Idiomatic Reward Function with Structured Judge Output.
        """
        # Get raw verdict from judge
        verdict = await judge(prompt, completion, answer, state)
        
        # Use the injected judge_parser to extract the structured fields
        parsed = judge_parser.parse(verdict)
        score_str = parsed.score
        
        formmatted_judge_prompt = JUDGE_PROMPT.format(
            question=prompt[-1]["content"],
            answer=answer,
            response=completion
        )

        # logger.info(f"========\nJudge Prompt: {formmatted_judge_prompt}\n")
        logger.info(f"========\nJudge Thought: {parsed.thought}\nJudge Score: {score_str}\n========")

        if score_str:
            try:
                numbers = re.findall(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", score_str)
                if numbers:
                    score = float(numbers[-1])
                    return max(0.0, min(1.0, score / 10.0))
            except ValueError:
                pass
        return 0.0
    
    rubric.add_class_object("judge_parser", judge_parser)
    rubric.add_reward_func(ethology_reward_func)

    return vf.SingleTurnEnv(
        eval_dataset=build_dataset,
        parser=parser,
        rubric=rubric,
        **kwargs
    )