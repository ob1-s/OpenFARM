"""Tier 4 — Veterinary Diagnostic Reasoning (Judge-based)."""

import os
import re
import logging

import verifiers as vf
from datasets import Dataset, load_dataset, Audio
from openai import AsyncOpenAI

from prompts import SYSTEM_PROMPT_VET, PROMPT_VETERINARY, JUDGE_PROMPT_REASONING
from utils import build_audio_content_part

logger = logging.getLogger("verifiers.soundwel_pvwb.veterinary")


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for open-ended veterinary reasoning.

    Uses a composite rubric:
    - deterministic exact-match on call_type, valence, context (weight 0.5 total)
    - LLM-as-judge on the analysis block (weight 0.5)
    """

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        def transform(row):
            sex = row["Sex"]
            age = row["Age Category"]

            prompt_text = PROMPT_VETERINARY.format(
                sex=sex, age_category=age
            )

            media_part = build_audio_content_part(row, modality)
            prompt =[
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_VET}],
                },
                {
                    "role": "user",
                    "content":[
                        {"type": "text", "text": prompt_text},
                        media_part,
                    ],
                }
            ]

            # Ground truth for deterministic scoring
            valence = "Positive" if row["Valence"] == "Pos" else "Negative"
            answer = row["Type"]  # HF or LF — used as primary answer

            # Helper to safely format floats, as some acoustic features may be missing (None)
            def fmt_float(val):
                return f"{val:.2f}" if val is not None else "Unknown"

            # Extra metadata in info for judge and sub-scores
            info = {
                "call_type": row["Type"],
                "valence": valence,
                "context": row["Context Category"],
                "age": age,
                "sex": sex,
                "duration": fmt_float(row.get("Dur")),
                "q50": fmt_float(row.get("Q50%")),
                "fpeak": fmt_float(row.get("FPeak")),
                "amp_mod_rate": fmt_float(row.get("AmpModRate")),
            }

            return {"prompt": prompt, "answer": answer, "info": info}

        ds = ds.map(transform, remove_columns=ds.column_names)
        ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))
        return ds

    parser = vf.XMLParser(
        fields=["analysis", "call_type", "valence", "context"],
        answer_field="call_type",
    )
    judge_parser = vf.XMLParser(fields=["thought", "score"])

    # --- Deterministic sub-rewards ---

    async def call_type_accuracy(completion, answer, parser, **kw) -> float:
        parsed = parser.parse_answer(completion)
        if parsed is None:
            return 0.0
        return 1.0 if parsed.strip().upper() == answer.strip().upper() else 0.0

    async def valence_accuracy(completion, info, parser, **kw) -> float:
        text = parser._content_to_text(completion[-1].get("content", ""))
        parsed = parser.parse(text)
        pred = getattr(parsed, "valence", None)
        if pred is None:
            return 0.0
        expected = info["valence"]
        return 1.0 if pred.strip().lower() == expected.strip().lower() else 0.0

    async def context_accuracy(completion, info, parser, **kw) -> float:
        text = parser._content_to_text(completion[-1].get("content", ""))
        parsed = parser.parse(text)
        pred = getattr(parsed, "context", None)
        if pred is None:
            return 0.0
        expected = info["context"]
        return 1.0 if pred.strip().lower() == expected.strip().lower() else 0.0

    # --- Judge-based reasoning reward ---

    # We instantiate the client here to pass it into JudgeRubric
    judge_client = AsyncOpenAI(
        api_key=os.getenv(judge_api_key_var, ""),
        base_url=judge_base_url,
    )

    # Idiomatic approach: We request judge_client, judge_model, and judge_prompt 
    # directly from the JudgeRubric class objects via DI.
    async def reasoning_quality(
        completion, info, parser, judge_parser, judge_client, judge_model, judge_prompt, **kw
    ) -> float:
        text = parser._content_to_text(completion[-1].get("content", ""))
        parsed = parser.parse(text)
        analysis = getattr(parsed, "analysis", None)

        if not analysis or not analysis.strip():
            return 0.0

        # Inject actual numeric context to enforce anti-hallucination bounds
        filled_prompt = judge_prompt.format(
            call_type=info["call_type"],
            valence=info["valence"],
            context=info["context"],
            age=info["age"],
            sex=info["sex"],
            duration=info["duration"],
            q50=info["q50"],
            fpeak=info["fpeak"],
            amp_mod_rate=info["amp_mod_rate"],
            analysis=analysis.strip(),
        )

        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": filled_prompt}],
            )
            verdict = response.choices[0].message.content or ""

            logger.info(f"Judge verdict: {verdict}")

            parsed_verdict = judge_parser.parse(verdict)
            score_str = getattr(parsed_verdict, "score", None)

            if score_str:
                numbers = re.findall(
                    r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", score_str
                )
                if numbers:
                    score = float(numbers[-1])
                    return max(0.0, min(1.0, score / 10.0))
        except Exception as e:
            logger.warning(f"Judge error: {e}")

        return 0.0

    # --- Assemble rubric ---

    rubric = vf.JudgeRubric(
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt=JUDGE_PROMPT_REASONING,
        parser=parser,
    )

    # Attach the secondary parser used specifically by the judge
    rubric.add_class_object("judge_parser", judge_parser)
    
    # Add rewards with appropriate weights
    rubric.add_reward_func(call_type_accuracy, weight=0.2)
    rubric.add_reward_func(valence_accuracy, weight=0.2)
    rubric.add_reward_func(context_accuracy, weight=0.1)
    rubric.add_reward_func(reasoning_quality, weight=0.5)

    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.0)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        parser=parser,
        rubric=rubric,
    )