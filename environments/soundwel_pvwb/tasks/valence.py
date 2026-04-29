"""Tier 2 — Valence Classification (Positive / Negative)."""

import logging

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_VALENCE
from utils import build_audio_content_part


logger = logging.getLogger("verifiers.env.soundwel_pvwb.valence")


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for binary valence classification."""

    def build_dataset() -> Dataset:
        ds = load_dataset("oliveirabruno01/soundwel-pig-vocalizations", split="train")
        ds = ds.cast_column("audio", Audio(decode=False))

        def transform(row):
            media_part = build_audio_content_part(row, modality)
            # Map dataset labels to full words for clarity
            valence = "Positive" if row["Valence"] == "Pos" else "Negative"
            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_ETHOLOGIST}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_VALENCE},
                        media_part,
                    ],
                },
            ]
            return {"prompt": prompt, "answer": valence}

        ds = ds.map(transform, remove_columns=ds.column_names)
        ds = ds.shuffle(seed=seed)
        if max_examples > 0:
            ds = ds.select(range(min(max_examples, len(ds))))
        return ds

    parser = vf.XMLParser(fields=["reasoning", "answer"], answer_field="answer")

    async def accuracy_reward(completion, answer, parser, **kw) -> float:
        parsed = parser.parse_answer(completion)
        if parsed is None:
            return 0.0
        return 1.0 if parsed.strip().lower() == answer.strip().lower() else 0.0

    # Group reward for balanced accuracy (tracks class-level performance)
    async def balanced_accuracy(completions, answers, **kw) -> list[float]:
        """Compute per-group balanced accuracy as a metric.

        Returns the balanced accuracy value for each rollout in the group
        (same value for all, since it is a group-level statistic).
        """
        correct_pos = 0
        total_pos = 0
        correct_neg = 0
        total_neg = 0

        for comp, ans in zip(completions, answers):
            parsed = parser.parse_answer(comp)
            pred = (parsed or "").strip().lower()
            if ans.strip().lower() == "positive":
                total_pos += 1
                if pred == "positive":
                    correct_pos += 1
            else:
                total_neg += 1
                if pred == "negative":
                    correct_neg += 1

        recall_pos = correct_pos / total_pos if total_pos > 0 else 0.0
        recall_neg = correct_neg / total_neg if total_neg > 0 else 0.0
        ba = (recall_pos + recall_neg) / 2.0

        return [ba] * len(completions)

    rubric = vf.Rubric(funcs=[accuracy_reward], parser=parser)
    rubric.add_metric(balanced_accuracy)

    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.0)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        parser=parser,
        rubric=rubric,
    )
