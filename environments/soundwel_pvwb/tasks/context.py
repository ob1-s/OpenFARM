"""Tier 3 — Context Classification (MCQ)."""

import random

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_CONTEXT
from utils import (
    LETTER_MAP,
    NEGATIVE_CONTEXTS,
    POSITIVE_CONTEXTS,
    build_audio_content_part,
    sample_distractors,
)


# All context categories used in the dataset
ALL_CONTEXT_CATEGORIES = sorted(set(NEGATIVE_CONTEXTS + POSITIVE_CONTEXTS))


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for context classification MCQ.

    Distractors are sampled from the same valence bucket to prevent the
    model from trivially eliminating options by valence alone.
    """

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        rng = random.Random(seed)

        def transform(row):
            correct = row["Context Category"]
            age = row["Age Category"]

            # Same-valence distractor pool
            if correct in NEGATIVE_CONTEXTS:
                pool = NEGATIVE_CONTEXTS
            elif correct in POSITIVE_CONTEXTS:
                pool = POSITIVE_CONTEXTS
            else:
                pool = ALL_CONTEXT_CATEGORIES

            distractors = sample_distractors(correct, pool, n=3, rng=rng)
            options = [correct] + distractors
            rng.shuffle(options)

            correct_idx = options.index(correct)
            correct_letter = LETTER_MAP[correct_idx]

            options_text = "\n".join(
                f"{LETTER_MAP[i]}. {opt}" for i, opt in enumerate(options)
            )

            prompt_text = PROMPT_CONTEXT.format(
                age_category=age, options_text=options_text
            )

            media_part = build_audio_content_part(row, modality)
            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_ETHOLOGIST}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        media_part,
                    ],
                }
            ]
            return {"prompt": prompt, "answer": correct_letter}

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
        return 1.0 if parsed.strip().upper() == answer.strip().upper() else 0.0

    rubric = vf.Rubric(funcs=[accuracy_reward], parser=parser)
    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.0)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        parser=parser,
        rubric=rubric,
    )
