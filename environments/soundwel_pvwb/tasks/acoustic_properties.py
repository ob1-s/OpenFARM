"""Tier 1 — Acoustic Property Estimation (MCQ)."""

import random

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_ACOUSTIC_PROPERTY
from utils import (
    ACOUSTIC_BINS,
    LETTER_MAP,
    build_acoustic_mcq,
    build_audio_content_part,
)


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for acoustic property MCQ estimation."""

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        rng = random.Random(seed)
        features = list(ACOUSTIC_BINS.keys())

        # Filter rows that have at least one valid acoustic feature
        ds = ds.filter(lambda row: any(row.get(f) is not None for f in features))

        def transform(row):
            # Pick a random acoustic feature for this row that is not None
            valid_features = [f for f in features if row.get(f) is not None]
            feature = rng.choice(valid_features)
            value = row[feature]

            question, correct_letter, options = build_acoustic_mcq(feature, value)

            options_text = "\n".join(
                f"{LETTER_MAP[i]}. {opt}" for i, opt in enumerate(options)
            )

            prompt_text = PROMPT_ACOUSTIC_PROPERTY.format(
                question=question, options_text=options_text
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
