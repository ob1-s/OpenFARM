"""Tier 2 — Pain Severity Ranking (Pairwise Comparison)."""

import random

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_PAIN_RANKING
from utils import HIGH_PAIN_CONTEXTS, LOW_PAIN_CONTEXTS, build_audio_content_part


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = 200,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for pairwise pain severity ranking.

    Constructs pairs of vocalizations: one from a high-pain context and one
    from a low-pain/positive context. The model must identify which is in
    more distress.
    """

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        rng = random.Random(seed)

        # Split rows by pain level
        high_pain_rows = []
        low_pain_rows = []
        for i in range(len(ds)):
            ctx = ds[i]["Context Category"]
            if ctx in HIGH_PAIN_CONTEXTS:
                high_pain_rows.append(i)
            elif ctx in LOW_PAIN_CONTEXTS:
                low_pain_rows.append(i)

        # Generate pairs
        rng.shuffle(high_pain_rows)
        rng.shuffle(low_pain_rows)

        n_pairs = min(len(high_pain_rows), len(low_pain_rows))
        if max_examples > 0:
            n_pairs = min(n_pairs, max_examples)

        pairs = []
        for i in range(n_pairs):
            hi_idx = high_pain_rows[i % len(high_pain_rows)]
            lo_idx = low_pain_rows[i % len(low_pain_rows)]

            # Randomly assign high-pain to A or B
            if rng.random() < 0.5:
                a_idx, b_idx = hi_idx, lo_idx
                correct = "A"
            else:
                a_idx, b_idx = lo_idx, hi_idx
                correct = "B"

            row_a = ds[a_idx]
            row_b = ds[b_idx]

            media_a = build_audio_content_part(row_a, modality)
            media_b = build_audio_content_part(row_b, modality)

            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_ETHOLOGIST}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_PAIN_RANKING},
                        media_a,
                        media_b,
                    ],
                }
            ]
            pairs.append({"prompt": prompt, "answer": correct})

        return Dataset.from_list(pairs)

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
