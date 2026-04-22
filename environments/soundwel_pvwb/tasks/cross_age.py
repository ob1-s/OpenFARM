"""Tier 3 — Cross-Age Generalization (OOD Valence Classification)."""

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_VALENCE
from utils import build_audio_content_part


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    eval_age: str = "GrFinishing",
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for cross-age valence generalization.

    Uses a different age group for eval than the majority training
    distribution (Piglet). This tests whether the model generalises
    across developmental stages where vocal tract anatomy differs.

    Args:
        eval_age: Age category to use as the OOD eval set. One of
            "GrFinishing", "Weaner", or "Piglet".
    """

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        # Filter to the target age group for OOD evaluation
        ds = ds.filter(lambda row: row["Age Category"] == eval_age)

        def transform(row):
            media_part = build_audio_content_part(row, modality)
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
                }
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

    rubric = vf.Rubric(funcs=[accuracy_reward], parser=parser)
    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=0.0)

    return vf.SingleTurnEnv(
        dataset=build_dataset,
        parser=parser,
        rubric=rubric,
    )
