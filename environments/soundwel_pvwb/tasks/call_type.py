"""Tier 1 — Call Type Classification (HF / LF)."""

import verifiers as vf
from datasets import Dataset, load_dataset, Audio

from prompts import SYSTEM_PROMPT_ETHOLOGIST, PROMPT_CALL_TYPE
from utils import build_audio_content_part


def build_task(
    modality: str = "audio",
    seed: int = 42,
    max_examples: int = -1,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for binary HF/LF call type classification."""

    def build_dataset() -> Dataset:
        ds = load_dataset(
            "oliveirabruno01/soundwel-pig-vocalizations", split="train"
        )
        ds = ds.cast_column("audio", Audio(decode=False))

        def transform(row):
            media_part = build_audio_content_part(row, modality)
            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_ETHOLOGIST}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT_CALL_TYPE},
                        media_part,
                    ],
                }
            ]
            return {"prompt": prompt, "answer": row["Type"]}

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
