"""Tier 2 — Valence Classification (Positive / Negative)."""

import logging
import verifiers as vf

from utils import get_dataset_builder, FARMRubric

logger = logging.getLogger("verifiers.env.soundwel_pvwb.valence")


def build_task(
    modality: str = "audio",
    data_subset: str = "all",
    use_escape_hatch: bool = False,
    seed: int = 42,
    max_examples: int = -1,
    **kwargs,
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for binary valence classification."""

    dataset_builder = get_dataset_builder(
        task_name="valence_classification",
        modality=modality,
        data_subset=data_subset,
        use_escape_hatch=use_escape_hatch,
        seed=seed,
        max_examples=max_examples,
    )

    rubric = FARMRubric(use_escape_hatch=use_escape_hatch)

    return vf.SingleTurnEnv(
        dataset=dataset_builder,
        parser=rubric.parser,
        rubric=rubric,
    )