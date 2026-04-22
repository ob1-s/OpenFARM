"""Task registry for the Soundwel PVWB benchmark suite.

Each task module exposes a ``build_task(...)`` function that returns a
``vf.SingleTurnEnv`` ready for inclusion in an ``EnvGroup``.
"""

from tasks.call_type import build_task as build_call_type_task
from tasks.acoustic_properties import build_task as build_acoustic_properties_task
from tasks.valence import build_task as build_valence_task
from tasks.pain_ranking import build_task as build_pain_ranking_task
from tasks.context import build_task as build_context_task
from tasks.cross_age import build_task as build_cross_age_task
from tasks.veterinary import build_task as build_veterinary_task

# Canonical registry: task_name -> (builder_fn, requires_judge)
TASK_REGISTRY: dict[str, tuple] = {
    "call_type_classification": (build_call_type_task, False),
    "acoustic_property_estimation": (build_acoustic_properties_task, False),
    "valence_classification": (build_valence_task, False),
    "pain_severity_ranking": (build_pain_ranking_task, False),
    "context_classification": (build_context_task, False),
    "cross_age_generalization": (build_cross_age_task, False),
    "veterinary_reasoning": (build_veterinary_task, True),
}

ALL_TASK_NAMES = list(TASK_REGISTRY.keys())

__all__ = [
    "TASK_REGISTRY",
    "ALL_TASK_NAMES",
]
