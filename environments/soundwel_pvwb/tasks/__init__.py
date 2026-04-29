"""Task registry for the Soundwel PVWB benchmark suite.

Each task module exposes a ``build_task(...)`` function that returns a
``vf.SingleTurnEnv`` ready for inclusion in an ``EnvGroup``.
"""

from tasks.call_type import build_task as build_call_type_task
from tasks.valence import build_task as build_valence_task

# Canonical registry: task_name -> builder_fn
TASK_REGISTRY = {
    "call_type_classification": build_call_type_task,
    "valence_classification": build_valence_task,
}

ALL_TASK_NAMES = list(TASK_REGISTRY.keys())

__all__ = [
    "TASK_REGISTRY",
    "ALL_TASK_NAMES",
]
