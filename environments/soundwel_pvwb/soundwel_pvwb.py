"""Porcine Vocal Welfare Benchmark (PVWB).

A multi-task benchmark suite evaluating Multimodal LLMs on pig vocalization
understanding, built on the Soundwel dataset.

Tasks span 2 core difficulty tiers:

    Tier 1 — Acoustic Perception
        call_type_classification       HF/LF binary classification

    Tier 2 — Emotional & Welfare Assessment
        valence_classification         Positive/Negative valence
"""

import logging
import verifiers as vf

from tasks import TASK_REGISTRY, ALL_TASK_NAMES

logger = logging.getLogger("verifiers.soundwel_pvwb")


def load_environment(
    task: str | list[str] = "valence_classification",
    modality: str = "audio",
    data_subset: str = "all",
    use_escape_hatch: bool = False,
    seed: int = 42,
    max_examples_per_task: int = -1,
    **kwargs,
) -> vf.Environment:
    """Load the Porcine Vocal Welfare Benchmark environment.

    Args:
        task: Which task(s) to include. Pass ``"all"`` for the full suite,
            a single task name string, or a list of task name strings.
            Valid names: call_type_classification, valence_classification.
        modality: Input modality — ``"audio"`` (raw wav), ``"vision"`` 
            (spectrogram image), or ``"multimodal"`` (both).
        data_subset: Stratification tier. Valid names: ``"all"``, ``"rich"``, 
            ``"hard_short"``, ``"hard_noisy"``, ``"average"``.
        use_escape_hatch: If True, injects instructions and allows the model 
            to classify poor quality samples as UNINTELLIGIBLE.
        seed: Random seed for dataset shuffling.
        max_examples_per_task: Maximum examples per task (-1 for all).
        **kwargs: Additional keyword arguments forwarded to sub-environments.

    Returns:
        A ``vf.EnvGroup`` containing one ``SingleTurnEnv`` per selected task,
        or a single ``SingleTurnEnv`` if only one task is selected.
    """
    # --- Resolve task selection ---
    if task == "all":
        selected_tasks = list(ALL_TASK_NAMES)
    elif isinstance(task, str):
        selected_tasks = [task]
    else:
        selected_tasks = list(task)

    # Validate task names
    for t in selected_tasks:
        if t not in TASK_REGISTRY:
            raise ValueError(f"Unknown task: {t!r}. Valid tasks: {ALL_TASK_NAMES}")

    # --- Build sub-environments ---
    envs: list[vf.Environment] = []
    env_names: list[str] = []

    # Common kwargs for all task builders
    common_kwargs = {
        "modality": modality,
        "data_subset": data_subset,
        "use_escape_hatch": use_escape_hatch,
        "seed": seed,
        "max_examples": max_examples_per_task,
    }

    for task_name in selected_tasks:
        builder_fn = TASK_REGISTRY[task_name]

        logger.info(f"Building task: {task_name}")
        env = builder_fn(**common_kwargs)
        envs.append(env)
        env_names.append(task_name)

    # --- Return single env or group ---
    if len(envs) == 1:
        return envs[0]

    return vf.EnvGroup(envs=envs, env_names=env_names)