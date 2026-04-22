"""Porcine Vocal Welfare Benchmark (PVWB).

A multi-task benchmark suite evaluating Multimodal LLMs on pig vocalization
understanding, built on the Soundwel dataset.

Tasks span 4 difficulty tiers:

    Tier 1 — Acoustic Perception
        call_type_classification       HF/LF binary classification
        acoustic_property_estimation   MCQ on duration/pitch/peak frequency

    Tier 2 — Emotional & Welfare Assessment
        valence_classification         Positive/Negative valence
        pain_severity_ranking          Pairwise pain comparison

    Tier 3 — Contextual Reasoning
        context_classification         MCQ context identification
        cross_age_generalization       OOD valence across age groups

    Tier 4 — Veterinary Diagnostic Reasoning
        veterinary_reasoning           Open-ended analysis with LLM-as-judge
"""

import logging

import verifiers as vf

from tasks import TASK_REGISTRY, ALL_TASK_NAMES

logger = logging.getLogger("verifiers.soundwel_pvwb")


def load_environment(
    task: str | list[str] = "all",
    modality: str = "audio",
    seed: int = 42,
    max_examples_per_task: int = -1,
    judge_model: str = "gpt-4o-mini",
    judge_base_url: str = "https://api.openai.com/v1",
    judge_api_key_var: str = "OPENAI_API_KEY",
    **kwargs,
) -> vf.Environment:
    """Load the Porcine Vocal Welfare Benchmark environment.

    Args:
        task: Which task(s) to include. Pass ``"all"`` for the full suite,
            a single task name string, or a list of task name strings.
            Valid names: call_type_classification, acoustic_property_estimation,
            valence_classification, pain_severity_ranking, context_classification,
            cross_age_generalization, veterinary_reasoning.
        modality: Input modality — ``"audio"`` (raw wav) or ``"spectrogram"``
            (spectrogram image).
        seed: Random seed for dataset shuffling and MCQ generation.
        max_examples_per_task: Maximum examples per task (-1 for all).
        judge_model: Model for LLM-as-judge in Tier 4.
        judge_base_url: API base URL for the judge model.
        judge_api_key_var: Environment variable name holding the judge API key.
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
            raise ValueError(
                f"Unknown task: {t!r}. Valid tasks: {ALL_TASK_NAMES}"
            )

    # --- Check if judge is needed ---
    needs_judge = any(
        TASK_REGISTRY[t][1] for t in selected_tasks  # [1] = requires_judge
    )
    if needs_judge:
        vf.ensure_keys([judge_api_key_var])

    # --- Build sub-environments ---
    envs: list[vf.Environment] = []
    env_names: list[str] = []

    # Common kwargs for all task builders
    common_kwargs = {
        "modality": modality,
        "seed": seed,
        "max_examples": max_examples_per_task,
    }

    # Judge kwargs only for tasks that need them
    judge_kwargs = {
        "judge_model": judge_model,
        "judge_base_url": judge_base_url,
        "judge_api_key_var": judge_api_key_var,
    }

    for task_name in selected_tasks:
        builder_fn, requires_judge = TASK_REGISTRY[task_name]

        task_kwargs = {**common_kwargs}
        if requires_judge:
            task_kwargs.update(judge_kwargs)

        logger.info(f"Building task: {task_name}")
        env = builder_fn(**task_kwargs)
        envs.append(env)
        env_names.append(task_name)

    # --- Return single env or group ---
    if len(envs) == 1:
        return envs[0]

    return vf.EnvGroup(envs=envs, env_names=env_names)
