"""Porcine Vocal Welfare Benchmark (PVWB).

A multi-task benchmark suite evaluating Multimodal LLMs on pig vocalization understanding.
Tasks span 2 core difficulty tiers:
    Tier 1 — Acoustic Perception (call_type_classification)
    Tier 2 — Emotional Assessment (valence_classification)
"""

import logging
import verifiers as vf

from utils import (
    build_multimodal_content_parts,
    get_dataset,
    balance_and_shuffle,
)

logger = logging.getLogger("verifiers.soundwel_pvwb")

# ============================================================================
# Declarative Prompt Templates (0-Shot Lean Baselines)
# ============================================================================

SYSTEM_PROMPT_ETHOLOGIST = (
    "You are an expert ethologist specializing in swine welfare and "
    "vocalization analysis."
)

MODALITY_INSTRUCTIONS = {
    "audio": "Listen to the provided pig vocalization.",
    "vision": "Look at the provided spectrogram of a pig vocalization.",
    "multimodal": "You are provided with BOTH an audio recording and its corresponding spectrogram image of a pig vocalization. Using both the audio and the visual data,",
}

ESCAPE_HATCH_NOTES = {
    "audio": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if the audio is completely missing, entirely silent, or too short to hear anything, you must classify it as UNINTELLIGIBLE.\n\n",
    "vision": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if the spectrogram is completely empty or missing, you must classify it as UNINTELLIGIBLE.\n\n",
    "multimodal": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if BOTH the audio and the spectrogram are completely missing or unintelligible, you must classify it as UNINTELLIGIBLE.\n\n",
}

CALL_TYPE_TEMPLATE = """{modality_instruction} Is this a High Frequency (HF) call or a Low Frequency (LF) call?

{escape_hatch_note}{format_instructions}"""

VALENCE_TEMPLATE = """{modality_instruction} Is the emotional valence of this animal Positive or Negative?

{escape_hatch_note}{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<reasoning>
[your step-by-step analysis]
</reasoning>
<answer>
[{options}]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[{options}]
</answer>"""

# ============================================================================
# Shared Rubric
# ============================================================================

REFUSAL_KEYWORDS = ["missing", "empty", "audio file", "please provide"]


class FARMRubric(vf.Rubric):
    """Centralized F.A.R.M. rubric handling granular parsing and validation states."""

    def __init__(self, use_escape_hatch: bool = False, require_reasoning: bool = False, format_reward_weight: float = 0.0):
        fields = ["reasoning", "answer"] if require_reasoning else ["answer"]
        self.parser = vf.XMLParser(fields=fields, answer_field="answer")
        super().__init__(parser=self.parser)
        self.use_escape_hatch = use_escape_hatch

        # 1. High-priority parser executes first (weight=0)
        self.add_reward_func(self.parse_and_categorize, weight=0.0)

        # 2. Granular tracker metrics (weight=0)
        self.add_metric(self.clean_refusals)
        self.add_metric(self.messy_refusals)
        self.add_metric(self.valid_attempts)

        # 3. Standard Accuracy Reward
        self.add_reward_func(self.accuracy_reward, weight=1.0)

        # 4. Built-in Format Check (weight=0, tracked as metric)
        format_reward = self.parser.get_format_reward_func()
        format_reward.__name__ = "format_reward"
        self.add_reward_func(format_reward, weight=format_reward_weight)

    async def parse_and_categorize(self, completion, answer, state):
        """Single-pass parsing to set the categorization in the shared rollout state."""
        raw_text = ""
        for msg in completion:
            if getattr(msg, "role", msg.get("role", "")) == "assistant":
                raw_text += str(getattr(msg, "content", msg.get("content", "")))

        parsed = self.parser.parse_answer(completion)
        pred = parsed.strip().upper() if parsed else None

        if pred:
            if self.use_escape_hatch and "UNINTELLIGIBLE" in pred:
                state["farm_outcome"] = "clean_refusal"
            else:
                state["farm_outcome"] = "valid_attempt"
                # Normalize prediction
                if "HF" in pred:
                    final_pred = "HF"
                elif "LF" in pred:
                    final_pred = "LF"
                elif "POSITIVE" in pred or "POS" in pred:
                    final_pred = "Positive"
                elif "NEGATIVE" in pred or "NEG" in pred:
                    final_pred = "Negative"
                else:
                    final_pred = pred

                state["farm_pred"] = final_pred
                state["farm_correct"] = final_pred.lower() == answer.strip().lower()
        else:
            if any(kw in raw_text.lower() for kw in REFUSAL_KEYWORDS):
                state["farm_outcome"] = "messy_refusal"
            else:
                state["farm_outcome"] = "format_failure"
        return 0.0

    async def clean_refusals(self, state):
        return 1.0 if state.get("farm_outcome") == "clean_refusal" else 0.0

    async def messy_refusals(self, state):
        return 1.0 if state.get("farm_outcome") == "messy_refusal" else 0.0

    async def valid_attempts(self, state):
        return 1.0 if state.get("farm_outcome") == "valid_attempt" else 0.0

    async def accuracy_reward(self, state):
        return 1.0 if state.get("farm_correct", False) else 0.0


# ============================================================================
# Task Builders
# ============================================================================


def build_call_type_env(
    modality: str,
    use_escape_hatch: bool,
    max_examples: int,
    dataset_name: str,
    train_split: str,
    test_split: str,
    seed: int,
    target_pad_seconds: float,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    require_reasoning: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for binary HF/LF classification."""

    def get_dataset_builder(split: str):
        def build():
            ds = get_dataset(dataset_name=dataset_name, split=split)
            ds = ds.filter(lambda x: x["Type"] is not None)

            def transform(row):
                age = row.get("Age Category", "pig")

                # Use provided overrides or default templates
                sys_prompt = (
                    system_prompt_override
                    if system_prompt_override is not None
                    else SYSTEM_PROMPT_ETHOLOGIST
                )
                user_template = (
                    user_prompt_override
                    if user_prompt_override is not None
                    else CALL_TYPE_TEMPLATE
                )

                format_instr = (
                    FORMAT_WITH_REASONING
                    if require_reasoning
                    else FORMAT_WITHOUT_REASONING
                )

                # Format the text prompt declaratively
                prompt_text = user_template.format(
                    modality_instruction=MODALITY_INSTRUCTIONS[modality],
                    escape_hatch_note=ESCAPE_HATCH_NOTES[modality]
                    if use_escape_hatch
                    else "",
                    format_instructions=format_instr.format(
                        options="HF, LF, or UNINTELLIGIBLE"
                        if use_escape_hatch
                        else "HF or LF"
                    ),
                    age_category=age,  # Extra context available for custom override prompts
                )

                # Build the multimodal payload
                media_parts = build_multimodal_content_parts(
                    row, modality, target_pad_seconds
                )
                prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_prompt}],
                    },
                    {
                        "role": "user",
                        "content": media_parts
                        + [{"type": "text", "text": prompt_text}],
                    },
                ]

                return {
                    "prompt": prompt,
                    "answer": row["Type"],
                    "info": {
                        "age_category": age,
                        "duration_raw": row.get("Dur"),
                        "am_rate": row.get("AMRate"),
                    },
                }

            logger.info(f"Applying mapping and formatting prompts (split={split})...")
            ds = ds.map(transform, remove_columns=ds.column_names, num_proc=1)
            return balance_and_shuffle(
                ds, "HF", "LF", max_examples, seed, strategy=balancing_strategy
            )

        return build

    rubric = FARMRubric(
        use_escape_hatch=use_escape_hatch, require_reasoning=require_reasoning, format_reward_weight=format_reward_weight
    )
    return vf.SingleTurnEnv(
        dataset=get_dataset_builder(train_split),
        eval_dataset=get_dataset_builder(test_split),
        parser=rubric.parser,
        rubric=rubric,
    )


def build_valence_env(
    modality: str,
    use_escape_hatch: bool,
    max_examples: int,
    dataset_name: str,
    train_split: str,
    test_split: str,
    seed: int,
    target_pad_seconds: float,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    require_reasoning: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
) -> vf.SingleTurnEnv:
    """Build a SingleTurnEnv for binary Positive/Negative valence classification."""

    def get_dataset_builder(split: str):
        def build():
            ds = get_dataset(dataset_name=dataset_name, split=split)
            ds = ds.filter(lambda x: x["Valence"] is not None)

            def transform(row):
                age = row.get("Age Category", "pig")

                sys_prompt = (
                    system_prompt_override
                    if system_prompt_override is not None
                    else SYSTEM_PROMPT_ETHOLOGIST
                )
                user_template = (
                    user_prompt_override
                    if user_prompt_override is not None
                    else VALENCE_TEMPLATE
                )

                format_instr = (
                    FORMAT_WITH_REASONING
                    if require_reasoning
                    else FORMAT_WITHOUT_REASONING
                )
                prompt_text = user_template.format(
                    modality_instruction=MODALITY_INSTRUCTIONS[modality],
                    escape_hatch_note=ESCAPE_HATCH_NOTES[modality]
                    if use_escape_hatch
                    else "",
                    format_instructions=format_instr.format(
                        options="Positive, Negative, or UNINTELLIGIBLE"
                        if use_escape_hatch
                        else "Positive or Negative"
                    ),
                    age_category=age,
                )

                media_parts = build_multimodal_content_parts(
                    row, modality, target_pad_seconds
                )
                prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_prompt}],
                    },
                    {
                        "role": "user",
                        "content": media_parts
                        + [{"type": "text", "text": prompt_text}],
                    },
                ]

                answer = "Positive" if row["Valence"] == "Pos" else "Negative"

                return {
                    "prompt": prompt,
                    "answer": answer,
                    "info": {
                        "age_category": age,
                        "duration_raw": row.get("Dur"),
                        "am_rate": row.get("AMRate"),
                    },
                }

            logger.info(f"Applying mapping and formatting prompts (split={split})...")
            ds = ds.map(transform, remove_columns=ds.column_names, num_proc=1)
            return balance_and_shuffle(
                ds, "Positive", "Negative", max_examples, seed, strategy=balancing_strategy
            )

        return build

    rubric = FARMRubric(
        use_escape_hatch=use_escape_hatch, require_reasoning=require_reasoning, format_reward_weight=format_reward_weight
    )
    return vf.SingleTurnEnv(
        dataset=get_dataset_builder(train_split),
        eval_dataset=get_dataset_builder(test_split),
        parser=rubric.parser,
        rubric=rubric,
    )


# ============================================================================
# Main Router
# ============================================================================

TASK_BUILDERS = {
    "call_type_classification": build_call_type_env,
    "valence_classification": build_valence_env,
}


def load_environment(
    task: str | list[str] = "valence_classification",
    modality: str = "audio",
    use_escape_hatch: bool = False,
    max_examples_per_task: int = -1,
    dataset_name: str = "oliveirabruno01/soundwel-pig-vocalizations",
    train_split: str = "train",
    test_split: str = "test",
    seed: int = 42,
    target_pad_seconds: float = 3.595,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    require_reasoning: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
    **kwargs,
) -> vf.Environment:
    """Load the Porcine Vocal Welfare Benchmark environment.

    Args:
        task: Which task(s) to include. Pass "all" for the full suite,
            a single task name string, or a list of task name strings.
        modality: "audio" (wav), "vision" (spectrogram image), or "multimodal" (both).
        use_escape_hatch: If True, allow the model to classify poor quality samples as UNINTELLIGIBLE.
        max_examples_per_task: Maximum examples per task (-1 for all).
        dataset_name: HuggingFace dataset ID or local path.
        train_split: Split to use for training.
        test_split: Split to use for testing.
        seed: Random seed for dataset shuffling.
        target_pad_seconds: Centrally pad audio arrays to this length.
        system_prompt_override: Optional custom string to replace system prompt.
        user_prompt_override: Optional custom template string for the user task prompt.
        require_reasoning: Whether to require reasoning in the model's response.
        format_reward_weight: Weight for the formatting reward.
        balancing_strategy: "proportional" or "balanced". "proportional" to class balance, "balanced" to balance classes.
    """
    if task == "all":
        selected_tasks = list(TASK_BUILDERS.keys())
    elif isinstance(task, str):
        selected_tasks = [task]
    else:
        selected_tasks = list(task)

    for t in selected_tasks:
        if t not in TASK_BUILDERS:
            raise ValueError(
                f"Unknown task: {t!r}. Valid tasks: {list(TASK_BUILDERS.keys())}"
            )

    common_kwargs = {
        "modality": modality,
        "use_escape_hatch": use_escape_hatch,
        "seed": seed,
        "max_examples": max_examples_per_task,
        "dataset_name": dataset_name,
        "train_split": train_split,
        "test_split": test_split,
        "target_pad_seconds": target_pad_seconds,
        "system_prompt_override": system_prompt_override,
        "user_prompt_override": user_prompt_override,
        "require_reasoning": require_reasoning,
        "format_reward_weight": format_reward_weight,
        "balancing_strategy": balancing_strategy,
    }

    envs = []
    for task_name in selected_tasks:
        logger.info(f"Building task: {task_name}")
        env = TASK_BUILDERS[task_name](**common_kwargs)
        envs.append(env)

    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=selected_tasks)
