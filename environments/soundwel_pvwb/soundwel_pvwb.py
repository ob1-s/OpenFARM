"""Porcine Vocal Welfare Benchmark (PVWB).

A multi-task benchmark suite evaluating Multimodal LLMs on pig vocalization understanding.
Tasks span 4 core evaluation tiers:
    Tier 1 — Acoustic Perception (call_type_classification)
    Tier 2 — Emotional Assessment (valence_classification)
    Tier 3 — Context Identification (context_classification)
    Tier 4 — Age Identification (age_classification)
"""

import logging
import re
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
    "audio": "<audio>Listen to the provided pig vocalization.",
    "vision": "Look at the provided spectrogram of a pig vocalization.",
    "multimodal": "<audio>You are provided with BOTH an audio recording and its corresponding spectrogram image of a pig vocalization. Using both the audio and the visual data,",
    "text": "Review the tabular acoustic and demographic data below.",
}

ESCAPE_HATCH_NOTES = {
    "audio": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if the audio is completely missing, entirely silent, or too short to hear anything, you must classify it as UNINTELLIGIBLE.\n\n",
    "vision": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if the spectrogram is completely empty or missing, you must classify it as UNINTELLIGIBLE.\n\n",
    "multimodal": "NOTE: The audio may be padded with silence to assist processing. If you hear *any* pig vocalization, evaluate it. Only if BOTH the audio and the spectrogram are completely missing or unintelligible, you must classify it as UNINTELLIGIBLE.\n\n",
    "text": "NOTE: If the tabular data is completely missing or empty, you must classify it as UNINTELLIGIBLE.\n\n",
}

CALL_TYPE_TEMPLATE = """{modality_instruction} Is this a High Frequency (HF) call or a Low Frequency (LF) call?
{tabular_data}
{escape_hatch_note}{format_instructions}"""

VALENCE_TEMPLATE = """{modality_instruction} Is the emotional valence of this animal Positive or Negative?
{tabular_data}
{escape_hatch_note}{format_instructions}"""

CONTEXT_TEMPLATE = """{modality_instruction} What is the context category of this vocalization? (e.g., Isolation, Reunion, Castration, etc.)
{tabular_data}
{escape_hatch_note}{format_instructions}"""

AGE_TEMPLATE = """{modality_instruction} What is the age category of the pig? (Piglet, Weaner, or GrFinishing)
{tabular_data}
{escape_hatch_note}{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<reasoning>[your step-by-step analysis]
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


def format_tabular_data(
    row: dict, label_column: str, extra_exclude: set | None = None
) -> str:
    """Format available dataset columns into a markdown list for prompt injection.
    Explicitly excludes the target label and any direct categorical proxies."""

    # Always exclude categorical labels that give away the task
    exclude = {
        "Valence",
        "Context",
        "Context Category",
        "Type",
        "Call Type (Freq)",
        label_column,
    }

    if extra_exclude:
        exclude.update(extra_exclude)

    # Safe list of demographics and raw acoustic features
    keys = [
        "Age Category",
        "Sex",
        "Dur",
        "F0Mean",
        "Fpeak",
        "WienEntrMean",
        "Harmonicity",
        "Q25",
        "Q50",
        "Q75",
        "AMVar",
        "AMRate",
        "AMExtent",
        "DFMin",
    ]

    lines = []
    for k in keys:
        if k not in exclude and k in row and row[k] is not None:
            lines.append(f"- {k}: {row[k]}")

    if not lines:
        return ""
    return "\n[Tabular Data]\n" + "\n".join(lines) + "\n"


class FARMRubric(vf.Rubric):
    """Centralized F.A.R.M. rubric handling granular parsing and validation states."""

    def __init__(
        self,
        classes: list[str] | None = None,
        use_escape_hatch: bool = False,
        require_reasoning: bool = False,
        format_reward_weight: float = 0.0,
        extra_refusal_keywords: list[str] | None = None,
    ):
        fields = ["reasoning", "answer"] if require_reasoning else ["answer"]
        self.parser = vf.XMLParser(fields=fields, answer_field="answer")
        super().__init__(parser=self.parser)
        self.use_escape_hatch = use_escape_hatch
        self.classes = classes or []

        self.refusal_keywords = REFUSAL_KEYWORDS.copy()
        if extra_refusal_keywords:
            self.refusal_keywords.extend([k.lower() for k in extra_refusal_keywords])

        # 1. High-priority parser executes first (weight=0)
        self.add_reward_func(self.parse_and_categorize, weight=0.0)

        # 2. Granular tracker metrics (weight=0)
        self.add_metric(self.clean_refusals)
        self.add_metric(self.messy_refusals)
        self.add_metric(self.valid_attempts)
        
        # Add dynamic trackers to surface mode-collapse directly in the Verifiers TUI
        for c in self.classes:
            self.add_metric(self._make_pred_tracker(c))
            self.add_metric(self._make_actual_tracker(c))

        # 3. Standard Accuracy Reward
        self.add_reward_func(self.accuracy_reward, weight=1.0)

        # 4. Built-in Format Check (weight=0, tracked as metric)
        format_reward = self.parser.get_format_reward_func()
        format_reward.__name__ = "format_reward"
        self.add_reward_func(format_reward, weight=format_reward_weight)

    def _make_pred_tracker(self, class_name: str):
        """Creates a tracking metric for how often the model predicts this class."""
        async def tracker(state: dict) -> float:
            pred = state.get("farm_pred", "")
            if not pred:
                return 0.0
            return 1.0 if str(pred).lower() == class_name.lower() else 0.0
        tracker.__name__ = f"pred_{class_name.lower()}"
        return tracker

    def _make_actual_tracker(self, class_name: str):
        """Creates a tracking metric for how often this class is the true target."""
        async def tracker(answer: str) -> float:
            if not answer:
                return 0.0
            return 1.0 if str(answer).lower() == class_name.lower() else 0.0
        tracker.__name__ = f"actual_{class_name.lower()}"
        return tracker

    async def parse_and_categorize(self, completion, answer, state):
        """Single-pass parsing to set the categorization in the shared rollout state."""
        raw_text = ""
        for msg in completion:
            if getattr(msg, "role", msg.get("role", "")) == "assistant":
                raw_text += str(getattr(msg, "content", msg.get("content", "")))

        parsed = self.parser.parse_answer(completion)
        pred = parsed.strip() if parsed else None

        if pred:
            if self.use_escape_hatch and "UNINTELLIGIBLE" in pred.upper():
                state["farm_outcome"] = "clean_refusal"
            else:
                state["farm_outcome"] = "valid_attempt"
                # Normalize common legacy binary predictions, avoiding context class overwriting
                pred_upper = pred.upper()
                if "POSITIVE" in pred_upper and "CONDITIONING" not in pred_upper:
                    final_pred = "Positive"
                elif "NEGATIVE" in pred_upper and "CONDITIONING" not in pred_upper:
                    final_pred = "Negative"
                elif "POS" in pred_upper and "POSITIVE" not in pred_upper:
                    # Only map 'POS' if it's the strict abbreviation
                    final_pred = "Positive" if pred_upper.strip() == "POS" else pred
                elif "NEG" in pred_upper and "NEGATIVE" not in pred_upper:
                    final_pred = "Negative" if pred_upper.strip() == "NEG" else pred
                elif re.search(r"\bHF\b", pred_upper) or "HIGH FREQUENCY" in pred_upper:
                    final_pred = "HF"
                elif re.search(r"\bLF\b", pred_upper) or "LOW FREQUENCY" in pred_upper:
                    final_pred = "LF"
                else:
                    final_pred = pred

                state["farm_pred"] = final_pred

                # Robust correctness check (handles multi-class answers inside verbose sentences)
                ans_clean = str(answer).strip().lower()
                pred_clean = final_pred.lower()
                state["farm_correct"] = (ans_clean == pred_clean) or (
                    ans_clean in pred_clean
                )
        else:
            if any(kw in raw_text.lower() for kw in self.refusal_keywords):
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
# Task Builder Factory
# ============================================================================

TASK_CONFIGS = {
    "call_type_classification": {
        "label_column": "Type",
        "template": CALL_TYPE_TEMPLATE,
        "options": "HF or LF",
        "classes": ["HF", "LF"],
        # LEAK PATCH: Frequency features strongly leak Call Type
        "extra_exclude": {"Q25", "Q50", "Q75", "F0Mean", "Fpeak", "DFMin"},
    },
    "valence_classification": {
        "label_column": "Valence",
        "template": VALENCE_TEMPLATE,
        "options": "Positive or Negative",
        "classes": ["Positive", "Negative"],
        "label_map": {"Pos": "Positive", "Neg": "Negative"},
    },
    "context_classification": {
        "label_column": "Context Category",
        "template": CONTEXT_TEMPLATE,
        "options": "AfterNursing, Barren, BeforeNursing, Castration, Crushing, Enriched, Fighting, Huddling, Isolation, MissedNursing, NegativeConditioning, NovelObject, PositiveConditioning, Restrain, Reunion, Run, or Surprise",
        "classes": [
            "AfterNursing", "Barren", "BeforeNursing", "Castration", "Crushing", 
            "Enriched", "Fighting", "Huddling", "Isolation", "MissedNursing", 
            "NegativeConditioning", "NovelObject", "PositiveConditioning", 
            "Restrain", "Reunion", "Run", "Surprise"
        ],
    },
    "age_classification": {
        "label_column": "Age Category",
        "template": AGE_TEMPLATE,
        "options": "Piglet, Weaner, or GrFinishing",
        "classes": ["Piglet", "Weaner", "GrFinishing"],
    },
}


def build_pvwb_env(
    task_config: dict,
    modality: str,
    include_tabular_data: bool,
    use_escape_hatch: bool,
    max_examples: int,
    dataset_name: str,
    train_split: str,
    test_split: str,
    seed: int,
    target_pad_seconds: float,
    system_prompt_override: str | None,
    user_prompt_override: str | None,
    extra_refusal_keywords: list[str] | None,
    require_reasoning: bool = False,
    format_reward_weight: float = 0.0,
    balancing_strategy: str = "proportional",
) -> vf.SingleTurnEnv:
    """Builds a SingleTurnEnv dynamically based on the requested PVWB task config."""
    label_col = task_config["label_column"]
    label_map = task_config.get("label_map", {})

    def get_dataset_builder(split: str):
        def build():
            ds = get_dataset(dataset_name=dataset_name, split=split)
            ds = ds.filter(lambda x: x[label_col] is not None)

            # 1. Balance and subsample BEFORE mapping.
            # This avoids converting massive amounts of audio to base64 if we only need a few samples.
            ds = balance_and_shuffle(
                ds, label_col, max_examples, seed, strategy=balancing_strategy
            )

            # 2. Transform the subsampled rows.
            def transform(row):
                sys_prompt = (
                    system_prompt_override
                    if system_prompt_override is not None
                    else SYSTEM_PROMPT_ETHOLOGIST
                )
                user_template = (
                    user_prompt_override
                    if user_prompt_override is not None
                    else task_config["template"]
                )

                format_instr = (
                    FORMAT_WITH_REASONING
                    if require_reasoning
                    else FORMAT_WITHOUT_REASONING
                )

                options_str = task_config["options"]
                if use_escape_hatch:
                    options_str = (
                        options_str.replace(" or ", ", ") + ", or UNINTELLIGIBLE"
                    )

                # Pass the label_col so we don't accidentally leak the answer!
                extra_exclude = task_config.get("extra_exclude")
                tabular_data_str = (
                    format_tabular_data(row, label_col, extra_exclude)
                    if include_tabular_data
                    else ""
                )

                prompt_text = user_template.format(
                    modality_instruction=MODALITY_INSTRUCTIONS[modality],
                    tabular_data=tabular_data_str,
                    escape_hatch_note=ESCAPE_HATCH_NOTES.get(modality, "")
                    if use_escape_hatch
                    else "",
                    format_instructions=format_instr.format(options=options_str),
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

                ans_raw = row[label_col]
                answer = label_map.get(ans_raw, ans_raw)

                return {
                    "prompt": prompt,
                    "answer": answer,
                    "info": {
                        "age_category": row.get("Age Category", "pig"),
                        "duration_raw": row.get("Dur"),
                        "am_rate": row.get("AMRate"),
                    },
                }

            logger.info(f"Applying mapping and formatting prompts (split={split})...")
            ds = ds.map(transform, remove_columns=ds.column_names, num_proc=1)

            return ds

        return build

    rubric = FARMRubric(
        classes=task_config.get("classes", []),
        use_escape_hatch=use_escape_hatch,
        require_reasoning=require_reasoning,
        format_reward_weight=format_reward_weight,
        extra_refusal_keywords=extra_refusal_keywords,
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


def load_environment(
    task: str | list[str] = "valence_classification",
    modality: str = "audio",
    include_tabular_data: bool = False,
    use_escape_hatch: bool = False,
    extra_refusal_keywords: list[str] | None = None,
    max_examples_per_task: int = -1,
    dataset_name: str = "oliveirabruno01/soundwel-pig-vocalizations",
    train_split: str = "train",
    test_split: str = "test",
    seed: int = 42,
    target_pad_seconds: float = 3.592,
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
        modality: "audio" (wav), "vision" (spectrogram image), "multimodal" (both), or "text".
        include_tabular_data: If True, inject acoustic/demographic features into the prompt. (Forced True if modality="text").
        use_escape_hatch: If True, allow the model to classify poor quality samples as UNINTELLIGIBLE.
        extra_refusal_keywords: List of extra refusal keywords to append to the default list.
        max_examples_per_task: Maximum examples per task (-1 for all).
        dataset_name: HuggingFace dataset ID or local path.
        train_split: Split to use for training.
        test_split: Split to use for testing.
        seed: Random seed for dataset shuffling.
        target_pad_seconds: Centrally pad audio arrays to this length. Set to 0.0 or -1.0 to disable padding.
        system_prompt_override: Optional custom string to replace system prompt.
        user_prompt_override: Optional custom template string for the user task prompt.
        require_reasoning: Whether to require reasoning in the model's response.
        format_reward_weight: Weight for the formatting reward.
        balancing_strategy: "proportional" or "balanced". "proportional" to class balance, "balanced" to balance classes.
    """
    if task == "all":
        selected_tasks = list(TASK_CONFIGS.keys())
    elif isinstance(task, str):
        selected_tasks = [task]
    else:
        selected_tasks = list(task)

    for t in selected_tasks:
        if t not in TASK_CONFIGS:
            raise ValueError(
                f"Unknown task: {t!r}. Valid tasks: {list(TASK_CONFIGS.keys())}"
            )

    if modality == "text":
        include_tabular_data = True

    common_kwargs = {
        "modality": modality,
        "include_tabular_data": include_tabular_data,
        "use_escape_hatch": use_escape_hatch,
        "extra_refusal_keywords": extra_refusal_keywords,
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
        env = build_pvwb_env(task_config=TASK_CONFIGS[task_name], **common_kwargs)
        envs.append(env)

    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=selected_tasks)
