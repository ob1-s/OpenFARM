import io
import base64
import logging
import numpy as np
import verifiers as vf

logger = logging.getLogger("verifiers.catmeows_audio_context")

# ============================================================================
# Declarative Prompt Templates
# ============================================================================

SYSTEM_PROMPT_ETHOLOGIST = (
    "You are a feline bioacoustics evaluator. Classify the source context "
    "of cat vocalizations. Do not diagnose pain, disease, stress, or overall welfare."
)

ESCAPE_HATCH_NOTE = (
    "NOTE: The audio may be padded with silence to assist processing. If you hear *any* "
    "cat vocalization, evaluate it. Only if the audio is completely missing, entirely silent, "
    "or too short to hear anything, you must classify it as UNINTELLIGIBLE.\n\n"
)

TASK_TEMPLATE = """<audio>Listen to the provided cat vocalization.
Choose the most likely source dataset context for this recording.
This is a forced-choice context classification task, not a clinical or welfare diagnosis.
{tabular_data}
{escape_hatch_note}{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[brief acoustic evidence for the selected context]
</explanation>
<answer>
[{options}]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[{options}]
</answer>"""

# ============================================================================
# Shared Utilities
# ============================================================================


def pad_audio_array(
    audio_array: np.ndarray, sr: int, target_seconds: float
) -> np.ndarray:
    """Centrally zero-pads an audio array to reach a safe VLM duration threshold."""
    target_length = int(target_seconds * sr)
    current_length = len(audio_array)

    if current_length >= target_length:
        return audio_array

    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    return np.pad(
        audio_array, (pad_left, pad_right), mode="constant", constant_values=0
    )


def encode_audio_to_base64(audio_dict: dict, target_pad_seconds: float = 0.0) -> str:
    """JIT decodes audio bytes, applies padding, and converts to base64 WAV."""
    import soundfile as sf

    # 1. Extract raw bytes (since decode=False usually)
    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
    elif "path" in audio_dict and audio_dict["path"] is not None:
        with open(audio_dict["path"], "rb") as f:
            audio_bytes = f.read()
    else:
        # Fallback if somehow already decoded
        audio_bytes = None
        array = audio_dict.get("array")
        sr = audio_dict.get("sampling_rate")

    # 2. JIT Decode using soundfile if we have bytes
    if audio_bytes is not None:
        array, sr = sf.read(io.BytesIO(audio_bytes))

    # 3. Apply central zero-padding if requested
    if target_pad_seconds > 0:
        array = pad_audio_array(array, sr, target_pad_seconds)

    # 4. Write back to WAV format
    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV")
    padded_bytes = buf.getvalue()

    return base64.b64encode(padded_bytes).decode("utf-8")


def format_tabular_data(row: dict) -> str:
    """Format available dataset columns into a markdown list for prompt injection."""
    keys = [
        "duration_sec",
        "rms_energy",
        "peak_abs_amplitude",
        "zero_crossing_rate",
        "spectral_centroid_hz",
        "breed",
        "sex_status",
    ]
    lines = []
    for k in keys:
        if k in row and row[k] is not None:
            lines.append(f"- {k}: {row[k]}")

    if not lines:
        return ""
    return "\n[Tabular Data]\n" + "\n".join(lines) + "\n"


# ============================================================================
# Core Rubric
# ============================================================================


class CatMeowsRubric(vf.Rubric):
    """Minimal rubric for CatMeows context classification."""

    def __init__(
        self,
        require_explanation: bool = False,
        format_reward_weight: float = 0.0,
    ):
        fields = ["explanation", "answer"] if require_explanation else ["answer"]
        self.parser = vf.XMLParser(fields=fields, answer_field="answer")
        super().__init__(parser=self.parser)

        self.add_reward_func(self.accuracy_reward, weight=1.0)

        format_reward = self.parser.get_format_reward_func()
        format_reward.__name__ = "format_reward"
        self.add_reward_func(format_reward, weight=format_reward_weight)

    @staticmethod
    def normalize_prediction(pred: str) -> str:
        pred_lower = pred.strip().lower()
        if "isolation" in pred_lower:
            return "isolation_unfamiliar_environment"
        if "brushing" in pred_lower:
            return "brushing"
        if "waiting" in pred_lower or "food" in pred_lower:
            return "waiting_for_food"
        return pred_lower

    async def accuracy_reward(self, completion, answer) -> float:
        parsed = self.parser.parse_answer(completion) or ""
        if not parsed.strip():
            return 0.0

        pred = self.normalize_prediction(parsed)
        ans_clean = str(answer).strip().lower()
        return 1.0 if ans_clean == pred else 0.0


# ============================================================================
# Environment Loader
# ============================================================================


def load_environment(
    modality: str = "audio",  # 'audio' or 'text'
    include_tabular_data: bool = False,
    use_escape_hatch: bool = False,
    max_examples: int = -1,
    dataset_name: str = "oliveirabruno01/openfarm-catmeows",
    dataset_revision: str | None = None,
    train_split: str = "train",
    test_split: str = "test",
    seed: int = 42,
    target_pad_seconds: float = 0.0,
    system_prompt_override: str | None = None,
    user_prompt_override: str | None = None,
    require_explanation: bool = False,
    format_reward_weight: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """Load the CatMeows Audio Context Classification Benchmark."""

    if modality == "text":
        include_tabular_data = True

    def get_dataset_builder(split: str):
        def build():
            from datasets import load_dataset, Audio

            logger.info(
                f"Loading raw HF dataset from {dataset_name} (split={split})..."
            )
            try:
                ds = load_dataset(dataset_name, split=split, revision=dataset_revision)
            except Exception as e:
                logger.warning(
                    f"Failed to load split '{split}', falling back to 'train'. Error: {e}"
                )
                ds = load_dataset(
                    dataset_name, split="train", revision=dataset_revision
                )

            # Keep decode=False so HF doesn't crash if torchcodec/libsndfile are missing locally
            if "audio" in ds.column_names:
                ds = ds.cast_column("audio", Audio(decode=False))

            if seed:
                ds = ds.shuffle(seed=seed)
            if max_examples > 0:
                ds = ds.select(range(min(max_examples, len(ds))))

            def transform(row):
                sys_prompt = (
                    system_prompt_override
                    if system_prompt_override is not None
                    else SYSTEM_PROMPT_ETHOLOGIST
                )
                user_template = (
                    user_prompt_override
                    if user_prompt_override is not None
                    else TASK_TEMPLATE
                )
                format_instr = (
                    FORMAT_WITH_REASONING
                    if require_explanation
                    else FORMAT_WITHOUT_REASONING
                )

                options_str = (
                    "brushing, isolation_unfamiliar_environment, or waiting_for_food"
                )
                if use_escape_hatch:
                    options_str += ", or UNINTELLIGIBLE"

                tabular_data_str = (
                    format_tabular_data(row) if include_tabular_data else ""
                )

                prompt_text = user_template.format(
                    tabular_data=tabular_data_str,
                    escape_hatch_note=ESCAPE_HATCH_NOTE if use_escape_hatch else "",
                    format_instructions=format_instr.format(options=options_str),
                )

                # Assemble multi-modal or text-only messages
                content_parts = []
                if modality == "audio" and "audio" in row:
                    b64 = encode_audio_to_base64(row["audio"], target_pad_seconds)
                    content_parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": b64, "format": "wav"},
                        }
                    )

                content_parts.append({"type": "text", "text": prompt_text})

                prompt = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": sys_prompt}],
                    },
                    {"role": "user", "content": content_parts},
                ]

                answer = str(row.get("context", "")).strip()

                return {
                    "prompt": prompt,
                    "answer": answer,
                    "info": {
                        "cat_id": row.get("cat_id"),
                        "duration_sec": row.get("duration_sec"),
                    },
                }

            logger.info(f"Applying mapping and formatting prompts (split={split})...")
            return ds.map(transform, remove_columns=ds.column_names, num_proc=1)

        return build

    rubric = CatMeowsRubric(
        require_explanation=require_explanation,
        format_reward_weight=format_reward_weight,
    )

    return vf.SingleTurnEnv(
        dataset=get_dataset_builder(train_split),
        eval_dataset=get_dataset_builder(test_split),
        parser=rubric.parser,
        rubric=rubric,
        **kwargs,
    )
