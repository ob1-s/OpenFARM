"""Shared utilities for the Soundwel PVWB benchmark suite."""

import io
import base64
from PIL import Image

import verifiers as vf
import logging

logger = logging.getLogger("verifiers.env.soundwel_pvwb.utils")


def encode_audio_to_base64(audio_dict: dict) -> str:
    """Convert an HF Audio feature dict to a base64-encoded WAV data URI."""
    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
    elif "path" in audio_dict and audio_dict["path"] is not None:
        with open(audio_dict["path"], "rb") as f:
            audio_bytes = f.read()
    else:
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
        audio_bytes = buf.getvalue()

    return base64.b64encode(audio_bytes).decode("utf-8")


def encode_spectrogram_to_base64(image: Image.Image) -> str:
    """Convert a PIL spectrogram image to a base64 JPEG data URI."""
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def build_multimodal_content_parts(row: dict, modality: str) -> list[dict]:
    """Build the appropriate sequenced multimodal content parts for a dataset row."""
    parts = []
    
    # Notebook findings dictate Audio should precede Image when Multimodal
    if modality in ["audio", "multimodal"]:
        b64 = encode_audio_to_base64(row["audio"])
        parts.append({
            "type": "input_audio",
            "input_audio": {"data": b64, "format": "wav"},
        })
        
    if modality in ["vision", "multimodal"]:
        b64_uri = encode_spectrogram_to_base64(row["spectrogram"])
        parts.append({
            "type": "image_url",
            "image_url": {"url": b64_uri},
        })
        
    return parts


def get_dataset_builder(
    task_name: str,
    modality: str,
    data_subset: str,
    use_escape_hatch: bool,
    seed: int,
    max_examples: int,
) -> vf.DatasetBuilder:
    """Lazy Dataset Builder handling HF dataset loading, stratification, and prompt mapping."""
    
    def build():
        from datasets import load_dataset, Audio
        from prompts import SYSTEM_PROMPT_ETHOLOGIST, build_task_prompt
        
        ds = load_dataset("oliveirabruno01/soundwel-pig-vocalizations", split="train")

        # CRITICAL FIX: Cast audio with decode=False IMMEDIATELY after loading
        # This prevents HF datasets from crashing on missing audio libraries during .filter()
        ds = ds.cast_column("audio", Audio(decode=False))

        # F.A.R.M. Stratification Logic
        if data_subset == "rich":
            ds = ds.filter(lambda x: x["Dur"] is not None and x["Harmonicity"] is not None and x["Dur"] >= 0.5 and x["Harmonicity"] >= 3.5)
        elif data_subset == "hard_short":
            ds = ds.filter(lambda x: x["Dur"] is not None and x["Dur"] < 0.3)
        elif data_subset == "hard_noisy":
            ds = ds.filter(lambda x: x["Dur"] is not None and x["Harmonicity"] is not None and x["Dur"] >= 0.3 and x["Harmonicity"] < 0.0)
        elif data_subset == "average":
            ds = ds.filter(lambda x: x["Dur"] is not None and x["Harmonicity"] is not None and not (x["Dur"] >= 0.5 and x["Harmonicity"] >= 3.5) and not (x["Dur"] < 0.3) and not (x["Dur"] >= 0.3 and x["Harmonicity"] < 0.0))

        # Filter missing target labels
        if task_name == "call_type_classification":
            ds = ds.filter(lambda x: x["Type"] is not None)
        else:
            ds = ds.filter(lambda x: x["Valence"] is not None)

        prompt_text = build_task_prompt(task_name, modality, use_escape_hatch)

        def transform(row):
            media_parts = build_multimodal_content_parts(row, modality)
            prompt = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": SYSTEM_PROMPT_ETHOLOGIST}],
                },
                {
                    "role": "user",
                    "content": media_parts + [{"type": "text", "text": prompt_text}],
                },
            ]

            if task_name == "call_type_classification":
                answer = row["Type"]
            else:
                answer = "Positive" if row["Valence"] == "Pos" else "Negative"

            return {"prompt": prompt, "answer": answer}

        ds = ds.map(transform, remove_columns=ds.column_names)
        ds = ds.shuffle(seed=seed)
        
        if max_examples > 0:
            from datasets import concatenate_datasets
            
            # Split by the standardized "answer" column we just created in the transform step
            if task_name == "call_type_classification":
                class_a = ds.filter(lambda x: x["answer"] == "HF")
                class_b = ds.filter(lambda x: x["answer"] == "LF")
            else: # valence_classification
                class_a = ds.filter(lambda x: x["answer"] == "Positive")
                class_b = ds.filter(lambda x: x["answer"] == "Negative")
            
            # Calculate how many examples to take per class (50/50 split)
            target_per_class = max_examples // 2
            
            # Safely select up to the available amount in the minority class
            take_a = min(target_per_class, len(class_a))
            take_b = min(target_per_class, len(class_b))
            
            # If one class is severely lacking, you might want to fill the rest with the majority class
            # to hit the exact max_examples requested (optional):
            # if take_a < target_per_class: take_b = min(max_examples - take_a, len(class_b))
            # if take_b < target_per_class: take_a = min(max_examples - take_b, len(class_a))
            
            logger.info(f"Taking {take_a} examples from class {class_a} and {take_b} examples from class {class_b} (out of {len(ds)} total examples)")

            ds = concatenate_datasets([
                class_a.select(range(take_a)), 
                class_b.select(range(take_b))
            ])
            # Reshuffle so they aren't grouped by class
            ds = ds.shuffle(seed=seed)
            
        return ds

    return build


# Shared F.A.R.M. Evaluation Rubric 
REFUSAL_KEYWORDS = ["missing", "empty", "audio file", "please provide"]

class FARMRubric(vf.Rubric):
    """Centralized F.A.R.M. rubric handling granular parsing and validation states."""
    
    def __init__(self, use_escape_hatch: bool = False):
        self.parser = vf.XMLParser(fields=["reasoning", "answer"], answer_field="answer")
        super().__init__(parser=self.parser)
        self.use_escape_hatch = use_escape_hatch

        # 1. High-priority parser executes first
        self.add_reward_func(self.parse_and_categorize, weight=0.0)
        
        # 2. Granular tracker metrics (Zero weight)
        self.add_metric(self.clean_refusals)
        self.add_metric(self.messy_refusals)
        self.add_metric(self.format_failures)
        self.add_metric(self.valid_attempts)
        
        # 3. Standard Accuracy Reward 
        self.add_reward_func(self.accuracy_reward, weight=1.0)
        
        # 4. Built-in Format Check
        format_reward = self.parser.get_format_reward_func()
        format_reward.__name__ = "format_reward"
        self.add_reward_func(format_reward, weight=0.0)

    async def parse_and_categorize(self, completion, answer, state):
        """Single-pass parsing to set the categorization in the shared rollout state."""
        raw_text = ""
        for msg in completion:
            if isinstance(msg, dict):
                if msg.get("role") == "assistant":
                    raw_text += str(msg.get("content", ""))
            else:
                if getattr(msg, "role", "") == "assistant":
                    raw_text += str(getattr(msg, "content", ""))
        raw_text_lower = raw_text.lower()
        
        parsed = self.parser.parse_answer(completion)
        pred = parsed.strip().upper() if parsed else None
        
        if pred:
            if self.use_escape_hatch and "UNINTELLIGIBLE" in pred:
                state["farm_outcome"] = "clean_refusal"
            else:
                state["farm_outcome"] = "valid_attempt"
                # Normalize prediction
                if "HF" in pred: final_pred = "HF"
                elif "LF" in pred: final_pred = "LF"
                elif "POSITIVE" in pred or "POS" in pred: final_pred = "Positive"
                elif "NEGATIVE" in pred or "NEG" in pred: final_pred = "Negative"
                else: final_pred = pred
                
                state["farm_pred"] = final_pred
                state["farm_correct"] = (final_pred.lower() == answer.strip().lower())
        else:
            if any(kw in raw_text_lower for kw in REFUSAL_KEYWORDS):
                state["farm_outcome"] = "messy_refusal"
            else:
                state["farm_outcome"] = "format_failure"
        return 0.0

    async def clean_refusals(self, state): return 1.0 if state.get("farm_outcome") == "clean_refusal" else 0.0
    async def messy_refusals(self, state): return 1.0 if state.get("farm_outcome") == "messy_refusal" else 0.0
    async def format_failures(self, state): return 1.0 if state.get("farm_outcome") == "format_failure" else 0.0
    async def valid_attempts(self, state): return 1.0 if state.get("farm_outcome") == "valid_attempt" else 0.0
    
    async def accuracy_reward(self, state):
        return 1.0 if state.get("farm_correct", False) else 0.0