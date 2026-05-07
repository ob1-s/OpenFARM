"""Shared media processing and dataset utilities for the Soundwel PVWB suite."""

import io
import base64
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger("verifiers.env.soundwel_pvwb.utils")


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


def encode_audio_to_base64(audio_dict: dict, target_pad_seconds: float = 3.595) -> str:
    """JIT decodes audio bytes, applies padding, and converts to base64 WAV."""
    import soundfile as sf

    # 1. Extract raw bytes (since decode=False)
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

    # 3. Apply paper-aligned central zero-padding
    if target_pad_seconds > 0:
        array = pad_audio_array(array, sr, target_pad_seconds)

    # 4. Write back to WAV format
    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV")
    padded_bytes = buf.getvalue()

    return base64.b64encode(padded_bytes).decode("utf-8")


def encode_spectrogram_to_base64(image: Image.Image) -> str:
    """Convert a PIL spectrogram image to a base64 JPEG data URI."""
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def build_multimodal_content_parts(
    row: dict, modality: str, target_pad_seconds: float = 3.595
) -> list[dict]:
    """Build the sequenced multimodal content parts for a dataset row."""
    parts = []

    if modality in ["audio", "multimodal"]:
        b64 = encode_audio_to_base64(row["audio"], target_pad_seconds)
        parts.append(
            {
                "type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"},
            }
        )

    if modality in ["vision", "multimodal"]:
        b64_uri = encode_spectrogram_to_base64(row["spectrogram"])
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": b64_uri},
            }
        )

    return parts


def get_dataset(dataset_name: str, split: str = "train"):
    """Loads the base dataset split without algorithmic difficulty filtering."""
    from datasets import load_dataset, Audio

    logger.info(f"Loading raw HF dataset from {dataset_name} (split={split})...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        logger.warning(
            f"Failed to load split '{split}', falling back to 'train'. Error: {e}"
        )
        ds = load_dataset(dataset_name, split="train")

    # Keep decode=False so HF doesn't crash requiring torchcodec globally
    ds = ds.cast_column("audio", Audio(decode=False))

    return ds


def balance_and_shuffle(
    ds,
    class_a_label: str,
    class_b_label: str,
    max_examples: int,
    seed: int,
    strategy: str = "proportional",
):
    """
    Subsamples and shuffles the dataset.
    strategy="balanced" forces a 50/50 split (may aggressively downsample majority class).
    strategy="proportional" respects the natural class imbalance of the dataset.
    """
    if max_examples > 0:
        from datasets import concatenate_datasets

        class_a = ds.filter(lambda x: x["answer"] == class_a_label)
        class_b = ds.filter(lambda x: x["answer"] == class_b_label)

        if strategy == "balanced":
            take_a = min(max_examples // 2, len(class_a))
            take_b = min(max_examples // 2, len(class_b))
        else:  # proportional
            total_avail = len(class_a) + len(class_b)
            if total_avail == 0:
                return ds.shuffle(seed=seed)
            ratio_a = len(class_a) / total_avail
            take_a = min(int(max_examples * ratio_a), len(class_a))
            # allocate the remainder to b, bounded by b's total size
            take_b = min(max_examples - take_a, len(class_b))

        ds = concatenate_datasets(
            [class_a.select(range(take_a)), class_b.select(range(take_b))]
        )

    return ds.shuffle(seed=seed)
