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


def encode_audio_to_base64(audio_dict: dict, target_pad_seconds: float = 0.0) -> str:
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
    row: dict, modality: str, target_pad_seconds: float = 0.0
) -> list[dict]:
    """Build the sequenced multimodal content parts for a dataset row."""
    parts = []

    if modality == "text":
        return parts

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
    label_column: str,
    max_examples: int,
    seed: int,
    strategy: str = "proportional",
):
    """
    Subsamples and shuffles the dataset.
    strategy="balanced" forces an equal split across all unique classes.
    strategy="proportional" respects the natural class imbalance of the dataset.
    """
    if max_examples <= 0:
        return ds.shuffle(seed=seed)

    from datasets import concatenate_datasets

    unique_labels = list(set(ds[label_column]))
    unique_labels = [label for label in unique_labels if label is not None]

    class_datasets = {
        label: ds.filter(lambda x: x[label_column] == label) for label in unique_labels
    }
    total_avail = sum(len(c_ds) for c_ds in class_datasets.values())

    if total_avail == 0:
        return ds.shuffle(seed=seed)

    selected_datasets = []
    if strategy == "balanced":
        take_per_class = max_examples // len(unique_labels)
        remainder = max_examples % len(unique_labels)
        for i, (label, c_ds) in enumerate(class_datasets.items()):
            take = min(take_per_class + (1 if i < remainder else 0), len(c_ds))
            selected_datasets.append(c_ds.select(range(take)))
    else:  # proportional
        taken_total = 0
        for i, (label, c_ds) in enumerate(class_datasets.items()):
            if i == len(unique_labels) - 1:
                # Last class takes whatever is left (bounded by its size)
                take = min(max_examples - taken_total, len(c_ds))
            else:
                ratio = len(c_ds) / total_avail
                take = min(int(max_examples * ratio), len(c_ds))
            selected_datasets.append(c_ds.select(range(take)))
            taken_total += take

    ds = concatenate_datasets(selected_datasets)
    return ds.shuffle(seed=seed)
