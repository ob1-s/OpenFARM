import base64
import io
from typing import Any

import numpy as np
from PIL import Image as PILImage


def pad_audio_array(
    audio_array: np.ndarray, sr: int, target_seconds: float
) -> np.ndarray:
    target_length = int(target_seconds * sr)
    current_length = len(audio_array)
    if current_length >= target_length:
        return audio_array

    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(audio_array, (pad_left, pad_right), mode="constant")


def clip_audio_array(
    audio_array: np.ndarray,
    sr: int,
    max_seconds: float | None,
    strategy: str = "start",
) -> np.ndarray:
    if max_seconds is None or max_seconds <= 0:
        return audio_array

    max_samples = int(max_seconds * sr)
    if max_samples <= 0:
        return audio_array
    if len(audio_array) <= max_samples:
        return audio_array

    strategy = (strategy or "start").lower()
    if strategy == "center":
        start = max(0, (len(audio_array) - max_samples) // 2)
        return audio_array[start : start + max_samples]

    if strategy == "start":
        start = 0
        return audio_array[start : start + max_samples]

    if strategy == "even_windows":
        window_count = min(5, max_samples)
        base_window = max_samples // window_count
        remainder = max_samples % window_count
        window_lengths = [
            base_window + (1 if idx < remainder else 0) for idx in range(window_count)
        ]

        centers = np.linspace(0.1, 0.9, num=window_count) * (len(audio_array) - 1)
        windows = []
        for center, window_length in zip(centers, window_lengths):
            start = int(round(center - (window_length / 2)))
            start = min(max(0, start), len(audio_array) - window_length)
            windows.append(audio_array[start : start + window_length])

        return np.concatenate(windows)[:max_samples]

    raise ValueError(
        f"Unknown audio_clip_strategy {strategy!r}. "
        "Valid strategies: start, center, even_windows."
    )


def audio_dict_to_array(audio_dict: dict) -> tuple[np.ndarray, int]:
    """Decode a HF Audio row dict into an audio array and sampling rate."""
    import soundfile as sf

    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
        array, sr = sf.read(io.BytesIO(audio_bytes))
    elif "path" in audio_dict and audio_dict["path"] is not None:
        array, sr = sf.read(audio_dict["path"])
    else:
        array = audio_dict.get("array")
        sr = audio_dict.get("sampling_rate")

    if array is None or sr is None:
        raise ValueError("Audio row does not contain bytes, path, or decoded array.")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, int(sr)


def prepare_audio_array(
    audio_dict: dict,
    target_pad_seconds: float = 0.0,
    max_audio_seconds: float | None = None,
    audio_clip_strategy: str = "start",
) -> tuple[np.ndarray, int]:
    array, sr = audio_dict_to_array(audio_dict)
    array = clip_audio_array(array, sr, max_audio_seconds, audio_clip_strategy)
    if target_pad_seconds > 0:
        array = pad_audio_array(array, sr, target_pad_seconds)
    return array, sr


def encode_audio_to_base64(
    audio_dict: dict,
    target_pad_seconds: float = 0.0,
    max_audio_seconds: float | None = None,
    audio_clip_strategy: str = "start",
) -> str:
    import soundfile as sf

    array, sr = prepare_audio_array(
        audio_dict,
        target_pad_seconds=target_pad_seconds,
        max_audio_seconds=max_audio_seconds,
        audio_clip_strategy=audio_clip_strategy,
    )
    buf = io.BytesIO()
    sf.write(buf, array, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def spectrogram_from_audio(audio_array: np.ndarray, sr: int) -> PILImage.Image:
    """Generate a compact spectrogram image using only numpy and Pillow."""
    del sr  # The visual benchmark needs relative time/frequency energy, not axes.

    if audio_array.size == 0:
        audio_array = np.zeros(2048, dtype=np.float32)

    frame_size = 1024
    hop = 256
    if len(audio_array) < frame_size:
        audio_array = pad_audio_array(audio_array, 1, frame_size)

    n_frames = 1 + max(0, (len(audio_array) - frame_size) // hop)
    frames = np.lib.stride_tricks.sliding_window_view(audio_array, frame_size)[::hop]
    frames = frames[:n_frames]
    window = np.hanning(frame_size).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).T
    spec = np.log1p(spec)

    low, high = np.percentile(spec, [5, 99])
    if high <= low:
        high = low + 1e-6
    spec = np.clip((spec - low) / (high - low), 0.0, 1.0)
    spec = np.flipud(spec)

    gray = (spec * 255).astype(np.uint8)
    image = PILImage.fromarray(gray, mode="L").convert("RGB")
    return image.resize((512, 384), resample=PILImage.BILINEAR)


def coerce_image(value: Any) -> PILImage.Image:
    if isinstance(value, PILImage.Image):
        return value
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return PILImage.open(io.BytesIO(value["bytes"]))
        if value.get("path") is not None:
            return PILImage.open(value["path"])
    raise ValueError("Unsupported spectrogram image value.")


def encode_image_to_base64(image: PILImage.Image) -> str:
    buf = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def build_media_parts(
    row: dict,
    modality: str,
    target_pad_seconds: float,
    max_audio_seconds: float | None,
    audio_clip_strategy: str,
) -> list[dict]:
    parts = []
    if modality == "text":
        return parts

    audio_array = None
    sr = None

    if modality in {"audio", "multimodal"}:
        audio_b64 = encode_audio_to_base64(
            row["audio"],
            target_pad_seconds=target_pad_seconds,
            max_audio_seconds=max_audio_seconds,
            audio_clip_strategy=audio_clip_strategy,
        )
        parts.append(
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}}
        )

    if modality in {"vision", "multimodal"}:
        if "spectrogram" in row and row["spectrogram"] is not None:
            image = coerce_image(row["spectrogram"])
        else:
            audio_array, sr = prepare_audio_array(
                row["audio"],
                target_pad_seconds=target_pad_seconds,
                max_audio_seconds=max_audio_seconds,
                audio_clip_strategy=audio_clip_strategy,
            )
            image = spectrogram_from_audio(audio_array, sr)

        parts.append(
            {"type": "image_url", "image_url": {"url": encode_image_to_base64(image)}}
        )

    return parts
