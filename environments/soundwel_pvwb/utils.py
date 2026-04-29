"""Shared utilities for the Soundwel PVWB benchmark suite."""

import io
import base64

from PIL import Image


def encode_audio_to_base64(audio_dict: dict) -> str:
    """Convert an HF Audio feature dict to a base64-encoded WAV data URI.

    Args:
        audio_dict: HF datasets Audio dict with 'array', 'sampling_rate', and
                    optionally 'bytes' keys.

    Returns:
        Base64 data URI string suitable for OpenAI input_audio content parts.
    """
    if "bytes" in audio_dict and audio_dict["bytes"] is not None:
        audio_bytes = audio_dict["bytes"]
    elif "path" in audio_dict and audio_dict["path"] is not None:
        with open(audio_dict["path"], "rb") as f:
            audio_bytes = f.read()
    else:
        # Fallback: encode numpy array to WAV bytes
        import soundfile as sf

        buf = io.BytesIO()
        sf.write(buf, audio_dict["array"], audio_dict["sampling_rate"], format="WAV")
        audio_bytes = buf.getvalue()

    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return b64


def encode_spectrogram_to_base64(image: Image.Image) -> str:
    """Convert a PIL spectrogram image to a base64 JPEG data URI.

    Args:
        image: PIL Image object (spectrogram).

    Returns:
        Base64 data URI string for image_url content parts.
    """
    buffered = io.BytesIO()
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def build_audio_content_part(row: dict, modality: str = "audio") -> dict:
    """Build the appropriate multimodal content part for a dataset row.

    Args:
        row: A single row from the Soundwel dataset.
        modality: "audio" or "spectrogram".

    Returns:
        A dict suitable for inclusion in an OpenAI chat message content list.
    """
    if modality == "audio":
        b64 = encode_audio_to_base64(row["audio"])
        return {
            "type": "input_audio",
            "input_audio": {"data": b64, "format": "wav"},
        }
    else:
        b64_uri = encode_spectrogram_to_base64(row["spectrogram"])
        return {
            "type": "image_url",
            "image_url": {"url": b64_uri},
        }
