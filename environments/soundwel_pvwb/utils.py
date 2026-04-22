"""Shared utilities for the Soundwel PVWB benchmark suite."""

import io
import base64
import random

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


# ---------------------------------------------------------------------------
# MCQ helpers
# ---------------------------------------------------------------------------

# Acoustic property bins (derived from dataset quartiles)
ACOUSTIC_BINS = {
    "Dur": {
        "label": "duration",
        "unit": "seconds",
        "bins": [
            ("Very short", 0.0, 0.15),
            ("Short", 0.15, 0.30),
            ("Medium", 0.30, 0.60),
            ("Long", 0.60, float("inf")),
        ],
    },
    "F0Mean": {
        "label": "fundamental frequency (pitch)",
        "unit": "Hz",
        "bins": [
            ("Very low", 0.0, 70.0),
            ("Low", 70.0, 110.0),
            ("Medium", 110.0, 200.0),
            ("High", 200.0, float("inf")),
        ],
    },
    "Fpeak": {
        "label": "peak frequency",
        "unit": "Hz",
        "bins": [
            ("Low", 0.0, 400.0),
            ("Medium-low", 400.0, 1000.0),
            ("Medium-high", 1000.0, 2500.0),
            ("High", 2500.0, float("inf")),
        ],
    },
}

LETTER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}
LETTER_MAP_INV = {v: k for k, v in LETTER_MAP.items()}


def classify_acoustic_value(feature: str, value: float) -> int:
    """Return the bin index (0-3) for a given acoustic feature value."""
    bins = ACOUSTIC_BINS[feature]["bins"]
    for i, (_, lo, hi) in enumerate(bins):
        if lo <= value < hi:
            return i
    return len(bins) - 1


def build_acoustic_mcq(feature: str, value: float) -> tuple[str, str, list[str]]:
    """Build an MCQ question string, correct letter, and option list.

    Returns:
        (question_fragment, correct_letter, options_list)
    """
    bins = ACOUSTIC_BINS[feature]["bins"]
    correct_idx = classify_acoustic_value(feature, value)

    options = []
    for label, lo, hi in bins:
        if hi == float("inf"):
            options.append(f"{label} (>{lo:.0f} {ACOUSTIC_BINS[feature]['unit']})")
        else:
            options.append(
                f"{label} ({lo:.0f}–{hi:.0f} {ACOUSTIC_BINS[feature]['unit']})"
            )

    correct_letter = LETTER_MAP[correct_idx]
    label = ACOUSTIC_BINS[feature]["label"]
    question = f"Which of the following best describes the {label} of this vocalization?"

    return question, correct_letter, options


# ---------------------------------------------------------------------------
# Context MCQ helpers
# ---------------------------------------------------------------------------

# Pain/distress hierarchy for context classification
NEGATIVE_CONTEXTS = [
    "Castration", "Crushing", "Restrain", "Fighting",
    "Isolation", "MissedNursing", "NegativeConditioning",
]
POSITIVE_CONTEXTS = [
    "Enriched", "PositiveConditioning", "Run",
    "AfterNursing", "BeforeNursing", "Reunion",
    "Huddling", "Surprise", "NovelObject",
]

# Known high-pain contexts for pairwise ranking
HIGH_PAIN_CONTEXTS = {"Castration", "Crushing", "Restrain", "Fighting"}
LOW_PAIN_CONTEXTS = {"Enriched", "PositiveConditioning", "Run", "AfterNursing", "Reunion"}


def sample_distractors(
    correct: str,
    pool: list[str],
    n: int = 3,
    rng: random.Random | None = None,
) -> list[str]:
    """Sample n distractors from pool, excluding the correct answer.

    Args:
        correct: The correct answer string.
        pool: List of possible answers to draw from.
        n: Number of distractors.
        rng: Optional random.Random instance for reproducibility.

    Returns:
        List of n distractor strings.
    """
    if rng is None:
        rng = random.Random()
    candidates = [c for c in pool if c != correct]
    return rng.sample(candidates, min(n, len(candidates)))
