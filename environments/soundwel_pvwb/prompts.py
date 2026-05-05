"""Centralized prompt templates for the Soundwel PVWB benchmark suite."""

SYSTEM_PROMPT_ETHOLOGIST = (
    "You are an expert ethologist specializing in swine welfare and "
    "vocalization analysis. You have deep knowledge of pig vocal "
    "communication, including the distinction between high-frequency (HF) "
    "calls (screams, squeals — associated with acute pain, fear, distress) "
    "and low-frequency (LF) calls (grunts, barks — associated with "
    "lower-arousal states, both positive and negative)."
)

def build_task_prompt(task_name: str, modality: str, use_escape_hatch: bool) -> str:
    """Dynamically construct the task prompt across all axes."""
    
    # 1. Resolve modality subject
    if modality == "audio":
        subject = "Listen to the provided pig vocalization."
        escape_context = "audio is completely missing, entirely silent, or too short to hear anything"
    elif modality == "vision":
        subject = "Look at the provided spectrogram of a pig vocalization."
        escape_context = "spectrogram is completely empty or missing"
    else: # multimodal
        subject = "You are provided with BOTH an audio recording and its corresponding spectrogram image of a pig vocalization. Using both the audio and the visual data,"
        escape_context = "BOTH the audio and the spectrogram are completely missing or unintelligible"

    # 2. Resolve task question
    if task_name == "call_type_classification":
        question = "Is this a High Frequency (HF) call (scream or squeal, typically above 1 kHz fundamental frequency) or a Low Frequency (LF) call (grunt or bark, typically below 1 kHz)?"
        options_text = "HF, LF, or UNINTELLIGIBLE" if use_escape_hatch else "HF or LF"
        reasoning_hint = "[your acoustic analysis]"
    else: # valence_classification
        question = "Based on the acoustic properties of the call — including frequency, duration, harmonicity, and amplitude modulation — is the emotional valence of this animal Positive or Negative?"
        options_text = "Positive, Negative, or UNINTELLIGIBLE" if use_escape_hatch else "Positive or Negative"
        reasoning_hint = "[your analysis of acoustic features and emotional state]"

    parts = [f"{subject} {question}\n"]

    # 3. Escape hatch rules
    if use_escape_hatch:
        parts.append(f"If the {escape_context}, you must classify it as UNINTELLIGIBLE.\n")

    # 4. XML Formatting Schema
    parts.append(
        f"Respond using this format:\n"
        f"<reasoning>\n{reasoning_hint}\n</reasoning>\n"
        f"<answer>\n[{options_text}]\n</answer>"
    )

    return "\n".join(parts)