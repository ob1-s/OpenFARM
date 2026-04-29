"""Centralized prompt templates for the Soundwel PVWB benchmark suite."""

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_ETHOLOGIST = (
    "You are an expert ethologist specializing in swine welfare and "
    "vocalization analysis. You have deep knowledge of pig vocal "
    "communication, including the distinction between high-frequency (HF) "
    "calls (screams, squeals — associated with acute pain, fear, distress) "
    "and low-frequency (LF) calls (grunts, barks — associated with "
    "lower-arousal states, both positive and negative)."
)

# ---------------------------------------------------------------------------
# Task prompt templates
# ---------------------------------------------------------------------------

PROMPT_CALL_TYPE = (
    "Listen to the provided pig vocalization. Is this a High Frequency (HF) call "
    "(scream or squeal, typically above 1 kHz fundamental frequency) or a "
    "Low Frequency (LF) call (grunt or bark, typically below 1 kHz)?\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your acoustic analysis]\n</reasoning>\n"
    "<answer>\n[HF or LF]\n</answer>"
)

PROMPT_VALENCE = (
    "Listen to the provided pig vocalization. Based on the acoustic properties of "
    "the call — including frequency, duration, harmonicity, and amplitude "
    "modulation — is the emotional valence of this animal Positive or "
    "Negative?\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your analysis of acoustic features and emotional state]\n</reasoning>\n"
    "<answer>\n[Positive or Negative]\n</answer>"
)
