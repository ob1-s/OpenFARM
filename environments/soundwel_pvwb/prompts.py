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

SYSTEM_PROMPT_VET = (
    "You are a veterinary diagnostician with expertise in precision "
    "livestock farming and swine welfare assessment. You use established "
    "ethological frameworks to interpret pig vocalizations, including "
    "frequency type (HF/LF), duration, harmonic structure, and amplitude "
    "modulation patterns."
)

# ---------------------------------------------------------------------------
# Task prompt templates
# ---------------------------------------------------------------------------

PROMPT_CALL_TYPE = (
    "Listen to this pig vocalization. Is this a High Frequency (HF) call "
    "(scream or squeal, typically above 1 kHz fundamental frequency) or a "
    "Low Frequency (LF) call (grunt or bark, typically below 1 kHz)?\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your acoustic analysis]\n</reasoning>\n"
    "<answer>\n[HF or LF]\n</answer>"
)

PROMPT_ACOUSTIC_PROPERTY = (
    "Listen to this pig vocalization. {question}\n\n"
    "{options_text}\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your acoustic analysis]\n</reasoning>\n"
    "<answer>\n[A, B, C, or D]\n</answer>"
)

PROMPT_VALENCE = (
    "Listen to this pig vocalization. Based on the acoustic properties of "
    "the call — including frequency, duration, harmonicity, and amplitude "
    "modulation — is the emotional valence of this animal Positive or "
    "Negative?\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your analysis of acoustic features and emotional state]\n</reasoning>\n"
    "<answer>\n[Positive or Negative]\n</answer>"
)

PROMPT_PAIN_RANKING = (
    "You will hear two pig vocalizations.\n\n"
    "**Vocalization A:**\n"
    "(first audio below)\n\n"
    "**Vocalization B:**\n"
    "(second audio below)\n\n"
    "Which pig is experiencing higher distress or pain? Consider the "
    "frequency type (HF vs LF), call duration, pitch, and energy.\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your comparative analysis]\n</reasoning>\n"
    "<answer>\n[A or B]\n</answer>"
)

PROMPT_CONTEXT = (
    "You are an AI assistant for precision livestock farming. "
    "This pig is a {age_category}. Based on the vocalization, which "
    "scenario is most likely occurring?\n\n"
    "{options_text}\n\n"
    "Respond using this format:\n"
    "<reasoning>\n[your analysis]\n</reasoning>\n"
    "<answer>\n[A, B, C, or D]\n</answer>"
)

PROMPT_VETERINARY = (
    "This call was recorded from a {sex} {age_category}. "
    "Analyze this pig vocalization and provide a comprehensive assessment.\n\n"
    "You must:\n"
    "1. Identify whether this is a High Frequency (HF) or Low Frequency (LF) call\n"
    "2. Assess the emotional valence (Positive or Negative)\n"
    "3. Infer the most likely behavioral context\n"
    "4. Explain the physiological and emotional state\n\n"
    "Respond using this format:\n"
    "<analysis>\n[your detailed ethological analysis]\n</analysis>\n"
    "<call_type>\n[HF or LF]\n</call_type>\n"
    "<valence>\n[Positive or Negative]\n</valence>\n"
    "<context>\n[most likely context category]\n</context>"
)

# ---------------------------------------------------------------------------
# Judge prompt for Tier 4 veterinary reasoning
# ---------------------------------------------------------------------------

JUDGE_PROMPT_REASONING = """Evaluate this veterinary analysis of a pig vocalization.

[GROUND TRUTH METADATA]
- Call Type: {call_type}
- Valence: {valence}
- Context: {context}
- Age: {age}
- Sex: {sex}

[GROUND TRUTH ACOUSTIC FEATURES]
- Duration: {duration} seconds
- Spectral Center of Gravity (Q50%, proxy for pitch): {q50} Hz
- Peak Frequency: {fpeak} Hz
- Amplitude Modulation Rate: {amp_mod_rate} Hz
(Note: The model did not see these exact numbers, but its acoustic description must align with them. E.g., if duration is 0.15s, it should not describe a "prolonged, long-lasting" call. If Q50% is 3500Hz, it should not describe a "low-pitched" call.)

[MODEL'S ANALYSIS]
{analysis}

Score the analysis from 0 to 10 based on ethological and veterinary accuracy:
- Does it correctly identify HF vs LF call characteristics?
- Does it correctly reason about emotional valence?
- ANTI-HALLUCINATION CHECK: Does the acoustic description logically align with the ground truth acoustic features provided above? Penalize heavily if the model hallucinates acoustic properties.
- Does it demonstrate understanding of pig vocal communication mechanisms (e.g., HF calls from vocal fold vibration under tension, LF calls from relaxed vocal apparatus)?
- Is the contextual inference reasonable given the acoustic description?
- Does it avoid hallucinating fake ethological concepts?

Respond in the following format:
<thought>
Briefly explain your reasoning for the score. Explicitly state whether the model's acoustic description aligns with the ground truth acoustic features.
</thought>
<score>
A single number from 0 to 10
</score>"""
