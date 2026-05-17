from typing import Any

# ============================================================================
# Prompt Templates
# ============================================================================

SYSTEM_PROMPT = (
    "You are an animal bioacoustics evaluator. Classify vocalization recordings "
    "according to the requested source-dataset task. Do not infer pain, disease, "
    "stress, or welfare status unless that is exactly what the task label means."
)

MODALITY_INSTRUCTIONS = {
    "audio": "<audio>Listen to {subject}.",
    "vision": "Look at the provided spectrogram of {subject}.",
    "multimodal": (
        "<audio>You are provided with both an audio recording and its "
        "corresponding spectrogram image for {subject}."
    ),
    "text": "Review the tabular acoustic/source metadata for {subject}.",
}

ESCAPE_HATCH_NOTES = {
    "audio": (
        "NOTE: If the audio is completely missing, entirely silent, or too short "
        "to hear any animal vocalization, classify it as UNINTELLIGIBLE.\n\n"
    ),
    "vision": (
        "NOTE: If the spectrogram is missing or visually empty, classify it as "
        "UNINTELLIGIBLE.\n\n"
    ),
    "multimodal": (
        "NOTE: If both the audio and spectrogram are missing or unintelligible, "
        "classify it as UNINTELLIGIBLE.\n\n"
    ),
    "text": (
        "NOTE: If the tabular metadata is completely missing or empty, classify "
        "it as UNINTELLIGIBLE.\n\n"
    ),
}

TASK_TEMPLATE = """{modality_instruction}
{question}
{tabular_data}
{escape_hatch_note}{format_instructions}"""

FORMAT_WITH_REASONING = """Respond using this format:
<explanation>
[reasoning]
</explanation>
<answer>
[{options}]
</answer>"""

FORMAT_WITHOUT_REASONING = """Respond using this format:
<answer>
[{options}]
</answer>"""


# ============================================================================
# Task Registry
# ============================================================================

SOUNDWEL_DATASET = "oliveirabruno01/soundwel-pig-vocalizations"
CATMEOWS_DATASET = "oliveirabruno01/openfarm-catmeows"
LAYING_HEN_DATASET = "oliveirabruno01/openfarm-laying-hen-stress"

SOUNDWEL_TABULAR_KEYS = [
    "Age Category",
    "Sex",
    "Dur",
    "F0Mean",
    "Fpeak",
    "WienEntrMean",
    "Harmonicity",
    "Q25",
    "Q50",
    "Q75",
    "AMVar",
    "AMRate",
    "AMExtent",
    "DFMin",
]

CATMEOWS_TABULAR_KEYS = [
    "duration_sec",
    "rms_energy",
    "peak_abs_amplitude",
    "zero_crossing_rate",
    "spectral_centroid_hz",
    "breed",
    "sex_status",
]

LAYING_HEN_TABULAR_KEYS = [
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "mfcc_segment_count",
]

TASK_CONFIGS: dict[str, dict[str, Any]] = {
    "soundwel_call_type": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Type",
        "subject": "a pig vocalization",
        "question": "Is this a High Frequency (HF) call or a Low Frequency (LF) call?",
        "options": ["HF", "LF"],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "exclude_columns": {"Q25", "Q50", "Q75", "F0Mean", "Fpeak", "DFMin"},
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_valence": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Valence",
        "label_map": {"Pos": "Positive", "Neg": "Negative"},
        "subject": "a pig vocalization",
        "question": "Is the emotional valence of this animal Positive or Negative?",
        "options": ["Positive", "Negative"],
        "answer_aliases": {
            "Positive": ["positive", "pos"],
            "Negative": ["negative", "neg"],
        },
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_context": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Context Category",
        "subject": "a pig vocalization",
        "question": "What is the source context category of this vocalization?",
        "options": [
            "AfterNursing",
            "Barren",
            "BeforeNursing",
            "Castration",
            "Crushing",
            "Enriched",
            "Fighting",
            "Huddling",
            "Isolation",
            "MissedNursing",
            "NegativeConditioning",
            "NovelObject",
            "PositiveConditioning",
            "Restrain",
            "Reunion",
            "Run",
            "Surprise",
        ],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Age Category", "Dur", "AMRate"],
    },
    "soundwel_age": {
        "dataset_name": SOUNDWEL_DATASET,
        "label_column": "Age Category",
        "subject": "a pig vocalization",
        "question": "What is the age category of the pig?",
        "options": ["Piglet", "Weaner", "GrFinishing"],
        "tabular_keys": SOUNDWEL_TABULAR_KEYS,
        "info_keys": ["Dur", "AMRate"],
    },
    "catmeows_context": {
        "dataset_name": CATMEOWS_DATASET,
        "label_column": "context",
        "subject": "a cat vocalization",
        "question": (
            "Choose the most likely source dataset context for this recording. "
            "This is not a clinical pain, stress, or welfare diagnosis."
        ),
        "options": [
            "brushing",
            "isolation_unfamiliar_environment",
            "waiting_for_food",
        ],
        "answer_aliases": {
            "brushing": ["brushing"],
            "isolation_unfamiliar_environment": [
                "isolation_unfamiliar_environment",
                "isolation",
                "unfamiliar environment",
            ],
            "waiting_for_food": ["waiting_for_food", "waiting", "food"],
        },
        "tabular_keys": CATMEOWS_TABULAR_KEYS,
        "info_keys": ["cat_id", "duration_sec"],
        "target_pad_seconds": 0.0,
    },
    "laying_hen_stress_context": {
        "dataset_name": LAYING_HEN_DATASET,
        "label_column": "stress_context",
        "subject": "a laying-hen vocalization recording",
        "question": "Choose the source experimental stress-context label for this recording.",
        "options": [
            "poststress_control",
            "prestress_control",
            "poststress_treatment",
            "prestress_treatment",
        ],
        "tabular_keys": LAYING_HEN_TABULAR_KEYS,
        "info_keys": [
            "source_audio_id",
            "experimental_day_group",
            "duration_sec",
            "source_duration_sec",
            "excerpt_strategy",
        ],
        "max_audio_seconds": 15.0,
        "audio_clip_strategy": "even_windows",
    },
    "laying_hen_stress_binary": {
        "dataset_name": LAYING_HEN_DATASET,
        "label_column": "stress_binary",
        "subject": "a laying-hen vocalization recording",
        "question": "Classify whether this recording represents a stress response.",
        "options": ["stress_response", "not_stress_response"],
        "tabular_keys": LAYING_HEN_TABULAR_KEYS,
        "info_keys": [
            "source_audio_id",
            "experimental_day_group",
            "duration_sec",
            "source_duration_sec",
            "excerpt_strategy",
        ],
        "max_audio_seconds": 15.0,
        "audio_clip_strategy": "even_windows",
    },
}

TASK_ALIASES = {
    "call_type_classification": "soundwel_call_type",
    "valence_classification": "soundwel_valence",
    "context_classification": "soundwel_context",
    "age_classification": "soundwel_age",
    "catmeows_audio_context": "catmeows_context",
    "cat_context": "catmeows_context",
    "hen_stress_context": "laying_hen_stress_context",
    "hen_stress_binary": "laying_hen_stress_binary",
}
