from __future__ import annotations

import base64
import io
import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from openai import OpenAI
from PIL import Image

try:
    import spaces
except ImportError:
    class _SpacesNoop:
        def GPU(self, *args, **kwargs):
            if args and callable(args[0]):
                return args[0]

            def decorator(fn):
                return fn

            return decorator

    spaces = _SpacesNoop()
else:
    if not hasattr(spaces, "GPU"):
        class _SpacesNoop:
            def GPU(self, *args, **kwargs):
                if args and callable(args[0]):
                    return args[0]

                def decorator(fn):
                    return fn

                return decorator

        spaces = _SpacesNoop()


APP_DIR = Path(__file__).parent
GUIDE_DIR = APP_DIR / "species_guides"
DEFAULT_MODEL = os.getenv("OPENFARM_MODEL", "gemma-4-E2B-it")
LOCAL_MODEL_ID = os.getenv(
    "OPENFARM_LOCAL_MODEL",
    "unsloth/gemma-4-E2B-it-unsloth-bnb-4bit",
)
USE_LOCAL_GEMMA = os.getenv("OPENFARM_USE_LOCAL_GEMMA", "0").lower() in {"1", "true", "yes"}
DEFAULT_BASE_URL = os.getenv("OPENFARM_LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY = (
    os.getenv("OPENFARM_LLM_API_KEY")
    or os.getenv("PRIME_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)

SPECIES = ["cat", "dog", "horse", "sheep", "cattle", "pig", "rabbit"]
ANNOTATION_LABELS = [
    "Agree",
    "Partly agree",
    "Disagree",
    "Needs more context",
    "Unsafe overclaim",
    "Missed concern",
    "Other / free text",
]
ASSESSMENT_LABELS = [
    "Needs more context",
    "Neutral or baseline",
    "Possible stress or fear",
    "Possible discomfort",
    "Possible pain cue",
    "Positive welfare cue",
]
ASSESSMENT_TOOL = {
    "type": "function",
    "function": {
        "name": "record_assessment",
        "description": "Record a structured animal-welfare assessment for the current observation.",
        "parameters": {
            "type": "object",
            "properties": {
                "label": {"type": "string", "enum": ASSESSMENT_LABELS},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "summary": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "uncertainty": {"type": "array", "items": {"type": "string"}},
                "next_steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["label", "confidence", "summary", "evidence", "uncertainty", "next_steps"],
        },
    },
}

LEADERBOARD_ROWS = [
    {
        "Use case": "General multimodal assistant",
        "Model": os.getenv("OPENFARM_MODEL", "Gemma 4 E2B or OpenFARM finetune"),
        "Status": "Configured by Space secret",
        "Best for": "Live demo chat, images, profile context",
        "Notes": "Server-side endpoint keeps the public Space no-login.",
    },
    {
        "Use case": "Text RL training artifact",
        "Model": "Gemma 4 E2B + Unsloth GRPO",
        "Status": "Notebook",
        "Best for": "AWW + dog pain + bioacoustic metadata EnvGroup",
        "Notes": "See notebooks/unsloth/RL_Gemma4_Text_Verifiers_OpenFARM_AWW.ipynb.",
    },
    {
        "Use case": "External multimodal baseline",
        "Model": "Gemini 3 Flash Preview",
        "Status": "Prime eval configs",
        "Best for": "Fast smoke tests across visual/audio envs",
        "Notes": "Replace this table with exported Prime eval results after final sweeps.",
    },
]
_LOCAL_MODEL = None
_LOCAL_PROCESSOR = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_guides() -> dict[str, str]:
    guides: dict[str, str] = {}
    for path in GUIDE_DIR.glob("*.md"):
        guides[path.stem] = path.read_text(encoding="utf-8")
    return guides


GUIDES = load_guides()


def initial_state() -> dict[str, Any]:
    thread_id = "thread-001"
    return {
        "threads": {
            thread_id: {
                "id": thread_id,
                "title": "New welfare chat",
                "created_at": now_iso(),
                "updated_at": now_iso(),
                "species": "cat",
                "animal_id": "",
                "messages": [],
                "turns": [],
                "assessments": [],
            }
        },
        "active_thread_id": thread_id,
        "next_thread": 2,
        "next_assessment": 1,
        "animals": {},
        "annotations": [],
    }


def ensure_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if not state or "threads" not in state:
        return initial_state()
    if not state.get("active_thread_id") and state["threads"]:
        state["active_thread_id"] = next(iter(state["threads"]))
    return state


def species_guide(species: str) -> str:
    return GUIDES.get(species, "")


def species_preview(species: str) -> str:
    guide = species_guide(species)
    if not guide:
        return "No guide loaded for this species."
    lines = [line for line in guide.splitlines() if line.strip()]
    return "\n".join(lines[:10])


def file_path(file_value: Any) -> str | None:
    if not file_value:
        return None
    if isinstance(file_value, str):
        return file_value
    if isinstance(file_value, dict):
        return file_value.get("path") or file_value.get("name")
    return getattr(file_value, "path", None) or getattr(file_value, "name", None)


def file_name(file_value: Any) -> str:
    path = file_path(file_value)
    return Path(path).name if path else "attachment"


def file_kind(file_value: Any) -> str:
    path = (file_path(file_value) or "").lower()
    if path.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
        return "image"
    if path.endswith((".wav", ".mp3", ".m4a", ".ogg", ".flac")):
        return "audio"
    if path.endswith((".mp4", ".mov", ".webm", ".mkv")):
        return "video"
    return "file"


def encode_bytes_data_uri(path: str, mime: str) -> str:
    data = Path(path).read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii")


def image_data_uri(path: str, max_side: int = 1152) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def media_parts(files: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    parts: list[dict[str, Any]] = []
    summaries: list[str] = []
    for file_value in files or []:
        path = file_path(file_value)
        if not path:
            continue
        name = Path(path).name
        kind = file_kind(path)
        summaries.append(f"{kind}: {name}")
        if kind == "image":
            parts.append({"type": "image_url", "image_url": {"url": image_data_uri(path)}})
        elif kind == "audio":
            ext = Path(path).suffix.lower().lstrip(".") or "wav"
            audio_b64 = base64.b64encode(Path(path).read_bytes()).decode("ascii")
            parts.append(
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": ext}}
            )
        elif kind == "video":
            mime = "video/mp4" if path.lower().endswith(".mp4") else "video/webm"
            parts.append({"type": "video_url", "video_url": {"url": encode_bytes_data_uri(path, mime)}})
    return parts, summaries


def local_multimodal_content(text: str, files: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    summaries: list[str] = []
    for file_value in files or []:
        path = file_path(file_value)
        if not path:
            continue
        kind = file_kind(path)
        summaries.append(f"{kind}: {Path(path).name}")
        if kind == "image":
            content.append({"type": "image", "image": Image.open(path).convert("RGB")})
    return content, summaries


def thread_label(thread: dict[str, Any]) -> str:
    species = thread.get("species") or "species?"
    title = thread.get("title") or thread["id"]
    return f"{thread['id']} | {title} | {species}"


def thread_choices(state: dict[str, Any]) -> list[str]:
    threads = sorted(
        state["threads"].values(),
        key=lambda item: item.get("updated_at", ""),
        reverse=True,
    )
    return [thread_label(thread) for thread in threads]


def label_to_thread_id(label: str | None, state: dict[str, Any]) -> str:
    if label:
        candidate = label.split("|", 1)[0].strip()
        if candidate in state["threads"]:
            return candidate
    return state.get("active_thread_id") or next(iter(state["threads"]))


def current_thread(state: dict[str, Any]) -> dict[str, Any]:
    state = ensure_state(state)
    return state["threads"][state["active_thread_id"]]


def normalize_message_value(value: dict[str, Any] | None) -> tuple[str, list[Any]]:
    if not value:
        return "", []
    if isinstance(value, str):
        return value.strip(), []
    text = str(value.get("text") or "").strip()
    files = value.get("files") or []
    return text, files


def animal_context(state: dict[str, Any], animal_id: str) -> dict[str, Any] | None:
    animal_id = (animal_id or "").strip()
    if not animal_id:
        return None
    return state.get("animals", {}).get(animal_id)


def build_system_prompt(species: str) -> str:
    guide = species_guide(species)[:4600]
    return textwrap.dedent(
        f"""
        You are OpenFARM Welfare Assistant, a cautious multimodal animal-welfare
        support tool for vets, caregivers, researchers, and annotators.

        You may ask follow-up questions, compare against animal profile context,
        or make an ASSESSMENT call when there is enough evidence to summarize
        the current welfare concern. Never diagnose disease, prescribe treatment,
        or claim pain from one image alone. Recommend veterinary care when signs
        are urgent, severe, or worsening.

        When you make a final assessment, include exactly one XML block:
        <assessment>
        {{"label": "...", "confidence": 0.0, "summary": "...", "evidence": ["..."], "uncertainty": ["..."], "next_steps": ["..."]}}
        </assessment>

        If your runtime supports tool calls, you may instead call the
        record_assessment tool with the same fields. In either case, keep the
        user-visible answer concise and contestable.

        Valid labels: {", ".join(ASSESSMENT_LABELS)}.
        Outside that XML block, speak naturally and concisely to the user.

        Species guide:
        {guide}
        """
    ).strip()


def build_context_text(
    state: dict[str, Any],
    thread: dict[str, Any],
    animal_id: str,
    species: str,
) -> str:
    animal = animal_context(state, animal_id)
    animal_lines = "No saved animal profile selected."
    if animal:
        history = animal.get("history", [])
        recent = "\n".join(
            f"- {item.get('timestamp')}: {item.get('label')}: {item.get('summary')}"
            for item in history[-5:]
        )
        animal_lines = textwrap.dedent(
            f"""
            Saved animal profile:
            - ID: {animal.get('id')}
            - Name: {animal.get('name') or 'not provided'}
            - Species: {animal.get('species') or species}
            - Notes: {animal.get('notes') or 'not provided'}
            Recent profile history:
            {recent or 'No assessment history saved to this animal.'}
            """
        ).strip()

    previous = []
    for turn in thread.get("turns", [])[-6:]:
        role = turn.get("role")
        text = turn.get("text", "")
        previous.append(f"- {role}: {text[:300]}")
    return textwrap.dedent(
        f"""
        Current optional context:
        - Thread ID: {thread['id']}
        - Species: {species}
        - Animal ID: {animal_id or 'not selected'}

        {animal_lines}

        Recent conversation:
        {chr(10).join(previous) if previous else 'No prior turns in this thread.'}
        """
    ).strip()


def parse_assessment(text: str) -> tuple[str, dict[str, Any] | None]:
    match = re.search(r"<assessment>\s*(.*?)\s*</assessment>", text, flags=re.S | re.I)
    if not match:
        return text.strip(), None
    visible = (text[: match.start()] + text[match.end() :]).strip()
    raw = match.group(1).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {
            "label": "Needs more context",
            "confidence": 0.0,
            "summary": raw,
            "evidence": [],
            "uncertainty": ["Assessment JSON could not be parsed."],
            "next_steps": ["Ask for a clearer follow-up observation."],
        }
    return visible or "Assessment recorded.", parsed


def parse_tool_assessment(message: Any) -> tuple[str, dict[str, Any] | None]:
    raw = getattr(message, "content", None) or ""
    for tool_call in getattr(message, "tool_calls", None) or []:
        function = getattr(tool_call, "function", None)
        if getattr(function, "name", "") != "record_assessment":
            continue
        try:
            assessment = json.loads(getattr(function, "arguments", "") or "{}")
        except json.JSONDecodeError:
            assessment = {
                "label": "Needs more context",
                "confidence": 0.0,
                "summary": getattr(function, "arguments", "") or "",
                "evidence": [],
                "uncertainty": ["Tool-call arguments could not be parsed as JSON."],
                "next_steps": ["Ask for a clearer follow-up observation."],
            }
        return raw.strip() or "Assessment recorded.", assessment
    return parse_assessment(raw)


def load_local_model():
    global _LOCAL_MODEL, _LOCAL_PROCESSOR
    if _LOCAL_MODEL is not None and _LOCAL_PROCESSOR is not None:
        return _LOCAL_MODEL, _LOCAL_PROCESSOR

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    _LOCAL_PROCESSOR = AutoProcessor.from_pretrained(
        LOCAL_MODEL_ID,
        trust_remote_code=True,
    )
    _LOCAL_MODEL = AutoModelForImageTextToText.from_pretrained(
        LOCAL_MODEL_ID,
        device_map="auto",
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    _LOCAL_MODEL.eval()
    return _LOCAL_MODEL, _LOCAL_PROCESSOR


@spaces.GPU(duration=120)
def local_gemma_reply(
    state: dict[str, Any],
    thread: dict[str, Any],
    text: str,
    files: list[Any],
    animal_id: str,
    species: str,
    force_reassessment: str = "",
) -> str:
    import torch

    prompt_text = (
        build_context_text(state, thread, animal_id, species)
        + "\n\nCurrent user message:\n"
        + (text or "(no text)")
        + (f"\n\nContest/revision request:\n{force_reassessment}" if force_reassessment else "")
    )
    content, summaries = local_multimodal_content(prompt_text, files)
    if summaries:
        content[0]["text"] += "\n\nAttached media:\n" + "\n".join(f"- {item}" for item in summaries)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt(species)}]},
        {"role": "user", "content": content},
    ]
    model, processor = load_local_model()

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=900,
            do_sample=True,
            temperature=0.35,
            top_p=0.9,
        )
    prompt_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][prompt_len:]
    return processor.decode(generated, skip_special_tokens=True).strip()


def offline_reply(
    state: dict[str, Any],
    thread: dict[str, Any],
    text: str,
    files: list[Any],
    animal_id: str,
    species: str,
) -> str:
    _, summaries = media_parts(files)
    animal = animal_context(state, animal_id)
    baseline_note = (
        "A saved animal profile is available, so compare against that individual baseline."
        if animal
        else "No saved animal profile is selected, so keep confidence low."
    )
    return textwrap.dedent(
        f"""
        I can help structure this welfare assessment. {baseline_note}

        I see this turn includes: {', '.join(summaries) if summaries else 'text only'}.
        For {species}, the useful next checks are behavior, posture, appetite,
        movement, recent procedures, medication, and whether this differs from a
        normal day for this individual.

        <assessment>
        {{"label": "Needs more context", "confidence": 0.35, "summary": "Demo-mode assessment: the app has no model endpoint configured, but the workflow captured the observation and profile context.", "evidence": ["User note: {text[:120].replace(chr(34), chr(39))}", "Media attached: {len(files or [])}"], "uncertainty": ["No live model endpoint is configured.", "Single observations should not be treated as diagnosis."], "next_steps": ["Add baseline images or a saved animal profile.", "Record appetite, posture, movement, and recent clinical context."]}}
        </assessment>
        """
    ).strip()


def call_model(
    state: dict[str, Any],
    thread: dict[str, Any],
    text: str,
    files: list[Any],
    animal_id: str,
    species: str,
    force_reassessment: str = "",
) -> tuple[str, dict[str, Any] | None, str]:
    if USE_LOCAL_GEMMA:
        try:
            raw = local_gemma_reply(state, thread, text, files, animal_id, species, force_reassessment)
            visible, assessment = parse_assessment(raw)
            return visible, assessment, f"Local ZeroGPU model: {LOCAL_MODEL_ID}"
        except Exception as exc:
            if not API_KEY:
                visible, assessment = parse_assessment(
                    offline_reply(state, thread, text, files, animal_id, species)
                )
                return visible, assessment, f"Local Gemma failed; demo fallback active: {exc}"

    if not API_KEY:
        visible, assessment = parse_assessment(
            offline_reply(state, thread, text, files, animal_id, species)
        )
        return visible, assessment, "Demo mode: configure Space secrets for a live model."

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_context_text(state, thread, animal_id, species)
            + "\n\nCurrent user message:\n"
            + (text or "(no text)")
            + (f"\n\nContest/revision request:\n{force_reassessment}" if force_reassessment else ""),
        }
    ]
    parts, _ = media_parts(files)
    content.extend(parts)

    client = OpenAI(api_key=API_KEY, base_url=DEFAULT_BASE_URL)
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": build_system_prompt(species)},
            {"role": "user", "content": content},
        ],
        tools=[ASSESSMENT_TOOL],
        tool_choice="auto",
        temperature=0.25,
        max_tokens=1000,
    )
    visible, assessment = parse_tool_assessment(response.choices[0].message)
    return visible, assessment, f"Model: {DEFAULT_MODEL}"


def assessment_card(assessment: dict[str, Any] | None) -> str:
    if not assessment:
        return "No assessment call in the latest assistant message."

    def bullets(items: list[Any] | None) -> str:
        values = [str(item).strip() for item in (items or [])[:5] if str(item).strip()]
        return "\n".join(f"- {item}" for item in values) if values else "- Not specified"

    return "\n\n".join(
        [
            "### Latest Assessment",
            f"**Label:** {assessment.get('label', 'Needs more context')}",
            f"**Confidence:** {assessment.get('confidence', 0)}",
            f"**Summary:** {assessment.get('summary', '')}",
            "**Evidence**\n" + bullets(assessment.get("evidence", [])),
            "**Uncertainty**\n" + bullets(assessment.get("uncertainty", [])),
            "**Next Steps**\n" + bullets(assessment.get("next_steps", [])),
        ]
    ).strip()


def conversations_frame(state: dict[str, Any]) -> pd.DataFrame:
    state = ensure_state(state)
    rows = []
    for thread in sorted(state["threads"].values(), key=lambda x: x.get("updated_at", ""), reverse=True):
        latest = thread.get("assessments", [])[-1] if thread.get("assessments") else {}
        rows.append(
            {
                "thread": thread["id"],
                "title": thread.get("title", ""),
                "animal": thread.get("animal_id", ""),
                "species": thread.get("species", ""),
                "turns": len(thread.get("turns", [])),
                "latest_label": latest.get("label", ""),
                "updated": thread.get("updated_at", ""),
            }
        )
    return pd.DataFrame(rows)


def animals_frame(state: dict[str, Any]) -> pd.DataFrame:
    state = ensure_state(state)
    rows = []
    for animal in state.get("animals", {}).values():
        rows.append(
            {
                "id": animal.get("id", ""),
                "name": animal.get("name", ""),
                "species": animal.get("species", ""),
                "history_items": len(animal.get("history", [])),
                "notes": animal.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def annotations_frame(state: dict[str, Any]) -> pd.DataFrame:
    state = ensure_state(state)
    rows = []
    for item in state.get("annotations", []):
        rows.append(
            {
                "time": item.get("timestamp", ""),
                "thread": item.get("thread_id", ""),
                "assessment": item.get("assessment_id", ""),
                "verdict": item.get("verdict", ""),
                "polarity": item.get("polarity", ""),
                "notes": item.get("notes", ""),
            }
        )
    return pd.DataFrame(rows)


def active_outputs(state: dict[str, Any]):
    thread = current_thread(state)
    latest = thread.get("assessments", [])[-1] if thread.get("assessments") else None
    choices = thread_choices(state)
    active = thread_label(thread)
    return (
        gr.update(choices=choices, value=active),
        thread.get("messages", []),
        assessment_card(latest),
        conversations_frame(state),
        annotations_frame(state),
        animals_frame(state),
    )


def new_thread(state: dict[str, Any]):
    state = ensure_state(state)
    idx = int(state.get("next_thread", 1))
    thread_id = f"thread-{idx:03d}"
    state["next_thread"] = idx + 1
    state["threads"][thread_id] = {
        "id": thread_id,
        "title": "New welfare chat",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "species": "cat",
        "animal_id": "",
        "messages": [],
        "turns": [],
        "assessments": [],
    }
    state["active_thread_id"] = thread_id
    radio, messages, latest, conv, ann, animals = active_outputs(state)
    return state, radio, messages, latest, conv, ann, animals, "Started a new thread."


def switch_thread(selected: str, state: dict[str, Any]):
    state = ensure_state(state)
    state["active_thread_id"] = label_to_thread_id(selected, state)
    radio, messages, latest, conv, ann, animals = active_outputs(state)
    return state, radio, messages, latest, conv, ann, animals, "Thread loaded."


def send_message(
    message: dict[str, Any] | None,
    state: dict[str, Any],
    selected_thread: str,
    animal_id: str,
    species: str,
):
    state = ensure_state(state)
    state["active_thread_id"] = label_to_thread_id(selected_thread, state)
    thread = current_thread(state)
    text, files = normalize_message_value(message)
    if not text and not files:
        radio, messages, latest, conv, ann, animals = active_outputs(state)
        return state, radio, messages, latest, conv, ann, animals, gr.update(value=None), "Write a message or attach media."

    _, summaries = media_parts(files)
    shown_text = text or "(media only)"
    if summaries:
        shown_text += "\n\nAttached media:\n" + "\n".join(f"- {item}" for item in summaries)
    thread["messages"].append({"role": "user", "content": shown_text})
    thread["turns"].append({"role": "user", "text": text, "media": summaries, "timestamp": now_iso()})
    thread["species"] = species
    thread["animal_id"] = (animal_id or "").strip()
    if thread.get("title") == "New welfare chat":
        title_seed = text.strip() or ", ".join(summaries) or "Welfare chat"
        thread["title"] = title_seed[:44]

    assistant_text, assessment, status = call_model(
        state=state,
        thread=thread,
        text=text,
        files=files,
        animal_id=animal_id,
        species=species,
    )
    if assessment:
        assessment_id = f"assessment-{state['next_assessment']:03d}"
        state["next_assessment"] += 1
        assessment.update(
            {
                "id": assessment_id,
                "thread_id": thread["id"],
                "animal_id": (animal_id or "").strip(),
                "species": species,
                "timestamp": now_iso(),
            }
        )
        thread["assessments"].append(assessment)
        assistant_text += f"\n\nAssessment call recorded: `{assessment_id}`."

    thread["messages"].append({"role": "assistant", "content": assistant_text})
    thread["turns"].append({"role": "assistant", "text": assistant_text, "timestamp": now_iso()})
    thread["updated_at"] = now_iso()

    radio, messages, latest, conv, ann, animals = active_outputs(state)
    return state, radio, messages, latest, conv, ann, animals, gr.update(value=None), status


def annotate_latest(
    state: dict[str, Any],
    selected_thread: str,
    verdict: str,
    custom_verdict: str,
    polarity: str,
    notes: str,
):
    state = ensure_state(state)
    state["active_thread_id"] = label_to_thread_id(selected_thread, state)
    thread = current_thread(state)
    latest = thread.get("assessments", [])[-1] if thread.get("assessments") else None
    if not latest:
        radio, messages, card, conv, ann, animals = active_outputs(state)
        return state, radio, messages, card, conv, ann, animals, "No assessment to annotate yet."
    custom_verdict = (custom_verdict or "").strip()
    resolved_verdict = custom_verdict if custom_verdict else verdict
    state["annotations"].append(
        {
            "timestamp": now_iso(),
            "thread_id": thread["id"],
            "assessment_id": latest.get("id", ""),
            "verdict": resolved_verdict,
            "verdict_option": verdict,
            "polarity": polarity,
            "notes": notes or "",
        }
    )
    radio, messages, card, conv, ann, animals = active_outputs(state)
    return state, radio, messages, card, conv, ann, animals, "Annotation saved."


def contest_and_continue(
    state: dict[str, Any],
    selected_thread: str,
    animal_id: str,
    species: str,
    custom_verdict: str,
    contest_note: str,
):
    state = ensure_state(state)
    state["active_thread_id"] = label_to_thread_id(selected_thread, state)
    thread = current_thread(state)
    latest = thread.get("assessments", [])[-1] if thread.get("assessments") else None
    if not latest:
        radio, messages, card, conv, ann, animals = active_outputs(state)
        return state, radio, messages, card, conv, ann, animals, "No assessment to contest yet."

    custom_verdict = (custom_verdict or "").strip()
    contest_note = (contest_note or "").strip()
    user_text = contest_note or custom_verdict or "I contest the previous assessment. Please revise it and explain what changed."
    thread["messages"].append({"role": "user", "content": user_text})
    thread["turns"].append({"role": "user", "text": user_text, "media": [], "timestamp": now_iso()})

    assistant_text, assessment, status = call_model(
        state=state,
        thread=thread,
        text=user_text,
        files=[],
        animal_id=animal_id,
        species=species,
        force_reassessment=f"Previous assessment: {json.dumps(latest)}\nUser contest: {user_text}",
    )
    if assessment:
        assessment_id = f"assessment-{state['next_assessment']:03d}"
        state["next_assessment"] += 1
        assessment.update(
            {
                "id": assessment_id,
                "thread_id": thread["id"],
                "animal_id": (animal_id or "").strip(),
                "species": species,
                "timestamp": now_iso(),
                "revises": latest.get("id", ""),
            }
        )
        thread["assessments"].append(assessment)
        assistant_text += f"\n\nRevised assessment call recorded: `{assessment_id}`."
    thread["messages"].append({"role": "assistant", "content": assistant_text})
    thread["turns"].append({"role": "assistant", "text": assistant_text, "timestamp": now_iso()})
    thread["updated_at"] = now_iso()
    radio, messages, card, conv, ann, animals = active_outputs(state)
    return state, radio, messages, card, conv, ann, animals, status


def save_animal(state, animal_id, animal_name, animal_species, animal_notes):
    state = ensure_state(state)
    animal_id = (animal_id or animal_name or f"animal-{len(state['animals']) + 1:03d}").strip()
    existing = state["animals"].get(animal_id, {})
    state["animals"][animal_id] = {
        "id": animal_id,
        "name": (animal_name or existing.get("name") or animal_id).strip(),
        "species": animal_species,
        "notes": (animal_notes or existing.get("notes") or "").strip(),
        "history": existing.get("history", []),
        "updated_at": now_iso(),
    }
    return state, animals_frame(state), f"Saved animal profile `{animal_id}`."


def add_latest_to_animal(state, selected_thread, animal_id, animal_name, animal_species, animal_notes):
    state = ensure_state(state)
    state["active_thread_id"] = label_to_thread_id(selected_thread, state)
    thread = current_thread(state)
    latest = thread.get("assessments", [])[-1] if thread.get("assessments") else None
    if not latest:
        return state, animals_frame(state), "No assessment to add to an animal history."
    state, _, _ = save_animal(state, animal_id, animal_name, animal_species, animal_notes)
    resolved_id = (animal_id or animal_name or "").strip()
    if not resolved_id:
        resolved_id = next(reversed(state["animals"]))
    state["animals"][resolved_id]["history"].append(
        {
            "timestamp": now_iso(),
            "assessment_id": latest.get("id", ""),
            "label": latest.get("label", ""),
            "summary": latest.get("summary", ""),
            "thread_id": thread["id"],
        }
    )
    return state, animals_frame(state), f"Added `{latest.get('id')}` to `{resolved_id}` history."


def export_session(state: dict[str, Any]):
    state = ensure_state(state)
    return {
        "schema": "openfarm_welfare_assistant_demo_v1",
        "exported_at": now_iso(),
        "threads": state.get("threads", {}),
        "animals": state.get("animals", {}),
        "annotations": state.get("annotations", []),
    }


def leaderboard_frame() -> pd.DataFrame:
    return pd.DataFrame(LEADERBOARD_ROWS)


CSS = """
:root {
  --ofarm-green: #087a55;
}
.gradio-container {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.ofarm-title h1 {
  margin: 0 0 0.15rem;
  letter-spacing: 0;
}
.ofarm-title p {
  color: var(--body-text-color-subdued);
  margin-top: 0.25rem;
  opacity: 0.82;
}
.ofarm-sidebar {
  border-right: 1px solid var(--border-color-primary);
  padding-right: 0.75rem;
}
.ofarm-card {
  border: 1px solid var(--border-color-primary);
  background: var(--block-background-fill);
  color: var(--body-text-color);
  border-radius: 8px;
  padding: 0.85rem;
}
.ofarm-card h3 {
  margin-top: 0;
}
.ofarm-card ul {
  margin-bottom: 0.75rem;
}
.ofarm-subtle-note {
  color: var(--body-text-color-subdued);
  margin: -0.25rem 0 0.9rem;
  font-size: 0.86rem;
  opacity: 0.74;
}
button.primary {
  background: var(--ofarm-green) !important;
}
button {
  transition: transform 140ms cubic-bezier(0.23, 1, 0.32, 1);
}
button:active {
  transform: scale(0.98);
}
"""


state0 = initial_state()

with gr.Blocks(css=CSS, title="OpenFARM Welfare Assistant") as demo:
    app_state = gr.State(state0)

    gr.Markdown(
        """
        # OpenFARM Welfare Assistant
        Chat-first animal welfare assessment, contestable model assessments, ephemeral profiles, and annotation export.
        """,
        elem_classes=["ofarm-title"],
    )
    gr.Markdown(
        "Demo session only: conversations, animal profiles, and annotations stay in browser/app memory and may disappear after refresh or restart.",
        elem_classes=["ofarm-subtle-note"],
    )

    with gr.Tabs():
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=2, min_width=240, elem_classes=["ofarm-sidebar"]):
                    new_btn = gr.Button("New chat", variant="primary")
                    thread_radio = gr.Radio(
                        choices=thread_choices(state0),
                        value=thread_label(current_thread(state0)),
                        label="Conversations",
                    )
                    with gr.Accordion("Optional context", open=True):
                        chat_animal_id = gr.Textbox(label="Animal ID", placeholder="miso-cat-001")
                        chat_species = gr.Radio(SPECIES, value="cat", label="Species")
                        species_notes = gr.Markdown(species_preview("cat"))

                with gr.Column(scale=5, min_width=420):
                    chatbot = gr.Chatbot(
                        label="OpenFARM chat",
                        type="messages",
                        height=560,
                        show_copy_button=True,
                        placeholder="Start with a question, image, audio clip, or video.",
                    )
                    composer = gr.MultimodalTextbox(
                        label="Message",
                        placeholder="Ask about welfare context or upload image/audio/video...",
                        file_count="multiple",
                        sources=["upload", "microphone"],
                        submit_btn=True,
                    )
                    status = gr.Textbox(label="Status", interactive=False)

                with gr.Column(scale=3, min_width=320):
                    latest_card = gr.Markdown(
                        "No assessment call in the latest assistant message.",
                        elem_classes=["ofarm-card"],
                    )
                    gr.Markdown("### Contest or Annotate")
                    annotation_verdict = gr.Radio(
                        ANNOTATION_LABELS,
                        value="Partly agree",
                        label="Vet/annotator verdict",
                    )
                    annotation_more = gr.Textbox(
                        label="Other verdict or correction",
                        lines=2,
                        placeholder="Optional free-form verdict, label, or correction...",
                    )
                    annotation_polarity = gr.Radio(
                        ["positive", "negative", "neutral"],
                        value="neutral",
                        label="Annotation polarity",
                    )
                    annotation_notes = gr.Textbox(
                        label="Annotation notes",
                        lines=3,
                        placeholder="Why is the assessment right, incomplete, or wrong?",
                    )
                    with gr.Row():
                        annotate_btn = gr.Button("Save annotation", variant="primary")
                        contest_btn = gr.Button("Contest and continue")
                    add_history_btn = gr.Button("Add latest assessment to animal history")

        with gr.Tab("Animals"):
            with gr.Row():
                with gr.Column(scale=2):
                    animal_id = gr.Textbox(label="Animal ID", placeholder="miso-cat-001")
                    animal_name = gr.Textbox(label="Name", placeholder="Miso")
                    animal_species = gr.Radio(SPECIES, value="cat", label="Species")
                    animal_notes = gr.Textbox(
                        label="Profile notes",
                        lines=5,
                        placeholder="Baseline temperament, morphology, conditions, normal face, medication...",
                    )
                    save_animal_btn = gr.Button("Save animal profile", variant="primary")
                    animal_status = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=5):
                    animals_table = gr.Dataframe(
                        value=animals_frame(state0),
                        interactive=False,
                        label="Session animal profiles",
                    )

        with gr.Tab("Leaderboard"):
            gr.Markdown(
                "Model choice view for the live assistant. Replace or extend this with exported Prime eval results before final publishing."
            )
            gr.Dataframe(value=leaderboard_frame(), interactive=False, label="OpenFARM model matrix")

        with gr.Tab("Archive"):
            conversations_table = gr.Dataframe(
                value=conversations_frame(state0),
                interactive=False,
                label="Threads",
            )
            annotations_table = gr.Dataframe(
                value=annotations_frame(state0),
                interactive=False,
                label="Annotations",
            )
            export_btn = gr.Button("Export session JSON")
            export_json = gr.JSON(label="Session export")

    species_notes_inputs = [chat_species]
    chat_species.change(species_preview, inputs=species_notes_inputs, outputs=species_notes)

    new_btn.click(
        new_thread,
        inputs=app_state,
        outputs=[
            app_state,
            thread_radio,
            chatbot,
            latest_card,
            conversations_table,
            annotations_table,
            animals_table,
            status,
        ],
    )
    thread_radio.change(
        switch_thread,
        inputs=[thread_radio, app_state],
        outputs=[
            app_state,
            thread_radio,
            chatbot,
            latest_card,
            conversations_table,
            annotations_table,
            animals_table,
            status,
        ],
    )
    composer.submit(
        send_message,
        inputs=[composer, app_state, thread_radio, chat_animal_id, chat_species],
        outputs=[
            app_state,
            thread_radio,
            chatbot,
            latest_card,
            conversations_table,
            annotations_table,
            animals_table,
            composer,
            status,
        ],
    )
    annotate_btn.click(
        annotate_latest,
        inputs=[
            app_state,
            thread_radio,
            annotation_verdict,
            annotation_more,
            annotation_polarity,
            annotation_notes,
        ],
        outputs=[
            app_state,
            thread_radio,
            chatbot,
            latest_card,
            conversations_table,
            annotations_table,
            animals_table,
            status,
        ],
    )
    contest_btn.click(
        contest_and_continue,
        inputs=[app_state, thread_radio, chat_animal_id, chat_species, annotation_more, annotation_notes],
        outputs=[
            app_state,
            thread_radio,
            chatbot,
            latest_card,
            conversations_table,
            annotations_table,
            animals_table,
            status,
        ],
    )
    save_animal_btn.click(
        save_animal,
        inputs=[app_state, animal_id, animal_name, animal_species, animal_notes],
        outputs=[app_state, animals_table, animal_status],
    )
    add_history_btn.click(
        add_latest_to_animal,
        inputs=[
            app_state,
            thread_radio,
            chat_animal_id,
            animal_name,
            chat_species,
            animal_notes,
        ],
        outputs=[app_state, animals_table, status],
    )
    export_btn.click(export_session, inputs=app_state, outputs=export_json)


if __name__ == "__main__":
    demo.launch()
