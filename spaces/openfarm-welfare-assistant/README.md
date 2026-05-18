---
title: OpenFARM Welfare Assistant
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: apache-2.0
python_version: 3.11
suggested_hardware: zero-a10g
---

# OpenFARM Welfare Assistant

Public Gradio demo for multimodal animal-welfare assessment, contestable model
assessments, lightweight annotation, and ephemeral animal profile tracking.

This Space can use an OpenAI-compatible Gemma/OpenFARM endpoint when configured.
The app code also includes a local 4-bit Gemma 4 E2B path for ZeroGPU, but the
public Space is currently deployed in CPU/demo mode because ZeroGPU access is
account-gated by Hugging Face.

## Demo Shape

- Chat-first interface with per-turn text and media uploads.
- Thread list for reopening prior conversations during the same Space session.
- Optional animal ID and species context on each chat.
- Assistant responses can include a structured `ASSESSMENT` call.
- Users can annotate, contest, continue from, or save an assessment to an animal
  profile history.
- Animals, annotations, conversations, and export are stored in memory only for
  the public demo session.
- Leaderboard tab is ready for exported Prime eval results and model-choice
  notes.

## Space Secrets

Set these in the Hugging Face Space settings:

- `OPENFARM_USE_LOCAL_GEMMA`: set to `1` for local ZeroGPU inference after
  adding the local model dependencies and enabling ZeroGPU hardware.
- `OPENFARM_LOCAL_MODEL`: defaults to
  `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`.
- `OPENFARM_LLM_API_KEY`: OpenAI-compatible API key.
- `OPENFARM_LLM_BASE_URL`: OpenAI-compatible base URL.
- `OPENFARM_MODEL`: multimodal model name, for example a Gemma 4 endpoint or
  an OpenFARM finetune.

The app stays usable in demo mode if the API key is absent.
