"""Instantiate OpenFARM Verifiers environments without loading any remote dataset.

Run after installing all packages from environments/*:
    python scripts/smoke_env_construction.py

This only checks Python imports and environment construction. It is NOT a
data-load check, evaluation, model test, or evidence of label/split validity.
"""
from importlib import import_module

ENVS = (
    ("openfarm_aww", {"judge_mode":"none"}),
    ("openfarm_bioacustics", {"task":"catmeows_context","modality":"vision"}),
    ("openfarm_dog_pain_triage", {"judge_mode":"none"}),
    ("openfarm_horse_grimace", {"judge_mode":"none"}),
    ("openfarm_sheep_apr", {"judge_mode":"none"}),
    ("openfarm_zoo_arousal_eval", {"task":"arousal","modality":"filmstrip"}),
)

for package, kwargs in ENVS:
    module = import_module(package)
    load_environment = getattr(module, "load_environment")
    env = load_environment(**kwargs)
    if not hasattr(env,"rubric"):
        raise AssertionError(f"{package}: no rubric on constructed environment")
    print(f"OK (construction only): {package}: {type(env).__name__}")
