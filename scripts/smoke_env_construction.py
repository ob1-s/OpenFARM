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
    ("openfarm_sheep_apr", {"judge_mode":"none",
                            "allow_uncertified_sheep_eval":True}),
    ("openfarm_zoo_arousal_eval", {"task":"arousal","modality":"filmstrip"}),
)

# Quarantine defaults are deliberate evaluation safeguards; the diagnostic
# opt-in below is only to test module construction, not to certify a dataset.
from openfarm_sheep_apr import load_environment as sheep_env
try:
    sheep_env(judge_mode="none")
except ValueError as exc:
    assert "quarantined" in str(exc).lower(), str(exc)
else:
    raise AssertionError("Sheep APR quarantine unexpectedly disabled")

for package, kwargs in ENVS:
    module = import_module(package)
    load_environment = getattr(module, "load_environment")
    env = load_environment(**kwargs)
    if not hasattr(env,"rubric"):
        raise AssertionError(f"{package}: no rubric on constructed environment")
    print(f"OK (construction only): {package}: {type(env).__name__}")
