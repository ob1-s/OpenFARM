"""Verify that executable suite tasks construct and retired environments refuse runs.

This is an offline import/construction test, not a model or data-quality check.
"""
from importlib import import_module

ACTIVE = (
    ("openfarm_bioacustics", {"task": "catmeows_context", "modality": "vision"}),
    ("openfarm_horse_grimace", {"task": "region_score", "judge_mode": "none"}),
)
RETIRED = (
    ("openfarm_aww", {}),
    ("openfarm_dog_pain_triage", {}),
    ("openfarm_sheep_apr", {}),
    ("openfarm_zoo_arousal_eval", {}),
)

for package, kwargs in RETIRED:
    load = import_module(package).load_environment
    try:
        load(**kwargs)
    except RuntimeError as exc:
        if "RETIRED FROM ACTIVE" not in str(exc):
            raise
    else:
        raise AssertionError(f"{package}: retired evaluator still executable")
    print(f"OK (retired): {package}")

bio=import_module("openfarm_bioacustics").load_environment
for task in ("ungulate_valence","laying_hen_stress_binary","soundwel_call_type","soundwel_valence","soundwel_context","soundwel_age","laying_hen_stress_context"):
    try:bio(task=task)
    except ValueError as exc:
        if "retired" not in str(exc):raise
    else:raise AssertionError(f"Retired bioacoustic task still executable: {task}")
print("OK: all six retired bioacoustic tasks denied")

horse=import_module("openfarm_horse_grimace").load_environment
try:horse(task="binary_pain")
except ValueError as exc:
    if "Retired" not in str(exc):raise
else:raise AssertionError("Horse binary pain unexpectedly available")

for package, kwargs in ACTIVE:
    env=import_module(package).load_environment(**kwargs)
    if not hasattr(env,"rubric"):
        raise AssertionError(f"{package}: no rubric on constructed environment")
    print(f"OK (construction only): {package}: {kwargs}")
