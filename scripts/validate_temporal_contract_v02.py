"""Scientific time-availability evidence gate for historical System-1 tasks.

Not a generic LLM guess from column names. The primary paper and original
source metadata determine *when a feature exists*. A timed outcome after a
sample does not make experimental behavior recorded during exposure an input
available at the sample time.
"""
from __future__ import annotations
from datetime import datetime

# PigWeb: pre saliva is collected before exposure; annotated behaviors are
# recorded DURING the intervening experimental stimulus. Raw behavior CSV has
# the PRE sample date/time but does not independently prove end-of-observation.
PIGWEB_AVAILABILITY={
    "pre_cortisol_ng_ml":"pre_baseline",
    "pre_oxytocin_pg_ml":"pre_baseline",
    "observed_locomotor_play":"during_stimulus_end_unknown",
    "observed_escape_events":"during_stimulus_end_unknown",
    "observed_inactive_fraction":"during_stimulus_end_unknown",
    "observed_tailcurled_fraction":"during_stimulus_end_unknown",
    "observed_tailtucked_fraction":"during_stimulus_end_unknown",
    "observed_earforward_fraction":"during_stimulus_end_unknown"
}

def pre_only_features(raw:dict) -> dict:
    """Fail closed for a PRE timestamp; no speculative exposure of after-PRE."""
    unknown=set(raw)-set(PIGWEB_AVAILABILITY)
    if unknown:raise ValueError(f"Undocumented source feature timing: {sorted(unknown)}")
    return {k:raw[k] for k in raw if PIGWEB_AVAILABILITY[k]=="pre_baseline"}

def assert_pre_only(features:dict):
    if not features or any(PIGWEB_AVAILABILITY.get(k)!="pre_baseline" for k in features):
        raise ValueError("Pre-stimulus historical forecast contains later/unknown-time observation")

def assert_future_clock(pre_iso:str,post_iso:str):
    if datetime.fromisoformat(post_iso)<=datetime.fromisoformat(pre_iso):
        raise ValueError("No strictly later outcome time")

def future_feeder_bout_targets(feed:list[float|None],t:int,horizon:int=15):
    """Pure-Python reference definition for source-independent fixture tests."""
    if t<0 or horizon<1 or t>=len(feed):
        raise ValueError("Invalid source-relative sample index/horizon")
    def measured(v):
        return v is not None and isinstance(v,(float,int)) and float("-inf")<v<float("inf")
    current=int(feed[t]>0) if measured(feed[t]) else None
    if t+horizon>=len(feed) or any(not measured(v) for v in feed[t:t+horizon+1]):
        return current,None
    starts=[feed[i]>0 and feed[i-1]<=0 for i in range(t+1,t+horizon+1)]
    return current,int(any(starts))
