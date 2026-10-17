#!/usr/bin/env python3
"""Cheap, fixed-threshold baselines for the Aarhus first-fresh-tail-damage
7-day warning task built by index_l3_care_build_aarhus_historical.py.

No model training and no threshold tuning on held-out data. Every rule uses
only causal prior features of the built states file.

Denominator discipline:
- Window-level metrics use EVALUABLE prediction windows only (target_status in
  {positive, negative}). Censored and not_measured windows are never rule
  metrics; total-window prevalence is still reported with its denominator
  for transparency.
- Event-level metrics de-duplicate overlapping windows belonging to the same
  first-onset (batch, animal_id, event_ts): a recorded onset counts as detected
  iff ANY of its evaluable windows alerted; lead time is measured from the
  EARLIEST alerting window's prediction_ts inside that onset's horizon.
- Alert-episode metrics collapse repeated detached per-animal window alerts:
  consecutive alerted windows of one animal separated by <= COOLDOWN_DAYS form
  one episode (non-consecutive alerts start a new episode).

Unit note: false-positive rates are per 1,000 evaluable SCORING windows
(weekday expert scoring events), not per 1,000 calendar pig-days and not
per deduplicated alert after suppression. "Treatment" columns are assigned
husbandry conditions of the source trial (Control/Enrichment/Roughage/Space),
not treatments administered to an animal; clinical surveillance features
require expert inspection of pen-mates and the animal's own ear.
"""
from __future__ import annotations
import argparse, json
from datetime import timedelta
from pathlib import Path

import pandas as pd

COOLDOWN_DAYS = 3


def load_states(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["prediction_ts"] = pd.to_datetime(df["prediction_ts"])
    df["event_ts"] = pd.to_datetime(df["event_ts"], errors="coerce")
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    need = {"batch", "animal_id", "primary_pen", "split_group", "prediction_ts",
            "target", "target_status", "treatment", "split",
            "pen_mate_fresh_tail_days_prior7", "own_ear_fresh_days_prior"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"states file missing columns {sorted(missing)}")
    dup = df.duplicated(subset=["batch", "animal_id", "prediction_ts"]).sum()
    if dup:
        raise ValueError(f"{dup} duplicate prediction keys")
    return df


def eval_rule(sub: pd.DataFrame, score: pd.Series, threshold: float) -> dict:
    evaluable = sub["target_status"].isin(["positive", "negative"]).to_numpy()
    posv = sub["target"].to_numpy() == 1.0
    alert = (score.to_numpy() >= threshold) & evaluable
    tpw = int((alert & posv).sum())
    fpw = int((alert & ~posv).sum())
    mpw = int((~alert & posv).sum())
    n_eval = int(evaluable.sum())
    precision = tpw / (tpw + fpw) if tpw + fpw else None
    recall = tpw / (tpw + mpw) if tpw + mpw else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall is not None
          and precision + recall > 0 else None)
    out = {
        "evaluable_windows": n_eval,
        "true_positive_windows": tpw,
        "false_positive_windows": fpw,
        "missed_positive_windows": mpw,
        "precision_windows": precision,
        "recall_windows": recall,
        "f1_windows": f1,
        "false_positive_windows_per_1000_evaluable_scoring_windows":
            round(fpw / n_eval * 1000, 3),
        "mean_lead_days_alerted_positive_windows": None,
        "onsets_total_distinct_first_onsets": 0,
        "onsets_detected_distinct": 0,
        "onset_recall": None,
        "mean_first_alert_lead_days": None,
        "alert_episodes_cooldown_3d_per_pig": 0,
        "alerted_animals_any_window": 0,
        "mean_episodes_per_alerted_animal": None,
        "mean_alert_episodes_per_pen_cohort": None,
    }
    flagged = sub[alert]
    hit = flagged[posv[alert]]
    if len(hit):
        lead = ((pd.to_datetime(hit["event_ts"]) - hit["prediction_ts"])
                .dt.total_seconds() / 86400.0)
        out["mean_lead_days_alerted_positive_windows"] = round(float(lead.mean()), 2)
    onsets_total = (sub[posv].groupby(["batch", "animal_id"])["event_ts"]
                    .nunique().sum())
    out["onsets_total_distinct_first_onsets"] = int(onsets_total)
    if len(hit):
        per_onset = (hit.assign(event_key=hit["event_ts"])
                     .groupby(["batch", "animal_id", "event_key"])
                     .agg(first_alert=("prediction_ts", "min")).reset_index())
        out["onsets_detected_distinct"] = int(len(per_onset))
        lead_ev = ((per_onset["event_key"] - per_onset["first_alert"])
                   .dt.total_seconds() / 86400.0)
        out["mean_first_alert_lead_days"] = round(float(lead_ev.mean()), 2)
    if out["onsets_total_distinct_first_onsets"]:
        out["onset_recall"] = (out["onsets_detected_distinct"]
                               / out["onsets_total_distinct_first_onsets"])
    if len(flagged):
        grouped = flagged.groupby(["batch", "animal_id"])["prediction_ts"]
        episodes = 0
        for _, ts in grouped:
            ts = sorted(ts.tolist())
            episodes += 1
            for a, b in zip(ts[:-1], ts[1:]):
                if b - a > timedelta(days=COOLDOWN_DAYS):
                    episodes += 1
        animals = int(flagged[["batch", "animal_id"]].drop_duplicates().shape[0])
        pens = int(flagged["split_group"].nunique())
        out["alert_episodes_cooldown_3d_per_pig"] = episodes
        out["alerted_animals_any_window"] = animals
        out["mean_episodes_per_alerted_animal"] = round(episodes / animals, 2)
        out["mean_alert_episodes_per_pen_cohort"] = round(episodes / pens, 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path,
                    default=Path("data/index_l3_care/aarhus/states.csv"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    states = load_states(args.states)
    prior = float(states[states["split"] == "train"]["target"].mean())
    rules = {
        "constant_prior_train_prevalence_threshold_0.5_no_alerts":
            (lambda sub: pd.Series(prior, index=sub.index), 0.5),
        "rule_pen_mate_flag_ge1":
            (lambda sub: sub["pen_mate_fresh_tail_days_prior7"], 1.0),
        "rule_pen_mate_flag_ge3":
            (lambda sub: sub["pen_mate_fresh_tail_days_prior7"], 3.0),
        "rule_pen_mates_distinct_ge2":
            (lambda sub: sub["pen_mates_flagged_prior7"], 2.0),
        "rule_own_ear_fresh_ge1":
            (lambda sub: sub["own_ear_fresh_days_prior"], 1.0),
        "rule_ear_or_pen_mate_ge1":
            (lambda sub: sub[["pen_mate_fresh_tail_days_prior7",
                              "own_ear_fresh_days_prior"]].max(axis=1), 1.0),
    }
    report = {
        "prior_train_prevalence_per_window": round(prior, 6),
        "n_rule_thresholds_tuned_on_test": 0,
        "unit_note": ("false-positive rates are per 1000 evaluable weekday "
                      "scoring windows; not calendar pig-days, not "
                      "deduplicated alerts; onset recall deduplicates "
                      "overlapping windows of one (animal, event_ts)"),
        "splits": {},
    }
    for split in ("train", "calibration", "test"):
        sub = states[states["split"] == split].reset_index(drop=True)
        posv = sub["target"] == 1.0
        entry = {
            "total_windows": int(len(sub)),
            "evaluable_windows": int(sub["target_status"].isin(["positive", "negative"]).sum()),
            "status_counts": {k: int(v) for k, v in sub["target_status"].value_counts().items()},
            "prevalence_total_windows_incl_censored": round(float(posv.mean()), 6),
            "evaluable_positives": int(posv.sum()),
            "prevalence_evaluable_windows": round(float((posv).sum()
                    / max(1, int(sub["target_status"].isin(["positive", "negative"]).sum()))), 6),
            "distinct_first_onsets_in_split": int(sub[posv].groupby(["batch", "animal_id"])["event_ts"].nunique().sum()),
            "rules": {},
        }
        for name, (fn, thr) in rules.items():
            entry["rules"][name] = eval_rule(sub, fn(sub), thr)
        report["splits"][split] = entry
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    test = report["splits"]["test"]
    if not (test["distinct_first_onsets_in_split"] > 0
            and all("true_positive_windows" in m for m in test["rules"].values())):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
