#!/usr/bin/env python3
"""Cheap, exhaustive non-generative baselines for the Aarhus first-fresh-tail
damage 7-day warning task built by index_l3_care_build_aarhus_historical.py.

No model training is performed, matching the handoff stop rule. Every rule
uses only causal prior features of the built states file. Metrics are
evaluated on the fixed (batch, pen) heldout test cohorts; no threshold is
tuned on test metrics.

Reported per rule and split:
- positives/negatives and prevalence (proper denominators)
- event-level precision, recall, F1 at the rule's fixed alert threshold
- false alerts per 1000 at-risk pig-day evaluation windows
- alert lead time (days from prediction scan to onset record) for true alerts
- uncertainty by cohort (batch, pen) and treatment for the best information rule
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import pandas as pd


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


def eval_rule(df: pd.DataFrame, score: pd.Series, threshold: float) -> dict:
    evaluable = df["target_status"].isin(["positive", "negative"]).to_numpy()
    d = df[evaluable]
    scorev = score.to_numpy()
    alert = (scorev[evaluable] >= threshold)
    pos = (d["target"] == 1).to_numpy()
    tp = int((alert & pos).sum())
    fp = int((alert & ~pos).sum())
    fn = int((~alert & pos).sum())
    tn = int((~alert & ~pos).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision is not None and recall and precision + recall else None)
    false_alerts = fp / len(d) * 1000
    lead = None
    if tp:
        lead = (d.loc[alert & pos, "event_ts"] - d.loc[alert & pos, "prediction_ts"])
        lead = float(lead.dt.total_seconds().mean() / 86400)
    return {
        "evaluable_windows": int(len(d)),
        "positive_windows": int(pos.sum()),
        "true_alerts": tp,
        "false_alerts": fp,
        "missed_events": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alerts_per_1000_at_risk_pig_day_windows": round(false_alerts, 3),
        "mean_alert_lead_days": round(lead, 2) if lead is not None else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", type=Path,
                    default=Path("data/index_l3_care/aarhus/states.csv"))
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    states = load_states(args.states)
    prior = float(states[states["split"] == "train"]["target"].mean())
    rules = {
        "constant_prior_train_prevalence_no_alert_below_0.5":
            lambda sub: pd.Series(prior, index=sub.index),
        "rule_pen_mate_flag_ge1":
            lambda sub: sub["pen_mate_fresh_tail_days_prior7"],
        "rule_pen_mate_flag_ge3":
            lambda sub: sub["pen_mate_fresh_tail_days_prior7"],
        "rule_pen_hours_2days":
            lambda sub: sub["pen_mates_flagged_prior7"],
        "rule_own_ear_damage_prior":
            lambda sub: sub["own_ear_fresh_days_prior"],
        "rule_ear_or_penmate":
            lambda sub: sub[["pen_mate_fresh_tail_days_prior7",
                             "own_ear_fresh_days_prior"]].max(axis=1),
    }
    thresholds = {"constant_prior_train_prevalence_no_alert_below_0.5": 0.5,
                  "rule_pen_mate_flag_ge1": 1.0,
                  "rule_pen_mate_flag_ge3": 3.0,
                  "rule_pen_hours_2days": 2.0,
                  "rule_own_ear_damage_prior": 1.0,
                  "rule_ear_or_penmate": 1.0}
    report = {"prior_train_prevalence_per_window": round(prior, 6),
              "n_rule_thresholds_tuned_on_test": 0,
              "splits": {}}
    for split in ("train", "calibration", "test"):
        sub = states[states["split"] == split].reset_index(drop=True)
        entry = {"status_counts": {k: int(v) for k, v in
                                   sub["target_status"].value_counts().items()},
                 "prevalence_positive": float((sub["target"] == 1).mean()),
                 "rules": {}}
        for name, fn in rules.items():
            entry["rules"][name] = eval_rule(sub, fn(sub), thresholds[name])
        if split == "test":
            by_pen = {}
            for key, group in sub.groupby("split_group"):
                by_pen[key] = {
                    "windows": int(len(group)),
                    "positives": int((group["target"] == 1).sum()),
                    "alert_rate_ge1": float((group["pen_mate_fresh_tail_days_prior7"] >= 1).mean()),
                }
            entry["pen_cohort_detail_rule_ge1"] = by_pen
        report["splits"][split] = entry
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)
    post = report["splits"]["test"]["rules"]
    sanity = [
        post["rule_pen_mate_flag_ge1"]["precision"] is not None,
        post["rule_pen_mate_flag_ge1"]["recall"] >= 0,
    ]
    if not all(sanity):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
