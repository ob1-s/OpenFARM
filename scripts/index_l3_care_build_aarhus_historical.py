#!/usr/bin/env python3
"""Build time-causal first-fresh-tail-damage warning states from the Coutant et
al. 2026 Aarhus longitudinal finisher-pig clinical dataset (Zenodo 18785731,
CC BY 4.0).

The label is an independently recorded clinical outcome: trained experimenters
score every enrolled pig on weekdays with RFID-linked rows and include the scan
timestamp per row. No caregiver action timestamps exist in the release, so this
is a future clinical deterioration task (L2->L3 candidate), never an approved
caregiver-action benchmark.

Contracts enforced by construction and unit-tested:
- prediction_ts is the animal's own scan time; observable state includes that
  scan and everything earlier; the target counts fresh tail damage recorded on
  later scans within a 7-day horizon, so no later label can enter an earlier
  state.
- positives come only from real scored rows; never-treated animals keep no
  pseudo-onset date and stay clean-at-risk negatives.
- windows with later coverage ending before the horizon are "censored"; windows
  with no later scans are "not_measured"; neither is ever a negative.
- animals already flagged on their first observed day have no reliable clean
  prior, so they are excluded from the at-risk cohort and reported separately.
- PenID is reused across batches: cohort identity is (batch, pen) and pen
  features/splits always include the batch.

Outputs (default data/index_l3_care/aarhus, Gitignored): states.csv,
provenance.json. No source rows are committed to the repository.
"""
from __future__ import annotations
import argparse, hashlib, io, json, zipfile
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
import urllib.request

import pandas as pd

ZENODO_RECORD_URL = "https://zenodo.org/records/18785731"
ZENODO_ZIP_URL = (
    "https://zenodo.org/api/records/18785731/files/"
    "Data%20-%20Effects%20of%20single%20extensification%20factors%20on%20the%20occurrence"
    "%20of%20tail%20and%20ear%20biting%2C%20pen%20fouling%20and%20stomach%20ulcers%20in"
    "%20finisher%20pigs.zip/content")
ZIP_MD5 = "7d07b5597b935abea7999c26adec9869"
MEMBER_SHA256 = {
    "2-tail-ear-damage/tail-ear-damage.csv":
        "6ad08458e705c66c94e3f95091e0594f88ec8d307053c09abc1d7b4348dc749e",
    "1-pen-fouling/pen-fouling.csv":
        "2da365f7530fb25d75b51ce5a1cdbf8214f12f0005eb54bb3a500db690d1a270",
}
TAIL_MEMBER = "2-tail-ear-damage/tail-ear-damage.csv"
FOULING_MEMBER = "1-pen-fouling/pen-fouling.csv"
SPLIT_SALT = "openfarm-index-l3-care-aarhus-v0"
HORIZON_DAYS = 7
CENSOR_GRACE = timedelta(hours=36)

REQUIRED_TAIL_COLUMNS = ("Batch", "PenID", "Treatment", "PigsInPen", "Period", "Week",
                         "Weekday", "Day", "Observer", "AnimalID", "Sex",
                         "TailStatus_arrival", "IntactTail_arrival", "DamagedTail_arrival",
                         "RegDateTime", "taildamage_fresh", "eardamage_fresh")
REQUIRED_FOULING_COLUMNS = ("Batch", "PenID", "Week", "Day", "Score", "PenFouling")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_members(zip_bytes: bytes) -> dict[str, bytes]:
    if hashlib.md5(zip_bytes).hexdigest() != ZIP_MD5:
        raise ValueError("source zip md5 mismatch")
    members = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name, want in MEMBER_SHA256.items():
            if name not in zf.namelist():
                raise ValueError(f"missing archive member {name}")
            got = sha256_hex(zf.read(name))
            if got != want:
                raise ValueError(f"{name} sha256 {got} != {want}")
            members[name] = zf.read(name)
    return members


def download_zip_bytes() -> bytes:
    req = urllib.request.Request(ZENODO_ZIP_URL, headers={
        "User-Agent": "OpenFARM-academic-provenance-review/0.3"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = resp.read()
    if hashlib.md5(data).hexdigest() != ZIP_MD5:
        raise ValueError("downloaded zip md5 mismatch")
    return data


def read_tail_csv(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    missing = [c for c in REQUIRED_TAIL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"tail source missing columns {missing}")
    df = df.copy()
    df["ts"] = pd.to_datetime(df["RegDateTime"], format="ISO8601", utc=True)
    df["ts"] = df["ts"].dt.tz_localize(None)
    for col in ("taildamage_fresh", "eardamage_fresh", "IntactTail_arrival", "DamagedTail_arrival"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["taildamage_fresh", "eardamage_fresh"]].isna().any().any():
        raise ValueError("blank fresh-damage flags are not allowed by the source contract")
    for col in ("taildamage_fresh", "eardamage_fresh"):
        bad = ~df[col].isin([0, 1])
        if bad.any():
            raise ValueError(f"unexpected {col} values {sorted(df.loc[bad, col].unique())}")
    df["taildamage_fresh"] = df["taildamage_fresh"].astype(int)
    df["eardamage_fresh"] = df["eardamage_fresh"].astype(int)
    df["Day"] = df["Day"].astype(int)
    df["Batch"] = df["Batch"].astype(int)
    df["AnimalID"] = df["AnimalID"].astype(int)
    if df.duplicated(subset=["Batch", "AnimalID", "Day"]).any():
        raise ValueError("duplicate (batch, animal, day) scoring rows")
    return df


def read_fouling_csv(data: bytes) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(data))
    missing = [c for c in REQUIRED_FOULING_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"fouling source missing columns {missing}")
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["PenFouling"] = pd.to_numeric(df["PenFouling"], errors="coerce")
    df["Batch"] = df["Batch"].astype(int)
    df["PenID"] = df["PenID"].astype(int)
    df["Day"] = df["Day"].astype(int)
    return df


def primary_pen_map(tail: pd.DataFrame) -> tuple[dict, list[dict]]:
    primary = (tail.groupby(["Batch", "AnimalID"], sort=True)["PenID"]
               .agg(lambda s: int(s.mode().iloc[0])))
    pmap = {key: int(val) for key, val in primary.items()}
    t = tail.assign(primary_pen=[pmap[(int(b), int(a))] for b, a in
                                 zip(tail["Batch"], tail["AnimalID"])])
    amb = t[t["PenID"] != t["primary_pen"]]
    report = (amb.groupby(["Batch", "AnimalID"])
              .agg(ambiguous_rows=("Day", "size"),
                   pens=("PenID", lambda s: sorted(set(int(x) for x in s))))
              .reset_index().to_dict(orient="records"))
    return pmap, report


def cohort_pen_animal_map(tail: pd.DataFrame, pmap: dict) -> dict[tuple[int, int], list[int]]:
    tail = tail.copy()
    tail["primary_pen"] = [pmap[(int(b), int(a))] for b, a in zip(tail["Batch"], tail["AnimalID"])]
    out: dict[tuple[int, int], set[int]] = defaultdict(set)
    core = tail[["Batch", "primary_pen", "AnimalID"]]
    for b, p, a in core.itertuples(index=False, name=None):
        out[(int(b), int(p))].add(int(a))
    return {key: sorted(val) for key, val in out.items()}


def cohort_split(cohort_keys: list[str]) -> dict[str, str]:
    """Per-batch deterministic split: cohorts ranked by sha256(salt|key); bottom
    ~20% test and next ~10% calibration within each batch."""
    by_batch: dict[int, list[str]] = defaultdict(list)
    for key in cohort_keys:
        by_batch[int(key.split("-", 1)[0])].append(key)
    membership: dict[str, str] = {}
    for batch, keys in by_batch.items():
        ranked = sorted(keys, key=lambda k: hashlib.sha256(
            (SPLIT_SALT + "|" + k).encode()).hexdigest())
        n = len(ranked)
        n_test = max(1, round(0.20 * n))
        n_cal = max(1, round(0.10 * n))
        for rank, key in enumerate(ranked):
            membership[key] = ("test" if rank < n_test
                               else "calibration" if rank < n_test + n_cal
                               else "train")
    return membership


def build_states(tail: pd.DataFrame, fouling: pd.DataFrame,
                 pen_animals: dict[tuple[int, int], list[int]],
                 horizon_days: int = HORIZON_DAYS) -> tuple[pd.DataFrame, dict]:
    horizon = timedelta(days=horizon_days)
    tail = tail.copy()
    rows_all = tail.sort_values(["Batch", "AnimalID", "ts", "Day"]).reset_index(drop=True)
    fouling_index: dict[tuple[int, int], pd.DataFrame] = {}
    for (b, p), sub in fouling.groupby(["Batch", "PenID"]):
        fouling_index[(int(b), int(p))] = sub.sort_values("Day").reset_index(drop=True)
    pen_frames: dict[tuple[int, int], pd.DataFrame] = {}
    for (b, p), sub in rows_all.groupby(["Batch", "primary_pen"]):
        pen_frames[(int(b), int(p))] = sub.sort_values("ts").reset_index(drop=True)

    provenance: dict = {
        "horizon_days": horizon_days,
        "windows_by_status": Counter(),
        "events_by_split_cohort": Counter(),
        "animals_flagged_at_first_observed_day": 0,
        "onset_animals_with_clean_prior": 0,
        "cohort_totals": {},
    }

    candidates: list[dict] = []
    onset_animals: set[tuple[int, int]] = set()
    for (batch, animal), rows in rows_all.groupby(["Batch", "AnimalID"], sort=True):
        rows = rows.reset_index(drop=True)
        first_ts = rows["ts"].iloc[0]
        primary_pen = int(rows["primary_pen"].iloc[0])
        cohort_key = f"{batch}-{primary_pen}"
        flags = rows["taildamage_fresh"].to_numpy()
        times = pd.DatetimeIndex(rows["ts"])
        if len(rows) and flags[0] == 1:
            provenance["animals_flagged_at_first_observed_day"] += 1
            continue

        for i in range(len(rows)):
            tail_flag = int(flags[i])
            if tail_flag == 1:
                onset_animals.add((batch, int(animal)))
                break
            t0 = times[i]
            fut_mask = (times > t0) & (times <= t0 + horizon)
            fut_flags = flags[fut_mask].astype(int)
            fut_ts = times[fut_mask]
            n_obs = int(fut_mask.sum())
            hit_positions = [j for j, f in enumerate(fut_flags) if f == 1]
            if hit_positions:
                status, target = "positive", 1.0
                event_ts = pd.Timestamp(fut_ts[hit_positions[0]])
            elif n_obs == 0:
                status, target, event_ts = "not_measured", None, None
            else:
                covered_last = pd.Timestamp(fut_ts.max())
                if t0 + horizon - covered_last > CENSOR_GRACE:
                    status, target, event_ts = "censored", None, None
                else:
                    status, target, event_ts = "negative", 0.0, None
            pen_frame = pen_frames[(batch, primary_pen)]
            p_ts = pen_frame["ts"].to_numpy()
            lo = p_ts.searchsorted(t0 - horizon, side="left")
            hi = p_ts.searchsorted(t0, side="left")
            ph = pen_frame.iloc[lo:hi]
            ph = ph[ph["AnimalID"] != animal]
            pen_fresh_days = int(ph["taildamage_fresh"].sum())
            pen_flagged_pigs = int(ph.loc[ph["taildamage_fresh"] == 1, "AnimalID"].nunique())
            d_hi = p_ts.searchsorted(t0 - timedelta(days=1), side="left")
            pen_scans_day = int(max(0, hi - d_hi))
            fkey = (batch, primary_pen)
            day0 = int(rows["Day"].iloc[i])
            findex = fouling_index.get(fkey)
            fouling_score, fouling_days = None, None
            if findex is not None:
                sub = findex[findex["Day"] < day0]
                if len(sub):
                    fouling_score = float(sub["Score"].iloc[-1])
                    last7 = sub[sub["Day"] > day0 - HORIZON_DAYS]
                    fouling_days = (float((last7["PenFouling"] == 1).sum())
                                    if len(last7) else None)
            ear_prior = int(rows["eardamage_fresh"].to_numpy()[: i + 1].sum())
            candidates.append({
                "batch": batch,
                "animal_id": int(animal),
                "primary_pen": primary_pen,
                "split_group": cohort_key,
                "prediction_ts": t0.isoformat(),
                "prediction_day": day0,
                "prediction_weekday": str(rows["Weekday"].iloc[i]),
                "horizon_days": horizon_days,
                "target": target,
                "target_status": status,
                "event_ts": "" if event_ts is None else event_ts.isoformat(),
                "arrival_intact": (None if pd.isna(rows["IntactTail_arrival"].iloc[i])
                                   else float(rows["IntactTail_arrival"].iloc[i])),
                "arrival_damaged": (None if pd.isna(rows["DamagedTail_arrival"].iloc[i])
                                    else float(rows["DamagedTail_arrival"].iloc[i])),
                "treatment": str(rows["Treatment"].iloc[i]),
                "sex_male": 1.0 if str(rows["Sex"].iloc[i]) == "Castrated" else 0.0,
                "days_in_farm": float((t0 - first_ts) / timedelta(days=1)),
                "own_ear_fresh_days_prior": float(ear_prior),
                "pen_mate_fresh_tail_days_prior7": float(pen_fresh_days),
                "pen_mates_flagged_prior7": float(pen_flagged_pigs),
                "pen_mate_count": float(len(pen_animals.get(fkey, [])) - 1),
                "pen_scans_prior_day": float(pen_scans_day),
                "pen_fouling_score_prior": fouling_score,
                "pen_fouled_days_prior7": fouling_days,
                "pigs_in_pen_current": float(rows["PigsInPen"].iloc[i]),
            })
            provenance["windows_by_status"][status] += 1
            if status == "positive":
                provenance["events_by_split_cohort"][cohort_key] += 1
    provenance["onset_animals_with_clean_prior"] = len(onset_animals)
    states = pd.DataFrame(candidates)
    return states, provenance


def attach_split(states: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    cohort_keys = sorted({key for key in states["split_group"].unique()})
    membership = cohort_split(cohort_keys)
    states = states.copy()
    states["split"] = states["split_group"].map(membership)
    return states, membership


def main() -> None:
    ap = argparse.ArgumentParser(description="build Aarhus historical warning states")
    ap.add_argument("--zip", type=Path, default=None,
                    help="path to an already-downloaded source zip")
    ap.add_argument("--cache-dir", type=Path, default=Path("data/index_l3_care/aarhus_source"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/index_l3_care/aarhus"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    if args.zip is None:
        cache_zip = args.cache_dir / "aarhus_source.zip"
        if args.no_cache or not cache_zip.exists():
            args.cache_dir.mkdir(parents=True, exist_ok=True)
            data = download_zip_bytes()
            cache_zip.write_bytes(data)
        args.zip = cache_zip
    members = load_members(args.zip.read_bytes())
    tail = read_tail_csv(members[TAIL_MEMBER])
    fouling = read_fouling_csv(members[FOULING_MEMBER])
    pmap, ambiguous_report = primary_pen_map(tail)
    pen_animals = cohort_pen_animal_map(tail, pmap)
    tail = tail.assign(primary_pen=[pmap[(int(b), int(a))] for b, a in
                                    zip(tail["Batch"], tail["AnimalID"])])
    states, prov = build_states(tail, fouling, pen_animals)
    if states.empty:
        raise SystemExit("no candidate windows built; aborting")
    states, membership = attach_split(states)
    provenance = {
        "task": ("L2->L3 candidate: first fresh tail damage within 7 days from "
                 "causal prior clinically-scored history; NOT a caregiver-action "
                 "benchmark (no action timestamps in source)"),
        "record": ZENODO_RECORD_URL,
        "zip_md5": ZIP_MD5,
        "member_sha256": MEMBER_SHA256,
        "ambiguous_pen_animals": ambiguous_report,
        "tail_rows_source": int(len(tail)),
        "animals_source": int(tail[["Batch", "AnimalID"]].drop_duplicates().shape[0]),
        "cohorts_source": int(states["split_group"].nunique()),
        "split_membership": membership,
        "windows_total": int(len(states)),
        "windows_by_status": dict(prov["windows_by_status"]),
        "events_by_split_cohort": dict(prov["events_by_split_cohort"]),
        "animals_flagged_at_first_observed_day": prov["animals_flagged_at_first_observed_day"],
        "onset_animals_with_clean_prior": prov["onset_animals_with_clean_prior"],
        "splits": {},
    }
    for split in ("train", "calibration", "test"):
        sub = states[states["split"] == split]
        provenance["splits"][split] = {
            "windows": int(len(sub)),
            "status_counts": {k: int(v) for k, v in sub["target_status"].value_counts().items()},
            "positives": int((sub["target_status"] == "positive").sum()),
            "cohorts": int(sub["split_group"].nunique()),
            "animals": int(sub[["batch", "animal_id"]].drop_duplicates().shape[0]),
        }
    states_buf = states.to_csv(index=False).encode()
    provenance["states_csv_sha256"] = sha256_hex(states_buf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    states.to_csv(args.output_dir / "states.csv", index=False)
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(provenance, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
