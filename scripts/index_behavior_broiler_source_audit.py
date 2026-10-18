"""Big Broiler source audit (Operator C, index-behavior events).

Verifies actual access to the official labelled release, streams ALL 102
annotation CSVs via HTTP byte-range (no bulk media download), cross-checks
every parsed CSV against the checksummed dataset_manifest.csv row counts,
caches parsed label/geometry rows OUTSIDE Git, and emits a label inventory
needed to judge welfare-relevant event-task feasibility (feather pecking,
other damaging social interactions, Out_Of_Sight observation constraint).

Fail-closed: wrong file count, checksum mismatch, or per-CSV row count
mismatch against the manifest aborts with nonzero exit.

Local cache layout (env-var BIG_BROILER_CACHE, default data/broiler/cache):
  manifest.csv                 pinned manifest bytes
  cache/<stem>.parquet         one per video, pinned column schema
  audit_records.json           per-video parse records (rows, hashes, checks)
"""
from __future__ import annotations

import argparse, hashlib, json, re, sys, time
from pathlib import Path

import pandas as pd

MANIFEST_SHA256 = "c90e25a13bbb373ab9dd4a0411ae56fe76dde180d4c52b6eb2d943ccae222554"
SOURCE_URL = "https://datasets.ilabt.imec.be/broilertracking/big-broiler_v1.zip"
UPSTREAM_REPO = "https://github.com/decide-ugent/big-broiler-dataset"
FPS = 25
CORE_COLS = ["frame", "track", "x1", "y1", "x2", "y2", "actions_ground", "postures_ground"]
ANNOTATOR_COLS = [
    "actions_annotator_1", "postures_annotator_1",
    "actions_annotator_2", "postures_annotator_2",
    "actions_annotator_3", "postures_annotator_3",
]
CACHE_SCHEMA = CORE_COLS + ANNOTATOR_COLS  # always present as nullable strings
PARSED_SET = []


def parse_labels(value: object) -> list[str]:
    """,/;/| separated multi-labels; NaN -> no recorded label (NOT a negative)."""
    if value is None or (isinstance(value, float) and value != value):
        return []
    parts = re.split(r"[;,|]", str(value))
    return [p.strip() for p in parts if p.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(cache: Path) -> pd.DataFrame:
    manifest_path = cache / "manifest.csv"
    if not manifest_path.exists():
        raise SystemExit(f"Missing cached manifest {manifest_path}; run the access probe first")
    data = manifest_path.read_bytes()
    digest = sha256_bytes(data)
    if digest != MANIFEST_SHA256:
        raise SystemExit(f"Manifest checksum changed: {digest}")
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 102:
        raise SystemExit(f"Manifest rows {len(manifest)} != 102")
    return manifest


def fetch_annotation_names() -> list[str]:
    from remotezip import RemoteZip

    with RemoteZip(SOURCE_URL, initial_buffer_size=4 * 1024 * 1024) as z:
        names = sorted(n for n in z.namelist() if "annotations/" in n and n.endswith(".csv"))
    if len(names) != 102:
        raise SystemExit(f"Expected 102 annotation CSVs, found {len(names)} (refusing shifted source)")
    return names


def scan_one_video(z, name: str, stem: str, meta: pd.Series, cache: Path, force: bool) -> dict:
    """Stream one CSV, verify against manifest, cache parquet, return parse record."""
    out_path = cache / f"{stem}.parquet"
    rec = cache / f"{stem}.rows.json"
    if out_path.exists() and rec.exists() and not force:
        return json.loads(rec.read_text())
    df = pd.read_csv(z.open(name), low_memory=False)
    for col in CACHE_SCHEMA:
        if col not in df.columns and col in ANNOTATOR_COLS:
            df[col] = None
        elif col not in df.columns and col in CORE_COLS:
            raise SystemExit(f"{name}: missing required column {col}")
    df = df[CACHE_SCHEMA]
    n_rows = len(df)
    if n_rows != int(meta["total_csv_rows"]):
        raise SystemExit(f"{name}: parsed {n_rows} rows, manifest says {meta['total_csv_rows']}")
    frames = df["frame"].to_numpy()
    rec_data = {
        "stem": stem,
        "name": name,
        "rows": n_rows,
        "fps": FPS,
        "video_nb_frames_manifest": int(meta["video_nb_frames"]),
        "frame_min": int(frames.min()),
        "frame_max": int(frames.max()),
        "unique_frames": int(df["frame"].nunique()),
        "tracks": int(df["track"].nunique()),
        "duplicated_frame_track_rows": int(df.duplicated(["frame", "track"]).sum()),
        "bbox_outside_unit": int(
            ((df[["x1", "y1", "x2", "y2"]] < 0) | (df[["x1", "y1", "x2", "y2"]] > 1.0000001)).any(axis=1).sum()
        ),
        "ai_labelled_rows": int(df["actions_ground"].notna().sum()),
        "empty_action_rows": int(df["actions_ground"].isna().sum()),
        "labelled_rows_missing_geometric_coverage_missing": int(
            df["actions_ground"].notna().sum() + df["actions_ground"].isna().sum() == n_rows
        ),
        "actions_labels": {},
        "postures_labels": {},
        "annotator_cols_present": [c for c in ANNOTATOR_COLS if df[c].notna().any()],
        "sha256_parquet": "",
    }
    actions = []
    postures = []
    for v in df["actions_ground"]:
        for lab in parse_labels(v):
            actions.append(lab)
    for v in df["postures_ground"]:
        for lab in parse_labels(v):
            postures.append(lab)
    rec_data["actions_rows"] = {lab: int(actions.count(lab)) for lab in sorted(set(actions))}
    rec_data["postures_rows"] = {lab: int(postures.count(lab)) for lab in sorted(set(postures))}
    # frame indices on which each action label appears (for window rarity)
    fp_masks = {}
    for lab in sorted(set(actions)):
        mask_lbl = df["actions_ground"].astype("string").fillna("")
        fp_masks[lab] = int(
            df.loc[mask_lbl.str.contains(lab, regex=False), "frame"].nunique()
        )
    rec_data["actions_unique_frames"] = fp_masks
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    rec_data["sha256_parquet"] = sha256_bytes(out_path.read_bytes())
    (cache / f"{stem}.rows.json").write_text(json.dumps(rec_data, indent=1) + "\n")
    return rec_data


def main(argv: list[str] | None = None) -> int:
    t0 = time.time()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/broiler/cache"))
    parser.add_argument("--report", type=Path, default=Path("reports/index_behavior_broiler_source_audit.json"))
    parser.add_argument("--force", action="store_true", help="re-fetch and overwrite cache entries")
    args = parser.parse_args(argv)

    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cache)
    names = fetch_annotation_names()
    records: list[dict] = []
    errors: list[str] = []
    from remotezip import RemoteZip

    for i, name in enumerate(names):
        stem = Path(name).stem
        meta = manifest.loc[manifest["stem"] == stem].iloc[0]
        for attempt in range(3):
            try:
                with RemoteZip(SOURCE_URL, initial_buffer_size=4 * 1024 * 1024) as z:
                    if name not in z.namelist():
                        raise SystemExit(f"Missing in archive: {name}")
                    rec = scan_one_video(z, name, stem, meta, cache, args.force)
                break
            except SystemExit:
                raise
            except Exception as exc:  # network retry
                if attempt == 2:
                    errors.append(f"{stem}: {type(exc).__name__}: {str(exc)[:200]}")
                    rec = None
                    break
                time.sleep(2 * (attempt + 1))
        if rec is not None:
            records.append(rec)
        print(f"[{i + 1:>3}/102] {stem} rows={rec['rows'] if rec else 'ERR'}", flush=True)

    if errors:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"errors": errors}, indent=1))
        print("BROILER_INDEX_AUDIT:" + json.dumps({"status": "ERRORS", "errors": errors}))
        return 2

    # aggregate inventory (aggregate counts only; rows stay in gitignored cache)
    totals: dict[str, dict] = {}
    postures_totals: dict[str, dict] = {}
    low_integrity = []
    for rec in records:
        for lab, rows in rec["actions_rows"].items():
            t = totals.setdefault(lab, {"rows": 0, "videos": 0, "unique_frames": 0})
            t["rows"] += rows
            t["videos"] += 1
            t["unique_frames"] += rec["actions_unique_frames"][lab]
        for lab, rows in rec["postures_rows"].items():
            t = postures_totals.setdefault(lab, {"rows": 0, "videos": 0})
            t["rows"] += rows
            t["videos"] += 1
        checks = [
            (rec["duplicated_frame_track_rows"] > 0, "duplicate_frame_track"),
            (rec["bbox_outside_unit"] > 0, "bbox_outside_unit_range"),
            (rec["frame_max"] + 1 > rec["video_nb_frames_manifest"], "frame_beyond_manifest"),
            (rec["unique_frames"] != rec["video_nb_frames_manifest"], "frame_index_gap"),
        ]
        for bad, why in checks:
            if bad:
                low_integrity.append({"stem": rec["stem"], "issue": why})

    agg = {
        "source_url": SOURCE_URL,
        "upstream": UPSTREAM_REPO,
        "manifest_sha256": MANIFEST_SHA256,
        "license": "AGPL-3.0 (source repo README/LICENSE); local cache, no rows published",
        "videos": len(records),
        "total_rows": sum(r["rows"] for r in records),
        "annotated_seconds": sum(r["rows"] for r in records) / FPS,
        "action_label_inventory_rows": {
            k: {**v, "annotated_seconds": round(v["rows"] / FPS, 2)}
            for k, v in sorted(totals.items(), key=lambda x: -x[1]["rows"])
        },
        "posture_label_inventory_rows": {
            k: {**v, "annotated_seconds": round(v["rows"] / FPS, 2)}
            for k, v in sorted(postures_totals.items(), key=lambda x: -x[1]["rows"])
        },
        "integrity_observations": low_integrity,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(agg, indent=1) + "\n")
    print("BROILER_INDEX_AUDIT:" + json.dumps({
        "status": "OK",
        "videos": agg["videos"],
        "total_rows": agg["total_rows"],
        "labels": {k: v["rows"] for k, v in list(agg["action_label_inventory_rows"].items())},
        "integrity_observations": len(low_integrity),
        "elapsed_sec": agg["elapsed_sec"],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
