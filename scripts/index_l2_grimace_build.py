"""GrimACE L2 source build (single canonical pipeline for the Score/Noul benchmark).

Three modes:
  --fetch-labels        Remote probe: fetch record metadata, parse the Data.zip
                        central directory via byte ranges, verify the advertised
                        md5, extract the 15 label CSVs + 2 xlsx (small members)
                        and run remote alignment checks. No archive download.
  --labels-only         Build the canonical label matrix from the probed tables.
  --zip FILE            Full build: verify whole-archive md5/sha256, map every
                        scored image to its actual ZIP member (name looked up
                        from the central directory, never guessed), record
                        per-image sha256/crc32/size, extract the scored subsets
                        into the local corpus, freeze splits.

Outputs:
  --out dir (default .prime_data/indexl2, NOT committed):
    manifest.csv, splits.json, aggregate_stats.json, build_report.json
  committed (docs/research/index-l2-perception/): split membership CSV, per-image
    sha256 registry, build report. No label values, no image bytes (apgl-v3).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import index_l2_grimace_lib as lib

ZENODO_ORIGIN = f"https://{lib.ZENODO_RECORD}"
DATA_ZIP_URL = "https://zenodo.org/api/records/18299984/files/Data.zip/content"


# ---------------------------------------------------------------------------
# remote ranged primitives
# ---------------------------------------------------------------------------

def fetch_range(url: str, start: int, end: int) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def remote_total_size(url: str) -> int:
    import urllib.request
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
    if not cr:
        raise RuntimeError("Zenodo ignored the Range request (unexpected)")
    return int(cr.split("/")[-1])


def fetch_central_directory(url: str, total: int) -> list[dict]:
    tail = fetch_range(url, total - 65536, total - 1)
    idx64 = tail.rfind(b"PK\x06\x07")
    if idx64 != -1:
        eocd64_off = struct.unpack("<Q", tail[idx64 + 8:idx64 + 16])[0]
        rec = fetch_range(url, eocd64_off, eocd64_off + 55)
        if rec[:4] != b"PK\x06\x06":
            raise RuntimeError("bad zip64 EOCD")
        cd_size, cd_off = struct.unpack("<QQ", rec[40:56])
    else:
        eocd = tail.rfind(b"PK\x05\x06")
        if eocd == -1:
            raise RuntimeError("no EOCD found")
        cd_size, cd_off = struct.unpack("<II", tail[eocd + 12:eocd + 20])
    if cd_size > 64 << 20:
        raise RuntimeError(f"central directory unexpectedly large: {cd_size}")
    cd = fetch_range(url, cd_off, cd_off + cd_size - 1) if cd_size else b""
    entries = []
    p = 0
    while p + 46 <= len(cd):
        if cd[p:p + 4] != b"PK\x01\x02":
            break
        nlen, elen, clen = struct.unpack("<HHH", cd[p + 28:p + 34])
        crc, csize, usize = struct.unpack("<III", cd[p + 16:p + 28])
        lho = struct.unpack("<I", cd[p + 42:p + 46])[0]
        method = cd[p + 10]
        extra = cd[p + 46 + nlen:p + 46 + nlen + elen]
        if csize == 0xFFFFFFFF:
            q = 0
            while q + 4 <= len(extra):
                hid, hsz = struct.unpack("<HH", extra[q:q + 4])
                if hid == 0x0001:
                    off = q + 4
                    vals = []
                    for need in (usize != 0xFFFFFFFF, csize != 0xFFFFFFFF, lho != 0xFFFFFFFF):
                        if need:
                            vals.append(struct.unpack("<Q", extra[off:off + 8])[0])
                            off += 8
                    r = 0
                    if usize == 0xFFFFFFFF:
                        usize = vals[r]; r += 1
                    if csize == 0xFFFFFFFF:
                        csize = vals[r]; r += 1
                    if lho == 0xFFFFFFFF:
                        lho = vals[r]; r += 1
                q += 4 + hsz
        name = cd[p + 46:p + 46 + nlen].decode("utf-8", "replace")
        entries.append({"name": name, "csize": csize, "usize": usize,
                        "crc": f"{crc:08x}", "lho": lho, "method": method})
        p += 46 + nlen + elen + clen
    return entries


def extract_member_remote(entries: list[dict], url: str, name: str) -> tuple[bytes, str]:
    """Ranged local-header + data fetch of one member; returns (raw, sha256)."""
    index = {e["name"]: e for e in entries}
    e = index[name]
    head = fetch_range(url, e["lho"], e["lho"] + 511)
    if head[:4] != b"PK\x03\x04":
        raise RuntimeError(f"bad local header for {name}")
    nlen, elen = struct.unpack("<HH", head[26:30])
    data_start = e["lho"] + 30 + nlen + elen
    blob = fetch_range(url, data_start, data_start + e["csize"] - 1)
    if e["method"] == 8:
        d = zlib.decompressobj(-15)
        raw = d.decompress(blob) + d.flush()
    else:
        raw = blob[:e["csize"]]
    assert len(raw) == e["usize"], f"uncompressed size mismatch {name}"
    assert f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}" == e["crc"], f"crc mismatch {name}"
    return raw, hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# small table parsing
# ---------------------------------------------------------------------------

def parse_float_cell(cell: str) -> float | None:
    s = (cell or "").strip()
    if s in ("", "-", "NA", "NR"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def parse_gt3_automatic_rows(text: str) -> list[dict]:
    """gt3_automatic_individual.csv automatic rows; scores stay float (model posteriors)."""
    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["annotator"] != "automatic":
            continue
        out.append({"video_name": row["video_name"], "animal_id": row["animal_id"],
                    "treatment": row["treatment"], "timepoint": row["timepoint"],
                    "frame": int(row["frame_index"]), "scores": {au: parse_float_cell(row[au]) for au in lib.AUS}})
    return out


# ---------------------------------------------------------------------------
# probe (--fetch-labels)
# ---------------------------------------------------------------------------

def run_probe(probe_dir: Path, url: str = DATA_ZIP_URL) -> dict:
    import urllib.request
    probe_dir.mkdir(parents=True, exist_ok=True)
    small_dir = probe_dir / "remote_small"
    small_dir.mkdir(exist_ok=True)

    # 1. record metadata (authoritative md5)
    with urllib.request.urlopen("https://zenodo.org/api/records/18299984", timeout=60) as r:
        record = json.loads(r.read())
    (probe_dir / "record.json").write_text(json.dumps(record))
    files = {f["key"]: f for f in record["files"]}
    rec_checksum = files["Data.zip"]["checksum"]  # "md5:7292..."
    assert rec_checksum == f"md5:{lib.DATA_ZIP_MD5}", rec_checksum

    # 2. central directory
    total = remote_total_size(url)
    entries = fetch_central_directory(url, total)
    (probe_dir / "remote_cd_entries.json").write_text(json.dumps(entries))
    by_prefix: dict[str, dict] = {}
    for e in entries:
        p = e["name"].split("/")
        if len(p) >= 3:
            key = "/".join(p[:4]) if p[1] == "MGS_scores" and p[2] == "images" else "/".join(p[:2])
            d = by_prefix.setdefault(key, {"png": 0, "bytes": 0, "sessions": set()})
            if e["name"].lower().endswith(".png"):
                d["png"] += 1
                d["bytes"] += e["usize"]
                if "/".join(e["name"].split("/")[:-1]) :
                    d["sessions"].add("/".join(e["name"].split("/")[:-1]))

    # 3. extract small members (15 CSVs + 2 xlsx)
    wanted = sorted({path for tables in lib.CSV_EXTERNALS.values() for path in tables.values()}
                    | {lib.CSV_GT_TRAIN, lib.CSV_GT_EXP34, lib.CSV_EXP34_HOHLBAUM,
                       "Data/MGS_scores/values/gt1_gt2_automatic_averaged.csv",
                       lib.XLSX_PART1, lib.XLSX_PART2})
    extracted = {}
    cd_index = {e["name"]: e for e in entries}
    for name in wanted:
        assert name in cd_index, f"missing from archive: {name}"
        raw, sha = extract_member_remote(entries, url, name)
        (small_dir / name.replace("/", "__")).write_bytes(raw)
        extracted[name] = sha

    # 4. remote alignment checks (image side via CD only)
    provider = lambda nm: (small_dir / nm.replace("/", "__")).read_text(encoding="utf-8")
    provider.__name__ = "probe_text_provider"
    provider.__qualname__ = "probe_text_provider"
    provider.__module__ = "index_l2_grimace_build"
    gt_rows = lib.parse_gt_train_csv(provider(lib.CSV_GT_TRAIN))
    gt1_keys = {(v, f) for r, v, f, _ in gt_rows if r == "gt1"}
    exp1_cd_keys = set()
    for e in entries:
        p = e["name"].split("/")
        if (len(p) == 6 and p[1] == "MGS_scores" and p[2] == "images"
                and p[3].startswith("GrimaceTest1") and e["name"].endswith(".png")):
            exp1_cd_keys.add((p[4], int(p[5][3:-4])))
    auto3 = parse_gt3_automatic_rows(provider(lib.CSV_GT_EXP34))
    hlb3 = lib.parse_gt3_hohlbaum_individual(provider(lib.CSV_EXP34_HOHLBAUM))
    exp3_keys_wild = []
    for e in entries:
        p = e["name"].split("/")
        if (len(p) == 6 and p[1] == "MGS_scores" and p[2] == "images"
                and p[3] == "GrimaceTest3Frames" and e["name"].endswith(".png")):
            exp3_keys_wild.append((p[4], p[5][3:-4]))
    weird = [(s, f) for s, f in exp3_keys_wild if not f.isdigit()]
    if weird:
        raise SystemExit(f"unexpected exp3 image names: {sorted(weird)[:5]}")
    exp3_cd_keys = {(s, int(f)) for s, f in exp3_keys_wild}

    alignment = {
        "cd_entries_total": len(entries),
        "cd_total_size": total,
        "exp1_gt1_scored": len(gt1_keys),
        "exp1_cd_images": len(exp1_cd_keys),
        "exp1_scored_equals_cd": gt1_keys == exp1_cd_keys,
        "exp3_cd_images": len(exp3_cd_keys),
        "exp3_sessions": len({s for s, _ in exp3_cd_keys}),
        "exp3_hohlbaum_rows": len(hlb3),
        "exp3_hohlbaum_missing_image": sorted({(v, f) for v, f, _ in hlb3} - exp3_cd_keys)[:5],
        "exp3_automatic_rows": len(auto3),
        "exp3_automatic_missing_image": sum(1 for a in auto3 if (a["video_name"], a["frame"]) not in exp3_cd_keys),
    }
    probe_report = {
        "url": url,
        "record_md5": rec_checksum,
        "alignment": alignment,
        "extracted_sha256": extracted,
        "prefix_counts": {k: {"png": v["png"], "bytes": v["bytes"], "sessions": len(v["sessions"])}
                          for k, v in sorted(by_prefix.items())},
    }
    (probe_dir / "probe_report.json").write_text(json.dumps(probe_report, indent=2, sort_keys=True))
    print(json.dumps(probe_report["alignment"], indent=2))
    return probe_report


# ---------------------------------------------------------------------------
# build (--labels-only / --zip FILE)
# ---------------------------------------------------------------------------

def strip_face0(session: str) -> str:
    return session.removesuffix("_face0")


def build(out_dir: Path, zip_path: Path | None, remote_dir: Path | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    anomalies: list[str] = []

    zf = None
    source_hashes = None
    if zip_path:
        md5, sha = hashlib.md5(), hashlib.sha256()
        with open(zip_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                md5.update(chunk); sha.update(chunk)
        if md5.hexdigest() != lib.DATA_ZIP_MD5:
            raise SystemExit(f"MD5 mismatch: got {md5.hexdigest()}")
        source_hashes = {"md5": md5.hexdigest(), "sha256": sha.hexdigest()}
        zf = zipfile.ZipFile(zip_path)
        print("archive verified:", source_hashes["md5"],
              "files:", len(zf.namelist()))

    def provider(name: str) -> str:
        if zf is not None:
            return zf.read(name).decode("utf-8")
        p = remote_dir / name.replace("/", "__")
        return p.read_text(encoding="utf-8")

    names_in_zip: set[str] = set(zf.namelist()) if zf is not None else set()

    # actual deposit member names keyed by (session, frame) - never constructed from guesses
    exp1_name: dict[tuple[str, int], str] = {}
    exp3_name: dict[tuple[str, int], str] = {}
    if zf is not None:
        for nm in names_in_zip:
            p = nm.split("/")
            if not nm.endswith(".png"):
                continue
            if (len(p) == 6 and p[1] == "MGS_scores" and p[2] == "images"
                    and p[3].startswith("GrimaceTest1")):
                exp1_name[(p[4], int(p[5][3:-4]))] = nm
            elif (len(p) == 6 and p[1] == "MGS_scores" and p[2] == "images"
                    and p[3] == "GrimaceTest3Frames"):
                exp3_name[(p[4], int(p[5][3:-4]))] = nm

    # --- rater tables ---
    gt_rows = lib.parse_gt_train_csv(provider(lib.CSV_GT_TRAIN))
    gt1_by_key = {(v, f): lab for r, v, f, lab in gt_rows if r == "gt1"}
    gt2_by_key = {(strip_face0(v), f): lab for r, v, f, lab in gt_rows if r == "gt2"}

    externals: dict[str, dict[tuple[str, int], dict[str, dict]]] = {"exp1": {}, "exp2": {}, "exp1r2": {}}
    for scope, tables in lib.CSV_EXTERNALS.items():
        exp = "exp2" if scope == "exp2" else "exp1"
        key_sets: dict[str, set] = {}
        for rat, name in sorted(tables.items()):
            rows = lib.parse_externals_csv(provider(name), rat)
            key_sets[rat] = {(r[0], r[1]) for r in rows}
            for session, frame, lab in rows:
                if exp == "exp2" and session.startswith("psFC"):
                    # Second annotation round over exp1 images (discovered: 103 of the
                    # 575 exp2-inscribed rows are psFC folders of exp1, 10 sessions).
                    externals["exp1r2"].setdefault((session, frame), {})[rat] = lab
                else:
                    externals[exp].setdefault((strip_face0(session), frame), {})[rat] = lab
        if len(key_sets) >= 2:
            sets_list = list(key_sets.values())
            if any(s != sets_list[0] for s in sets_list):
                raise SystemExit(f"rater key-sets diverge in {scope}: "
                                 f"{[(r, len(s)) for r, s in key_sets.items()]}")

    auto3_rows = parse_gt3_automatic_rows(provider(lib.CSV_GT_EXP34))
    hlb3 = lib.parse_gt3_hohlbaum_individual(provider(lib.CSV_EXP34_HOHLBAUM))
    meta3 = {}
    for r in auto3_rows:
        meta3[r["video_name"]] = {"animal_id": r["animal_id"], "treatment": r["treatment"],
                                  "timepoint": r["timepoint"]}

    expected_sessions = {s for s, _ in gt1_by_key} | {s for s, _ in gt2_by_key}
    averaged: dict[str, dict] = {}
    for row in csv.DictReader(io.StringIO(provider("Data/MGS_scores/values/gt1_gt2_automatic_averaged.csv"))):
        if row["annotator"] != "automatic":
            continue
        averaged[strip_face0(row["video_name"])] = row
    missing_meta = expected_sessions - set(averaged)
    if missing_meta:
        raise SystemExit(f"sessions without averaged metadata: {sorted(missing_meta)[:5]} (n={len(missing_meta)})")

    # --- alignment cross-checks ---
    if zf is not None:
        missing_exp1 = sorted(set(gt1_by_key) - set(exp1_name))
        if missing_exp1:
            raise SystemExit(f"exp1 scored images missing from archive: {missing_exp1[:5]} (n={len(missing_exp1)})")
        extra_exp1 = sorted(set(exp1_name) - set(gt1_by_key))
        if extra_exp1:
            anomalies.append(f"exp1 deposit images without any rater row: {extra_exp1[:5]} (n={len(extra_exp1)})")
    else:
        anomalies.append("labels-only build: image bytes unverified (sha256/size unknown in manifest)")
        # remote probe already validated alignment; double-check against probe report if present
        prp = remote_dir.parent / "probe_report.json"
        if prp.exists():
            al = json.loads(prp.read_text())["alignment"]
            if not al.get("exp1_scored_equals_cd"):
                raise SystemExit("remote probe did not confirm exp1 alignment; rerun --fetch-labels")

    if zf is not None:
        missing_h = sorted(((v, f) for v, f, _ in hlb3 if (v, f) not in exp3_name))
        if missing_h:
            anomalies.append(f"exp3 hohlbaum rows without deposit image: {missing_h[:5]} (n={len(missing_h)})")
        missing_auto = [(r["video_name"], r["frame"]) for r in auto3_rows if (r["video_name"], r["frame"]) not in exp3_name]
        if missing_auto:
            anomalies.append(
                "exp3 automatic rows referencing videos absent from the deposit image tree "
                f"(n={len(missing_auto)}; csv-only sessions: a12_c_24h_front_12 [deposit ships "
                "unscored sibling a12_c_24h_front_13 with disjoint frame numbers], a4_c_24h_front_5 [no images])")

    # exp3 automatic scores: companion evidence only (model posteriors, never a target)
    matched_auto = [r for r in auto3_rows if (r["video_name"], r["frame"]) in
                    (exp3_name if zf is not None else set(exp3_name))]
    with open(out_dir / "automatic_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "frame", "animal", "treatment", "timepoint", "scores_json"])
        for r in matched_auto:
            try:
                animal = lib.session_to_exp3_mouse(r["video_name"])
            except ValueError:
                animal = r["animal_id"]
            w.writerow([r["video_name"], r["frame"], animal, r["treatment"], r["timepoint"],
                        json.dumps(r["scores"], separators=(",", ":"))])

    # --- records ---
    records: list[dict] = []

    def zsha(zp: str) -> tuple[str, int, str]:
        if not zp:
            return "unknown", -1, "unknown"
        raw = zf.read(zp)
        return hashlib.sha256(raw).hexdigest(), len(raw), f"{zlib.crc32(raw) & 0xFFFFFFFF:08x}"

    for (session, frame), gt_lab in sorted(gt1_by_key.items()):
        meta = averaged[session]
        animal = lib.canonical_animal_exp1(session, meta["animal_id"])
        zp = exp1_name.get((session, frame), "") if zf is not None else ""
        sha, size, crc = zsha(zp)
        labels = {lib.TRAIN_RATER: dict(gt_lab)}
        labels.update(externals["exp1"].get((session, frame), {}))
        records.append({"experiment": "exp1", "session": session, "animal": animal,
                        "frame": frame, "zip_path": zp, "sha256": sha, "size": size, "crc32": crc,
                        "treatment": meta["treatment"], "timepoint": meta["timepoint"],
                        "labels": labels})

    for (session, frame) in sorted(set(gt2_by_key) | set(externals["exp2"])):
        labels = dict(externals["exp2"].get((session, frame), {}))
        if (session, frame) in gt2_by_key:
            labels[lib.EXP2_RATER] = dict(gt2_by_key[(session, frame)])
        meta = averaged[session]
        records.append({"experiment": "exp2", "session": session, "animal": meta["animal_id"],
                        "frame": frame, "zip_path": "", "sha256": "unknown", "size": -1, "crc32": "unknown",
                        "treatment": meta["treatment"], "timepoint": meta["timepoint"],
                        "labels": labels})
    anomalies.append("exp2 handled label-only: its scored pixels are absent from the deposit "
                     "(kept for cross-experiment rater-agreement evidence; excluded from the image benchmark)")

    # round-2 external annotations land on exp1 frames; validate and mirror image provenance
    exp1_lookup = {(r["session"], r["frame"]): r for r in records if r["experiment"] == "exp1"}
    for (session, frame) in sorted(externals["exp1r2"]):
        base = exp1_lookup.get((session, frame))
        if base is None:
            raise SystemExit(f"exp1 r2 annotation without exp1 base row: {(session, frame)}")
        records.append({"experiment": "exp1r2", "session": session, "animal": base["animal"],
                        "frame": frame, "zip_path": base["zip_path"], "sha256": base["sha256"],
                        "size": base["size"], "crc32": base["crc32"],
                        "treatment": base["treatment"], "timepoint": base["timepoint"],
                        "labels": externals["exp1r2"][(session, frame)]})

    exp3_keys = set(exp3_name) if zf is not None else {(v, f) for v, f, _ in hlb3}
    hlb3_by_key = {(v, f): lab for v, f, lab in hlb3}
    for (session, frame) in sorted(exp3_keys):
        meta = meta3.get(session, {})
        labels = {}
        h = hlb3_by_key.get((session, frame))
        if h:
            labels[lib.EXP3_EXTERNAL] = dict(h)
        zp = exp3_name.get((session, frame), "") if zf is not None else ""
        sha, size, crc = zsha(zp)
        tr, tp = meta.get("treatment", "c"), meta.get("timepoint", "unknown")
        if tp == "unknown":
            tr, tp = lib.session_to_exp3_timepoint(session)
        records.append({"experiment": "exp3", "session": session,
                        "animal": meta.get("animal_id", lib.session_to_exp3_mouse(session)),
                        "frame": frame, "zip_path": zp, "sha256": sha, "size": size, "crc32": crc,
                        "treatment": tr, "timepoint": tp, "labels": labels})

    # labels-only floor: 773 exp1 + 575 exp2 + 119 hohlbaum exp3 keys; zip build adds 1,606 more exp3 images
    if len(records) < 1400:
        raise SystemExit(f"record count implausibly low: {len(records)}")

    # --- manifest (full labels; local only) ---
    cols = ["experiment", "session", "animal", "frame", "zip_path", "sha256", "size", "crc32",
            "treatment", "timepoint", "labels_json"]
    with open(out_dir / "manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in records:
            w.writerow([r[c] for c in cols[:-1]] + [json.dumps(r["labels"], sort_keys=True, separators=(",", ":"))])

    label_digest = lib.manifest_digest([{"key": [r["experiment"], r["session"], r["frame"]],
                                         "labels": r["labels"]} for r in records])

    # --- splits (frozen deterministic rule) ---
    exp1_animals = sorted({r["animal"] for r in records if r["experiment"] == "exp1"})
    exp3_animals = sorted({r["animal"] for r in records if r["experiment"] == "exp3"})
    treatment1 = {r["animal"]: r["treatment"] for r in records if r["experiment"] == "exp1"}
    split1 = lib.stratified_animal_split(exp1_animals, treatment1, n_test=2, n_cal=1)
    split3 = lib.stratified_animal_split(exp3_animals, {a: "c" for a in exp3_animals}, n_test=2, n_cal=2)
    lib.validate_split(split1)
    lib.validate_split(split3)
    splits = {"exp1": split1, "exp3": split3}
    digests = {"exp1": lib.split_membership_digest(split1), "exp3": lib.split_membership_digest(split3)}
    (out_dir / "splits.json").write_text(json.dumps({
        "membership": splits, "digests": digests,
        "rule": "per treatment group sorted by animal id ASC: last 2 animals -> test, "
                "previous n_cal animals -> cal, rest -> train; deterministic, no RNG",
        "exp2": "excluded from image benchmark (labels-only)",
    }, indent=2))

    # --- aggregates ---
    stats = aggregate_stats(records)
    (out_dir / "aggregate_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True))

    # --- committed artifacts (provenance only) ---
    pub = Path(__file__).resolve().parents[1] / "docs/research/index-l2-perception"
    pub.mkdir(parents=True, exist_ok=True)
    with open(pub / "GRIMACE_split_membership.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["experiment", "animal", "treatment", "images", "split"])
        for exp, spl in (("exp1", split1), ("exp3", split3)):
            for part in ("train", "cal", "test"):
                for a in spl[part]:
                    rs = [r for r in records if r["experiment"] == exp and r["animal"] == a]
                    w.writerow([exp, a, rs[0]["treatment"] if rs else "unknown", len(rs), part])
    with open(pub / "GRIMACE_image_sha256_registry.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["zip_path", "sha256", "size_bytes", "crc32", "experiments", "first_frame",
                    "has_human_labels", "split"])
        reg: dict[str, dict] = {}
        for r in records:
            if not r["zip_path"]:
                continue
            e = reg.setdefault(r["zip_path"], {"sha256": r["sha256"], "size": r["size"],
                            "crc32": r["crc32"], "exps": set(), "first_frame": r["frame"],
                            "human": False, "split": ""})
            e["exps"].add(r["experiment"])
            e["human"] = e["human"] or bool(r["labels"])
            if r["experiment"] in ("exp1", "exp3") and not e["split"]:
                e["split"] = next((p for p in ("train", "cal", "test")
                                   if r["animal"] in splits[r["experiment"]][p]), "")
        for zp in sorted(reg):
            e = reg[zp]
            w.writerow([zp, e["sha256"], e["size"], e["crc32"],
                        "|".join(sorted(e["exps"])), e["first_frame"],
                        int(e["human"]), e["split"]])

    report = {
        "source": {"zenodo_record": lib.ZENODO_RECORD,
                   "origin": ZENODO_ORIGIN,
                   "data_zip_md5": lib.DATA_ZIP_MD5,
                   "data_zip_sha256": (source_hashes or {}).get("sha256"),
                   "labels_via_byte_range": zf is None},
        "counts": stats["counts"],
        "label_digest": label_digest,
        "split_digests": digests,
        "anomalies": anomalies,
    }
    (out_dir / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (pub / "GRIMACE_build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report, indent=2, sort_keys=True))
    return {"records": records, "splits": splits, "stats": stats, "report": report}


def aggregate_stats(records: list[dict]) -> dict:
    out = {"counts": {
        "images_exp1": sum(1 for r in records if r["experiment"] == "exp1"),
        "images_exp1r2_round2_rows": sum(1 for r in records if r["experiment"] == "exp1r2"),
        "images_exp2_labelonly": sum(1 for r in records if r["experiment"] == "exp2"),
        "images_exp3": sum(1 for r in records if r["experiment"] == "exp3"),
        "images_exp3_with_human_labels": sum(1 for r in records
                                             if r["experiment"] == "exp3" and r["labels"]),
        "mice_exp1": len({r["animal"] for r in records if r["experiment"] == "exp1"}),
        "mice_exp2": len({r["animal"] for r in records if r["experiment"] == "exp2"}),
        "mice_exp3": len({r["animal"] for r in records if r["experiment"] == "exp3"}),
        "sessions_exp1": len({r["session"] for r in records if r["experiment"] == "exp1"}),
        "sessions_exp3": len({r["session"] for r in records if r["experiment"] == "exp3"}),
    }}
    out["exp1_images_per_mouse"] = {
        mouse: {"images": sum(1 for r in records if r["experiment"] == "exp1" and r["animal"] == mouse),
                "sessions": len({r["session"] for r in records
                                 if r["experiment"] == "exp1" and r["animal"] == mouse})}
        for mouse in sorted({r["animal"] for r in records if r["experiment"] == "exp1"})}

    dist: dict[str, dict] = {}
    for r in records:
        for rat, lab in r["labels"].items():
            for au, v in lab.items():
                key = f"{r['experiment']}|{rat}|{au}"
                d = dist.setdefault(key, {"0": 0, "1": 0, "2": 0, "NOT_RATEABLE": 0})
                d["NOT_RATEABLE" if v is None else str(v)] += 1
    out["label_distributions"] = dist

    seen: dict[str, list[str]] = {}
    for r in records:
        if r["experiment"] == "exp1r2":
            continue  # shares bytes with its exp1 parent row by construction
        if r["sha256"] not in ("unknown", ""):
            seen.setdefault(r["sha256"], []).append(r["zip_path"])
    out["byte_identical_image_groups"] = [v for v in seen.values() if len(v) > 1]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-labels", action="store_true", help="remote probe + small-table extraction only")
    ap.add_argument("--labels-only", action="store_true", help="build label matrix from probed tables")
    ap.add_argument("--zip", type=Path)
    ap.add_argument("--probe-dir", type=Path, default=Path(".prime_data/grimace"))
    ap.add_argument("--out", type=Path, default=Path(".prime_data/indexl2"))
    args = ap.parse_args()
    if args.fetch_labels:
        run_probe(args.probe_dir)
        return
    remote_dir = args.probe_dir / "remote_small"
    build(args.out, None if args.labels_only else args.zip, remote_dir)


if __name__ == "__main__":
    main()
