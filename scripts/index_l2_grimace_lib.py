"""GrimACE Level-2 benchmark library: canonical types, label parsing, animal grouping, splits.

Source: Zenodo record 18299984 (Data.zip, md5 729262e0872aab38cab3da52b499672d),
Sturman et al., GrimACE (Lab Anim 2026, PMC13043301). Record-level license apgl-v3.

All parsing/grouping/splitting logic is stdlib-only and deterministic so the
offline unit tests can exercise it without the source archive.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

ZENODO_RECORD = "zenodo.org/records/18299984"
DATA_ZIP_MD5 = "729262e0872aab38cab3da52b499672d"

AUS = ["eye", "nose", "cheek", "ear", "whisker"]

# Local file names inside Data.zip (verified against the ZIP central directory).
CSV_GT_TRAIN = "Data/MGS_scores/values/gt1_gt2_automatic.csv"
CSV_GT_EXP34 = "Data/MGS_scores/values/gt3_automatic_individual.csv"
CSV_EXP34_HOHLBAUM = "Data/MGS_scores/values/gt3_hohlbaum_individual.csv"
CSV_EXTERNALS = {
    "exp1_part1": {
        "clara": "Data/MGS_scores/values/grimacetest1_part1_clara.csv",
        "yvonne": "Data/MGS_scores/values/grimacetest1_part1_yvonne.csv",
        "hohlbaum": "Data/MGS_scores/values/grimacetest1_part1_hohlbaum.csv",
    },
    "exp1_part2": {
        "clara": "Data/MGS_scores/values/grimacetest1_part2_clara.csv",
        "yvonne": "Data/MGS_scores/values/grimacetest1_part2_yvonne.csv",
        "hohlbaum": "Data/MGS_scores/values/grimacetest1_part2_hohlbaum.csv",
    },
    "exp2": {
        "clara": "Data/MGS_scores/values/grimacetest2_clara.csv",
        "yvonne": "Data/MGS_scores/values/grimacetest2_yvonne.csv",
        "hohlbaum": "Data/MGS_scores/values/grimacetest2_hohlbaum.csv",
    },
}
XLSX_PART1 = "Data/MGS_scores/images/GrimaceTest1_Part1/matched_folder_names.xlsx"
XLSX_PART2 = "Data/MGS_scores/images/GrimaceTest1_Part2/matched_folder_names_trainingset2.xlsx"

TRAIN_RATER = "gt1"      # exp1 ground-truth trainer annotations (472 exp2 rows are "gt2")
EXP2_RATER = "gt2"
EXP3_EXTERNAL = "hohlbaum"

# ---------------------------------------------------------------------------
# Canonical image key = (experiment, session, frame) where experiment in
# {"exp1","exp2","exp3"}; exp2 has no released scored images (labels only).
# Labels: per (image_key, AU) -> {rater: ordinal} with ordinal values {0,1,2} or None
# meaning not rateable/blank in the source spreadsheet (never silently 0).
# ---------------------------------------------------------------------------


@dataclass
class ImageRecord:
    experiment: str
    session: str          # source folder name, e.g. psFC3346day1_4h / a12_c_4h_front_39
    animal: str           # canonical animal id (merged across folder-tag anomalies)
    session_raw: str      # original folder label (may carry source typos)
    frame: int            # frame index == image number (imgNNNNN)
    zip_path: str         # member path inside Data.zip ("" when labels-only)
    sha256: str
    size: int
    crc32: str
    treatment: str        # m / b / c / lm / unknown (metadata only, never model input)
    timepoint: str        # p24/0h/4h/24h/48h/72h (metadata only)
    labels: dict = field(default_factory=dict)  # rater -> {AU: value|None}


def parse_label_cell(cell: str) -> int | None:
    """Cell -> 0/1/2 or None. Blank/whitespace/non-numeric means not rateable/missing."""
    s = (cell or "").strip()
    if s in ("", "-", "NA", "na", "n/a", "NR"):
        return None
    try:
        v = int(s)
    except ValueError:
        return None
    if v in (0, 1, 2):
        return v
    raise ValueError(f"unexpected ordinal label {v!r}")


def parse_externals_csv(text: str, auto_rater: str) -> list[tuple[str, int, dict]]:
    """Parse grimacetestN_{rater}.csv rows -> [(session, frame, labels)]."""
    import csv
    import io

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        session = row["folder"]
        image = row["image"]
        if not (image.startswith("img") and image.endswith(".png")):
            raise ValueError(f"unexpected image key {image!r}")
        frame = int(image[3:-4])
        labels = {au: parse_label_cell(row[au]) for au in AUS}
        out.append((session, frame, labels))
    return out


def parse_gt_train_csv(text: str) -> list[tuple[str, str, int, dict]]:
    """Parse gt1_gt2_automatic.csv -> [(rater, video_name, frame, labels)] (drops automatic rows)."""
    import csv
    import io

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        vn = row["video_name"]
        rater = vn.split("_", 1)[0]
        if rater == "automatic":
            continue
        video = vn[len(rater) + 1:]
        out.append((rater, video, int(row["frame_index"]),
                    {au: parse_label_cell(row[au]) for au in AUS}))
    return out


def parse_gt3_automatic_individual(text: str) -> list[tuple[str, int, str, str, dict]]:
    """Parse gt3_automatic_individual.csv -> [(video, animal_id, treatment, timepoint, labels)] for automatic rows."""
    import csv
    import io

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["annotator"] != "automatic":
            continue
        out.append((row["video_name"], row["animal_id"], row["treatment"], row["timepoint"],
                    {au: parse_label_cell(row[au]) for au in AUS}))
    return out


def parse_gt3_hohlbaum_individual(text: str) -> list[tuple[str, int, dict]]:
    import csv
    import io

    out = []
    for row in csv.DictReader(io.StringIO(text)):
        if row["annotator"] != "hohlbaum":
            continue
        out.append((row["video_name"], int(row["frame_index"]),
                    {au: parse_label_cell(row[au]) for au in AUS}))
    return out


def parse_averaged_csv(text: str, expected: set[str]) -> dict[str, dict[str, str]]:
    """Parse gt1_gt2_automatic_averaged.csv (automatic rows) → session metadata."""
    import csv
    import io

    meta: dict[str, dict[str, str]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if row["annotator"] != "automatic":
            continue
        vid = row["video_name"]
        if vid not in expected:
            raise ValueError(f"averaged row for unknown video {vid}")
        meta[vid] = {
            "animal_id": row["animal_id"],
            "treatment": row["treatment"],
            "timepoint": row["timepoint"],
            "num_frames": row["num_frames"],
        }
    return meta


# ---------------------------------------------------------------------------
# Animal grouping: folder names can carry typos (psFC3440 == mouse 3340 etc.).
# The averaged file's animal_id is authoritative for exp1 session folder names.
# ---------------------------------------------------------------------------

def canonical_animal_exp1(session_folder: str, averaged_animal_id: str | None) -> str:
    """Return canonical mouse id for an exp1 session folder.

    averaged_animal_id comes from_gt1_gt2_automatic_averaged.csv, whose animal_id
    column matches the real animal even when the folder tag has a typo
    (seen: psFC3440day0 → 3340, psFC3441* → 3341, psFC3442day0 → 3342, psFC3443day0 → 3343).
    """
    import re
    m = re.match(r"psFC(\d+)", session_folder)
    if m is None:
        raise ValueError(f"unparsable exp1 session {session_folder}")
    folder_tag = m.group(1)
    if averaged_animal_id is not None and averaged_animal_id != folder_tag:
        # folder tag anomaly; the averaged file is authoritative
        return averaged_animal_id
    return folder_tag


def canonical_animal_from_folder(folder: str) -> str:
    import re
    m = re.match(r"psFC(\d+)", folder)
    if m is None:
        raise ValueError(f"unparsable exp1 session {folder}")
    return m.group(1)


def canonical_animal_exp3(video_name: str, animal_id: str) -> str:
    """exp3 video names look like a12_c_4h_front_39; source animal_id col is '12'."""
    import re
    m = re.match(r"a(\d+)_", video_name)
    if m is None or m.group(1) != animal_id:
        raise ValueError(f"exp3 id mismatch: {video_name} vs {animal_id}")
    return animal_id


# ---------------------------------------------------------------------------
# Frozen animal-disjoint splits. Deterministic, stratified by treatment where
# available, chosen BEFORE examining any test-label statistics.
# ---------------------------------------------------------------------------

def stratified_animal_split(
    animals: list[str],
    treatment_of: dict[str, str],
    n_test: int = 2,
    n_cal: int = 1,
) -> dict[str, list[str]]:
    """For each treatment group (animals sorted ascending): last n_test -> test,
    previous n_cal -> calibration, rest -> train. Deterministic; no RNG."""
    groups: dict[str, list[str]] = {"train": [], "cal": [], "test": []}
    # numeric-aware sort ("3" < "12"); exp1 ids are all-width so this equals plain sort there
    id_key = lambda s: (len(s), s)
    for t in sorted({treatment_of[a] for a in animals if a in treatment_of}):
        group = sorted((a for a in animals if treatment_of.get(a) == t), key=id_key)
        test = group[-n_test:]
        cal = group[-n_test - n_cal:-n_test]
        train = group[:-n_test - n_cal]
        groups["test"].extend(test)
        groups["cal"].extend(cal)
        groups["train"].extend(train)
    # animals without known treatment -> train (and recorded as unknown in the manifest)
    known = {a for a in animals if a in treatment_of}
    for a in sorted(animals, key=id_key):
        if a not in known:
            groups["train"].append(a)
    return {k: sorted(v, key=id_key) for k, v in groups.items()}


def animal_split_digests(split: dict[str, list[str]]) -> dict[str, str]:
    out = {}
    for part, members in split.items():
        blob = json.dumps(members, separators=(",", ":")).encode()
        out[part] = hashlib.sha256(blob).hexdigest()
    return out


def validate_split(split: dict[str, list[str]]):
    assert set(split) == {"train", "cal", "test"}
    all_members = [a for v in split.values() for a in v]
    assert len(all_members) == len(set(all_members)), "animal appears in two splits"


# ---------------------------------------------------------------------------
# Consensus (secondary reference; per-rater evaluation remains primary).
# ---------------------------------------------------------------------------

def per_au_consensus(rater_labels: dict[str, int | None], min_raters: int = 2) -> int | None | str:
    """Deterministic strict-majority consensus for one (image, AU).

    Returns ordinal {0,1,2} when a value holds strictly more than half of the
    usable (non-None) human ratings; "NOT_RATEABLE" when every rater returned
    None; "NO_CONSENSUS" otherwise (including <min_raters usable ratings).
    """
    usable = [(r, v) for r, v in rater_labels.items() if v is not None]
    masked = [r for r, v in rater_labels.items() if v is None]
    if not rater_labels:
        return "NO_CONSENSUS"  # nothing to evaluate (distinct from all-raters-blank)
    if len(masked) == len(rater_labels):
        return "NOT_RATEABLE"
    if len(usable) < min_raters:
        return "NO_CONSENSUS"
    from collections import Counter
    c = Counter(v for _, v in usable)
    val, cnt = c.most_common(1)[0]
    if cnt * 2 > len(usable):
        return val
    return "NO_CONSENSUS"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_digest(rows: list[dict]) -> str:
    """Stable digest over a manifest (list of dicts serialized with sorted keys)."""
    h = hashlib.sha256()
    for r in sorted(rows, key=lambda d: json.dumps(d, sort_keys=True, separators=(",", ":"))):
        h.update(json.dumps(r, sort_keys=True, separators=(",", ":")).encode())
        h.update(b"\n")
    return h.hexdigest()


def split_membership_digest(split: dict[str, list[str]]) -> str:
    parts = []
    for part in ("train", "cal", "test"):
        parts.append(json.dumps(split[part], separators=(",", ":")))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def session_timepoint_from_averaged(meta: dict | None, session_folder: str) -> str:
    if meta is None:
        return "unknown"
    return meta.get("timepoint", "unknown")


def zip_image_path_exp1(session_folder: str, frame: int, part: str) -> str:
    return f"Data/MGS_scores/images/{part}/{session_folder}/img{frame:05d}.png"


def zip_image_path_exp3(session: str, frame: int) -> str:
    return f"Data/MGS_scores/images/GrimaceTest3Frames/{session}/img{frame}.png"


def locate_exp1_part(session_folder: str, available_parts: dict[str, str]) -> str:
    """available_parts maps session folder -> part name ('GrimaceTest1_Part1'...)."""
    if session_folder in available_parts:
        return available_parts[session_folder]
    raise KeyError(session_folder)


def session_to_exp3_mouse(session: str) -> str:
    import re
    m = re.match(r"a(\d+)_", session)
    if not m:
        raise ValueError(f"bad exp3 session {session}")
    return m.group(1)


def session_to_exp3_timepoint(session: str) -> tuple[str, str]:
    """a12_c_4h_front_39 -> ('c', '4h'); treatment words {c,m,b,lm}; timepoints {0h,4h,24h,48h,72h,p24}."""
    import re
    m = re.match(r"a\d+_([a-z]{1,2})_(0h|4h|24h|48h|72h|p24)_front_\d+$", session)
    if not m or m.group(1) not in {"c", "m", "b", "lm"}:
        raise ValueError(f"cannot parse exp3 session {session}")
    return m.group(1), m.group(2)
