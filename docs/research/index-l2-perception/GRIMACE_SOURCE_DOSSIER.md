# GrimACE L2 Direct Welfare Perception - Source Dossier

Operator: B, 2026-10-16. Task: validate and materialize a scientifically defensible,
animal-disjoint image benchmark from an original public source. Target: **GrimACE**
mouse grimace-scale image deposit (Zenodo 18299984), companion to Sturman et al.,
*Current Research in Biotechnology / Lab Anim* 2026 (PMC13043301, DOI 10.1016/j.mimet.2025.103092).

All claims below are machine-checked (`scripts/index_l2_grimace_build.py`,
`scripts/index_l2_grimace_benchmark.py`); reproduction commands at the end.

## 1. Source record

| File | Size | MD5 |
|---|---|---|
| `Data.zip` | 3,829,346,399 B | `729262e0872aab38cab3da52b499672d` (verified post-download; sha256 `d2e974b8be4111adbe932f5999e342b2b3a1541e707abb5f1de77089fa91b6f6`) |
| `GrimACE_Live_Demo.zip` | 5,128,940,644 B | `4e0a694d112ece5c2ba64db3626c1442` |
| `GrimACE_hardware.zip` | 1,165,209,184 B | (CD-listed; not needed) |
| `analysis_scripts.zip` | 8,347,519 B | `7f4902f557ba58a262a0a1cb5936fbf7` |
| `Demo_Video.mp4` | 104,590,417 B | `0877b397d599849a3161c2ef9f4a0c29` |
| `README.txt` | 12,771 B | `924689551e50f071727a410f38ce9298` |

- Record-level license: **apgl-v3** (unusual for a data deposit - conservative
  handling: no source bytes and no per-row label values are committed to Git;
  only manifests, hashes, split membership, and aggregate reports).
- The same files appear under the duplicated Zenodo record 15119195 (identical
  title/checksums). We cite 18299984.

## 2. Deposit structure (ZIP central directory, parsed remotely then verified locally)

`Data.zip` = 8,991 members / ~5.0 GB uncompressed:

- `Data/MGS_scores/images/GrimaceTest1_Part1|Part2/psFC*day*` - **773 scored 1280x1024 RGB PNGs**
  (35 + 39 session folders, 74 sessions total).
- `Data/MGS_scores/values/` - the ground-truth tables (CSV, CRLF) + 2 anonymization XLSX:
  - `gt1_gt2_automatic.csv` (1,246 rows: `automatic` training metadata + **773 `gt1_*`** + **472 `gt2_*`** expert rows; columns `video_name, frame_index, eye, nose, cheek, ear, whisker, average`)
  - `grimacetest1_part1|part2_{clara,yvonne,hohlbaum}.csv` (external raters, exp1: identical 773-frame key sets)
  - `grimacetest2_{clara,yvonne,hohlbaum}.csv` (575 rows each: **472 exp2-style** + **103 exp1 psFC folders**)
  - `gt1_gt2_automatic_averaged.csv` (video-level metadata incl. authoritative `animal_id`)
  - `gt3_automatic_individual.csv` (1,771 rows: `automatic` posteriors incl. per-row `treatment`,`timepoint` + **124 `hohlbaum`** rows over exp3 sessions)
  - (exp1 na / exp3 na details in section 4)
- `Data/MGS_scores/images/GrimaceTest3Frames/a*_c_*` - **1,725 control-only frames**, 53 folders.
- `Data/FaceCropperTrainingData/` (1,332 PNG + YOLO-format TXT pairs; PNGs are full frames; crops are train-time boxes, not pre-cropped pixels)
- `Data/FrameSelectorTrainingData/FramesR3200/` + two annotation CSVs (image quality scores; same 3,200 file set both rounds, columns `video_name,image_name,score0/1`).

Every deposit PNG is a **full 1280x1024 RGB frame**; the "crop" is a YOLO box in the
TXT sidecar (`cx cy w h` normalized). Anything claiming "cropped" refers to runtime
cropping, not the deliverable state.

**No HDF5 files are present in either deposit.** The training code
(github.com/ETHZ-INS/GrimACE_manuscript, AGPL) expects
`datasets/LoaoBBSurround1.h5` + `LoaoPaddedBB.h5` with `fold_ranges` (31-fold LOAO),
`set_names`, `num_samples` attributes - these artifacts were not distributed, so we
rebuild our own canonical corpus from the deposit originals (see benchmark card).

## 3. The paper's numbers reconcile exactly

- 773 (`gt1`) + 472 (`gt2`) = **1,245** = the paper's "1,245 expert-scored images from 31 mice" (15 exp1 + 16 exp2 mice).
- Treatment split: exp1 `m` = 8 mice (3340, 3342, 3346, 3349, 3354, 3369, 3371, 3379), `b` = 7 (3341, 3343, 3347, 3348, 3355, 3367, 3376) - matches the paper's 8M/7B.
- 31-fold LOAO animals = 15 (exp1) + 16 (exp2) mice.

## 4. Findings that a naive ingestion would get wrong

1. **exp1 mouse-id aliasing (leakage trap).** Session folders `psFC3440|3441|3442|
   3443*` are the **same mice** as `3340-3343` (proven by the `animal_id` column of
   `gt1_gt2_automatic_averaged.csv`; folder tag is authoritative only when it agrees).
2. **exp1 PNGs and CSV keys match exactly**: the union of Part1+Part2 deposit image
   (session, frame-number) keys equals the 773 `gt1` keys (machine-verified).
3. **exp2 scored pixels are absent from the deposit.** `grimacetest2_*` annotate 575
   frames but the a-mouse style names appear nowhere in the image tree. The 575
   rows decompose as 472 exp2-style keys (16 mice a3694..a3777) + **103 psFC rows
   that point at exp1 images** - i.e. a *second annotation round* over exp1 frames
   by the same three external raters (10 sessions, all keys validated to exist in
   the exp1 image tree). Externals are internally consistent in each round
   (identical key sets across clara/yvonne/hohlbaum per round).
4. **exp3 CSV/deposit session mismatches.** `gt3_automatic_individual.csv` +
   `gt3_hohlbaum_individual.csv` reference session `a12_c_24h_front_12` (33/2 rows)
   while the deposit ships sibling `a12_c_24h_front_13` (37 frames, fully disjoint
   frame numbers, no human rows), and `a4_c_24h_front_5` (49/3 rows) has no deposit
   folder at all. Net: 82 of 1,770 automatic rows and 5 of 124 hohlbaum rows have no
   matched pixels; 1,688 of 1,725 deposit frames have exactly one automatic row.
5. **exp3 hohlbaum labels exist only for eye & ear** (119 usable frames each;
   nose/cheek/whisker cells are blank for every exp3 human row).
6. **Blank cells are intentional**: gt1 has 722 blank AU cells (whisker 407, cheek
   166, nose 149), gt2 has 245. Blank = "not rateable", never silently 0. Item
   counts per AU are documented in `GRIMACE_build_report.json`.
7. **Anonymization XLSX quirk**: Part1's `matched_folder_names.xlsx` lists 75
   anonym names for 74 sessions and writes `psFC3379day30` where the deposit uses
   `psFC3379day31` (day3-family typo; exp1 also has day30/day31 folder-tag typos).
8. **Rater identity strata vs the paper.** The paper names an expert and two
   trainees; the deposit has `gt1` (also `gt2`) + clara + yvonne + hohlbaum.
   Whether clara/yvonne/hohlbaum are the two trainees plus a second expert, or
   external co-authors, is **not determinable from the deposit**. Recorded as an
   open item; it does not affect animal-disjoint splits (animal ids are rater-independent).

## 5. Limitations

- exp2: labels-only (pixels never released) - used for cross-experiment rater
  agreement only; excluded from the image benchmark.
- exp3: single-treatment (`c`) control cohort; human labels only on 119 pixel-backed
  frames (eye/ear). The `automatic` rows are model posteriors (floats), provided as
  companion evidence (alert-rate comparisons), **not** an eval target.
- One unscored 37-frame deposit folder (`a12_c_24h_front_13`) stays inventory-only.
- License: apgl-v3 - derived artifacts stay local; only aggregate outputs are
  committed to this repository.

## 6. Reproduction

```bash
# probe (remote byte-range central directory + label tables, no archive download)
python3 scripts/index_l2_grimace_build.py --fetch-labels
# labels-only canonical manifest from probed tables
python3 scripts/index_l2_grimace_build.py --labels-only
# full verified build (archive md5 must equal 729262e0872aab38cab3da52b499672d)
python3 scripts/index_l2_grimace_build.py --zip .prime_data/grimace/Data.zip
# offline tests (stdlib only)
python3 -m unittest tests.test_index_l2_grimace
# benchmark: label-only analytics, then pixel baselines
python3 scripts/index_l2_grimace_benchmark.py labels
uv run --with pillow,numpy,scikit-learn scripts/index_l2_grimace_benchmark.py pixels
```

Artifacts: `.prime_data/grimace/` (record.json, remote_cd_entries.json,
probe_report.json, remote_small/, Data.zip), `.prime_data/indexl2/` (manifest.csv,
splits.json, automatic_scores.csv, aggregate_stats.json, build_report.json).
Committed outputs live next to this file: `GRIMACE_split_membership.csv`,
`GRIMACE_image_sha256_registry.csv`, `GRIMACE_build_report.json`,
`GRIMACE_benchmark_report.json`.
