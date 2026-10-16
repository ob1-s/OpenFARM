# First real-file approval pass — 15 Oct 2026

**This document is updated from read-only downloads made on GitHub Actions.** It distinguishes scientific label quality, source access, legal conditions, and what we may actually enter into OpenFARM's train/eval suite.

**References**
- [Research source register](2026-10-animal-data-source-register.csv)
- [Admission rules](DATA_APPROVAL_POLICY.md)
- [Pig/horse provenance cross-check](2026-10-source-provenance-and-next-gates.md)
- [Reproducible initial audit](../../scripts/prospecting_microaudit.py)
- [Microaudit CI run](https://github.com/ob1-s/OpenFARM/actions/runs/38819485981)

## Sources with verified data access

### HotPig — ACCESS_VERIFIED (not TRAIN/EVAL approved)

Source: [Zenodo 17090997](https://zenodo.org/records/17090997), DOI 10.5281/zenodo.17090997.

Verified by programmatic **download of real bytes**, not just landing-page metadata:

- Zenodo record reports **CC BY 4.0**, open access.
- Actual file `series.zip` downloaded (**2,603,605 bytes**, SHA-256 beginning `1ba5125167224b96`). ZIP includes **24 individual P1–P24.csv files** and an explanatory README.
- Independently parsed the first three per-pig series, each with **23,040 minute rows**. Columns: `period`, `conditions`, `feed`, `unknown`, `standing`, `seating`, `lying`, `eating`, `drinking`, `cuddling`, `curious`, `idle`, `drink`, `eat`, `mate`, `state`.
- Also downloaded `metadata.txt` (**1,117 bytes**, content hash beginning `ef3b3910c1ea6705`).
- **Scientific label distinction:** minute-level posture/event/state values are **model-generated** from a detector; the 621MB `frames.zip` is a different resource with manual frame annotations; `feed` is from independent electronic feeders according to source documentation. Heat-treatment schedule is an **experimental condition**, NOT direct welfare distress ground truth.

**All-24-pig deep data check, verified in GitHub Actions run [38820334855](https://github.com/ob1-s/OpenFARM/actions/runs/38820334855):**

- The archive contains **24 valid CSVs × 23,040 rows = 552,960 minute records** across 16 days. These are **repeated minutes from 24 individuals**, not hundreds of thousands of independent animals/labels.
- Four experimental condition labels occur in the released series: `TN`, `HS`, `TU`, `TD` (and **2,142 null condition cells**). Every animal covers day/night periods and specified treatment phases.
- `state` is majority `idle` (**411,708/552,960**), followed by `active` (73,462), `drinking` (46,260), `unknown` (12,927), `curious` (4,918), and `cuddling` (3,685). These are detector/derived state categories, **not expert-confirmed emotional classes**.
- `feed` is numerical; inspected individuals have independent feeder values ranging zero to >0, not a constant feature. Some pigs have missing intervals in `feed` and related video-derived features (example P15 and P16: 66 missing rows apiece). Drop or mark invalid, never silently impute zeros.
- **Proposal requiring further validation:** `Noul(electronic feeder recorded feed>0 in this minute)` from *camera-derived posture/event features only*, with **16 train pigs, 4 calibration pigs and 4 held-out test pigs**, no `feed`, `conditions`, `period`, timestamp or pig ID exposed in the model state. A simple detector rule is being checked before treating this as an informative task. This is **instrument agreement**, not distress/pain detection or an alert policy.

**Provisional value:** bounded compact time-series states (no hours of raw video in the System-1 model); low-cost baseline-change decisions and instrument-validated feeding behavior. A training/eval approval still requires seeing actual condition/missingness distributions, ensuring pen/pig/session disjoint groups, label provenance, constructing a genuinely independent behavioral reference, and verifying a bounded welfare question. A CC BY publication license makes research/commercial reuse comparatively promising, **subject to checking any underlying video rights and intended use**.

### EquiFACS 2020 — ANNOTATION_ACCESS_VERIFIED (not TRAIN/EVAL approved)

Source: [Rashid et al. PLOS ONE 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7608869/), DOI 10.1371/journal.pone.0231608.

Downloaded directly from PLOS supplementary links (not the 2021 OpenFARM Lencioni/Mendeley image dataset):

- `s002.xlsx`: **45,555 bytes**, SHA-256 beginning `2df498362259cbd7`. Sheets: `FACS` (127 spreadsheet rows; wide facial-action structure across recordings) and `Pain` (**12 rows**, columns `Video no`, `Pain/no pain`, `Condition`, `Horse ID`).
- `s003.xlsx`: **105,566 bytes**, SHA-256 beginning `f33fe32b8bd12830`. Sheets: `FACS` (**1,449 spreadsheet rows**, columns including media, annotation date, annotator) and `PAIN` (**167 spreadsheet rows**, columns Media, Begin Time, End time, Duration, Pain, Annotator). **These counts are annotation-sheet rows, not the number of clinical subjects**.
- A second deep read confirmed **12 distinct experimental videos / six horses** with exactly **six ischemic-pain and six baseline** annotations. In the clinical `PAIN` sheet, the `Media` column has **22 distinct entries** (not 22 verified different horses); among the 167 interval annotation rows, **at least 77 are marked `Non-codeable`**, and other categorical strings differ only by trailing spaces. Do not treat these intervals as independent clinical samples; preserve genuine unknown/not-rateable intervals and reconcile media IDs against the published 21 clinical horses. Verified in [CI run 38820130470](https://github.com/ob1-s/OpenFARM/actions/runs/38820130470) (the run overall failed on a transient HotPig download; the EquiFACS file inspection succeeded).
- Original study: **six experimental horses and 21 clinical horses**, annotated AUs with onset/offset; three vets rated clinical pain, individual AU coding by certified expert.
- The paper publishes experimental source video as one roughly **880MB ZIP**, plus these spreadsheets. It does not clearly provide the 21 **clinical videos** in the same supplement.
- PLOS article CC BY; verify video/media and any constraints before redistributing a derived benchmark.

**Provisional value:** multiple **distinct**, evidence-supported typed System-1 decisions on AU activation, onset/offset and separately veterinary pain labels. Stronger source than static HGS score-threshold “pain.” Research train/eval approval requires validating exact video IDs/clip lengths, real AU event intervals, source quality, animal-heldout data, clinical media availability and rights, and a constrained AU/clinical question. No claim of a 1,449-example clinical video dataset.

## Sources NOT YET accessible at the file level

### Pig feeder/tail-biting
[Zenodo 5028016](https://zenodo.org/records/5028016). Zenodo metadata API confirms **open access under CC BY-NC 4.0**. The 4MB original TXT download **timed out** during this run. No row parsing or independent event/pen statistics validated yet. The paper/record documents that control pen `TBSTART` dates are **artificial assignments**, so a first seven-day risk target must use only **real** future onset dates, never fake control events; omit `PenType`, `TBSTART`, date-derived future hints and collection group from predictors. Noncommercial research may be possible, but product uses require license clearance.

### Agroscope pig social interactions/lesions
[Zenodo 5920843](https://zenodo.org/records/5920843). Source page lists expert ethogram, `activity.txt`, `animal.txt`, `confrontations.txt`, `conspecifics.txt`, `lesions.txt`, `objects.txt`, `rooting.txt`, and PDF explanations. This run timed out at Zenodo API; **neither the file contents nor its precise dataset license was certified**. No social-event→subsequent-lesion temporal inference is approved yet.

## What comes next for admission

1. From **HotPig**: inspect all 24 time series, measured feeder signal distribution, conditions, subject/time and missingness; separate expert frame training task from detector-produced minute pseudo labels; then test simple low-cost detection/change baselines with independent reference. No generic stress label.
2. From **EquiFACS**: reconstruct *actual onset/offset interval labels* from wide spreadsheets, join to the 12 experimental videos and separately understand the 21 clinical cases/167 annotation rows; verify potential horse-heldout split and clinical media rights.
3. Retry **pig feeding/social** sources with alternate Zenodo file URLs and bounded exponential backoff. Do not mark accessible by reading only the dataset card.
4. Declare candidate train and eval suites **separately** under [approval policy](DATA_APPROVAL_POLICY.md); inspect actual representative examples first. No existing source in this report yet has promotion to APPROVED_RESEARCH_TRAIN or APPROVED_RESEARCH_EVAL.

### Important note

The purpose is not to collect arbitrary high-accuracy classification tasks; it is to build **low-cost, calibrated, typed observation → welfare indicator → justified human-attention decision** tasks for a zero-shot+ animal well-being monitor, across feasible visual, audio, pose, time-series and clinical modalities. Inferred **affect** requires independent welfare and behavioral evidence. 
