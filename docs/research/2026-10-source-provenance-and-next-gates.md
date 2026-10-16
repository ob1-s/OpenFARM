# Source-provenance cross-check and first admission candidates

**Dated 2026-10-15. This note concerns OLD vs NEW OpenFARM provenance and source admission; it does not certify any new benchmark.**

## Were the highlighted new sources already used?

**Edinburgh Pig Behavior = NEW to OpenFARM's old task set.** The previous pig audio task came from the **SoundWel** vocalization archive at Zenodo record **8252482** (our historical preparation notebook `notebooks/data/soundwel_vocal_data.ipynb` and the bioacoustics README); the Edinburgh 2019 dataset is a *separate* RGB-D record of eight pigs with manual bounding-box/behavior annotations across 7,200 frames from 12 selected clips, and has no corresponding environment in the archived GitHub tree. Source: https://homepages.inf.ed.ac.uk/rbf/PIGDATA/ . Also distinct from the **2021 Agroscope/SLU pig feeding + outbreak** and **2023 Agroscope social interaction + lesion** tables; do not conflate the three pig sources.

**EquiFACS 2020 = NEW source and annotations, not the existing Horse Grimace crops.** Our surviving horse task originates with **Lencioni et al. 2021**, DOI 10.1371/journal.pone.0258672, raw annotated still images at Mendeley DOI 10.17632/t8rtzcgwxm.3. EquiFACS uses **Rashid et al. 2020**, DOI 10.1371/journal.pone.0231608: six experimental horses (baseline and acute nociceptive stimulation), plus 21 clinical horses, with a certified FACS coder's **timed AU onset/offset annotations** and **three-veterinarian clinical pain assessments**. The paper makes distinct S1/S2 XLSX supplementary files and an ~880MB experimental video ZIP available; original clinical videos not listed as a public supplement. Note: both literatures involve horses and overlapping concepts, not necessarily independent biology. **They must be tracked as separate study cohorts** with individual IDs and clips.

Research references:
- https://pmc.ncbi.nlm.nih.gov/articles/PMC7608869/  (2020 paper and support files)
- https://data.mendeley.com/datasets/t8rtzcgwxm/3 (2021 image dataset)
- https://zenodo.org/records/8252482 (old pig vocalization source)
- https://homepages.inf.ed.ac.uk/rbf/PIGDATA/ (Edinburgh pig video)
- https://zenodo.org/records/5028016 (pig feeder behavior/outbreak)
- https://zenodo.org/records/5920843 (pig social events and lesions)

## Admission queue (first execution pass)

The repo's [DATA_APPROVAL_POLICY.md](DATA_APPROVAL_POLICY.md) defines DISCOVERED → ACCESS_VERIFIED → TASK_VALIDATED → APPROVED_RESEARCH_TRAIN / APPROVED_RESEARCH_EVAL as **separate gates**. These are **not automatic** or conferred by appearing in this list.

1. **Pig feeder/outbreak table (4MB).** Retrieve raw TXT, verify license from Zenodo API, parse schema, inspect real group/date/outbreak distributions and missingness. Critically: **control pens have arbitrary pseudo-event dates** in TBSTART. Approval of causal alert prediction needs prospective (t)-censored windows; exclude TBSTART, PenType, post-onset exposure and experimental split labels from model input; measure independent outbreaks (27, not tens of thousands of pig-days) with pen/farm grouping. Leave-one-farm-out possible but only 2 farms: show huge uncertainty.
2. **Agroscope pig ethogram/lesions tables (~530kB total).** Verify source file schema and animal/session/time alignment, independent animal IDs, what is actually expert-coded. Start **observed interaction** vs **lesion severity** as *separate* decisions, and only make predictive hypotheses if intervention time and before/after are real. Human annotations as GT, no videos.
3. **HotPig minute data (series.zip 2.6MB).** Inspect ZIP directory, per-pig rows/features, independent electronic feeder field. Explicitly note minute posture labels are **YOLO predictions**, so not expert GT. Manually labeled frames (621MB) are a *different* potential perception task; no certification without sample inspection.
4. **EquiFACS two XLSX (44.5kB and 103.1kB).** Inspect actual tables, rater/animal/video keys, AU onset/offset annotations, clinical versus experimental references. Admission for timed AU event detection requires synchronized source video and horse-heldout splits; **clinical XLSX without public clinical videos cannot by itself approve a video-perception eval**. Original large experimental ZIP only if smaller tables confirm useful labels.

All four are being targeted by reproducible source microaudit script `scripts/prospecting_microaudit.py` in GitHub Actions. Schema inspection and license metadata are **not** independent verification of biological ground truth; do not elevate without inspecting representative example content and origin.

## Noncommercial research benchmark

CC BY-NC and CC BY-NC-SA can support genuinely **noncommercial research** with attribution, required license notice and SA terms for published adaptations. They do not mean **commercial product development is automatically allowed**, or that any public benchmark is intrinsically NC. Media reproduction, derivatives/redistribution and a public evaluation API can trigger different rights from linking to source files; preserve split/test integrity. Keep source media upstream until permitted, and distinguish evaluation-only rights from training rights. Consult licensor when use is ambiguous.

### Current checkpoint

The only **approved** artifacts here are our *prospecting method and metadata inventory*, **not any new OpenFARM training or evaluation data**. This is a deliberate fail-closed state until actual labels, subject grouping, licenses and file bytes are verified.
