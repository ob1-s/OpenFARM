# OpenFARM Decision Index — Independent Welfare Evidence: Calf Clinical Outcomes (16 October 2026)

**Lane:** ChatGPT cross-species independent evidence scouting. **Branch:** `research/index-independent-welfare-evidence-2026-10`. Complements, does not duplicate, Operators A (pig L3), B (visual pain), C (farm visual/audio behavior). **Status:** primary source metadata/protocol inspection and public GitHub schema verification; original Dryad/Zenodo binary data not downloaded. **No new tasks approved.**

## Summary: one strong new supervision hypothesis, with explicit access gate

**Priority 1: Adcock & Tucker 2026 (Dryad `10.5061/dryad.k3j9kd5qt`), calf disbudding healing and nociceptive response.** Released data metadata document **48 calves** with repeated two-sided observations: **2,050 mechanical nociceptive threshold (MNT) test rows × 28 columns** and **1,502 wound tissue assessment rows × 40 columns**. A painful mechanical response is recorded in measured Newtons, and separately a trained observer scored seven tissue types, including exudate, necrosis, granulation, crust, epithelium/healing, plus IRT. Records include `Calf`, `Side`, `Test_date`, treatment arm and procedure dates. These are **two meaningfully different measurement families** recorded on real calves across weeks, not merely derived animal labels. The 48 calves come from **12 matched experimental replicate cohorts**, including sham/disbudded, ages 3/35 days; 40 Holstein, eight Jersey. Study reports wound healing mean 62 ± 10 days and follow-up 42–91 d.

Source links:
- https://datadryad.org/dataset/doi:10.5061/dryad.k3j9kd5qt
- https://doi.org/10.5061/dryad.k3j9kd5qt
- Files: `Disbudding_Calves_2016_MNT_Dryad.xlsx` (218.09KB, Dryad file_stream/4850856) and `Disbudding_Calves_2016_Tissue_Dryad.xlsx` (277.95KB, file_stream/4850857), plus a 14.28KB author README.

**Access discrepancy, do not gloss over:** The Dryad landing page directly describes both files and complete data dictionaries. However fetching both file_stream endpoints here failed HTTP 403 and this environment has no direct outbound DNS. This is **ACCESS_METADATA_VERIFIED, RAW_BYTES_UNVERIFIED**; no counts/joins beyond the authors' README independently verified. Do not call this a runnable benchmark or say records/rights are checked.

### Proposed research questions only after original workbook audit

**A. Experimental welfare physiology, bounded interpretation (Level 2):**
- Input: **strictly earlier** (previous visit) wound tissue grades, local skin temperature, age, side and prior response measurements only if genuinely available. Exclude sham/disbudding treatment arm initially as context-only shortcut.
- Target: `Value(next-visit mechanical nociceptive threshold MNT_N)`, a continuous outcome with physical units. Compare with paired rump-reference site to account for individual response propensity. If Decision Index needs only Noul/Choice/Score, do **not** discretize MNT arbitrarily: first establish a clinically meaningful cutoff or keep an experimental numeric task outside frozen suite.
- Alternative target: `Noul(next-visit wound epithelialized/healed)`, **actual independently observed** new epithelium from author scored tissue, not an optimal intervention recommendation.
- Short example contract **not an actual source row**: `state = wound tissue/temperature measured on day 14`; `question = did this horn bud re-epithelialize at next scored visit?`; `reference = observer's later Healed=1`, grouped by `Calf` and replicate cohort, no future `Latency_heal_d` in inputs.
- Original wound tissue observations are reportedly scored from photographs, **but the deposited files on Dryad list spreadsheets only**, not the source photographs. This is *longitudinal clinical observation*, not an already-released pixel-based vision task.
- Important confound: assigned hot-iron procedure and sham treatment strongly alter both wound status and MNT; the task must show value beyond procedure/time priors. Pain sensitivity is not an unassailable subjective pain ground truth, and tests are at specific regions with measurement-rate exclusion criteria.

**B. Level-3-adjacent outcome (not action supervision):**
- If the follow-up wound healing and procedural reaction history passes, ask whether earliest available noninvasive wound scores predict prolonged healing / next-visit deterioration, with censored follow-up and animal-disjoint uncertainty.
- Scheduled lidocaine/meloxicam and disbudding treatments are **assigned protocol interventions**, not naturally selected veterinarian care decisions. No autonomous escalation recommendation labels verified.

### Hard admission gates (these are nonnegotiable)
1. Obtain original XLSX bytes; record SHA-256 and exact author-provided sheet dimensions. Parse `Data` and `Data Dictionary` separately; verify 48 unique calves, 12 replicates, two sides and ordered date ranges. Check repeated same-day per-site mechanical tests and invalid `Rate_N_s` exclusion (authors specify 0.1–0.6 N/s).
2. Verify joins by (`Calf`, `Side`, `Test_date`, anatomical test site), including that rump reference rows are separate from horn-wound rows and no many-to-many multiplication occurs. Never count per-algometer-test rows as independent animals.
3. Explicitly distinguish same-day cross-sectional questions from *next-visit* forecasts. For forecasts only data strictly earlier than target can enter state. Evaluate animal-disjoint and replicate-block holdouts; with only 12 cohorts expect wide uncertainty.
4. Inspect actual future transition counts and target missingness; healed state may be absorbing, but don't assume absent follow-up implies not healed. Evaluate sham controls separately; avoid treatment/time shortcut.
5. Check Dryad data rights and file-specific use provisions, then write deterministic source builder and small baseline, **only once actual bytes are accessible**.
6. The dataset can remain a useful **positive welfare-physiology seed** even if no unbiased low-cost noninvasive predictor exists; absence of pixels and tiny N prohibit product claims.

## Priority 2: Direct calf respiratory monitoring — honest deposit-vs-paper reconciliation

Mach et al., *AI-Powered Acoustic Surveillance for Early Detection of Calf Respiratory Disease*, published 2026, article DOI https://doi.org/10.1016/j.animal.2026.101928 ; public dataset https://doi.org/10.57745/FP5LA9 . Paper: **2,730h** 30-day recordings, **4 pens × 7 calves = 28**, artificial infection status/antibiotic-stratified housing, HuBERT trained on **1,045** human labeled cough/non-cough clips, reports cough change preceding clinical score 1–2 days.

**Crucial:** Open dataset listing shows **45 items** (including 44 audio-classified files and 1 small model checkpoint), not a verified release of all 1,045 training clips, and no verified released 2,730h raw streams **or independent dated per-pen clinical scores**. Visible `non_cough_001.wav` entry: 86.2KB, author description "calf vocalization with harmonic structure; no cough signature present"; other small examples indexed. Listed dataset license **Etalab Open License 2.0**, compatible with CC BY 2.0; review detailed terms before use.

Verdict: real public source metadata and visible audio examples, **not** a ready cough→future BRD task. At best short-clip cough `Noul` Level 1 and *hypothesis* that aggregate counts precede clinical changes. Four experimental pens are confounded with infection/treatment; no farm generalization. **If full time-stamped clinical series and original audio aren't in the deposition, reject claim of L2/L3 forecast readiness.** Distinguish released file count from paper corpus without inventing access.

## Priority 3: Alternative real multisensor timeline and cautionary negatives

### MmCows (Purdue), https://github.com/neis-lab/mmcows and https://huggingface.co/datasets/neis-lab/mmcows
- **Source truly published**, HF card **CC BY-NC-SA 4.0**, `sensor_data.zip` ~18.1GB, 14 days of UWB/inertial/temperature/lying/milk/weather, paired video across 16 cows (wearables 10) and `sub_data/health_records` named in published source tree; separate manually annotated activities for one day. The 2024 public README includes first-hand modality sampling: core temperature 60s, UWB 15s, neck IMMU 0.1s.
- **Critical unknown:** Whether `health_records` actually contain repeated veterinarian assessments/action times or merely static health history. Filename alone establishes nothing. Huge archive/16 cows/noncommercial rights: do NOT download 18GB before inspecting a byte-range ZIP index and tiny health-record member. If records are static, label as L1 behavior/instrumentation only. Both group size and source license limit benchmark/commercial usefulness.

### DigiMuh, https://github.com/zerotonin/digimuh
- Publishes an MIT-licensed *processing pipeline* for ~8.9GB dairy-cow multimodal records across 2021–2024, including HerdePlus diseases and animal-level sensors. However the repo labels selection sheets/private herd data and **does not publish original 8.9GB underlying farm records**. Code/open manuscript ≠ accessible disease-event/sensor pairs. **REJECT_ACCESS as a current standalone benchmark**; do not use synthetic `example` workbook as real data.

### CalvingDetection, https://github.com/huipingcao/CalvingDetection
- **Unlike most leads, actual primary text CSV sensor bytes are accessible on public GitHub.** Inspected raw `Processed_data/09BcollarData.csv` first seven source rows: hourly UTC `Rounded_Time`, `KNN2..5`, cumulative `payload.activityCount`, `Distance_moved`, `ActivityCount_Difference`. Repo has **15 actual per-cow processed CSVs** plus 15 raw sensor XLSX and `Calving date Brangus collars LoRa WAN CDRRC.xlsx` (9.6KB); repo license GPL-3.0 (verify dataset rights separately).
- Possible **actual-timestamp future event benchmark**: pre-calving monitoring vs recorded delivery date, with within-cow before/after time-series and animal-heldout testing. But calving is a predictable physiological life event, *not automatically an adverse welfare condition or independently judged need for intervention*. With only 15 cows / unknown sites and a binary event-date spreadsheet not yet parsed, no validated task. Flag for a future **operational event pilot**, not L3 care-action research.

### Synthetic false leads: hard reject
- `https://zenodo.org/records/17045457` explicitly states per-milking records are *dummy generated demo*, no real farm data. `https://github.com/uxeer/cow-health-vital-signs-datasets` explicitly synthetic 22k rows per condition, with health labels generated from value ranges. **Never train/evaluate welfare intelligence on those labels.**

## Cross-lane handoff vs ongoing A/B/C branches

- **Don't redirect A** from revising Aarhus PR #4 and Chou 2020 clinical escalation; avoid branch/file collisions.
- **Don't redirect B** from GrimACE original image acquisition.
- **Don't redirect C** from Big Broiler/ChickenVerse temporal event episodes.
- ChatGPT's independent lane concentrates on **calf wounds→independently measured pain sensitivity and future healing**, then verifies if respiratory sound/time-aligned clinical data genuinely exists.
- Admission count remains **zero new tasks** from this note. The added value is a testable source-grounded multireference hypothesis, a documented source-data mismatch, and one verified small original sensor CSV family, not inflated benchmark totals.

## Status and next action

**First scientific acquisition action:** try the two Dryad XLSX source files from a network where official file_stream downloads work; verify original hashes, physically sampled rows and per-animal/date joins; construct source truth table and group-split contract if passed. Do not promote or claim task was run before that gate. If Dryad remains inaccessible, pivot to a *small* public clinical+physiology source, not artificial proxy labels.

**Evidence integrity:** All figures above for unparsed XLSX/ZIP files are **author-provided descriptions, not independent row audits**. Source code observed directly for the `CalvingDetection` CSV and README and `MmCows` README; public dataset landing pages provide other metadata. No model trained, no historic benchmark changed.
