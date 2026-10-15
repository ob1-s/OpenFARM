# Final product-fit audit — Horse Grimace and CatMeows
**15 October 2026 · System-1 reboot checkpoint · supersedes initial two-task shortlist**

**Mission to optimize for:** animal well-being/behavior intelligence; affordable, lightweight, calibrated observation-level System-1 decisions; prospective detection of situations in which human attention is warranted. A commercial monitoring anchor must outperform credible **cheap zero-shot / frozen-encoder baselines** on domain-heldout observations at viable cost, and measure missed meaningful events and false alerts **per animal-day**, not merely curated classification accuracy.

This document separates three things which were previously conflated:
1. **Dataset integrity**: source/annotation identity, duplicates, splits, rights.
2. **Task construct validity**: does the target measure a real animal behavior or welfare-relevant signal, rather than a study condition?
3. **Product validation**: can a cheap model operate on natural sensor data with useful recall, calibration, latency, and acceptable false alarms?

Passing (1) does **not** imply (2) or (3). A probability over bad labels does not become animal empathy.

## Decision

| Source/task | Data integrity outcome | Welfare/behavior relevance | Initial framework role |
|---|---|---|---|
| **Horse Grimace / ordinal regional facial cue (0/1/2)** | Exact image duplicates in train repaired with independently tested source-byte hashing; published horse-heldout split checks pass; 5 source train horses, 2 source test horses | **Direct welfare-proximal visual signal**: pain-associated HGS-style facial units, although each region only partially describes an animal's state | **KEEP as a *seed welfare-proximal perception probe***, not a validated welfare benchmark, deployable pain detector, or system-level product test |
| **CatMeows / elicitation context** | Owner-disjoint train view repaired and tested, all class supports retained | **Behavioral context only**, NOT independently labeled emotion, distress or wellness; class requires inferring the experiment's specific setup from a meow | **DEMOTE to an optional audio-domain transfer/representation diagnostic**. Excluded from every default welfare suite, scorecard, and product go/no-go. Explicit opt-in only |
| **All other legacy sources** | See [repair-or-retire decisions](SUITE_DECISIONS.md) | Not defensible with current task and partition definitions | **RETIRED** pending independently reviewed replacement tasks/dataset versions |

**Bottom line: zero product-grade animal-welfare decision datasets survived.** This is a successful forensic audit rather than a failed training run. One scientifically relevant **seed** remains to exercise typed ordinal decisions; one optional audio modality diagnostic remains for representation probing. Move on to fresh source prospecting.

## Horse Grimace — why keep one constrained seed

**Authoritative source:** Lencioni et al., 2021, [Pain assessment in horses using automatic facial expression recognition through deep learning-based modeling](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0258672), DOI `10.1371/journal.pone.0258672`. Original labeled images are in [Mendeley Data v3](https://data.mendeley.com/datasets/t8rtzcgwxm/3), DOI `10.17632/t8rtzcgwxm.3`, CC BY 4.0.

**What is independently confirmed by the paper:**
- Seven horses (roughly one year old; six Brazilian sport horses, one Mangalarga Marchador) underwent routine castration. Cameras observed them at a feeder across pre-/postoperative days.
- Images were extracted from video; frames with appropriate lateral facial views were **manually selected**. A **trained human observer** assigned HGS-based 0/1/2 categories after cropping ears, eye region, or mouth/nostrils. The three regions group the six original HGS facial action units.
- The paper's classification results used an image-level random train/test procedure, **not animal-heldout evaluation**. Our prepared animal-heldout split is an improvement over that particular procedure, but still only tests **two held-out horses**.
- The paper itself calls for **more trained annotators** to reduce individual rater bias, better-balanced/higher-quality images, and other behavioral signals to avoid mistaking non-pain states for pain.

**Our repaired dataset:** The pinned prepared source is `oliveirabruno01/openfarm-horse-grimace-region@ba5799d2c29886e238f8a069fb3501d1d91444f5`. Actual image bytes verified. Dedup yields balanced train **945→829**, natural train **3,931→2,896**. Test views unchanged (279 and 916), with no shared exact media or horse IDs. Rebuild source-identifiable data with `scripts/prepare_horse_unique.py`; original source remains untouched. CI: https://github.com/ob1-s/OpenFARM/actions/runs/38728727960.

**Limits:**
- One expert's score is a human *reference observation*, not calibrated ground truth of pain. Reliability and adjudicated disagreement were **not measured in this prepared benchmark**.
- Repeated video frames (even after exact dedup) are highly correlated. 829 train images do **not** imply 829 independent clinical episodes; there are **only five source train animals**. Near-duplicate/time autocorrelation beyond identical bytes are unresolved.
- Only a procedure-related pain scenario and deliberately selected clear facial angles; real cameras may capture no animal, partial/occluded regions, night, moving animals, different barns/breeds, etc. We lack negative windows, longitudinal per-animal baselines, actual clinical outcomes, and action/alert outcomes.
- The published HGS research has species/condition limitations, and individual facial features can be confounded by blinking, fear, coat color, and lighting. See [original HGS validation](https://pmc.ncbi.nlm.nih.gov/articles/PMC3960217/) and [later clinical limitations](https://pmc.ncbi.nlm.nih.gov/articles/PMC10215503/).

**Appropriate candidate question:** `Score(0,1,2)` for source-defined *visible regional facial cue intensity*, with explicit crop-quality/OOD gates **only after obtaining independent quality labels**. Do **not** interpret score 2 as a direct animal pain diagnosis, and do not derive a clinical `Noul(pain)` from a score threshold. A sensible System-1 probe tests frozen multimodal encoders vs cheap, source-specific classifiers and a larger generative model on **animal-heldout** data, reporting effective sample/group sizes. No frontier ranking or calibrated severity claim from two animals.

**Promotion requirement:** Acquire multi-annotator consensus or rater distributions from independent veterinarians, substantial animal/site-heldout observations, realistic camera-quality negatives, a longitudinal follow-up/outcome study, and a validated decision-to-review policy. Then measure false alerts per animal-day and missed consequential events under realistic cost, not just test-row accuracy.

## CatMeows — why demote

**Authoritative sources:** [CatMeows Zenodo original](https://zenodo.org/records/4008297) and [Automatic Classification of Cat Vocalizations Emitted in Different Contexts](https://pmc.ncbi.nlm.nih.gov/articles/PMC6719916/), plus [human interpretation study](https://pmc.ncbi.nlm.nih.gov/articles/PMC7765146/).

**What the labels actually mean:** 440 meows from 21 cats, recorded during **brushing by the owner**, **waiting for the normal feeding routine**, or **being left alone for up to five minutes in an unfamiliar room** after an acclimatization period. Cats were recorded using a **collar-mounted small Bluetooth headset microphone**, not a typical room/farm monitoring setup. These conditions were imposed as experimental stimuli; no independent label of fear, pain, vocal urgency or welfare severity is supplied. Ten Maine Coons belonged to the same owner; the remaining cats were European Shorthairs owned separately.

The prepared source has 8kHz, 16-bit mono files (Nyquist 4kHz) and 13.43 minutes total sound. The pinned HF source `oliveirabruno01/openfarm-catmeows@57fe220b22eae46eb555af9f4169bd4677584cea` had an owner overlap despite cat ID separation. The new **training-only owner filter**, independently verified against real dataset rows in [CI](https://github.com/ob1-s/OpenFARM/actions/runs/38727625359), leaves balanced train **126** and test **75**, and raw train **212** and test **118**, with **0 owner, cat, or exact audio overlaps** and all three target classes represented.

**Additional important external quality evidence:** A later feline-vocalization ML study ([Scientific Reports 2025](https://www.nature.com/articles/s41598-025-17986-z)) specifically excluded CatMeows because of white noise, poor quality and waveform interruptions in low-budget Bluetooth recordings. This independently raises the threshold for using it to evaluate bioacoustic representation quality. The repaired split is better, but the data was not magically rerecorded.

**What it can still honestly measure:** Whether frozen or lightly tuned audio/spectrogram representations transfer to tiny owner-heldout, experimentally defined *elicitation-context* labels. It may be handy as a short, intentionally OOD audio debugging exercise; it **cannot** support claims about general animal empathy or welfare monitoring. We explicitly exclude it from the default evaluation suite; run `openfarm-bioacustics` only with `task="catmeows_context", allow_auxiliary_context_task=True`, never in the product scorecard.

## New product-data prospecting criterion

Require every proposed dataset/task to specify **before training**:
1. A concrete observable event/behavior or independently evaluated welfare-relevant outcome, with annotation protocol and rater reliability.
2. The model-accessible *state* (camera/audio/pose/time window) and observation sufficiency, with missingness, no-event examples, and explicit abstention/uncertainty.
3. Animal- and **site/recording session**-heldout validation; study-identity and context-only baseline; exact- and near-duplicate checks.
4. Multiple typed, defensible questions from the *same state* only when source labels actually support each; a label's deterministic rewrite is not new supervision.
5. A low-cost pretrained/frozen encoder baseline, a strong generative zero-shot comparator, calibrated non-AR decision head, inference per animal-day, and realistic deployment frequency/false-alert constraints.
6. Explicit deployment utility: event recall, precision, **false alarms per animal per day**, abstention/review workload, within-animal change sensitivity, licensing/rights, and a plausible route to product deployment.

**Do not count Horse or Cat as a finished product anchor.** Their retained status is **seed** and **optional diagnostic**, respectively. The next scientific research work should be an independent source sweep for behavior/event annotations closer to real monitoring.
