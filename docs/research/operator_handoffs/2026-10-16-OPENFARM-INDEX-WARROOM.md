# OpenFARM Decision Index — Parallel Research War Room
**16 October 2026, São Paulo time.** User is allocating 2–3 local coding/research agents over the next few days. Purpose: actual valuable train/eval tasks, **especially** independently grounded Level-3 (human-attention/care actions), without waiting for one visual dataset pipeline.

## Starting point
- Repo https://github.com/ob1-s/OpenFARM; checkpoint `7ad46922030dac34fab223753785ff2a4487b970`, draft PR #2 https://github.com/ob1-s/OpenFARM/pull/2, source research branch `research/2026-10-data-prospecting`.
- Frozen v0.1: HotPig instrument agreement (L1), human-labeled cattle sensor-to-behavior (L1), Meerkat manual ethogram geometry (L1), Horse Grimace pilot (L2), EquiFACS pilot (L2).
- v0.2 compiler: 130,005 source states, 623,174 valid typed question labels, 546 missing-evidence masks, **average only ~1.002 independently measured reference families per state**; PigWeb temporal leakage corrected; **zero confirmed causally available multichannel input states**.
- Latest welfare sprint: 5,529 cows/16 farms from overlapping 13,716-cow/50-farm study support **retrospective** metadata-to-human gait/BCS association only. No directly visual animal welfare perception and **zero approved Level-3 action policies**.
- Swedish pig first-treatment prospective pilot had only 3 heldout positive pen cohorts; pooled Swiss/Swedish source inconsistent 28 vs 27 outbreak cohorts. Quarantined. Avoid reinventing that source as 'approved Level 3'.
- Most original datasets retired after duplication/construct validity audits; no false happiness/distress from experimental conditions.

## Parallel lanes (OWNED paths; no shared-file collision)
| Who | Branch | Handoff | Main required output |
|---|---|---|---|
| **A** | `research/index-l3-care-events-2026-10` | [A care events](2026-10-A-L3-CARE-EVENTS.md) | Real source event/action histories, valid time-causal Level-3 or clinically adverse outcome feasibility; leakage audits |
| **B** | `research/index-l2-direct-welfare-2026-10` | [B direct perception](2026-10-B-L2-DIRECT-PERCEPTION.md) | Open real media + independently rated pain/lesion/gait source, clean animal-grouped Score evaluation |
| **C** | `research/index-behavior-to-care-2026-10` | [C behavior to care](2026-10-C-BEHAVIOR-TO-CARE.md) | True labeled farm video/audio harmful/relevant event episodes and observation metrics; linkage candidate |
| **ChatGPT scouting/review** | In conversation, no auto-background | Review source reliability and repo PRs after user supplies them; separate new lead reconnaissance | Source red-team and Index admission report |

### Operator start instructions
1. Open handoff for assigned lane and check repo branch `git status` / base. Use separate `git worktree` per branch if sharing local checkout; never switch same working tree under concurrently running agents.
2. Branches are pre-created from exactly one common base HEAD. Do not merge/rebase sibling branches, or cherry-pick unreviewed work. All local agent code/data/temp outputs isolated.
3. All data rights, source access and train/test rules in each handoff apply. Use exhaustive non-generative null baselines before expensive model inference.
4. When reporting completion send: candidate/source; independent labels and real input; original bytes reviewed? access/rights? group split and events; metrics with denominators and null; branch/PR/CI; PASS/FAIL/DEFER.
5. PRs target `research/2026-10-data-prospecting`, kept draft for scientific review. Don't merge/change `main`; don't mutate v0.1 or v0.2 during this sprint.

### Decision Index proposed index-card schema
`task_id | source_record_version_sha | specimen_species | unit(animal/pen/session) | modality | prediction_time | model_visible_observations | question | typed_output | original_reference | reference_family | observation_level(L1/L2/L3) | human_action_independent? | action_time | true_negatives/censoring | #sites/#animals/#episodes/#events | fixed_split | rights_research/rights_commercial | null+calibration | event_alert_budget | admission_verdict | failure_reason`.

### Acceptance and stop
- **Best-case**: one public-source Level-3 care/action dataset with genuine time-stamped observation/action and natural negatives plus a bounded L2 visual welfare task and one L1 event-task observation chain.
- **Acceptable honest result**: no Level-3 target passes (all negatives rigorously proven), **plus** an admitted real direct L2 perception seed and a feasible instrumentation path to collect needed L3 annotations.
- **Reject**: hundreds of paper mentions with no accessible files, induced pain group relabeled as 'pain' without independent clinical reference, alert threshold chosen from post-event features, pseudo timestamps, label leak or frame-level random splits.
- Review and promotion happen **after** each operator's PR. No automatic joint benchmark inflation.

## Promising external source finding to route correctly
- Heseker et al. (2024) pig screams + staff removal of biters, https://www.nature.com/articles/s41598-024-55336-7 : source *paper* contains true action descriptions, **the paper explicitly says data are available only on request and not publicly available due to privacy concerns** (Data Availability). Do not spend time attempting private recordings; preserve as example of a real staff-action study, but use public sources first.
- 2026 broiler footpad dermatitis / hock-burn study, https://doi.org/10.1016/j.psj.2026.106968 : 500 independent human multi-rater image assessments across two slaughterhouses. Its https://zenodo.org/records/19885754 release appears to contain PDF/HTML only, **not the study's images**. B should not claim usable benchmark until original data found.
- CowScreeningDB https://github.com/Shahid-Ismail/CowScreeningDB-A-public-database-for-lameness-detection has hours of real cow motion with vet lameness assessment, **but requires signed license/email/password to access source dataset**, not a public-first default.

## What NOT to optimize now
- Jev/Liquid/new model comparison on metadata cow task (frozen).
- Another model wrapper/one-vs-rest expansion, spontaneous v0.1 suite promotion, full 30+ GB archive download without evidence.
- Commercial claims before rights+field validation.

### Review rubric (0–2 each)
1. Ground truth is genuinely welfare/action-relevant and independent of features.
2. True predictor state exists and is causally available at prediction time.
3. Original accessible media/bytes have verified joins, IDs, timestamps and legality.
4. Generalization evaluation has independent animal/pen/site/video groups.
5. Realistic negatives, masks and calibration/alert cost can be evaluated.
Score does NOT alone grant approval; any fatal temporal, rights or label failure is veto.
