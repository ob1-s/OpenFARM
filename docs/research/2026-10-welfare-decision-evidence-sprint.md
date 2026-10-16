# Welfare Decision Evidence Sprint — evidence, not another source catalogue

**16 October 2026.** Companion to [44-source research inventory](2026-10-animal-data-source-register.csv), [data approval policy](DATA_APPROVAL_POLICY.md), and the frozen v0.2 query compiler. The explicit purpose of this sprint is to **admit only scientifically defensible welfare questions** whose answers come from distinct measurements or events, and to reject implausible action claims.

## What the existing research work has already demonstrated

1. **Human clinical outcomes, cross-sectional:** The [Brazil cow welfare study](https://zenodo.org/records/3408818) supplies **13,716 original cow gait and body-condition ratings from 50 farms**. The source is CC BY 4.0. These are **two distinct human reference families**, not three: severe lameness is just a threshold rewrite of the same gait grade. The compiler creates two questions per common cow/farm-management state, with farm-heldout source IDs, but the explanatory variables are **limited farm-management/demographic context**, not visual gait or measured weight. Success would demonstrate risk association, **not a capacity to observe an injured cow**.
2. **A stronger 16-farm observational risk assessment** uses **5,529 individual source cow records** with independently recorded breed, lactation and days in milk. Full leave-one-farm-out (LOFO) research baselines, [actual CI](https://github.com/ob1-s/OpenFARM/actions/runs/38868420045), beat a training-farm-only prevalence prior in **16/16 farms for clinical lameness**, **15/16 for severe lameness**, **12/16 for low body condition**. Adding milk production does **not consistently help** and could be a contemporaneous consequence of illness. This remains one study, retrospective; it has **no prospectively verified intervention need**.
3. **Observed care-related events:** Swedish pig feeder/historical [public CC BY-NC data](https://zenodo.org/records/5028016) were reconstructed by real original source pen-cohorts. **21 Swedish treated cohorts + 21 Swedish controls**, with three treated cohorts lacking ANY usable pre-treatment history. A candidate future-treatment target is an observed human care event (first tail-damage treatment), but selection is case/control and control pseudo-dates are artificial. It is **not** an unbiased population-rate or caretaker-escalation benchmark; only bounded historical case-control research can be considered. The 27 real treatment events in the combined study must not be misrepresented as independent Swedish prospective episodes.
4. **Affect-associated physiology:** Public [PigWeb 2026](https://github.com/AnjaEggert/PigWeb_cross-institutes) has independent cortisol and oxytocin assays with real pre/post timestamps from **124 pigs in four institutes**, two contexts per pig. Two distinct assay-reference questions, but **neither measures ground-truth pleasure, pain, suffering, or a need for a caretaker**. Four site-heldout research baselines exist. The physiological targets may help *welfare indicator validity*; an experiment condition is not an emotional label.
5. **Strong low-cost technical baseline, not welfare:** HotPig's 24-animal feeding sensor proxy and simple heads are valid perception / sensor-agreement engineering work, but not evidence that an animal needs attention.
6. **Rich time-local facial action, tiny sample:** EquiFACS structured features paired with codeable independent veterinary pain ratings provide a **54-interval / 18-media** exploratory pilot. Rater conflict/non-codeable flags and 5 heldout clinical media prohibit product/clinical inference.

## Independent supervised welfare question families

| Original reference | State you can realistically access | Honest decision questions | Strength | Main missing evidence |
|---|---|---|---|---|
| Human gait grade (cow) | Breed/lactation/DIM or same-visit farm management | Noul(source clinician gait >=3), Score(gait rubric). Derived severe threshold **shares source reference** | **Research retrospective welfare risk association**, 16-farm LOFO; not visual lameness detector | Same-time bias, no sensory direct observation, external prospective sites |
| Human body-condition grade (cow) | Same management/animal state | Noul(source BCS <3), Score(source original BCS) | **Second independent reference family**; weaker 12/16 farm Brier improvements | No body image/weight baseline, farm/site shift, action threshold |
| Human first tail-damage treatment (pig pen) | Only strictly pre-event feeder measurements | Noul(observed first treatment within next 7 days) | Real *recorded care event*, selected case-control **pilot** | True source prevalence, control censoring, prospective farms |
| Independent laboratory hormone assays (pig) | Precisely earlier PRE assay state | Noul(cortisol increases after exposure), Noul(oxytocin increases) | Two independent biochemical targets, cross-site benchmark candidate | Assay variability, construct validity of affect, heterogeneous institute response |
| Veterinary clinical pain rating (horse) | Earlier coder's AU event timelines with source media group | Noul(independent pain assessor codeable rating >0) | Intrinsically welfare-relevant, but only 5 pilot test media | Too few clinical groups, source AU rater not automatic, clinical validation |

**Crucial:** The same state can have many syntactic questions, but only real, **independent** label families count as new biological evidence. Type transformations (binary threshold of source ordinal) are still useful training/evaluation tasks, not independent discoveries.

## New acceptance gate introduced in this sprint

- [`scripts/gate_welfare_cow_lofo.py`](../../scripts/gate_welfare_cow_lofo.py) checks existing 16-whole-farm heldout results **without retraining or test-set selection**. To pass a *robust incremental information* diagnostic, a feature version must improve Brier over a train-only prevalence prior in **at least 12/16 heldout farms**, with **positive median per-farm and cow-weighted gain**.
- The CI workflow reruns [the independent full-farm benchmark](../../scripts/stress_test_cow_welfare_leave_one_farm_out.py) and publishes the result and a separate admission verdict. This doesn't turn the data into medical diagnoses or automatic intervention requirements.
- The human **gait and body condition** references are the only two original clinical families. This gate **never** promotes a threshold derivative to independent expert evidence.

## What actually earns new approval

**First:** independent welfare reference + a valid observation state with confirmed ex-ante availability + adequately independent animal/site holdouts + rights + trivial/source-only model controls.

**Then:** calibrated typed Noul/Choice/Score, proper probabilities and reference disagreement + reasonable prospective review utility. For action-taking, there must also be an independent caretaker outcome/decision policy under **natural sampling**, not balanced snippets.

**Explicit do-not-promote cases:** welfare state inferred solely from experimental condition; pain invented from posture; cow “lameness” predicted by directly exposing its source gait grade; tail-treatment controls whose fake dates are reinterpreted as future events; claiming dozens of independent cow records when they all share a farm; multiplying the independent evidence count by thresholding one ordinal score.

### Sprint disposition

- **Keep** cow gait and body-condition questions as *research observational welfare interpretation/association* data—more substantively welfare-related than the HotPig feeder agreement task.
- **Keep under bounded historical evaluation** Swedish first-treatment and PigWeb assays, with case-control and biological-construct warnings respectively. They are **not** independently validated caretaker-action or emotional labels.
- **Reject** all claims that this sprint has established a generalizable Level-3 escalation benchmark. That remains an unresolved scientific dependency.
- **Next falsification tests:** source-stratified context-only shortcuts; temporal feature availability checks; two independent assessor groups; per-farm gains rather than pooled minute/cow figures; reliability under site shift. Do not allocate frontier training merely to fill a scoreboard.

**Why this is useful:** it keeps the real scientific gold (human assessment, physiological assays, actual treatments) separate from convenient label multiplication. OpenFARM can now choose candidate questions for *animal welfare intelligence*, not just instrumentation accuracy.
