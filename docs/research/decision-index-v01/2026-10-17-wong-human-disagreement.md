# Wong 340-case three-veterinarian reference forensics

**Source original user-supplied Excel, SHA256** `4d4498fff6108f883fd209f4c996599980453a181ba5c39b27ce3d07ec044027` (9 columns, 340 cases). These statistics were recomputed directly from the pinned original XLSX via Python ZIP/XML extraction and an independent 102-case secret-backed GitHub runner. No private clinical case text, names or individual case-level predictions were committed.

## Original full-source results (not just 102 test cases)

| Rater | Immediate (out of 340) | Mean ordinal severity index (0 most urgent, 4 least urgent) |
|---|---:|---:|
| Veterinarian 1 | 23 (6.8%) | 2.568 |
| Veterinarian 2 | 26 (7.6%) | 2.253 |
| Veterinarian 3 | **71 (20.9%)** | **1.741** |

- 1,020 actual expert judgments on 340 clinical cases (do NOT call 1,020 independent patients).
- **58/340 (17.1%)** three-vet unanimous, **77/340 (22.6%) three different labels**.
- Pairwise exact agreement: vet 1–2 = **47.1%**, vet 1–3 = **26.8%**, vet 2–3 = **37.6%**. Thus median agreement is not the same as full human consensus.
- Any one vet Immediate: **76/340**. Median/majority Immediate: **30/340**. Vet 3 chooses Immediate much more often, even for the same presentations.
- Mean pairwise ordinal difference among raters on same case = **0.8255** ordinal bins.
- Averaged across the original clinical reference distribution, **0.41895** is the estimated empirical label-variance term `1 - sum(q_i^2)`, where `q` is the three-rater category distribution. This is the additive floor when replacing three-vote Brier `||p-q||²` with expected one-hot Brier against one randomly sampled of the three votes. It is **not** a universal irreducible error bound and is not a model score.

## Repeated test split 102 cases (separately recomputed)

- **17 unanimous cases**, **27 three-different cases**, **85 disagreement cases**.
- Vet 1 Immediate **6/102**, vet 2 **8/102**, vet 3 **26/102**; majority Immediate **9/102**, at least one vet Immediate **26/102**, unanimous Immediate **5/102**.
- Pairwise exact: vet1–2 = 50.0%, vet1–3 = 25.5%, vet2–3 = 31.4%.
- Empirical three-rater variance term = **0.42919** in expected single-vote Brier decomposition.
- Asymmetric rater thresholds are a plausible interpretation but cannot be distinguished from varying clinical expertise, information interpretation, bias or specialty with this evidence alone.
- No animal/patient ID: even the 340 original clinical cases may include repeat presentation records. Clinic/site generalization cannot be inferred.

## Scientific consequences and protocol modifications

1. Evaluate **each veterinarian as a separate reference**, preserve original vote vectors and report the signed ordinal bias of each model relative to each vet and their disagreement, before summarizing the median.
2. **Agreement versus outcome correctness:** individual vet opinions are *references*, not proven optimal clinical urgency. A model matching an aggressive rater's threshold may be penalized by median accuracy despite making clinically defensible decisions.
3. **Matched probability evaluation:** empirical three-vote Brier `sum_i(p_i-q_i)^2` and per-vet Brier `sum_i(p_i-1[vet=i])^2` are different metrics; for a random vet from these votes, expected latter equals former plus average `1-sum_i q_i^2`. This identity is exact for empirical three-vote frequencies; use it to interpret proper scores fairly.
4. Risk-sensitive outcomes should report missed **majority Immediate**, missed **any-vet Immediate** and additional escalation relative to each rater, with all denominators. Evaluating all unanimous immediate cases isn't sufficient to establish real-world care safety.
5. An optional research comparison is *hold-one-veterinarian-out*, asking how accurately each human agrees with the other two, but a disagreement between those two cannot be treated as consensus; report ties and missing decisions explicitly. No single “human ceiling” from this sample.
6. For training, preserving all three votes and optionally latent rater effects is scientifically stronger than collapsing the label to one unquestionable hard target.

**Reproducible tool:** `scripts/index_v01_wong_rater_audit.py` validates original source fingerprint and supports original XLSX input or private heldout test input through `WONG_TEST_CASES_B64`. No patient data are output in reports.

**Research only:** does not show any rater is objectively correct and must not be used to decide actual veterinary treatment.
