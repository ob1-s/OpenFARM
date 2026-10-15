# openfarm-horse-grimace

> **WELFARE-PROXIMAL SEED ONLY (final product-fit audit, 2026-10-15):** The ordinal `region_score` (0/1/2) is a reproducible limited perception probe, **not a validated welfare benchmark or monitoring system**. Its source contains just seven horses and single-observer HGS-style region labels from manually chosen video frames. The original `binary_pain` task remains retired. For any training experiment, **deduplicate** using `scripts/prepare_horse_unique.py`. See [product-fit verdict](../../docs/system1/FINAL_PRODUCT_FIT_AUDIT.md).

See [repair-or-retire decisions](../../docs/system1/SUITE_DECISIONS.md).

<p>
  <a href="https://github.com/ob1-s/happy-farm/tree/main/environments/openfarm_horse_grimace">
    <img align="left" hspace="4" src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
  </a>
  <a href="https://app.primeintellect.ai/dashboard/environments/openfarm/openfarm-horse-grimace">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Prime%20Intellect-Envs%20Hub-181717?style=for-the-badge&labelColor=181717&logoColor=white" alt="Prime Intellect Environments Hub">
  </a>
  <a href="https://huggingface.co/datasets/oliveirabruno01/openfarm-horse-grimace-region">
    <img align="left" hspace="4" src="https://img.shields.io/badge/Hugging%20Face-Dataset-181717?style=for-the-badge&logo=huggingface&logoColor=yellow&labelColor=181717" alt="Hugging Face Dataset">
  </a>
</p>
<br clear="all" />

### Overview
- **Environment ID**: `openfarm-horse-grimace`
- **Short description**: Horse Grimace Scale-style facial-region scoring from equine face crops.
- **Tags**: animal-welfare, horse, equine, facial-expression, grimace-scale, pain-assessment, vision, eval

### Datasets
- **Primary dataset**: [`oliveirabruno01/openfarm-horse-grimace-region`](https://huggingface.co/datasets/oliveirabruno01/openfarm-horse-grimace-region)
- **Source**: Mendeley Data, "Automatic Pain Assessment in Horses" ([`10.17632/t8rtzcgwxm.3`](https://doi.org/10.17632/t8rtzcgwxm.3)), CC-BY-4.0.
- **Related paper**: "Pain assessment in horses using automatic facial expression recognition through deep learning-based modeling" ([`10.1371/journal.pone.0258672`](https://doi.org/10.1371/journal.pone.0258672)).
- **Default split**: `test`

### Task
- **Type**: single-turn multimodal image classification
- **Default task**: `region_score`
- **Supported tasks**:
  - `region_score`: answer `0`, `1`, or `2` for an HGS-style facial-region cue.
  - `binary_pain`: answer `0` or `1`, derived from source scores (`0` = no cue, `1/2` = cue present).
- **Output format**: XML tags. Depending on `require_explanation`, models must output either `<answer>...</answer>` or both `<explanation>...</explanation>` and `<answer>...</answer>`.
- **Rubric**: exact-match answer reward, optional XML format reward, and optional grounded reasoning judge against an equine grimace-scale cheatsheet.

### Quickstart

```bash
prime eval run openfarm-horse-grimace
```

Configure task and split:

```bash
prime eval run openfarm-horse-grimace \
  -m google/gemini-3-flash-preview \
  -n 20 -r 3 \
  -a '{"task": "region_score", "test_split": "test", "require_explanation": true}'
```

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `dataset_id` | str | `"oliveirabruno01/openfarm-horse-grimace-region"` | Hugging Face dataset ID. |
| `dataset_revision` | str/null | `None` | Optional dataset revision. |
| `test_split` | str | `"test"` | Split to evaluate. |
| `task` | str | `"region_score"` | One of `region_score` or `binary_pain`. |
| `max_examples` | int | `-1` | Limit examples after shuffling. |
| `max_examples_per_task` | int/null | `None` | Compatibility alias for `max_examples`. |
| `seed` | int | `42` | Dataset shuffle seed. |
| `include_region_context` | bool | `true` | Include the source crop region in the prompt. |
| `require_explanation` | bool | `true` | Require an explanation before the answer. |
| `require_reasoning` | bool/null | `None` | Compatibility alias for `require_explanation`. |
| `accuracy_reward_weight` | float | `1.0` | Exact-match reward weight. |
| `judge_reward_weight` | float | `1.0` | Optional grounded-judge reward weight. |
| `format_reward_weight` | float | `0.0` | XML format reward weight. |
| `reward_weights` | dict/null | `None` | Optional aliases: `accuracy`, `judge`, `format` or full reward names. |
| `judge_mode` | str | `"none"` | `none`, `self`, or `external`. |
| `judge_model` | str | `"gpt-4o-mini"` | Model used when `judge_mode="external"`. |
| `system_prompt_override` | str/null | `None` | Replace the default system prompt. |
| `user_prompt_override` | str/null | `None` | Replace the default user prompt template. |
| `cheatsheet_version` | str | `"full"` | `full` or `short` for judge grounding. |
| `cheatsheet_override` | str/null | `None` | Replace the default judge cheatsheet. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Weighted scalar reward. |
| `accuracy_reward` | Exact match after answer normalization. |
| `format_reward` | XML format reward when enabled. |
| `judge_reward` | Optional biological-reasoning score against the equine cheatsheet. |
