# sheep-facial-expression

### Overview
- **Environment ID**: `sheep-facial-expression`
- **Short description**: Sheep Facial Expression Benchmark - Binary pain assessment.
- **Tags**: animal-behavior, animal-welfare, sheep, SFPES, livestock, vision, multimodal, eval

### Datasets
- **Primary dataset(s)**: oliveirabruno01/sheep-facial-expression-benchmark
- **Source links**: https://huggingface.co/datasets/oliveirabruno01/sheep-facial-expression-benchmark

### Task
- **Type**: single-turn

### Environment Arguments

| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| dataset_id | str | "oliveirabruno01/sheep-facial-expression-benchmark" | The ID of the dataset to use. |
| seed | int | 42 | The seed to use for dataset shuffling. |
| format_reward_weight | float | 0.0 | The reward to give for correct formatting. |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy_reward` | Exact match on target answer |
| `format_reward` | Reward for correct formatting |