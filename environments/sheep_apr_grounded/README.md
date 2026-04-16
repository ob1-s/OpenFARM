# sheep-apr-grounded

### Overview
- **Environment ID**: `sheep-apr-grounded`
- **Short description**: Sheep Pain Recognition Benchmark.
- **Tags**: animal behavior, animal welfare, livestock, automatic pain recognition, vision, multimodal, eval

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
| use_judge | bool | True | Whether to run grounded judge analysis |
| judge_model | str | "gpt-4o-mini" | The judge model |
| judge_base_url | str | "https://api.openai.com/v1" | The base url for the judge OpenAI client |
| judge_api_key_var | str | "OPENAI_API_KEY" | The environment variable that contains the judge's API key  |
| format_reward_weight | float | 0.0 | The reward to give for correct formatting. |
| reasoning_reward_weight | float | 1.0 | The reward to give for accurate recall and reasoning |

### Metrics

| Metric | Meaning |
| ------ | ------- |
| `reward` | Main scalar reward (weighted sum of criteria) |
| `accuracy_reward` | Exact match on target answer |
| `format_reward` | Reward for correct formatting |
| `reasoning_reward` | Reward for correct reasoning and recall |