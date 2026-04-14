import io
import base64
from PIL import Image
import verifiers as vf
from datasets import load_dataset


def encode_image_to_base64(image: Image.Image) -> str:
    """Helper to convert PIL images to base64 data URIs for multimodal LLMs."""
    buffered = io.BytesIO()
    # Convert to RGB if necessary (to handle PNG/RGBA)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"


def load_environment(
    dataset_id: str = "oliveirabruno01/sheep-facial-expression-benchmark",
    seed: int = 42,
    format_reward_weight: float = 0.0,
    **kwargs,
) -> vf.Environment:
    """
    Multimodal environment for sheep pain assessment.
    Labels: 0 = No Pain, 1 = Pain.

    Args:
        dataset_id (str): The ID of the dataset to use.
        seed (int): The seed to use for dataset shuffling.
        format_reward_weight (float): The reward to give for correct formatting.
        **kwargs: Additional keyword arguments to pass to the environment.
    """

    def build_dataset():
        ds = load_dataset(dataset_id, split="train")

        def transform(example):
            # 1. Convert PIL image to base64
            base64_image = encode_image_to_base64(example["image"])

            # 2. Create the multimodal prompt structure
            # verifiers handles message lists with 'type': 'image_url' for multimodal models
            instructions = (
                "You are a veterinary clinical evaluator. Assess this sheep for pain "
                "using the Sheep Pain Facial Expression Scale (SPFES).\n\n"
                "Provide your reasoning, then conclude with the score (0 or 1).\n\n"
                "Respond using this format:\n"
                "<reasoning>\n[your analysis of facial features]\n</reasoning>\n"
                "<score>\n[0 or 1]\n</score>"
            )
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instructions},
                        {"type": "image_url", "image_url": {"url": base64_image}},
                    ],
                }
            ]

            return {"prompt": prompt, "answer": str(example["label"])}

        return ds.map(transform, remove_columns=ds.column_names).shuffle(seed=seed)

    # Define the Parser to extract values from the XML tags
    parser = vf.XMLParser(fields=["reasoning", "score"], answer_field="score")

    # Reward function: Check if the parsed score matches the ground truth
    async def accuracy_reward(completion, answer, parser) -> float:
        parsed_answer = parser.parse_answer(completion)
        if parsed_answer is None:
            return 0.0

        # Clean string comparison (handles potential whitespace)
        return 1.0 if str(parsed_answer).strip() == str(answer).strip() else 0.0

    # Assemble the Rubric
    rubric = vf.Rubric(funcs=[accuracy_reward], parser=parser)

    # Add a format reward to encourage the model to follow the XML schema
    format_reward = parser.get_format_reward_func()
    format_reward.__name__ = "format_reward"
    rubric.add_reward_func(format_reward, weight=format_reward_weight)

    return vf.SingleTurnEnv(
        dataset=build_dataset, rubric=rubric, parser=parser, **kwargs
    )
