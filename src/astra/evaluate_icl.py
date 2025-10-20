"""
Evaluate in-context learning performance on classification tasks.

For a given classification rule, samples n few-shot examples from the dataset
and evaluates how well a model can predict labels for test samples.

Usage:
    python src/astra/evaluate_icl.py --rule contains_commas
    python src/astra/evaluate_icl.py --rule sentence_count_multiple
"""

import argparse
import random

from datasets import load_dataset
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer, accuracy, stderr
from inspect_ai.solver import generate, system_message


DATASET_FILE = "artifacts/datasets/classification_dataset.jsonl"
N_SHOT = 10  # Number of examples per class (total = 2*n)
N_TEST = 100  # Number of test samples
LOG_DIR = "./artifacts/logs"
MAX_CONNECTIONS = 50


load_dotenv()
random.seed(42)


def load_classification_dataset(dataset_path):
    """Load the dataset from JSONL file using HuggingFace datasets."""
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return list(dataset)


def create_few_shot_prompt(rule, examples, test_text):
    """Create a prompt with few-shot examples for classification."""
    prompt_parts = [
        "Classify the provided text as true or false using the given examples as a guide; you'll have to figure out what classification rule is being used in the examples. Answer only with 'true' or 'false'."
    ]

    # Add few-shot examples
    for example in examples:
        text = example["text"]
        label = str(example[rule]).lower()
        prompt_parts.append(f"Text: {text}")
        prompt_parts.append(f"Answer: {label}")
        prompt_parts.append("")

    # Add the test case
    prompt_parts.append(f"Text: {test_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def create_evaluation_samples(dataset, rule, n_shot, n_test):
    """
    Create evaluation samples by randomly sampling few-shot examples and test cases.

    Args:
        dataset: Full dataset
        rule: Classification rule to evaluate
        n_shot: Number of few-shot examples per class (total examples = 2 * n_shot)
        n_test: Number of test samples to evaluate
    """
    # Split dataset by label
    true_examples = [ex for ex in dataset if ex[rule] is True]
    false_examples = [ex for ex in dataset if ex[rule] is False]

    print(f"Dataset split: {len(true_examples)} true, {len(false_examples)} false")

    samples = []

    for _ in range(n_test):
        # Sample n examples from each class
        true_samples = random.sample(true_examples, n_shot)
        false_samples = random.sample(false_examples, n_shot)

        # Interleave true and false examples
        few_shot_examples = []
        for i in range(n_shot):
            few_shot_examples.append(true_samples[i])
            few_shot_examples.append(false_samples[i])

        # Sample a test example (from either class)
        test_example = random.choice(dataset)

        # Create prompt
        prompt = create_few_shot_prompt(rule, few_shot_examples, test_example["text"])

        # Create sample with the ground truth label as target
        sample = Sample(
            input=prompt,
            target=str(test_example[rule]).lower(),
            metadata={
                "rule": rule,
                "n_shot": n_shot,
                "total_examples": 2 * n_shot,
                "true_label": test_example[rule],
            },
        )
        samples.append(sample)

    return samples


@scorer(metrics=[accuracy(), stderr()])
def exact_match():
    """Score based on exact match of true/false prediction."""

    async def score(state, target):
        # Get the model's prediction
        prediction = state.output.completion.strip().lower()

        # Extract true/false from the prediction (handle various formats)
        if "true" in prediction and "false" not in prediction:
            prediction = "true"
        elif "false" in prediction and "true" not in prediction:
            prediction = "false"
        else:
            # If unclear, take the first word
            prediction = prediction.split()[0] if prediction else ""

        # Compare with target
        correct = prediction == target.text.lower()

        return Score(
            value=correct,
            answer=prediction,
            explanation=f"Predicted: {prediction}, Expected: {target.text}",
        )

    return score


@task
def evaluate_icl_task(rule):
    """Evaluate in-context learning for a specific classification rule."""
    # Load dataset
    dataset = load_classification_dataset(DATASET_FILE)
    print(f"Loaded {len(dataset)} samples from dataset")

    # Create evaluation samples
    samples = create_evaluation_samples(dataset, rule, N_SHOT, N_TEST)
    print(
        f"Created {len(samples)} test samples with {2 * N_SHOT} total examples ({N_SHOT} per class)"
    )

    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            system_message(
                "You are a helpful assistant that classifies text. "
                "Answer only with 'true' or 'false' based on the examples provided."
            ),
            generate(),
        ],
        scorer=exact_match(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate in-context learning for classification"
    )
    parser.add_argument(
        "--rule",
        type=str,
        required=True,
        help="Classification rule to evaluate (e.g., 'contains_commas', 'sentence_count_multiple')",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/anthropic/claude-sonnet-4.5",
        help=f"Model to evaluate",
    )
    args = parser.parse_args()

    print(f"Evaluating in-context learning performance")
    print(f"Rule: {args.rule}")
    print(f"Model: {args.model}")
    print(f"Few-shot examples: {N_SHOT} per class (total: {2 * N_SHOT})")
    print(f"Test samples: {N_TEST}")
    print()

    # Run the evaluation
    logs = eval(
        evaluate_icl_task(args.rule),
        model=args.model,
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
    )

    log = logs[0]
    print(f"\nEvaluation complete!")
    print(f"Log saved to: {log.location}")
