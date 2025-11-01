"""
Evaluate in-context learning performance on classification tasks.

For given classification rules, samples n few-shot examples from the dataset
and evaluates how well models can predict labels for test samples.

Usage:
    python src/astra/evaluate_icl.py --rules contains_commas --models openrouter/anthropic/claude-sonnet-4.5 --n-shot 10
    python src/astra/evaluate_icl.py --rules contains_commas contains_and --models openai/gpt-4 --n-shot 5 10
    python src/astra/evaluate_icl.py --rules all --models model1 model2 --n-shot 5 10 20
"""

import argparse
import random

from datasets import load_dataset
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, scorer, accuracy, stderr
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig


DATASET_DIR = "artifacts/datasets/isolated"
N_TEST = 100  # Number of test samples
LOG_DIR = "./artifacts/logs"
MAX_CONNECTIONS = 500


load_dotenv()
random.seed(42)


def load_classification_dataset(rule):
    """Load the isolated dataset for a specific rule from JSONL file."""
    dataset_path = f"{DATASET_DIR}/{rule}_dataset.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return list(dataset)


def create_few_shot_prompt(rule, examples, test_text):
    """Create a prompt with few-shot examples for classification."""
    prompt_parts = [
        "Classify the provided text as true or false using the given examples as a guide; you'll have to figure out what classification rule is being used in the examples. Answer only with 'true' or 'false'.",
        "",
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

        # Combine and randomize the order
        few_shot_examples = true_samples + false_samples
        random.shuffle(few_shot_examples)

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
def evaluate_icl_task(rule, n_shot):
    """Evaluate in-context learning for a specific classification rule."""
    # Load isolated dataset for this rule
    dataset = load_classification_dataset(rule)
    print(f"Loaded {len(dataset)} samples from {rule}_dataset.jsonl")

    # Create evaluation samples
    samples = create_evaluation_samples(dataset, rule, n_shot, N_TEST)
    print(
        f"Created {len(samples)} test samples with {2 * n_shot} total examples ({n_shot} per class)"
    )

    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            system_message(
                "You are a helpful assistant that classifies text. "
                "Answer only with 'true' or 'false' based on the examples provided."
            ),
            generate(
                config=GenerateConfig(extra_body={"reasoning": {"enabled": False}})
            ),
        ],
        scorer=exact_match(),
        name=f"icl_{rule}_n{n_shot}",
        metadata={"rule": rule, "n_shot": n_shot},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate in-context learning for classification"
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        required=True,
        help="Classification rules to evaluate (space-separated)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to evaluate (space-separated)",
    )
    parser.add_argument(
        "--n-shot",
        nargs="+",
        type=int,
        required=True,
        help="Number of few-shot examples per class (space-separated)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=5,
        help="Maximum number of tasks to run in parallel (default: 5)",
    )
    args = parser.parse_args()

    print(f"Evaluating in-context learning performance")
    print(f"Rules: {args.rules}")
    print(f"Models: {args.models}")
    print(f"N-shot values: {args.n_shot}")
    print(f"Test samples: {N_TEST}")
    print(f"Total evaluations: {len(args.rules) * len(args.models) * len(args.n_shot)}")
    print()

    # Create task configurations for all parameter combinations
    tasks = []
    for rule in args.rules:
        for n_shot in args.n_shot:
            tasks.append(evaluate_icl_task(rule, n_shot))

    # Run evaluations across all models and tasks
    logs = eval(
        tasks,
        model=args.models,
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
        max_tasks=args.max_tasks,
    )

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")

    for log in logs:
        task_name = log.eval.task
        model = log.eval.model

        if log.results and log.results.scores:
            # Scores are EvalScore objects with metrics inside
            for score in log.results.scores:
                if hasattr(score, 'metrics') and 'accuracy' in score.metrics:
                    acc_metric = score.metrics['accuracy']
                    print(f"{model:40s} | {task_name:30s} | {acc_metric.value:.2%}")
                    break

    print(f"\n{'='*80}")
    print(f"All logs saved to: {LOG_DIR}")
