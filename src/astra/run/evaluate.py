"""
Evaluate in-context learning performance on classification tasks.

For given classification rules, samples n few-shot examples from the dataset
and evaluates how well models can predict labels for test samples.

Optionally includes an "articulation" step where models explain/identify the rule
they used for classification (multiple-choice or free-form).

Usage (CLI arguments):
    # Standard evaluation (no articulation)
    python -m astra.run.evaluate --rules all --models openai/gpt-4 --n-shot 50

    # Multiple choice articulation
    python -m astra.run.evaluate --rules is_first_person --models openai/gpt-4 --n-shot 5 --articulation multi

    # Free-form articulation
    python -m astra.run.evaluate --rules has_many_verbs --models openai/gpt-4 --n-shot 10 --articulation free

    # Multiple rules and shot counts
    python -m astra.run.evaluate --rules contains_numbers is_title_case --models openai/gpt-4 --n-shot 5 10 --n-test 200

Usage (YAML config):
    # Create config.yaml:
    rules: [all]
    models: [openai/gpt-4]
    n_shot: [50]
    articulation: multi

    # Run with config:
    python -m astra.run.evaluate --config config.yaml

    # Override config values:
    python -m astra.run.evaluate --config config.yaml --seed 123
"""

import random
from typing import Literal

from confetti import BaseConfig
from datasets import load_dataset
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, system_message
from inspect_ai.model import GenerateConfig
from pydantic import Field

from astra.evaluate import (
    articulate_rule,
    generate_mc_options,
    classification_accuracy,
    articulation_choice_accuracy,
    articulation_quality,
)
from astra.generate import RULES


DATASET_DIR = "artifacts/datasets/isolated"
LOG_DIR = "./artifacts/logs"
MAX_CONNECTIONS = 500


load_dotenv()


def load_classification_dataset(rule):
    """Load the isolated dataset for a specific rule from JSONL file."""
    dataset_path = f"{DATASET_DIR}/{rule}_dataset.jsonl"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return list(dataset)


def split_dataset(dataset, rule, n_test):
    """
    Split dataset into train and test sets, stratified by label.

    Args:
        dataset: Full dataset
        rule: Classification rule (used to access labels)
        n_test: Number of samples to use for test set

    Returns:
        (train_dataset, test_dataset) tuple
    """
    # Split by label to ensure stratified split
    true_examples = [ex for ex in dataset if ex[rule] is True]
    false_examples = [ex for ex in dataset if ex[rule] is False]

    # Shuffle each class
    random.shuffle(true_examples)
    random.shuffle(false_examples)

    # Split test set evenly from each class
    n_test_per_class = n_test // 2

    train_dataset = true_examples[n_test_per_class:] + false_examples[n_test_per_class:]
    test_dataset = true_examples[:n_test_per_class] + false_examples[:n_test_per_class]

    # Shuffle train and test sets
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    return train_dataset, test_dataset


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


def create_evaluation_samples(train_dataset, test_dataset, rule, n_shot, articulation_type="none"):
    """
    Create evaluation samples by sampling few-shot examples from train and test cases from test.

    Args:
        train_dataset: Training dataset (for few-shot examples)
        test_dataset: Test dataset (for evaluation)
        rule: Classification rule to evaluate
        n_shot: Number of few-shot examples per class (total examples = 2 * n_shot)
        articulation_type: "none", "multi", or "free"
    """
    # Split train dataset by label for sampling few-shot examples
    train_true = [ex for ex in train_dataset if ex[rule] is True]
    train_false = [ex for ex in train_dataset if ex[rule] is False]

    print(f"Train dataset: {len(train_true)} true, {len(train_false)} false")
    print(f"Test dataset: {len(test_dataset)} samples")

    # Generate multiple choice options once for all samples in this task
    mc_options = None
    mc_correct_answer = None
    if articulation_type == "multi":
        mc_options, mc_correct_answer = generate_mc_options(rule)

    samples = []

    # Use each test example once
    for test_example in test_dataset:
        # Sample n examples from each class in the training set
        true_samples = random.sample(train_true, n_shot)
        false_samples = random.sample(train_false, n_shot)

        # Combine and randomize the order
        few_shot_examples = true_samples + false_samples
        random.shuffle(few_shot_examples)

        # Create prompt
        prompt = create_few_shot_prompt(rule, few_shot_examples, test_example["text"])

        # Build metadata
        metadata = {
            "rule": rule,
            "n_shot": n_shot,
            "total_examples": 2 * n_shot,
            "true_label": test_example[rule],
            "rule_description": RULES[rule].description,  # For articulation grading
        }

        # Add multiple choice metadata if applicable
        if articulation_type == "multi":
            metadata["mc_options"] = mc_options
            metadata["mc_correct_answer"] = mc_correct_answer

        # Create sample with the ground truth label as target
        sample = Sample(
            input=prompt,
            target=str(test_example[rule]).lower(),
            metadata=metadata,
        )
        samples.append(sample)

    return samples


@task
def evaluate_icl_task(rule, n_shot, n_test, articulation_type="none"):
    """Evaluate in-context learning for a specific classification rule with optional articulation."""
    # Load isolated dataset for this rule
    dataset = load_classification_dataset(rule)
    print(f"Loaded {len(dataset)} samples from {rule}_dataset.jsonl")

    # Split into train and test sets
    train_dataset, test_dataset = split_dataset(dataset, rule, n_test)
    print(f"Split into {len(train_dataset)} train and {len(test_dataset)} test samples")

    # Create evaluation samples with articulation metadata
    samples = create_evaluation_samples(train_dataset, test_dataset, rule, n_shot, articulation_type)
    print(
        f"Created {len(samples)} test samples with {2 * n_shot} few-shot examples ({n_shot} per class)"
    )

    # Build solver chain
    solvers = [
        system_message(
            "You are a helpful assistant that classifies text. "
            "Answer only with 'true' or 'false' based on the examples provided."
        ),
        generate(
            config=GenerateConfig(extra_body={"reasoning": {"enabled": False}})
        ),
    ]

    # Add articulation solver if enabled
    if articulation_type in ["multi", "free"]:
        # Get MC options from first sample (all samples have same options structure)
        mc_opts = samples[0].metadata.get("mc_options") if articulation_type == "multi" else None
        solvers.append(articulate_rule(articulation_type, mc_opts))

    # Build scorers - always score classification accuracy
    # Use custom scorer that extracts the first assistant message
    scorers = [classification_accuracy()]

    # Add articulation scorer if enabled
    if articulation_type == "multi":
        # For multiple choice, score the articulation response
        scorers.append(articulation_choice_accuracy())
    elif articulation_type == "free":
        # For free-form, use model grading against rule description
        scorers.append(articulation_quality())

    return Task(
        dataset=MemoryDataset(samples),
        solver=solvers,
        scorer=scorers,
        name=f"icl_{rule}_n{n_shot}_{articulation_type}",
        metadata={"rule": rule, "n_shot": n_shot, "n_test": n_test, "articulation": articulation_type},
    )


class Config(BaseConfig):
    """Configuration for ICL evaluation."""

    rules: list[str] = Field(
        description="Classification rules to evaluate. Use 'all' to evaluate all rules."
    )
    models: list[str] = Field(
        description="Models to evaluate (e.g., 'openai/gpt-4', 'openrouter/anthropic/claude-sonnet-4.5')"
    )
    n_shot: list[int] = Field(
        description="Number of few-shot examples per class"
    )
    n_test: int = Field(
        default=100,
        description="Number of test samples"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    articulation: Literal["none", "multi", "free"] = Field(
        default="none",
        description="Articulation mode: none (no articulation), multi (multiple choice), free (free-form explanation)"
    )
    max_tasks: int = Field(
        default=5,
        description="Maximum number of tasks to run in parallel"
    )


def main(config: Config):
    """Run ICL evaluation with the given configuration."""
    # Set random seed once
    random.seed(config.seed)

    # Handle "all" keyword for rules
    if "all" in config.rules:
        rules_to_evaluate = list(RULES.keys())
    else:
        rules_to_evaluate = config.rules

    print(f"Evaluating in-context learning performance")
    print(f"Rules: {rules_to_evaluate}")
    print(f"Models: {config.models}")
    print(f"N-shot values: {config.n_shot}")
    print(f"Test samples: {config.n_test}")
    print(f"Random seed: {config.seed}")
    print(f"Articulation mode: {config.articulation}")
    print(f"Total evaluations: {len(rules_to_evaluate) * len(config.models) * len(config.n_shot)}")
    print()

    # Create task configurations for all parameter combinations
    tasks = []
    for rule in rules_to_evaluate:
        for n_shot in config.n_shot:
            tasks.append(evaluate_icl_task(rule, n_shot, config.n_test, config.articulation))

    # Run evaluations across all models and tasks
    logs = eval(
        tasks,
        model=config.models,
        log_dir=LOG_DIR,
        max_connections=MAX_CONNECTIONS,
        max_tasks=config.max_tasks,
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


if __name__ == "__main__":
    main(Config())
