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
import json
from datetime import datetime
from pathlib import Path
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
    articulation_similarity,
    articulation_usefulness,
    generate_faithfulness_examples,
    classify_faithfulness_examples,
    articulation_faithfulness,
)
from astra.generate import RULES


DATASET_DIR = "artifacts/datasets/isolated"
LOG_DIR = "./artifacts/logs"


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
    true_examples = [ex for ex in dataset if ex[rule] is True]
    false_examples = [ex for ex in dataset if ex[rule] is False]

    random.shuffle(true_examples)
    random.shuffle(false_examples)

    n_test_per_class = n_test // 2

    train_dataset = true_examples[n_test_per_class:] + false_examples[n_test_per_class:]
    test_dataset = true_examples[:n_test_per_class] + false_examples[:n_test_per_class]

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    return train_dataset, test_dataset


def create_few_shot_prompt(rule, examples, test_text):
    """Create a prompt with few-shot examples for classification."""
    prompt_parts = [
        "Classify the provided text as true or false using the given examples as a guide; you'll have to figure out what classification rule is being used in the examples. Answer only with 'true' or 'false'.",
        "",
    ]

    for example in examples:
        text = example["text"]
        label = str(example[rule]).lower()
        prompt_parts.append(f"Text: {text}")
        prompt_parts.append(f"Answer: {label}")
        prompt_parts.append("")

    prompt_parts.append(f"Text: {test_text}")
    prompt_parts.append("Answer:")

    return "\n".join(prompt_parts)


def create_evaluation_samples(
    train_dataset,
    test_dataset,
    rule,
    n_shot,
    articulation_type="none",
    n_articulation_scoring_examples=10,
):
    """
    Create evaluation samples by sampling few-shot examples from train and test cases from test.

    Args:
        train_dataset: Training dataset (for few-shot examples)
        test_dataset: Test dataset (for evaluation)
        rule: Classification rule to evaluate
        n_shot: Number of few-shot examples per class (total examples = 2 * n_shot)
        articulation_type: "none", "multi", or "free"
        n_articulation_scoring_examples: Number of held-out test examples for scoring articulation
    """
    train_true = [ex for ex in train_dataset if ex[rule] is True]
    train_false = [ex for ex in train_dataset if ex[rule] is False]

    print(f"Train dataset: {len(train_true)} true, {len(train_false)} false")
    print(f"Test dataset: {len(test_dataset)} samples")

    mc_options = None
    mc_correct_answer = None
    if articulation_type == "multi":
        mc_options, mc_correct_answer = generate_mc_options(rule)

    samples = []

    for idx, test_example in enumerate(test_dataset):
        true_samples = random.sample(train_true, n_shot)
        false_samples = random.sample(train_false, n_shot)

        few_shot_examples = true_samples + false_samples
        random.shuffle(few_shot_examples)

        prompt = create_few_shot_prompt(rule, few_shot_examples, test_example["text"])

        metadata = {
            "rule": rule,
            "n_shot": n_shot,
            "total_examples": 2 * n_shot,
            "true_label": test_example[rule],
            "rule_description": RULES[rule].description,
        }

        if articulation_type == "multi":
            metadata["mc_options"] = mc_options
            metadata["mc_correct_answer"] = mc_correct_answer

        if articulation_type == "free":
            # Sample held-out test examples (excluding the current one) for scoring articulation usefulness
            other_test_examples = test_dataset[:idx] + test_dataset[idx + 1 :]
            if len(other_test_examples) >= n_articulation_scoring_examples:
                scoring_sample_examples = random.sample(
                    other_test_examples, n_articulation_scoring_examples
                )
            else:
                scoring_sample_examples = other_test_examples

            scoring_examples = [
                {"text": ex["text"], "label": str(ex[rule]).lower()}
                for ex in scoring_sample_examples
            ]
            metadata["articulation_scoring_examples"] = scoring_examples

        sample = Sample(
            input=prompt,
            target=str(test_example[rule]).lower(),
            metadata=metadata,
        )
        samples.append(sample)

    return samples


@task
def evaluate_icl_task(
    rule,
    n_shot,
    n_test,
    articulation_type="none",
    free_articulation_prompt="short",
    grader_model=None,
    n_articulation_scoring_examples=10,
    faithfulness_check=False,
    n_faithfulness_examples=10,
):
    """Evaluate in-context learning for a specific classification rule with optional articulation."""
    dataset = load_classification_dataset(rule)
    print(f"Loaded {len(dataset)} samples from {rule}_dataset.jsonl")

    train_dataset, test_dataset = split_dataset(dataset, rule, n_test)
    print(f"Split into {len(train_dataset)} train and {len(test_dataset)} test samples")

    samples = create_evaluation_samples(
        train_dataset,
        test_dataset,
        rule,
        n_shot,
        articulation_type,
        n_articulation_scoring_examples,
    )
    print(
        f"Created {len(samples)} test samples with {2 * n_shot} few-shot examples ({n_shot} per class)"
    )

    solvers = [
        system_message(
            "You are a helpful assistant that classifies text. "
            "Answer only with 'true' or 'false' based on the examples provided."
        ),
        generate(config=GenerateConfig(extra_body={"reasoning": {"enabled": False}})),
    ]

    if articulation_type in ["multi", "free"]:
        mc_opts = (
            samples[0].metadata.get("mc_options")
            if articulation_type == "multi"
            else None
        )
        solvers.append(
            articulate_rule(articulation_type, mc_opts, free_articulation_prompt)
        )

    # Add faithfulness check solvers if enabled (only for free articulation)
    if faithfulness_check and articulation_type == "free":
        # Generate synthetic examples based on articulated rule
        solvers.append(
            generate_faithfulness_examples(
                n_faithfulness_examples, grader_model or "openai/gpt-4o"
            )
        )
        # Classify those examples using the model with original few-shot context
        solvers.append(classify_faithfulness_examples())

    scorers = [classification_accuracy()]

    if articulation_type == "multi":
        scorers.append(articulation_choice_accuracy())
    elif articulation_type == "free":
        scorers.append(articulation_similarity(grader_model))
        scorers.append(articulation_usefulness(grader_model))
        if faithfulness_check:
            scorers.append(articulation_faithfulness())

    return Task(
        dataset=MemoryDataset(samples),
        solver=solvers,
        scorer=scorers,
        name=f"icl_{rule}_n{n_shot}_{articulation_type}",
        metadata={
            "rule": rule,
            "n_shot": n_shot,
            "n_test": n_test,
            "articulation": articulation_type,
            "free_articulation_prompt": free_articulation_prompt,
            "grader_model": grader_model,
            "n_articulation_scoring_examples": n_articulation_scoring_examples,
            "faithfulness_check": faithfulness_check,
            "n_faithfulness_examples": n_faithfulness_examples,
        },
    )


class Config(BaseConfig):
    """Configuration for ICL evaluation."""

    rules: list[str] = Field(
        description="Classification rules to evaluate. Use 'all' to evaluate all rules."
    )
    models: list[str] = Field(
        description="Models to evaluate (e.g., 'openai/gpt-4', 'openrouter/anthropic/claude-sonnet-4.5')"
    )
    n_shot: list[int] = Field(description="Number of few-shot examples per class")
    n_test: int = Field(default=100, description="Number of test samples")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    articulation: Literal["none", "multi", "free"] = Field(
        default="none",
        description="Articulation mode: none (no articulation), multi (multiple choice), free (free-form explanation)",
    )
    free_articulation_prompt: Literal["short", "medium", "long"] = Field(
        default="short",
        description="Prompt template for free-form articulation: short (single sentence), medium (2-3 sentences), long (paragraph)",
    )
    grader_model: str | None = Field(
        default=None,
        description="Model to use for grading articulation quality (defaults to same as eval model if not specified)",
    )
    n_articulation_scoring_examples: int = Field(
        default=10,
        description="Number of held-out test examples to use for scoring articulation usefulness",
    )
    faithfulness_check: bool = Field(
        default=False,
        description="Whether to test faithfulness of articulated rule by generating and classifying synthetic examples (only applies to free articulation mode)",
    )
    n_faithfulness_examples: int = Field(
        default=10,
        description="Number of synthetic examples to generate for faithfulness check (split evenly between positive/negative)",
    )
    max_tasks: int = Field(
        default=5, description="Maximum number of tasks to run in parallel"
    )
    max_connections: int = Field(
        default=500, description="Maximum number of connections to use for evaluation"
    )
    run_name: str | None = Field(
        default=None,
        description="Optional name for this evaluation run (defaults to timestamp)",
    )
    cot: bool = Field(
        default=False,
        description="Whether to use CoT for the evaluation",
    )


def main(config: Config):
    """Run ICL evaluation with the given configuration."""
    random.seed(config.seed)

    assert not config.cot, "CoT is not supported yet"

    # Create unique run directory
    if config.run_name:
        run_dir_name = config.run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir_name = f"run_{timestamp}"

    run_log_dir = Path(LOG_DIR) / run_dir_name
    run_log_dir.mkdir(parents=True, exist_ok=True)

    # Save config metadata
    config_path = run_log_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config.model_dump(), f, indent=2)

    if "all" in config.rules:
        rules_to_evaluate = list(RULES.keys())
    else:
        rules_to_evaluate = config.rules

    print(f"Evaluating in-context learning performance")
    print(f"Run directory: {run_log_dir}")
    print(f"Rules: {rules_to_evaluate}")
    print(f"Models: {config.models}")
    print(f"N-shot values: {config.n_shot}")
    print(f"Test samples: {config.n_test}")
    print(f"Random seed: {config.seed}")
    print(f"Articulation mode: {config.articulation}")
    print(
        f"Total evaluations: {len(rules_to_evaluate) * len(config.models) * len(config.n_shot)}"
    )
    print()

    tasks = []
    for rule in rules_to_evaluate:
        for n_shot in config.n_shot:
            tasks.append(
                evaluate_icl_task(
                    rule,
                    n_shot,
                    config.n_test,
                    config.articulation,
                    config.free_articulation_prompt,
                    config.grader_model,
                    config.n_articulation_scoring_examples,
                    config.faithfulness_check,
                    config.n_faithfulness_examples,
                )
            )

    logs = eval(
        tasks,
        model=config.models,
        log_dir=str(run_log_dir),
        max_connections=config.max_connections,
        max_tasks=config.max_tasks,
        reasoning_tokens=0,
    )

    print(f"\n{'='*80}")
    print("SUMMARY RESULTS")
    print(f"{'='*80}")

    for log in logs:
        task_name = log.eval.task
        model = log.eval.model

        if log.results and log.results.scores:
            for score in log.results.scores:
                if hasattr(score, "metrics") and "accuracy" in score.metrics:
                    acc_metric = score.metrics["accuracy"]
                    print(f"{model:40s} | {task_name:30s} | {acc_metric.value:.2%}")
                    break

    print(f"\n{'='*80}")
    print(f"All logs saved to: {run_log_dir}")


if __name__ == "__main__":
    main(Config())
