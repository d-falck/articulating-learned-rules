"""
Generate a classification dataset.

Creates paragraphs about various topics following different combinations of rules.
Each sample includes ground truth labels for various binary classification tasks.

Usage (CLI):
    python -m astra.run.generate_combined --model openai/gpt-4 --num-samples 1000

Usage (YAML):
    # Create config.yaml:
    model: openai/gpt-4
    num_samples: 1000
    seed: 42

    # Run:
    python -m astra.run.generate_combined --config config.yaml
"""

import itertools
import json
import random
from pathlib import Path

from confetti import BaseConfig
from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, system_message
from pydantic import Field

from astra.generate import RULES, TOPICS, self_check


OUTPUT_FILE = "artifacts/datasets/classification_dataset.jsonl"
MAX_CONCURRENCY = 100
LOG_DIR = "artifacts/logs"


load_dotenv()


def generate_rule_combinations(num_samples):
    """Generate all possible rule combinations and sample num_samples of them."""
    rule_names = list(RULES.keys())
    all_combinations = list(itertools.product([True, False], repeat=len(rule_names)))

    # Sample num_samples combinations (with replacement if needed)
    if num_samples <= len(all_combinations):
        # Sample without replacement when we have enough unique combinations
        selected = random.sample(all_combinations, num_samples)
    else:
        # Sample with replacement when we need more than the unique combinations
        selected = random.choices(all_combinations, k=num_samples)

    # Convert to list of dicts
    combinations = []
    for combo in selected:
        combo_dict = dict(zip(rule_names, combo))
        combinations.append(combo_dict)

    return combinations


def create_prompt(topic, rules_dict):
    """Create a prompt instructing the model to generate text following specific rules."""
    instructions = [f"Topic: {topic}", "", "Requirements:"]

    for rule_name, rule_value in rules_dict.items():
        instruction = RULES[rule_name].true if rule_value else RULES[rule_name].false
        instructions.append(f"- {instruction}")

    instructions.append("")
    instructions.append(
        "Generate a single sentence about this topic following ALL the requirements above. "
        "Adjust the length and complexity as needed to meet the requirements."
    )

    return "\n".join(instructions)


@task
def generate_classification_dataset(num_samples):
    """Generate dataset with short sentences following various rule combinations."""

    combinations = generate_rule_combinations(num_samples)

    samples = []
    for combo in combinations:
        topic = random.choice(TOPICS)
        prompt = create_prompt(topic, combo)

        # Store the ground truth labels in metadata
        sample = Sample(input=prompt, metadata=combo)
        samples.append(sample)

    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            system_message(
                "You are a helpful assistant that generates single sentences following specific constraints. "
                "Follow the given requirements EXACTLY. Pay special attention to:\n"
                "- Verb count: Count carefully. Common verbs include 'is', 'are', 'was', 'has', 'does', 'runs', 'makes', etc.\n"
                "- Word count: Count the total number of words in your sentence.\n"
                "Read the requirements carefully and follow them precisely."
            ),
            generate(),
            self_check(),
        ],
    )


def save_dataset_from_log(log):
    """Extract results from Inspect eval log and save to JSONL."""
    output_data = []
    for sample in log.samples:
        # Get the last generated text (after self-check correction)
        # sample.output.completion contains the final response
        text = sample.output.completion

        # Get ground truth labels from metadata
        labels = sample.metadata

        # Create output record
        record = {"text": text, **labels}
        output_data.append(record)

    # Save to JSONL
    output_path = Path(OUTPUT_FILE)
    with output_path.open("w") as f:
        for record in output_data:
            f.write(json.dumps(record) + "\n")

    print(f"\nDataset saved to {OUTPUT_FILE}")
    print(f"Total samples: {len(output_data)}")


class Config(BaseConfig):
    """Configuration for combined dataset generation."""

    model: str = Field(
        default="openrouter/anthropic/claude-sonnet-4.5",
        description="Model to use for generation"
    )
    num_samples: int = Field(
        default=1000,
        description="Number of samples to generate"
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


def main(config: Config):
    """Generate combined classification dataset with the given configuration."""
    # Set random seed
    random.seed(config.seed)

    print(f"Generating dataset with model: {config.model}")
    print(f"This will create {config.num_samples} samples...")
    print(f"Random seed: {config.seed}")

    # Run the evaluation with higher concurrency
    logs = eval(
        generate_classification_dataset(config.num_samples),
        model=config.model,
        log_dir=LOG_DIR,
        max_connections=MAX_CONCURRENCY,
    )

    # logs is a list of EvalLog objects (one per task)
    log = logs[0]

    # Save the dataset
    print(f"\nEvaluation complete!")
    print(f"Log saved to: {log.location}")
    save_dataset_from_log(log)


if __name__ == "__main__":
    main(Config())
