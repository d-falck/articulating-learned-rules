"""
Generate isolated classification datasets.

For each classification rule, creates a separate dataset with only that rule varying.
This makes patterns much easier to identify in few-shot learning.

Usage (CLI):
    python -m astra.run.generate_isolated --model openai/gpt-4 --num-samples 1000
    python -m astra.run.generate_isolated --rules is_first_person contains_numbers --num-samples 500

Usage (YAML):
    # Create config.yaml:
    model: openai/gpt-4
    num_samples: 1000
    rules: [all]  # or specific rules like [is_first_person, contains_numbers]
    seed: 42

    # Run:
    python -m astra.run.generate_isolated --config config.yaml
"""

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


OUTPUT_DIR = "artifacts/datasets/isolated"
MAX_CONCURRENCY = 100
LOG_DIR = "artifacts/logs"


load_dotenv()


def create_prompt(topic, rule_name, rule_value):
    """Create a prompt for generating text following a specific rule."""
    instruction = RULES[rule_name].true if rule_value else RULES[rule_name].false

    prompt = [
        f"Topic: {topic}",
        "",
        "Requirement:",
        f"- {instruction}",
        "",
        "Generate a single sentence about this topic following the requirement above. "
        "Adjust the length and complexity as needed to meet the requirement.",
    ]

    return "\n".join(prompt)


@task
def generate_isolated_dataset(rule_name, num_samples_per_class):
    """Generate dataset for a single classification rule."""

    samples = []

    # Generate positive examples (rule=True)
    for _ in range(num_samples_per_class):
        topic = random.choice(TOPICS)
        prompt = create_prompt(topic, rule_name, True)
        sample = Sample(
            input=prompt, metadata={"rule": rule_name, "label": True, "topic": topic}
        )
        samples.append(sample)

    # Generate negative examples (rule=False)
    for _ in range(num_samples_per_class):
        topic = random.choice(TOPICS)
        prompt = create_prompt(topic, rule_name, False)
        sample = Sample(
            input=prompt, metadata={"rule": rule_name, "label": False, "topic": topic}
        )
        samples.append(sample)

    # Shuffle to mix positive and negative
    random.shuffle(samples)

    return Task(
        dataset=MemoryDataset(samples),
        solver=[
            system_message(
                "You are a helpful assistant that generates single sentences following specific constraints. "
                "Follow the given requirement EXACTLY. "
                "Read the requirement carefully and follow it precisely."
            ),
            generate(),
            self_check(),
        ],
        name=f"isolated_{rule_name}",
    )


def save_dataset_from_log(log, rule_name):
    """Extract results from Inspect eval log and save to JSONL."""
    output_data = []
    for sample in log.samples:
        # Get the generated text (after self-check correction)
        text = sample.output.completion

        # Get metadata
        metadata = sample.metadata

        # Create output record with only the rule we're testing
        record = {
            "text": text,
            rule_name: metadata["label"],  # Only include this rule's label
            "topic": metadata.get("topic", "unknown"),
        }
        output_data.append(record)

    # Save to JSONL
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{rule_name}_dataset.jsonl"

    with output_file.open("w") as f:
        for record in output_data:
            f.write(json.dumps(record) + "\n")

    print(f"  Saved {len(output_data)} samples to {output_file}")

    # Print label distribution
    true_count = sum(1 for r in output_data if r[rule_name] is True)
    false_count = sum(1 for r in output_data if r[rule_name] is False)
    print(f"  Distribution: {true_count} true, {false_count} false")


class Config(BaseConfig):
    """Configuration for isolated dataset generation."""

    model: str = Field(
        default="openrouter/anthropic/claude-sonnet-4.5",
        description="Model to use for generation"
    )
    num_samples: int = Field(
        default=1000,
        description="Total number of samples per rule (split evenly between positive and negative)"
    )
    rules: list[str] | None = Field(
        default=None,
        description="Specific rules to generate. Use 'all' or leave empty to generate all rules."
    )
    seed: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


def main(config: Config):
    """Generate isolated classification datasets with the given configuration."""
    # Set random seed
    random.seed(config.seed)

    # Determine which rules to generate
    if config.rules is None or "all" in config.rules:
        rules_to_generate = list(RULES.keys())
    else:
        rules_to_generate = config.rules

    num_samples_per_class = config.num_samples // 2

    print(f"Generating isolated datasets with model: {config.model}")
    print(f"Rules to generate: {', '.join(rules_to_generate)}")
    print(f"Samples per rule: {config.num_samples} ({num_samples_per_class} positive, {num_samples_per_class} negative)")
    print(f"Random seed: {config.seed}")
    print()

    # Generate dataset for each rule
    for rule_name in rules_to_generate:
        print(f"Generating dataset for rule: {rule_name}")

        # Run the evaluation
        logs = eval(
            generate_isolated_dataset(rule_name, num_samples_per_class),
            model=config.model,
            log_dir=LOG_DIR,
            max_connections=MAX_CONCURRENCY,
        )

        # Save the dataset
        log = logs[0]
        print(f"  Log saved to: {log.location}")
        save_dataset_from_log(log, rule_name)
        print()

    print(f"All datasets saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main(Config())
