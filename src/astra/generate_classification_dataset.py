"""
Generate a classification dataset.

Creates paragraphs about various topics following different combinations of rules.
Each sample includes ground truth labels for various binary classification tasks.

Usage:
    python src/generate_classification_dataset.py --model anthropic/claude-3-5-sonnet-20241022
    python src/generate_classification_dataset.py --model openai/gpt-4
"""

import argparse
import itertools
import json
import random
from pathlib import Path

from dotenv import load_dotenv
from inspect_ai import Task, eval, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.solver import generate, solver, system_message
from inspect_ai.model import ChatMessageUser


TOPICS = [
    # Everyday Objects
    "coffee",
    "phones",
    "books",
    "shoes",
    "clocks",
    # Common Activities
    "walking",
    "cooking",
    "sleeping",
    "driving",
    "reading",
    # Simple Concepts
    "weather",
    "music",
    "colors",
    "food",
    "money",
    # Places
    "parks",
    "stores",
    "libraries",
    "restaurants",
    "offices",
]


RULES = {
    "sentence_count_multiple": {
        "true": "Write multiple sentences (2 or more)",
        "false": "Write exactly one sentence",
    },
    "contains_questions": {
        "true": "Include at least one question mark",
        "false": "Do not use any question marks",
    },
    "contains_exclamations": {
        "true": "Include at least one exclamation mark",
        "false": "Do not use any exclamation marks",
    },
    "contains_commas": {
        "true": "Include at least one comma",
        "false": "Do not use any commas",
    },
    "contains_numbers": {
        "true": "Include at least one digit (0-9)",
        "false": "Do not use any digits",
    },
    "starts_with_the": {
        "true": "Start the text with the word 'the' (uppercase or lowercase)",
        "false": "Start the text with a different word (not 'the' or 'The')",
    },
    "all_lowercase": {
        "true": "Write everything in lowercase (no capital letters at all)",
        "false": "Use normal capitalization",
    },
    "contains_quotes": {
        "true": "Include at least one quotation mark",
        "false": "Do not use any quotation marks",
    },
    "word_count_long": {
        "true": "Write over 40 words",
        "false": "Write under 25 words",
    },
    "contains_and": {
        "true": "Include the word 'and' at least once",
        "false": "Do not use the word 'and'",
    },
}


NUM_SAMPLES = 1000
OUTPUT_FILE = "artifacts/datasets/classification_dataset.jsonl"
MAX_CONCURRENCY = 50
LOG_DIR = "artifacts/logs"


load_dotenv()
random.seed(42)


def generate_rule_combinations():
    """Generate all possible rule combinations and sample NUM_SAMPLES of them."""
    rule_names = list(RULES.keys())
    all_combinations = list(itertools.product([True, False], repeat=len(rule_names)))

    # Sample NUM_SAMPLES combinations (or all if fewer than NUM_SAMPLES)
    selected = random.sample(all_combinations, min(NUM_SAMPLES, len(all_combinations)))

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
        instruction = RULES[rule_name]["true" if rule_value else "false"]
        instructions.append(f"- {instruction}")

    instructions.append("")
    instructions.append(
        "Generate a paragraph about this topic following ALL the requirements above."
    )

    return "\n".join(instructions)


@solver
def self_check():
    """
    Solver that asks the model to verify its answer meets requirements and correct if needed.
    """

    async def solve(state, generate):
        # Build the self-check prompt
        check_prompt = (
            "Please carefully review your previous response and check if it follows ALL the requirements listed above. "
            "If it does not fully satisfy all requirements, generate a corrected version. "
            "If it does satisfy all requirements, output the exact same text again. "
            "Only output the paragraph text, nothing else."
        )

        # Append the check prompt to the conversation using ChatMessageUser
        state.messages.append(ChatMessageUser(content=check_prompt))

        # Generate the corrected/verified response
        return await generate(state)

    return solve


@task
def generate_classification_dataset():
    """Generate dataset with paragraphs following various rule combinations."""

    combinations = generate_rule_combinations()

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
                "You are a helpful assistant that generates text following specific constraints. "
                "Follow the given requirements exactly."
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate classification dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/anthropic/claude-sonnet-4.5",
        help="Model to use for generation",
    )
    args = parser.parse_args()

    print(f"Generating dataset with model: {args.model}")
    print(f"This will create {NUM_SAMPLES} samples...")

    # Run the evaluation with higher concurrency
    logs = eval(
        generate_classification_dataset(),
        model=args.model,
        log_dir=LOG_DIR,
        max_connections=MAX_CONCURRENCY,
    )

    # logs is a list of EvalLog objects (one per task)
    log = logs[0]

    # Save the dataset
    print(f"\nEvaluation complete!")
    print(f"Log saved to: {log.location}")
    save_dataset_from_log(log)
