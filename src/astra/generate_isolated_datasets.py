"""
Generate isolated classification datasets.

For each classification rule, creates a separate dataset with only that rule varying.
This makes patterns much easier to identify in few-shot learning.

Usage:
    python src/astra/generate_isolated_datasets.py --model anthropic/claude-3-5-sonnet-20241022
    python src/astra/generate_isolated_datasets.py --num-samples 500
"""

import argparse
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
    "ends_with_question": {
        "true": "End the text with a question mark (?)",
        "false": "End the text with something other than a question mark",
    },
    "contains_numbers": {
        "true": "Include at least one digit (0-9)",
        "false": "Do not use any digits",
    },
    "is_title_case": {
        "true": "Write In Title Case Where Every Word Starts With A Capital Letter",
        "false": "Use normal sentence capitalization (only capitalize the first word and proper nouns)",
    },
    "contains_quotes": {
        "true": "Include at least one quotation mark",
        "false": "Do not use any quotation marks",
    },
    "has_many_verbs": {
        "true": "Include at least 5 different verbs (action/being words). Verbs include: runs, jumps, eats, sleeps, thinks, is, are, was, makes, has, does, goes, sees, feels, walks, talks, etc. Example: 'Dogs run, jump, play, eat, and sleep happily' (contains 5 verbs)",
        "false": "Write using only nouns and adjectives - no action or being words at all. NO verbs like: is, are, was, has, runs, does, etc. Example: 'Red apples' or 'Big coffee cup'",
    },
    "contains_hashtag": {
        "true": "Include at least one hashtag symbol (#)",
        "false": "Do not use any hashtag symbols",
    },
    "is_very_short": {
        "true": "Write a very short sentence with fewer than 7 words. Example: 'Coffee tastes really great today.' (5 words)",
        "false": "Write a longer sentence with more than 20 words. Add details, descriptions, and complexity to reach the word count.",
    },
    "is_first_person": {
        "true": "Write in first person using words like 'I', 'me', 'my', 'we', 'our', or 'us'",
        "false": "Write in the third person; do not use any first person pronouns (no 'I', 'me', 'my', 'we', 'our', 'us')",
    },
    "has_repeated_word": {
        "true": "Include one word repeated 4-5 times in a row (like 'the the the the')",
        "false": "Do not repeat any word more than once in a row",
    },
    "contains_rhyme": {
        "true": "Include as many words as possible that rhyme (like 'cat' and 'hat' or 'day' and 'way')",
        "false": "Do not include any rhyming words",
    },
}


OUTPUT_DIR = "artifacts/datasets/isolated"
MAX_CONCURRENCY = 100
LOG_DIR = "artifacts/logs"


load_dotenv()
random.seed(42)


def create_prompt(topic, rule_name, rule_value):
    """Create a prompt for generating text following a specific rule."""
    instruction = RULES[rule_name]["true" if rule_value else "false"]

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


@solver
def self_check():
    """
    Solver that asks the model to verify its answer meets requirements and correct if needed.
    """

    async def solve(state, generate):
        # Build the self-check prompt
        check_prompt = (
            "Review your previous response and check if it follows the requirement.\n\n"
            "Important: DO NOT explain your reasoning or show your work. "
            "DO NOT include phrases like 'Let me check' or 'Corrected version:'. "
            "ONLY output the final sentence and nothing else.\n\n"
            "If your response satisfies the requirement, output the exact same text. "
            "If it doesn't, output ONLY the corrected sentence.\n\n"
            "Output format: Just the sentence, no explanations."
        )

        # Append the check prompt to the conversation using ChatMessageUser
        state.messages.append(ChatMessageUser(content=check_prompt))

        # Generate the corrected/verified response
        return await generate(state)

    return solve


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate isolated classification datasets"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openrouter/anthropic/claude-sonnet-4.5",
        help="Model to use for generation",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Total number of samples per rule (default: 500, split 250/250)",
    )
    parser.add_argument(
        "--rules",
        nargs="+",
        default=None,
        help="Specific rules to generate (default: all rules)",
    )
    args = parser.parse_args()

    # Determine which rules to generate
    rules_to_generate = args.rules if args.rules else list(RULES.keys())
    num_samples_per_class = args.num_samples // 2

    print(f"Generating isolated datasets with model: {args.model}")
    print(f"Rules to generate: {', '.join(rules_to_generate)}")
    print(
        f"Samples per rule: {args.num_samples} ({num_samples_per_class} positive, {num_samples_per_class} negative)"
    )
    print()

    # Generate dataset for each rule
    for rule_name in rules_to_generate:
        print(f"Generating dataset for rule: {rule_name}")

        # Run the evaluation
        logs = eval(
            generate_isolated_dataset(rule_name, num_samples_per_class),
            model=args.model,
            log_dir=LOG_DIR,
            max_connections=MAX_CONCURRENCY,
        )

        # Save the dataset
        log = logs[0]
        print(f"  Log saved to: {log.location}")
        save_dataset_from_log(log, rule_name)
        print()

    print(f"All datasets saved to: {OUTPUT_DIR}")
