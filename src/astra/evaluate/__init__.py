"""Evaluation utilities and scorers for in-context learning."""

from .solvers import (
    articulate_rule,
    generate_faithfulness_examples,
    classify_faithfulness_examples,
)
from .helpers import generate_mc_options
from .scorers import (
    classification_accuracy,
    articulation_choice_accuracy,
    articulation_similarity,
    articulation_usefulness,
    articulation_faithfulness,
)
from .constants import ARTICULATION_PROMPTS, FAITHFULNESS_EXAMPLE_GENERATION_PROMPT

__all__ = [
    "articulate_rule",
    "generate_faithfulness_examples",
    "classify_faithfulness_examples",
    "generate_mc_options",
    "classification_accuracy",
    "articulation_choice_accuracy",
    "articulation_similarity",
    "articulation_usefulness",
    "articulation_faithfulness",
    "ARTICULATION_PROMPTS",
    "FAITHFULNESS_EXAMPLE_GENERATION_PROMPT",
]
