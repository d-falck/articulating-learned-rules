"""Helper functions for evaluation tasks."""

import random
from astra.generate import RULES


def generate_mc_options(correct_rule, num_options=4):
    """
    Generate multiple choice options for rule articulation.

    Args:
        correct_rule: The correct rule name
        num_options: Total options (default 4: 1 correct + 3 distractors)

    Returns:
        (mc_dict, correct_letter): Dict mapping letters to rules, and correct answer letter
    """
    all_rules = list(RULES.keys())
    all_rules.remove(correct_rule)

    # TODO: Generate more plausible distractors instead of random selection
    # Could consider: rules with similar characteristics, rules the model might confuse,
    # or rules that have some overlap with the correct rule
    distractors = random.sample(all_rules, num_options - 1)

    options = [correct_rule] + distractors
    random.shuffle(options)

    letters = ['A', 'B', 'C', 'D'][:num_options]
    mc_dict = {letters[i]: rule for i, rule in enumerate(options)}
    correct_letter = [k for k, v in mc_dict.items() if v == correct_rule][0]

    return mc_dict, correct_letter
