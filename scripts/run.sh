#!/bin/bash

# Evaluate in-context learning for all 10 classification rules
# Model: gpt-5-mini
# N-shot: 20 (20 examples per class = 40 total examples)
# Test samples: 100 (default)

python src/astra/evaluate_icl.py \
  --rules \
    contains_questions \
    contains_exclamations \
    contains_commas \
    contains_numbers \
    starts_with_the \
    all_lowercase \
    contains_quotes \
    contains_and \
    ends_with_period \
    contains_verb \
  --models openai/gpt-5-mini \
  --n-shot 20 \
  --max-tasks 10
