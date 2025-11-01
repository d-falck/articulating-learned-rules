#!/bin/bash

# Evaluate in-context learning for all 10 classification rules
# Model: gpt-5-mini
# N-shot: 20 (20 examples per class = 40 total examples)
# Test samples: 100 (default)

python src/astra/evaluate_icl.py \
  --rules \
    ends_with_question \
    contains_numbers \
    is_title_case \
    contains_quotes \
    has_many_verbs \
    contains_hashtag \
    is_very_short \
    is_first_person \
    has_repeated_word \
    contains_rhyme \
  --models openai/gpt-4.1-2025-04-14 \
  --n-shot 50 \
  --max-tasks 10
