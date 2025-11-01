#!/bin/bash
#
# Evaluate in-context learning (without articulation)
#
# This script runs the standard ICL evaluation across all classification rules.

uv run -m astra.run.evaluate \
  --rules all \
  --models openai/gpt-4.1-2025-04-14 \
  --n-shot 15 \
  --max-tasks 10
