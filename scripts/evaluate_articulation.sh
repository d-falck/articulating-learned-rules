#!/bin/bash
#
# Evaluate in-context learning with articulation (multiple choice)
#
# This script runs the ICL evaluation with multiple choice articulation,
# where the model must identify which rule it used from a list of 4 options.

uv run -m astra.run.evaluate \
  --rules all \
  --models openai/gpt-4.1-2025-04-14 \
  --n-shot 15 \
  --articulation free \
  --max-tasks 10
