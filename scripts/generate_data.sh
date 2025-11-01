#!/bin/bash
#
# Generate isolated classification datasets
#
# This script generates isolated datasets for all classification rules.

uv run -m astra.run.generate_isolated \
  --model openrouter/anthropic/claude-sonnet-4.5 \
  --num-samples 1000