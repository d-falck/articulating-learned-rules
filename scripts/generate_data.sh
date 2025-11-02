#!/bin/bash

uv run -m astra.run.generate_isolated \
  --model openrouter/anthropic/claude-sonnet-4.5 \
  --num-samples 1000