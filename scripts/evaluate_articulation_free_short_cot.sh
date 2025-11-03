#!/bin/bash

uv run -m astra.run.evaluate --config etc/evaluate_base_mini.yaml --articulation free --free-articulation-prompt short --cot true --run-name "articulation_free_short_cot" --faithfulness-check true --n-faithfulness-examples 10