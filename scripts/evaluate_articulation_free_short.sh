#!/bin/bash

uv run -m astra.run.evaluate --config etc/evaluate_base.yaml --articulation free --free-articulation-prompt short --run-name "articulation_free_short" --faithfulness-check true --n-faithfulness-examples 10