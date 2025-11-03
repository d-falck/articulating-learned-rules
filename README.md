# Articulating Learned Classification Rules

Evaluation framework for testing in-context learning performance on classification tasks with optional rule articulation.

## Overview

This project evaluates how well language models can:
1. Learn classification rules from few-shot examples
2. Apply those rules to test samples
3. Articulate the rules they've learned (optional)

The framework supports two articulation modes:
- **Multiple-choice**: Models select from predefined rule descriptions
- **Free-form**: Models explain the rule in their own words

## Setup

1. Install dependencies using [uv](https://github.com/astral-sh/uv):
```bash
uv sync
```

2. Configure environment variables:
```bash
cp .env.example .env
# Add your OPENROUTER_API_KEY to .env
```

## Experimental Scripts

All scripts are located in the [scripts/](scripts/) directory. Results are saved to `artifacts/logs/{run-name}/`.

### Data Generation

- [generate_data.sh](scripts/generate_data.sh)
  - Generates isolated classification datasets
  - Creates 1000 samples using Claude Sonnet 4.5
  - Run before evaluations if datasets don't exist

### Evaluation Scripts

All evaluation scripts use the `astra.run.evaluate` module with different configurations:

#### Free-form Articulation (varying prompt lengths)

- [evaluate_articulation_free_short.sh](scripts/evaluate_articulation_free_short.sh)
  - Config: `evaluate_base.yaml`
  - Free-form articulation with short prompts
  - Includes faithfulness checking (10 examples)
  - Results: `artifacts/logs/articulation_free_short/`

- [evaluate_articulation_free_short_cot.sh](scripts/evaluate_articulation_free_short_cot.sh)
  - Config: `evaluate_base_mini.yaml` (subset of models)
  - Free-form articulation with chain-of-thought reasoning
  - Includes faithfulness checking (10 examples)
  - Results: `artifacts/logs/articulation_free_short_cot/`

- [evaluate_articulation_free_med.sh](scripts/evaluate_articulation_free_med.sh)
  - Config: `evaluate_base_mini.yaml`
  - Free-form articulation with medium-length prompts
  - Results: `artifacts/logs/articulation_free_med/`

- [evaluate_articulation_free_long.sh](scripts/evaluate_articulation_free_long.sh)
  - Config: `evaluate_base_mini.yaml`
  - Free-form articulation with long prompts
  - Results: `artifacts/logs/articulation_free_long/`

#### Multiple-choice Articulation

- [evaluate_articulation_multi.sh](scripts/evaluate_articulation_multi.sh)
  - Config: `evaluate_base_mini.yaml`
  - Multiple-choice rule selection
  - Results: `artifacts/logs/articulation_multi/`

### Visualization

- [visualize.sh](scripts/visualize.sh)
  - Generates visualizations from evaluation results
  - Example usage: Update the `--run-dir` parameter to point to your results
  - Default: `artifacts/logs/articulation_multi`

## Configuration Files

Located in [etc/](etc/):

- [evaluate_base.yaml](etc/evaluate_base.yaml)
  - Full experimental configuration
  - Models: GPT-4.1, GPT-4.1-nano, Claude Sonnet/Haiku 4.5, Gemini 2.5 Flash variants
  - Shot counts: 5, 10, 15
  - 100 test samples per configuration

- [evaluate_base_mini.yaml](etc/evaluate_base_mini.yaml)
  - Smaller subset for faster iteration
  - Models: GPT-4.1, GPT-4.1-nano only
  - Shot counts: 5, 15
  - 100 test samples per configuration

### Key Configuration Parameters

- `models`: List of models to evaluate
- `n_shot`: Number of few-shot examples (can specify multiple)
- `articulation`: none | multi | free
- `free_articulation_prompt`: short | medium | long
- `cot`: Enable chain-of-thought reasoning
- `faithfulness_check`: Verify model's explanation matches its predictions
- `n_test`: Number of test samples
- `rules`: Which classification rules to test (use "all" for all available rules)

## Usage Examples

### Run a specific evaluation
```bash
./scripts/evaluate_articulation_free_short.sh
```

### Run with custom config
```bash
uv run -m astra.run.evaluate --config etc/evaluate_base.yaml --run-name my_experiment
```

### Override config parameters
```bash
uv run -m astra.run.evaluate \
  --config etc/evaluate_base_mini.yaml \
  --articulation free \
  --free-articulation-prompt medium \
  --n-test 50 \
  --run-name quick_test
```

### Visualize results
```bash
uv run -m astra.run.visualize --run-dir artifacts/logs/articulation_free_short
```

## Project Structure

- [src/astra/evaluate/](src/astra/evaluate/) - Evaluation logic (solvers, scorers)
- [src/astra/generate/](src/astra/generate/) - Dataset generation
- [src/astra/run/](src/astra/run/) - Main entry points (evaluate, visualize, generate)
- [scripts/](scripts/) - Experimental run scripts
- [etc/](etc/) - Configuration files
- [artifacts/logs/](artifacts/logs/) - Evaluation results (generated)
- [notebooks/](notebooks/) - Jupyter notebooks for analysis

## Output

Evaluation results include:
- Model predictions and accuracy scores
- Articulation responses (if enabled)
- Faithfulness check results (if enabled)
- Detailed logs in JSON format
- Visualization plots (when using visualize.sh)
