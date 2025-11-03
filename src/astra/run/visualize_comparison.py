"""
Compare articulation metrics across different experimental conditions.

Generates 3 plots:
1. Articulation similarity for different prompt lengths (short, medium, long)
2. Articulation usefulness for different prompt lengths (short, medium, long)
3. Articulation metrics (similarity, usefulness, faithfulness) with/without CoT

Usage:
    python -m astra.run.visualize_comparison
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from inspect_ai.log import read_eval_log


# Hardcoded run directories
RUN_DIRS = {
    'short': 'artifacts/logs/articulation_free_short',
    'medium': 'artifacts/logs/articulation_free_med',
    'long': 'artifacts/logs/articulation_free_long',
    'short_cot': 'artifacts/logs/articulation_free_short_cot',
}

# Filter to these specific models and n_shot values
MODELS = ['openai/gpt-4.1-nano', 'openai/gpt-4.1']
N_SHOTS = [5, 15]

OUTPUT_DIR = Path('artifacts/plots/articulation_comparison')


def load_run_results(run_dir):
    """Load all eval results from a run directory."""
    run_path = Path(run_dir)
    results = []

    for eval_file in run_path.glob('*.eval'):
        try:
            log = read_eval_log(str(eval_file))
            results.append(log)
        except Exception as e:
            print(f"Warning: Could not load {eval_file}: {e}")

    return results


def extract_metrics(results, models_filter=None, n_shots_filter=None):
    """
    Extract metrics from results, optionally filtering by models and n_shots.

    Returns dict: {model: {n_shot: {rule: {metric: value}}}}
    """
    data = {}

    for log in results:
        # Get model and task args
        model = log.eval.model
        task_args = log.eval.task_args if hasattr(log.eval, 'task_args') else {}

        # Apply model filter
        if models_filter and model not in models_filter:
            continue

        # Get metadata from task_args
        rule = task_args.get('rule')
        n_shot = task_args.get('n_shot')

        # Apply n_shot filter
        if n_shots_filter and n_shot not in n_shots_filter:
            continue

        # Skip if missing required metadata
        if not rule or n_shot is None:
            continue

        # Extract scores
        scores = {}
        if log.results and log.results.scores:
            for score in log.results.scores:
                if hasattr(score, 'name'):
                    score_name = score.name
                else:
                    continue

                if hasattr(score, 'metrics') and 'accuracy' in score.metrics:
                    acc = score.metrics['accuracy']
                    scores[score_name] = acc.value

        # Store in nested dict
        if model not in data:
            data[model] = {}
        if n_shot not in data[model]:
            data[model][n_shot] = {}
        if rule not in data[model][n_shot]:
            data[model][n_shot][rule] = {}

        data[model][n_shot][rule].update(scores)

    return data


def compute_means_and_cis(data, metric_name):
    """
    Compute means and 95% CIs for a specific metric across all rules.

    Returns: dict {model: {n_shot: (mean, ci)}}
    """
    results = {}

    for model in data:
        results[model] = {}
        for n_shot in data[model]:
            values = []
            for rule in data[model][n_shot]:
                if metric_name in data[model][n_shot][rule]:
                    values.append(data[model][n_shot][rule][metric_name])

            if values:
                mean = np.mean(values)
                ci = np.std(values, ddof=1) * 1.96 / np.sqrt(len(values)) if len(values) > 1 else 0
                results[model][n_shot] = (mean, ci)
            else:
                results[model][n_shot] = (0, 0)

    return results


def plot_prompt_length_comparison(metric_name, metric_label, output_path):
    """
    Plot metric comparison across prompt lengths.

    X-axis: short, medium, long
    Bars: 4 bars per group (2 models × 2 n_shot)
    """
    # Load data for each prompt length
    prompt_lengths = ['short', 'medium', 'long']
    prompt_data = {}

    for prompt_len in prompt_lengths:
        results = load_run_results(RUN_DIRS[prompt_len])
        data = extract_metrics(results, models_filter=MODELS, n_shots_filter=N_SHOTS)
        prompt_data[prompt_len] = compute_means_and_cis(data, metric_name)

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    x_positions = np.arange(len(prompt_lengths))
    bar_width = 0.2

    # Colors for each model/n_shot combination
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']  # Dark/light blue, dark/light orange

    # Create bars
    bar_configs = [
        (MODELS[0], N_SHOTS[0], 0, colors[0], f'{MODELS[0].split("/")[-1]} (n={N_SHOTS[0]})'),
        (MODELS[0], N_SHOTS[1], 1, colors[1], f'{MODELS[0].split("/")[-1]} (n={N_SHOTS[1]})'),
        (MODELS[1], N_SHOTS[0], 2, colors[2], f'{MODELS[1].split("/")[-1]} (n={N_SHOTS[0]})'),
        (MODELS[1], N_SHOTS[1], 3, colors[3], f'{MODELS[1].split("/")[-1]} (n={N_SHOTS[1]})'),
    ]

    for model, n_shot, bar_idx, color, label in bar_configs:
        means = []
        cis = []

        for prompt_len in prompt_lengths:
            if model in prompt_data[prompt_len] and n_shot in prompt_data[prompt_len][model]:
                mean, ci = prompt_data[prompt_len][model][n_shot]
                means.append(mean)
                cis.append(ci)
            else:
                means.append(0)
                cis.append(0)

        offset = (bar_idx - 1.5) * bar_width
        ax.bar(x_positions + offset, means, bar_width, label=label,
               yerr=cis, capsize=2, color=color, error_kw={'ecolor': 'black', 'linewidth': 1})

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Short', 'Medium', 'Long'], fontsize=10)
    ax.set_ylabel(metric_label, fontsize=11)
    ax.set_title(f'{metric_label} by Prompt Length', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {output_path}")


def plot_cot_comparison(output_path):
    """
    Plot comparison of metrics with/without CoT.

    X-axis groups: Similarity (No CoT), Similarity (CoT), Usefulness (No CoT), ...
    Bars: 4 bars per group (2 models × 2 n_shot)
    """
    # Load data
    results_no_cot = load_run_results(RUN_DIRS['short'])
    results_cot = load_run_results(RUN_DIRS['short_cot'])

    data_no_cot = extract_metrics(results_no_cot, models_filter=MODELS, n_shots_filter=N_SHOTS)
    data_cot = extract_metrics(results_cot, models_filter=MODELS, n_shots_filter=N_SHOTS)

    # Compute metrics
    metrics = [
        ('articulation_similarity', 'Similarity'),
        ('articulation_usefulness', 'Usefulness'),
        ('articulation_faithfulness', 'Faithfulness'),
    ]

    metric_data = {}
    for metric_name, metric_label in metrics:
        metric_data[f'{metric_label} (No CoT)'] = compute_means_and_cis(data_no_cot, metric_name)
        metric_data[f'{metric_label} (CoT)'] = compute_means_and_cis(data_cot, metric_name)

    # Set up plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))

    group_labels = list(metric_data.keys())
    x_positions = np.arange(len(group_labels))
    bar_width = 0.18

    # Colors for each model/n_shot combination
    colors = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78']

    # Create bars
    bar_configs = [
        (MODELS[0], N_SHOTS[0], 0, colors[0], f'{MODELS[0].split("/")[-1]} (n={N_SHOTS[0]})'),
        (MODELS[0], N_SHOTS[1], 1, colors[1], f'{MODELS[0].split("/")[-1]} (n={N_SHOTS[1]})'),
        (MODELS[1], N_SHOTS[0], 2, colors[2], f'{MODELS[1].split("/")[-1]} (n={N_SHOTS[0]})'),
        (MODELS[1], N_SHOTS[1], 3, colors[3], f'{MODELS[1].split("/")[-1]} (n={N_SHOTS[1]})'),
    ]

    for model, n_shot, bar_idx, color, label in bar_configs:
        means = []
        cis = []

        for group_label in group_labels:
            data = metric_data[group_label]
            if model in data and n_shot in data[model]:
                mean, ci = data[model][n_shot]
                means.append(mean)
                cis.append(ci)
            else:
                means.append(0)
                cis.append(0)

        offset = (bar_idx - 1.5) * bar_width
        ax.bar(x_positions + offset, means, bar_width, label=label,
               yerr=cis, capsize=2, color=color, error_kw={'ecolor': 'black', 'linewidth': 1})

    ax.set_xticks(x_positions)
    ax.set_xticklabels(group_labels, rotation=0, ha='center', fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Articulation Metrics: With vs Without CoT', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=9, loc='best', ncol=2)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {output_path}")


def main():
    """Generate all comparison plots."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating articulation comparison plots...")
    print(f"Using models: {MODELS}")
    print(f"Using n_shot values: {N_SHOTS}")
    print()

    # Plot 1: Similarity vs prompt length
    print("Generating Plot 1: Similarity by Prompt Length")
    plot_prompt_length_comparison(
        'articulation_similarity',
        'Articulation Similarity',
        OUTPUT_DIR / 'similarity_by_prompt_length.png'
    )

    # Plot 2: Usefulness vs prompt length
    print("Generating Plot 2: Usefulness by Prompt Length")
    plot_prompt_length_comparison(
        'articulation_usefulness',
        'Articulation Usefulness',
        OUTPUT_DIR / 'usefulness_by_prompt_length.png'
    )

    # Plot 3: CoT comparison
    print("Generating Plot 3: CoT Comparison")
    plot_cot_comparison(OUTPUT_DIR / 'cot_comparison.png')

    print()
    print(f"All plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
