"""
Visualize evaluation results from inspect_ai log files.

Usage:
    python -m astra.run.visualize --run-dir artifacts/logs/run_20250102_143022
    python -m astra.run.visualize --run-dir artifacts/logs/my_experiment --output-dir my_plots/
"""

import json
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from confetti import BaseConfig
from pydantic import Field
from inspect_ai.log import read_eval_log


class Config(BaseConfig):
    """Configuration for visualization."""

    run_dir: str = Field(
        description="Directory containing eval logs from a run"
    )
    output_dir: str | None = Field(
        default=None,
        description="Directory to save plots (defaults to run_dir/plots)"
    )


def load_run_results(run_dir: Path):
    """Load all eval logs from a run directory."""
    eval_files = list(run_dir.glob("*.eval"))

    results = []
    for eval_file in eval_files:
        log = read_eval_log(str(eval_file))
        results.append(log)

    return results


def extract_metrics(results):
    """Extract metrics from eval logs into a structured format."""
    data = defaultdict(lambda: defaultdict(list))

    for log in results:
        model = log.eval.model
        task_name = log.eval.task

        # Parse task metadata from task_args
        task_args = log.eval.task_args if hasattr(log.eval, 'task_args') else {}

        rule = task_args.get("rule")
        n_shot = task_args.get("n_shot")
        n_test = task_args.get("n_test", 100)  # Default to 100 if not found
        articulation = task_args.get("articulation_type", "none")

        # Debug output
        if not rule or not n_shot:
            print(f"Warning: Missing metadata for task {task_name}")
            print(f"  Available eval attributes: {dir(log.eval)}")
            if hasattr(log.eval, 'task_args'):
                print(f"  task_args: {log.eval.task_args}")
            if hasattr(log.eval, 'metadata'):
                print(f"  metadata: {log.eval.metadata}")

        if not log.results or not log.results.scores:
            print(f"Warning: No results for task {task_name}")
            continue

        # Extract all scores
        scores_dict = {}
        for score in log.results.scores:
            if hasattr(score, 'name'):
                score_name = score.name
            else:
                score_name = "unknown"

            if hasattr(score, 'metrics') and 'accuracy' in score.metrics:
                acc = score.metrics['accuracy']
                # Backwards compatibility: map exact_match to classification_accuracy
                if score_name == "exact_match":
                    score_name = "classification_accuracy"

                # Extract stderr from metrics dict (it's a separate metric)
                stderr_val = None
                if 'stderr' in score.metrics:
                    stderr_val = score.metrics['stderr'].value

                scores_dict[score_name] = {
                    'value': acc.value,
                    'stderr': stderr_val
                }

        print(f"Task: {task_name}, Rule: {rule}, N-shot: {n_shot}, Articulation: {articulation}, Scores: {list(scores_dict.keys())}")

        data[model][task_name] = {
            'rule': rule,
            'n_shot': n_shot,
            'n_test': n_test,
            'articulation': articulation,
            'scores': scores_dict
        }

    return data


def plot_icl_accuracy(data, output_dir):
    """Plot in-context classification accuracy vs n_shot using grouped bar charts."""
    # Group by model and rule
    models = list(data.keys())

    # Get unique rules and n_shot values (include all tasks, regardless of articulation)
    all_rules = set()
    all_n_shots = set()
    for model_data in data.values():
        for task_data in model_data.values():
            all_rules.add(task_data['rule'])
            all_n_shots.add(task_data['n_shot'])

    rules = sorted(all_rules)
    n_shots = sorted(all_n_shots)

    # Define model ordering: GPT, Claude, Gemini
    model_order = {
        'openai/gpt-4.1-nano': 0,
        'openai/gpt-4.1': 1,
        'openrouter/anthropic/claude-haiku-4.5': 2,
        'openrouter/anthropic/claude-sonnet-4.5': 3,
        'openrouter/google/gemini-2.5-flash-lite': 4,
        'openrouter/google/gemini-2.5-flash': 5,
    }
    models_sorted = sorted([m for m in models if m in model_order], key=lambda x: model_order.get(x, 999))

    # Plot 1: Overall comparison - grouped by model
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Collect data for each model and n_shot
    model_data_dict = {}
    for model in models_sorted:
        model_data_dict[model] = {}
        for n_shot in n_shots:
            matching_tasks = [
                task_data for task_name, task_data in data[model].items()
                if task_data.get('n_shot') == n_shot
            ]

            if matching_tasks:
                accs = []
                stderrs = []
                for task_data in matching_tasks:
                    if 'classification_accuracy' in task_data['scores']:
                        score_data = task_data['scores']['classification_accuracy']
                        accs.append(score_data['value'])
                        stderrs.append(score_data['stderr'] or 0)

                if accs:
                    mean_acc = np.mean(accs)
                    # stderr of the mean
                    stderr = np.std(accs) / np.sqrt(len(accs))
                    ci_95 = stderr * 1.96  # 95% confidence interval
                    model_data_dict[model][n_shot] = {'acc': mean_acc, 'ci': ci_95}

    # Create grouped bar chart
    bar_width = 0.8 / len(n_shots) if n_shots else 0.2
    x_positions = np.arange(len(models_sorted))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_shots)))

    for i, n_shot in enumerate(n_shots):
        accs = []
        cis = []
        for model in models_sorted:
            if n_shot in model_data_dict[model]:
                accs.append(model_data_dict[model][n_shot]['acc'])
                cis.append(model_data_dict[model][n_shot]['ci'])
            else:
                accs.append(0)
                cis.append(0)

        offset = (i - len(n_shots)/2 + 0.5) * bar_width
        bars = ax.bar(x_positions + offset, accs, bar_width, label=f'n={n_shot}',
                     yerr=cis, capsize=2, color=colors[i], error_kw={'ecolor': 'black', 'linewidth': 1})

    # Shorten model names for x-axis
    short_model_names = [m.split('/')[-1] for m in models_sorted]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Classification Accuracy', fontsize=11)
    ax.set_title('Overall ICL Performance by Model', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(title='n_shot', fontsize=10, title_fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'icl_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: One subplot per model showing all rules - grouped by rule
    # Order models: GPT nano, GPT, Claude haiku, Claude sonnet, Gemini flash-lite, Gemini flash
    models_by_order = models_sorted  # Use same ordering as overall plot

    n_rows = (len(models_by_order) + 1) // 2  # Wrap to 2 columns
    n_cols = min(2, len(models_by_order))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 4 * n_rows))
    if len(models_by_order) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(n_shots)))

    for model_idx, model in enumerate(models_by_order):
        ax = axes[model_idx]

        # Collect data for each rule and n_shot
        rule_data_dict = {}
        for rule in rules:
            rule_data_dict[rule] = {}
            for n_shot in n_shots:
                matching_tasks = [
                    task_data for task_name, task_data in data[model].items()
                    if task_data.get('rule') == rule
                    and task_data.get('n_shot') == n_shot
                ]

                if matching_tasks:
                    accs = []
                    stderrs = []
                    for task_data in matching_tasks:
                        if 'classification_accuracy' in task_data['scores']:
                            score_data = task_data['scores']['classification_accuracy']
                            accs.append(score_data['value'])
                            stderrs.append(score_data['stderr'] or 0)

                    if accs:
                        mean_acc = np.mean(accs)
                        # Calculate stderr and 95% CI
                        if len(accs) > 1:
                            stderr = np.std(accs) / np.sqrt(len(accs))
                        else:
                            stderr = stderrs[0]
                        ci_95 = stderr * 1.96
                        rule_data_dict[rule][n_shot] = {'acc': mean_acc, 'ci': ci_95}

        # Create grouped bar chart
        bar_width = 0.8 / len(n_shots) if n_shots else 0.2
        x_positions = np.arange(len(rules))

        for i, n_shot in enumerate(n_shots):
            accs = []
            cis = []
            for rule in rules:
                if n_shot in rule_data_dict[rule]:
                    accs.append(rule_data_dict[rule][n_shot]['acc'])
                    cis.append(rule_data_dict[rule][n_shot]['ci'])
                else:
                    accs.append(0)
                    cis.append(0)

            offset = (i - len(n_shots)/2 + 0.5) * bar_width
            bars = ax.bar(x_positions + offset, accs, bar_width, label=f'n={n_shot}',
                         yerr=cis, capsize=2, color=colors[i], error_kw={'ecolor': 'black', 'linewidth': 1})

        ax.set_xticks(x_positions)
        # Only show x-axis labels on bottom row
        if model_idx >= len(models_by_order) - n_cols:
            ax.set_xticklabels(rules, rotation=45, ha='right', fontsize=11)
        else:
            ax.set_xticklabels([])

        # Only show y-axis label on left column
        if model_idx % n_cols == 0:
            ax.set_ylabel('Classification Accuracy', fontsize=12)

        ax.tick_params(axis='y', labelsize=11)
        short_model_name = model.split('/')[-1]
        ax.set_title(f'{short_model_name}', fontsize=13)

        # Only show legend on first subplot
        if model_idx == 0:
            ax.legend(title='n_shot', fontsize=11, title_fontsize=11, loc='best')

        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(len(models_by_order), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'icl_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_articulation_multi_choice(data, output_dir):
    """Plot multi-choice articulation accuracy vs classification accuracy."""
    models = list(data.keys())

    # Get unique rules
    all_rules = set()
    for model_data in data.values():
        for task_data in model_data.values():
            if task_data['articulation'] == 'multi':
                all_rules.add(task_data['rule'])

    rules = sorted(all_rules)

    # Plot 1: Overall comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for model in models:
        class_accs = []
        artic_accs = []

        for task_name, task_data in data[model].items():
            if task_data.get('articulation') != 'multi':
                continue

            scores = task_data['scores']
            if 'classification_accuracy' in scores and 'articulation_choice_accuracy' in scores:
                class_accs.append(scores['classification_accuracy']['value'])
                artic_accs.append(scores['articulation_choice_accuracy']['value'])

        if class_accs:
            ax.scatter(class_accs, artic_accs, label=model.split('/')[-1], alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy', fontsize=11)
    ax.set_ylabel('Articulation Choice Accuracy', fontsize=11)
    ax.set_title('Multi-Choice Articulation: Overall', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_multi_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: One subplot per model showing all rules
    n_rows = (len(models) + 1) // 2  # Wrap to 2 columns
    n_cols = min(2, len(models))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if len(models) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        for rule in rules:
            class_accs = []
            artic_accs = []

            for task_name, task_data in data[model].items():
                if task_data.get('articulation') != 'multi':
                    continue
                if task_data.get('rule') != rule:
                    continue

                scores = task_data['scores']
                if 'classification_accuracy' in scores and 'articulation_choice_accuracy' in scores:
                    class_accs.append(scores['classification_accuracy']['value'])
                    artic_accs.append(scores['articulation_choice_accuracy']['value'])

            if class_accs:
                ax.scatter(class_accs, artic_accs, label=rule, alpha=0.6, s=100)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Classification Accuracy', fontsize=10)
        ax.set_ylabel('Articulation Choice Accuracy', fontsize=10)
        ax.set_title(f'{model.split("/")[-1]}', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_multi_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_articulation_free_form(data, output_dir):
    """Plot free-form articulation metrics."""
    models = list(data.keys())

    # Get unique rules
    all_rules = set()
    for model_data in data.values():
        for task_data in model_data.values():
            if task_data['articulation'] == 'free':
                all_rules.add(task_data['rule'])

    rules = sorted(all_rules)

    # Define model ordering: GPT, Claude, Gemini
    model_order = {
        'openai/gpt-4.1-nano': 0,
        'openai/gpt-4.1': 1,
        'openrouter/anthropic/claude-haiku-4.5': 2,
        'openrouter/anthropic/claude-sonnet-4.5': 3,
        'openrouter/google/gemini-2.5-flash-lite': 4,
        'openrouter/google/gemini-2.5-flash': 5,
    }
    models_sorted = sorted([m for m in models if m in model_order], key=lambda x: model_order.get(x, 999))

    # ===== METRICS COMPARISON BAR CHART (n=15 only) =====
    # Plot: Accuracy vs Usefulness vs Faithfulness for each model at n=15
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Collect data for each model at n=15
    metric_names = ['Accuracy', 'Usefulness', 'Faithfulness']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green

    model_metrics = {}
    for model in models_sorted:
        model_metrics[model] = {
            'accuracy': [],
            'usefulness': [],
            'faithfulness': []
        }

        for task_name, task_data in data[model].items():
            if task_data.get('articulation') != 'free':
                continue
            if task_data.get('n_shot') != 15:
                continue

            scores = task_data['scores']
            if 'classification_accuracy' in scores:
                model_metrics[model]['accuracy'].append(scores['classification_accuracy']['value'])
            if 'articulation_usefulness' in scores:
                model_metrics[model]['usefulness'].append(scores['articulation_usefulness']['value'])
            if 'articulation_faithfulness' in scores:
                model_metrics[model]['faithfulness'].append(scores['articulation_faithfulness']['value'])

    # Compute means and 95% CIs for each metric
    x_positions = np.arange(len(models_sorted))
    bar_width = 0.25

    for i, (metric_key, metric_name, color) in enumerate(zip(['accuracy', 'usefulness', 'faithfulness'], metric_names, colors)):
        means = []
        cis = []

        for model in models_sorted:
            values = model_metrics[model][metric_key]
            if values:
                mean_val = np.mean(values)
                ci = np.std(values, ddof=1) * 1.96 / np.sqrt(len(values)) if len(values) > 1 else 0
                means.append(mean_val)
                cis.append(ci)
            else:
                means.append(0)
                cis.append(0)

        offset = (i - 1) * bar_width
        ax.bar(x_positions + offset, means, bar_width, label=metric_name,
               yerr=cis, capsize=2, color=color, error_kw={'ecolor': 'black', 'linewidth': 1})

    # Shorten model names for x-axis
    short_model_names = [m.split('/')[-1] for m in models_sorted]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(short_model_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Free-Form Articulation Metrics (n=15)', fontsize=12)
    ax.tick_params(axis='y', labelsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== SIMILARITY PLOTS =====
    # Plot 1: Overall similarity comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for model in models:
        class_accs = []
        similarity_scores = []

        for task_name, task_data in data[model].items():
            if task_data.get('articulation') != 'free':
                continue

            scores = task_data['scores']
            if 'classification_accuracy' in scores and 'articulation_similarity' in scores:
                class_accs.append(scores['classification_accuracy']['value'])
                similarity_scores.append(scores['articulation_similarity']['value'])

        if class_accs:
            ax.scatter(class_accs, similarity_scores, label=model.split('/')[-1], alpha=0.6, s=100)

    ax.set_xlabel('Classification Accuracy', fontsize=11)
    ax.set_ylabel('Articulation Similarity to True Rule', fontsize=11)
    ax.set_title('Free-Form Articulation Similarity: Overall', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_similarity_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Similarity by rule for each model
    n_rows = (len(models) + 1) // 2  # Wrap to 2 columns
    n_cols = min(2, len(models))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if len(models) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        for rule in rules:
            class_accs = []
            similarity_scores = []

            for task_name, task_data in data[model].items():
                if task_data.get('articulation') != 'free':
                    continue
                if task_data.get('rule') != rule:
                    continue

                scores = task_data['scores']
                if 'classification_accuracy' in scores and 'articulation_similarity' in scores:
                    class_accs.append(scores['classification_accuracy']['value'])
                    similarity_scores.append(scores['articulation_similarity']['value'])

            if class_accs:
                ax.scatter(class_accs, similarity_scores, label=rule, alpha=0.6, s=100)

        ax.set_xlabel('Classification Accuracy', fontsize=10)
        ax.set_ylabel('Articulation Similarity to True Rule', fontsize=10)
        ax.set_title(f'{model.split("/")[-1]}', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_similarity_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== USEFULNESS PLOTS =====
    # Plot 3: Overall usefulness comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for model in models:
        class_accs = []
        usefulness_scores = []

        for task_name, task_data in data[model].items():
            if task_data.get('articulation') != 'free':
                continue

            scores = task_data['scores']
            if 'classification_accuracy' in scores and 'articulation_usefulness' in scores:
                class_accs.append(scores['classification_accuracy']['value'])
                usefulness_scores.append(scores['articulation_usefulness']['value'])

        if class_accs:
            ax.scatter(class_accs, usefulness_scores, label=model.split('/')[-1], alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy', fontsize=11)
    ax.set_ylabel('Articulation Usefulness (Accuracy on Held-out)', fontsize=11)
    ax.set_title('Free-Form Articulation Usefulness: Overall', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_usefulness_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Usefulness by rule for each model
    n_rows = (len(models) + 1) // 2  # Wrap to 2 columns
    n_cols = min(2, len(models))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if len(models) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        for rule in rules:
            class_accs = []
            usefulness_scores = []

            for task_name, task_data in data[model].items():
                if task_data.get('articulation') != 'free':
                    continue
                if task_data.get('rule') != rule:
                    continue

                scores = task_data['scores']
                if 'classification_accuracy' in scores and 'articulation_usefulness' in scores:
                    class_accs.append(scores['classification_accuracy']['value'])
                    usefulness_scores.append(scores['articulation_usefulness']['value'])

            if class_accs:
                ax.scatter(class_accs, usefulness_scores, label=rule, alpha=0.6, s=100)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Classification Accuracy', fontsize=10)
        ax.set_ylabel('Articulation Usefulness (Accuracy on Held-out)', fontsize=10)
        ax.set_title(f'{model.split("/")[-1]}', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_usefulness_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== FAITHFULNESS PLOTS =====
    # Plot 5: Overall faithfulness comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))

    for model in models:
        class_accs = []
        faithfulness_scores = []

        for task_name, task_data in data[model].items():
            if task_data.get('articulation') != 'free':
                continue

            scores = task_data['scores']
            if 'classification_accuracy' in scores and 'articulation_faithfulness' in scores:
                class_accs.append(scores['classification_accuracy']['value'])
                faithfulness_scores.append(scores['articulation_faithfulness']['value'])

        if class_accs:
            ax.scatter(class_accs, faithfulness_scores, label=model.split('/')[-1], alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy', fontsize=11)
    ax.set_ylabel('Articulation Faithfulness (Accuracy on Generated Examples)', fontsize=11)
    ax.set_title('Free-Form Articulation Faithfulness: Overall', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_faithfulness_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 6: Faithfulness by rule for each model
    n_rows = (len(models) + 1) // 2  # Wrap to 2 columns
    n_cols = min(2, len(models))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if len(models) == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        for rule in rules:
            class_accs = []
            faithfulness_scores = []

            for task_name, task_data in data[model].items():
                if task_data.get('articulation') != 'free':
                    continue
                if task_data.get('rule') != rule:
                    continue

                scores = task_data['scores']
                if 'classification_accuracy' in scores and 'articulation_faithfulness' in scores:
                    class_accs.append(scores['classification_accuracy']['value'])
                    faithfulness_scores.append(scores['articulation_faithfulness']['value'])

            if class_accs:
                ax.scatter(class_accs, faithfulness_scores, label=rule, alpha=0.6, s=100)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('Classification Accuracy', fontsize=10)
        ax.set_ylabel('Articulation Faithfulness (Accuracy on Generated Examples)', fontsize=10)
        ax.set_title(f'{model.split("/")[-1]}', fontsize=11)
        ax.tick_params(axis='both', labelsize=9)
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    # Hide unused subplots
    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_faithfulness_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(config: Config):
    """Generate visualizations from eval logs."""
    run_dir = Path(config.run_dir)

    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        return

    # Set output directory
    if config.output_dir:
        output_dir = Path(config.output_dir)
    else:
        # Default: artifacts/plots/<run_name>
        run_name = run_dir.name
        output_dir = Path("artifacts/plots") / run_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {run_dir}")
    results = load_run_results(run_dir)
    print(f"Loaded {len(results)} eval logs")

    print("Extracting metrics...")
    data = extract_metrics(results)

    print(f"Generating plots in: {output_dir}")

    # Generate plots based on what data is available
    has_icl = any(
        'classification_accuracy' in task_data.get('scores', {})
        for model_data in data.values()
        for task_data in model_data.values()
    )

    has_multi = any(
        task_data.get('articulation') == 'multi'
        for model_data in data.values()
        for task_data in model_data.values()
    )

    has_free = any(
        task_data.get('articulation') == 'free'
        for model_data in data.values()
        for task_data in model_data.values()
    )

    if has_icl:
        print("  - Generating ICL accuracy plots...")
        plot_icl_accuracy(data, output_dir)

    if has_multi:
        print("  - Generating multi-choice articulation plots...")
        plot_articulation_multi_choice(data, output_dir)

    if has_free:
        print("  - Generating free-form articulation plots...")
        plot_articulation_free_form(data, output_dir)

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    main(Config())
