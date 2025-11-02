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
    """Plot in-context classification accuracy vs n_shot."""
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

    # Plot 1: Overall comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for model in models:
        model_n_shots = []
        model_accs = []
        model_stderrs = []

        for n_shot in n_shots:
            # Aggregate across all rules
            matching_tasks = [
                task_data for task_name, task_data in data[model].items()
                if task_data.get('n_shot') == n_shot
            ]

            if matching_tasks:
                # Average across all matching tasks
                accs = []
                stderrs = []
                for task_data in matching_tasks:
                    if 'classification_accuracy' in task_data['scores']:
                        score_data = task_data['scores']['classification_accuracy']
                        accs.append(score_data['value'])
                        stderrs.append(score_data['stderr'] or 0)

                if accs:
                    model_n_shots.append(n_shot)
                    model_accs.append(np.mean(accs))
                    # stderr of the mean = std / sqrt(n)
                    model_stderrs.append(np.std(accs) / np.sqrt(len(accs)))

        if model_n_shots:
            ax.errorbar(model_n_shots, model_accs, yerr=model_stderrs,
                       marker='o', label=model, capsize=5)

    ax.set_xlabel('n_shot')
    ax.set_ylabel('Classification Accuracy')
    ax.set_title('Overall ICL Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'icl_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: One subplot per model showing all rules
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 6))
    if len(models) == 1:
        axes = [axes]

    for model_idx, model in enumerate(models):
        ax = axes[model_idx]

        for rule in rules:
            rule_n_shots = []
            rule_accs = []
            rule_stderrs = []

            for n_shot in n_shots:
                matching_tasks = [
                    task_data for task_name, task_data in data[model].items()
                    if task_data.get('rule') == rule
                    and task_data.get('n_shot') == n_shot
                ]

                if matching_tasks:
                    # Average across all matching tasks (usually just 1)
                    accs = []
                    stderrs = []
                    for task_data in matching_tasks:
                        if 'classification_accuracy' in task_data['scores']:
                            score_data = task_data['scores']['classification_accuracy']
                            accs.append(score_data['value'])
                            stderrs.append(score_data['stderr'] or 0)

                    if accs:
                        rule_n_shots.append(n_shot)
                        rule_accs.append(np.mean(accs))
                        if len(accs) > 1:
                            rule_stderrs.append(np.std(accs) / np.sqrt(len(accs)))
                        else:
                            rule_stderrs.append(stderrs[0])

            if rule_n_shots:
                ax.errorbar(rule_n_shots, rule_accs, yerr=rule_stderrs,
                           marker='o', label=rule, capsize=3, alpha=0.7)

        ax.set_xlabel('n_shot')
        ax.set_ylabel('Classification Accuracy')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

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
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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
            ax.scatter(class_accs, artic_accs, label=model, alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy')
    ax.set_ylabel('Articulation Choice Accuracy')
    ax.set_title('Multi-Choice Articulation: Overall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_multi_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: One subplot per model showing all rules
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

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
        ax.set_xlabel('Classification Accuracy')
        ax.set_ylabel('Articulation Choice Accuracy')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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

    # ===== SIMILARITY PLOTS =====
    # Plot 1: Overall similarity comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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
            ax.scatter(class_accs, similarity_scores, label=model, alpha=0.6, s=100)

    ax.set_xlabel('Classification Accuracy')
    ax.set_ylabel('Articulation Similarity to True Rule')
    ax.set_title('Free-Form Articulation Similarity: Overall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_similarity_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Similarity by rule for each model
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

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

        ax.set_xlabel('Classification Accuracy')
        ax.set_ylabel('Articulation Similarity to True Rule')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_similarity_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== USEFULNESS PLOTS =====
    # Plot 3: Overall usefulness comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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
            ax.scatter(class_accs, usefulness_scores, label=model, alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy')
    ax.set_ylabel('Articulation Usefulness (Accuracy on Held-out)')
    ax.set_title('Free-Form Articulation Usefulness: Overall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_usefulness_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 4: Usefulness by rule for each model
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

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
        ax.set_xlabel('Classification Accuracy')
        ax.set_ylabel('Articulation Usefulness (Accuracy on Held-out)')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_usefulness_by_rule.png', dpi=300, bbox_inches='tight')
    plt.close()

    # ===== FAITHFULNESS PLOTS =====
    # Plot 5: Overall faithfulness comparison across models
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

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
            ax.scatter(class_accs, faithfulness_scores, label=model, alpha=0.6, s=100)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('Classification Accuracy')
    ax.set_ylabel('Articulation Faithfulness (Accuracy on Generated Examples)')
    ax.set_title('Free-Form Articulation Faithfulness: Overall')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / 'articulation_free_faithfulness_overall.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 6: Faithfulness by rule for each model
    fig, axes = plt.subplots(1, len(models), figsize=(8 * len(models), 8))
    if len(models) == 1:
        axes = [axes]

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
        ax.set_xlabel('Classification Accuracy')
        ax.set_ylabel('Articulation Faithfulness (Accuracy on Generated Examples)')
        ax.set_title(f'{model}')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

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
