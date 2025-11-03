#!/usr/bin/env python3
"""
Fix faithfulness scores in CoT evaluation logs by re-extracting answers from CoT reasoning.
"""

import re
from pathlib import Path
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.scorer import Score, accuracy, stderr

# Directory with CoT evaluation logs
LOG_DIR = Path("artifacts/logs/articulation_free_short_cot")


def extract_cot_answer(text):
    """Extract final answer from CoT reasoning text."""
    patterns = [
        r"Final Answer:\s*(true|false)",
        r"final answer:\s*(true|false)",
        r"Answer:\s*(true|false)",
        r"answer:\s*(true|false)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).lower()

    # Fallback: look for true/false at the end
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line_lower = line.lower().strip()
        if 'true' in line_lower and 'false' not in line_lower:
            return 'true'
        elif 'false' in line_lower and 'true' not in line_lower:
            return 'false'

    return None


def recompute_faithfulness(log):
    """Recompute faithfulness scores for a log."""
    try:
        if not log.samples:
            return log, None

        # Check if this log has faithfulness predictions
        has_faithfulness = False
        for sample in log.samples:
            if sample.metadata and 'faithfulness_predictions' in sample.metadata:
                has_faithfulness = True
                break

        if not has_faithfulness:
            return log, None

        # Recompute faithfulness for each sample
        total_correct = 0
        total_count = 0

        for sample in log.samples:
            if not sample.metadata or 'faithfulness_predictions' not in sample.metadata:
                continue

            predictions = sample.metadata['faithfulness_predictions']

            for pred in predictions:
                expected = pred['expected_label']
                raw_predicted = pred['predicted_label']

                # Re-extract the final answer from CoT reasoning
                extracted = extract_cot_answer(raw_predicted)
                predicted = extracted if extracted else raw_predicted.lower()

                # Check if it matches
                pattern = r'\b(true|false)\b'
                matches = re.findall(pattern, predicted.lower())
                if matches:
                    predicted_label = matches[0]
                    if predicted_label == expected:
                        total_correct += 1
                total_count += 1

        if total_count == 0:
            return log, None

        # Compute new accuracy
        new_accuracy = total_correct / total_count

        print(f"  Recomputed faithfulness: {new_accuracy:.4f} ({total_correct}/{total_count})")

        # Don't modify the log - just return it with the computed values for display
        old_value = None
        if log.results and log.results.scores:
            for score in log.results.scores:
                if score.name == 'articulation_faithfulness':
                    old_value = score.value
                    break

        return log, (old_value, new_accuracy)

    except Exception as e:
        print(f"  Exception in recompute_faithfulness: {e}")
        import traceback
        traceback.print_exc()
        return log, None


def main():
    """Recompute and display faithfulness scores from CoT logs."""
    eval_files = list(LOG_DIR.glob('*.eval'))

    print(f"Found {len(eval_files)} eval files in {LOG_DIR}")
    print()
    print("Note: This script only DISPLAYS recomputed faithfulness scores.")
    print("To actually fix the scores, the evaluation would need to be re-run.")
    print()

    results = []
    for eval_file in eval_files:
        try:
            log = read_eval_log(str(eval_file))

            # Recompute faithfulness
            _, change = recompute_faithfulness(log)

            if change:
                old_val, new_val = change
                print(f"{eval_file.name}")
                print(f"  Old faithfulness: {old_val:.4f}")
                print(f"  New faithfulness: {new_val:.4f}")
                print(f"  Change: {new_val - old_val:+.4f}")
                print()
                results.append((eval_file.name, old_val, new_val))
            else:
                print(f"{eval_file.name}: No faithfulness score found (skipped)")
                print()

        except Exception as e:
            print(f"Error processing {eval_file.name}: {e}")
            print()

    print()
    print(f"Analyzed {len(results)} files with faithfulness scores")
    if results:
        avg_old = sum(old for _, old, _ in results) / len(results)
        avg_new = sum(new for _, _, new in results) / len(results)
        print(f"Average old faithfulness: {avg_old:.4f}")
        print(f"Average new faithfulness: {avg_new:.4f}")
        print(f"Average improvement: {avg_new - avg_old:+.4f}")


if __name__ == "__main__":
    main()
