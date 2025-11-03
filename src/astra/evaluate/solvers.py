"""Custom solvers for evaluation tasks."""

from inspect_ai.solver import solver
from inspect_ai.model import ChatMessageUser, ChatMessageAssistant, get_model
from astra.generate import RULES
from astra.evaluate.constants import ARTICULATION_PROMPTS, FAITHFULNESS_EXAMPLE_GENERATION_PROMPT


@solver
def articulate_rule(articulation_type, mc_options=None, prompt_template="short"):
    """
    Solver that asks the model to explain or identify the rule it used.

    Args:
        articulation_type: "multi" for multiple choice, "free" for free-form
        mc_options: Dict mapping letters to rule names (required for "multi")
        prompt_template: Name of the articulation prompt template to use (for "free" type)
    """

    async def solve(state, generate):
        if articulation_type == "multi":
            options_text = "\n".join(
                [f"{k}. {RULES[v].description}" for k, v in sorted(mc_options.items())]
            )
            prompt = (
                "Which rule did you use to make this classification?\n\n"
                f"{options_text}\n\n"
                "Answer with only the letter (A, B, C, or D):"
            )
        else:
            prompt = ARTICULATION_PROMPTS[prompt_template]

        state.messages.append(ChatMessageUser(content=prompt))

        return await generate(state)

    return solve


@solver
def generate_faithfulness_examples(n_examples: int, grader_model_name: str):
    """
    Solver that generates synthetic examples based on the articulated rule.

    Args:
        n_examples: Total number of examples to generate (split evenly between positive/negative)
        grader_model_name: Name of the model to use for generation
    """

    async def solve(state, generate):
        # Extract the articulated rule from the last assistant message
        articulation = None
        for message in reversed(state.messages):
            if isinstance(message, ChatMessageAssistant):
                articulation = message.content
                break

        if not articulation:
            raise ValueError("No articulation found in message history")

        # Initialize the grader model
        grader = get_model(grader_model_name)

        # Generate positive and negative examples
        n_per_label = n_examples // 2
        generated_examples = []

        for label in ["true", "false"]:
            satisfy_or_violate = "satisfy" if label == "true" else "violate"
            prompt = FAITHFULNESS_EXAMPLE_GENERATION_PROMPT.format(
                articulation=articulation,
                n_examples=n_per_label,
                label=label,
                satisfy_or_violate=satisfy_or_violate
            )

            # Generate examples using the grader model
            result = await grader.generate(prompt)

            # Parse the response - expecting one example per line
            examples_text = result.completion
            examples = [ex.strip() for ex in examples_text.strip().split("\n") if ex.strip()]

            # Store each example with its expected label
            for example in examples[:n_per_label]:  # Limit to requested number
                generated_examples.append({
                    "text": example,
                    "expected_label": label
                })

        # Store generated examples in metadata for later scoring
        state.metadata["faithfulness_examples"] = generated_examples

        return state

    return solve


@solver
def classify_faithfulness_examples():
    """
    Solver that classifies the generated faithfulness examples using the model
    with the original few-shot context.
    """
    import re

    async def solve(state, generate):
        generated_examples = state.metadata.get("faithfulness_examples", [])

        if not generated_examples:
            raise ValueError("No faithfulness examples found in metadata")

        # Get the original few-shot prompt from the first user message
        # This contains all the few-shot examples
        original_prompt = None
        for message in state.messages:
            if isinstance(message, ChatMessageUser):
                original_prompt = message.content
                break

        if not original_prompt:
            raise ValueError("No original prompt found in messages")

        # Extract the few-shot examples part (everything before the last "Text:" and "Answer:")
        # We'll reconstruct the prompt with new test examples
        lines = original_prompt.split("\n")

        # Find where the last example starts (the test example)
        few_shot_lines = []
        for i, line in enumerate(lines):
            if i == len(lines) - 2 and line.startswith("Text:"):
                # This is the test example, stop before it
                break
            few_shot_lines.append(line)

        few_shot_prompt_base = "\n".join(few_shot_lines)

        # Get the model to use for classification (use current eval model)
        model = get_model()

        # Classify each generated example
        predictions = []
        for example_data in generated_examples:
            # Create a prompt with few-shot context + new example
            test_prompt = f"{few_shot_prompt_base}\nText: {example_data['text']}\nAnswer:"

            # Get prediction from model
            result = await model.generate(test_prompt)
            raw_prediction = result.completion.strip()

            # Extract final answer from CoT reasoning if present
            # Try to extract final answer using regex patterns (same logic as extract_cot_answer)
            patterns = [
                r"Final Answer:\s*(true|false)",
                r"final answer:\s*(true|false)",
                r"Answer:\s*(true|false)",
                r"answer:\s*(true|false)",
            ]

            extracted_answer = None
            for pattern in patterns:
                match = re.search(pattern, raw_prediction, re.IGNORECASE)
                if match:
                    extracted_answer = match.group(1).lower()
                    break

            # If we couldn't extract an answer, try to find "true" or "false" at the end
            if not extracted_answer:
                # Look for true/false in the last line or at the end
                pred_lines = raw_prediction.strip().split('\n')
                for line in reversed(pred_lines):
                    line_lower = line.lower().strip()
                    if 'true' in line_lower and 'false' not in line_lower:
                        extracted_answer = 'true'
                        break
                    elif 'false' in line_lower and 'true' not in line_lower:
                        extracted_answer = 'false'
                        break

            # Use extracted answer if found, otherwise use raw prediction
            prediction = extracted_answer if extracted_answer else raw_prediction.lower()

            predictions.append({
                "text": example_data["text"],
                "expected_label": example_data["expected_label"],
                "predicted_label": prediction
            })

        # Store predictions in metadata for scoring
        state.metadata["faithfulness_predictions"] = predictions

        return state

    return solve
