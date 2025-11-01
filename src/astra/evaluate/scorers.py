"""Custom scorers for evaluation tasks."""

from inspect_ai.scorer import scorer, match, choice, accuracy, stderr
from inspect_ai.model import ChatMessageAssistant


@scorer(metrics=[accuracy(), stderr()])
def classification_accuracy():
    """
    Score classification accuracy using the first assistant response.

    When articulation is enabled, this extracts the first assistant message
    (the classification) and scores it using match(), ignoring later messages.
    """
    base_scorer = match(location="any")

    async def score(state, target):
        for msg in state.messages:
            if isinstance(msg, ChatMessageAssistant):
                original_completion = state.output.completion
                state.output.completion = msg.content

                result = await base_scorer(state, target)

                state.output.completion = original_completion

                return result

        return await base_scorer(state, target)

    return score


@scorer(metrics=[accuracy(), stderr()], name="articulation_choice_accuracy")
def articulation_choice_accuracy():
    """
    Score multiple choice articulation accuracy.

    Gets the correct answer from metadata and checks if the model selected it.
    """
    import re
    from inspect_ai.scorer import Score, CORRECT, INCORRECT

    async def score(state, target):
        mc_correct_answer = state.metadata.get("mc_correct_answer")

        if mc_correct_answer is None:
            return Score(
                value=INCORRECT,
                explanation="No mc_correct_answer in metadata"
            )

        response = state.output.completion if state.output else ""

        pattern = r'\b([A-D])\b'
        matches = re.findall(pattern, response.upper())

        if not matches:
            return Score(
                value=INCORRECT,
                answer=response,
                explanation=f"No choice letter found in response: '{response}'"
            )

        model_answer = matches[0]
        is_correct = model_answer == mc_correct_answer.upper()

        return Score(
            value=CORRECT if is_correct else INCORRECT,
            answer=model_answer,
            explanation=f"Expected: {mc_correct_answer}, Got: {model_answer}"
        )

    return score


@scorer(metrics=[accuracy(), stderr()], name="articulation_quality")
def articulation_quality():
    """
    Score free-form articulation quality on a scale from 0 to 1.

    Uses model grading to rate how well the explanation matches the true rule description.
    """
    from textwrap import dedent
    from inspect_ai.scorer import Score
    from inspect_ai.model import get_model

    async def score(state, target):
        rule_description = state.metadata.get("rule_description", "")
        model_explanation = state.output.completion if state.output else ""

        grading_prompt = dedent(f"""
            Rate how well this explanation captures the classification rule.

            True rule: {rule_description}

            Model's explanation: {model_explanation}

            Provide:
            1. A score from 0.0 to 1.0 where:
               - 1.0: Perfect match - captures the exact rule
               - 0.7-0.9: Good - captures the main idea with minor differences
               - 0.4-0.6: Partial - identifies some aspects but misses key details
               - 0.1-0.3: Poor - identifies wrong pattern or vague description
               - 0.0: Completely wrong or unrelated

            2. A brief explanation of your rating

            Format your response as:
            Score: [number]
            Explanation: [brief explanation]
        """).strip()

        grader_model = get_model()
        result = await grader_model.generate(grading_prompt)
        grader_response = result.completion.strip()

        try:
            score_line = [line for line in grader_response.split('\n') if line.startswith('Score:')][0]
            score_value = float(score_line.split(':')[1].strip())
            score_value = max(0.0, min(1.0, score_value))
        except (ValueError, AttributeError, IndexError):
            score_value = 0.0

        try:
            explanation_line = [line for line in grader_response.split('\n') if line.startswith('Explanation:')][0]
            explanation = explanation_line.split(':', 1)[1].strip()
        except IndexError:
            explanation = grader_response

        return Score(
            value=score_value,
            answer=model_explanation,
            explanation=explanation
        )

    return score
