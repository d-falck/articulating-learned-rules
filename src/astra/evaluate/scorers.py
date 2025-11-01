"""Custom scorers for evaluation tasks."""

from inspect_ai.scorer import scorer, match, choice, model_graded_qa, accuracy, stderr
from inspect_ai.model import ChatMessageAssistant


@scorer(metrics=[accuracy(), stderr()])
def classification_accuracy():
    """
    Score classification accuracy using the first assistant response.

    When articulation is enabled, this extracts the first assistant message
    (the classification) and scores it using match(), ignoring later messages.
    """
    # Get the underlying match scorer
    base_scorer = match(location="any")

    async def score(state, target):
        # Find the first assistant message in the conversation
        for msg in state.messages:
            if isinstance(msg, ChatMessageAssistant):
                # Temporarily replace state.output.completion with the first assistant message
                original_completion = state.output.completion
                state.output.completion = msg.content

                # Score using the built-in match scorer
                result = await base_scorer(state, target)

                # Restore original completion
                state.output.completion = original_completion

                return result

        # If no assistant message found, delegate to base scorer
        return await base_scorer(state, target)

    return score


@scorer(metrics=[accuracy(), stderr()], name="articulation_choice_accuracy")
def articulation_choice_accuracy():
    """
    Score multiple choice articulation accuracy.

    Extracts the correct answer from metadata and scores against it.
    """
    import re
    from inspect_ai.scorer import Score, CORRECT, INCORRECT

    async def score(state, target):
        # Extract the correct multiple choice answer from metadata
        mc_correct_answer = state.metadata.get("mc_correct_answer")

        if mc_correct_answer is None:
            return Score(
                value=INCORRECT,
                explanation="No mc_correct_answer found in metadata"
            )

        # Get the model's response (final output after articulation)
        response = state.output.completion if state.output else ""

        # Extract the letter from the response (look for A, B, C, or D)
        # The choice scorer looks for uppercase letters
        pattern = r'\b([A-D])\b'
        matches = re.findall(pattern, response.upper())

        if not matches:
            return Score(
                value=INCORRECT,
                answer=response,
                explanation=f"No choice letter found in response: '{response}'"
            )

        # Take the first letter found
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
    Score free-form articulation quality.

    Thin wrapper around model_graded_qa() scorer with a more descriptive name.
    """
    return model_graded_qa()
