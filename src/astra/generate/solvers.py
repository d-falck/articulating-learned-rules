"""Common solvers for dataset generation."""

from inspect_ai.solver import solver
from inspect_ai.model import ChatMessageUser


@solver
def self_check():
    """
    Solver that asks the model to verify its answer meets requirements and correct if needed.
    """

    async def solve(state, generate):
        # Build the self-check prompt
        check_prompt = (
            "Review your previous response and check if it follows ALL the requirements.\n\n"
            "Important: DO NOT explain your reasoning or show your work. "
            "DO NOT include phrases like 'Let me check' or 'Corrected version:'. "
            "ONLY output the final sentence(s) and nothing else.\n\n"
            "If your response satisfies all requirements, output the exact same text. "
            "If it doesn't, output ONLY the corrected sentence(s).\n\n"
            "Output format: Just the sentence(s), no explanations."
        )

        # Append the check prompt to the conversation using ChatMessageUser
        state.messages.append(ChatMessageUser(content=check_prompt))

        # Generate the corrected/verified response
        return await generate(state)

    return solve
