"""Custom solvers for evaluation tasks."""

from inspect_ai.solver import solver
from inspect_ai.model import ChatMessageUser
from astra.generate import RULES


@solver
def articulate_rule(articulation_type, mc_options=None):
    """
    Solver that asks the model to explain or identify the rule it used.

    Args:
        articulation_type: "multi" for multiple choice, "free" for free-form
        mc_options: Dict mapping letters to rule names (required for "multi")
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
            prompt = (
                "Explain what rule you used to classify this text. "
                "Describe the pattern or characteristic you identified."
            )

        state.messages.append(ChatMessageUser(content=prompt))

        return await generate(state)

    return solve
