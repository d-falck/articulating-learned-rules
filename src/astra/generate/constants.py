"""Constants for dataset generation."""

from pydantic import BaseModel


class Rule(BaseModel):
    """Definition of a classification rule."""

    description: str  # Human-readable description of what the rule checks
    true: str  # Instruction for generating text that satisfies the rule
    false: str  # Instruction for generating text that does not satisfy the rule


TOPICS = [
    # Everyday Objects
    "coffee",
    "phones",
    "books",
    "shoes",
    "clocks",
    # Common Activities
    "walking",
    "cooking",
    "sleeping",
    "driving",
    "reading",
    # Simple Concepts
    "weather",
    "music",
    "colors",
    "food",
    "money",
    # Places
    "parks",
    "stores",
    "libraries",
    "restaurants",
    "offices",
]


RULES: dict[str, Rule] = {
    "ends_with_question": Rule(
        description="The text ends with a question mark (?)",
        true="End the text with a question mark (?)",
        false="End the text with something other than a question mark",
    ),
    "contains_numbers": Rule(
        description="The text contains at least one numerical digit (0-9)",
        true="Include at least one digit (0-9)",
        false="Do not use any digits",
    ),
    "is_title_case": Rule(
        description="The text is written in Title Case (every word starts with a capital letter)",
        true="Write In Title Case Where Every Word Starts With A Capital Letter",
        false="Use normal sentence capitalization (only capitalize the first word and proper nouns)",
    ),
    "contains_quotes": Rule(
        description="The text contains at least one quotation mark",
        true="Include at least one quotation mark",
        false="Do not use any quotation marks",
    ),
    "has_many_verbs": Rule(
        description="The text contains at least 5 different verbs (action or being words)",
        true="Include at least 5 different verbs (action/being words). Verbs include: runs, jumps, eats, sleeps, thinks, is, are, was, makes, has, does, goes, sees, feels, walks, talks, etc. Example: 'Dogs run, jump, play, eat, and sleep happily' (contains 5 verbs)",
        false="Write using only nouns and adjectives - no action or being words at all. NO verbs like: is, are, was, has, runs, does, etc. Example: 'Red apples' or 'Big coffee cup'",
    ),
    "contains_hashtag": Rule(
        description="The text contains at least one hashtag symbol (#)",
        true="Include at least one hashtag symbol (#)",
        false="Do not use any hashtag symbols",
    ),
    "is_very_short": Rule(
        description="The text is very short (fewer than 7 words)",
        true="Write a very short sentence with fewer than 7 words. Example: 'Coffee tastes really great today.' (5 words)",
        false="Write a longer sentence with more than 20 words. Add details, descriptions, and complexity to reach the word count.",
    ),
    "is_first_person": Rule(
        description="The text is written in first person (uses 'I', 'me', 'my', 'we', 'our', or 'us')",
        true="Write in first person using words like 'I', 'me', 'my', 'we', 'our', or 'us'",
        false="Write in the third person; do not use any first person pronouns (no 'I', 'me', 'my', 'we', 'our', 'us')",
    ),
    "has_repeated_word": Rule(
        description="The text contains a word repeated 4-5 times in a row",
        true="Include one word repeated 4-5 times in a row (like 'the the the the')",
        false="Do not repeat any word more than once in a row",
    ),
    "contains_rhyme": Rule(
        description="The text contains multiple words that rhyme with each other",
        true="Include as many words as possible that rhyme (like 'cat' and 'hat' or 'day' and 'way')",
        false="Do not include any rhyming words",
    ),
}
