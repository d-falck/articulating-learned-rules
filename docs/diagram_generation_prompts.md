# Mermaid Diagram Generation Prompts

This file contains three prompts for generating mermaid diagrams to illustrate the key methodologies in our articulation evaluation research. Each prompt is designed to create a visually appealing, easy-to-understand diagram suitable for a research paper.

---

## Prompt 1: Data Generation Process Diagram

**Context**: We need a mermaid diagram that illustrates how we generate synthetic classification datasets using LLMs.

**Your task**: Create a fun, colorful, and understandable mermaid flowchart diagram that shows the data generation process. The diagram should:

1. Show both generation approaches (isolated datasets and combined datasets) as parallel paths
2. Include the key steps: prompt creation, LLM generation, self-check validation, and dataset saving
3. Highlight the 10 classification rules being used
4. Show example inputs and outputs at key stages
5. Use colors and visual styling to make it engaging
6. Include icons or emojis where appropriate to make it more accessible

**Key information to include**:

**Starting Point**:
- Input: 10 classification rules (ends_with_question, contains_numbers, is_title_case, contains_quotes, has_many_verbs, contains_hashtag, is_very_short, is_first_person, has_repeated_word, contains_rhyme)
- Input: 20 topics (coffee, phones, books, weather, music, etc.)

**Two Parallel Paths**:

*Path 1: Isolated Dataset Generation*
- For each rule: generate 500 positive + 500 negative samples
- One dataset file per rule (10 files total)
- Output: 10,000 total samples

*Path 2: Combined Dataset Generation*
- Generate samples with random combinations of all 10 rules
- One unified dataset file
- Output: 1,000 samples with 10 binary labels each

**Shared Generation Steps** (both paths):
1. Select random topic
2. Create prompt with topic + rule instructions
3. Send to LLM (Claude Sonnet 4.5)
4. LLM generates text following rules
5. Self-check: Model validates its output
6. Save to JSONL file

**Output Format Examples**:
- Isolated: `{"text": "Do you like coffee?", "ends_with_question": true, "topic": "coffee"}`
- Combined: `{"text": "...", "ends_with_question": true, "contains_numbers": false, ...}`

**Styling suggestions**:
- Use different colors for the two generation paths
- Add a legend showing what the 10 rules are
- Make decision points (branching) clear
- Show data flowing from top to bottom
- Include small example texts at generation steps

Please create a mermaid diagram (flowchart or graph style) that captures this process in a visually appealing and academically appropriate way.

---

## Prompt 2: Articulation Similarity and Usefulness Measurement Diagram

**Context**: We need a mermaid diagram that illustrates how we measure the quality of AI model explanations using two complementary metrics: similarity (semantic match to true rule) and usefulness (practical applicability).

**Your task**: Create a fun, colorful, and understandable mermaid diagram that shows both measurement processes. The diagram should:

1. Show the evaluation flow starting from few-shot classification
2. Illustrate both similarity scoring and usefulness scoring as parallel processes
3. Show the grader model's role in both metrics
4. Include example prompts, explanations, and scores
5. Use colors to distinguish between different components (evaluation model, grader model, test data)
6. Make the distinction between the two metrics clear

**Key information to include**:

**Starting Context**:
- Evaluation model has few-shot examples
- Model classifies a test example
- Model articulates the rule it used

**Similarity Measurement Process**:
1. Input: Model's articulation + True rule description
2. Grader model receives both
3. Grading prompt: "Rate how well this explanation captures the rule (0.0-1.0)"
4. Grader provides:
   - Score (1.0 = perfect, 0.7-0.9 = good, 0.4-0.6 = partial, 0.1-0.3 = poor, 0.0 = wrong)
   - Explanation of rating
5. Output: Similarity score (0.0-1.0)

**Example for similarity**:
- True rule: "The text ends with a question mark (?)"
- Model articulation: "The text ends with a question mark"
- Score: 1.0 (perfect match)

**Usefulness Measurement Process**:
1. Input: Model's articulation + 10 held-out test examples
2. For each held-out example:
   - Grader receives: articulated rule + example text
   - Classification prompt: "Given this rule: {articulation}, classify this text as true/false"
   - Grader predicts: true or false
   - Compare to ground truth label
3. Count correct predictions
4. Output: Usefulness score = (correct / 10)

**Example for usefulness**:
- Model articulation: "The text ends with a question mark"
- Test example: "Do you like coffee?"
- Grader prediction: true
- Ground truth: true
- Result: Correct! (contributes to accuracy)

**Key Design Elements**:
- Show the model being evaluated in one color
- Show the grader model in another color
- Use arrows to show data flow
- Include a small table or legend explaining the score ranges
- Show that similarity is holistic (1 grader call) while usefulness is aggregated (10 grader calls)

**Articulation Prompt Variations** (show these as options):
- Short: "Explain the rule in one sentence"
- Medium: "Explain the rule in 1-2 sentences"
- Long: "Write a short paragraph"

Please create a mermaid diagram (flowchart, sequence diagram, or graph) that illustrates these two complementary measurement approaches in a clear and engaging way.

---

## Prompt 3: Articulation Faithfulness Measurement Diagram

**Context**: We need a mermaid diagram that illustrates how we test whether a model's actual behavior is faithful to (consistent with) the explanation it provided. This is the "do you practice what you preach" test.

**Your task**: Create a fun, colorful, and understandable mermaid diagram that shows the faithfulness measurement process. The diagram should:

1. Show the complete 4-stage pipeline from articulation to scoring
2. Highlight the circular/feedback nature: model explains, we generate tests based on that explanation, then test the model
3. Show the role of both the evaluation model and the grader model
4. Include concrete examples at each stage
5. Use visual styling to emphasize the "faithfulness" concept (consistency between words and actions)
6. Make it clear how this differs from similarity and usefulness

**Key information to include**:

**The 4 Stages**:

**Stage 1: Model Articulates Rule**
- Input: Few-shot examples + test case
- Model classifies the test case
- Model explains: "The text ends with a question mark"
- This articulation is stored

**Stage 2: Generate Synthetic Test Cases**
- Input: Model's articulation
- Grader model (GPT-4o) generates edge cases:
  - 5 examples that SHOULD match the rule (positive)
    - Example: "Do you like coffee?"
    - Example: "What time is it?"
  - 5 examples that SHOULD NOT match the rule (negative)
    - Example: "I like coffee"
    - Example: "What time is it"
- Focus: boundary cases, edge cases, tricky examples
- Output: 10 synthetic test examples with expected labels

**Stage 3: Model Classifies Generated Examples**
- Input: Original few-shot context + each synthetic example
- Evaluation model classifies each of the 10 examples (using same few-shot context as before)
- Stores predictions for each

**Stage 4: Calculate Faithfulness Score**
- For each synthetic example:
  - Compare model's prediction to expected label
  - If they match: count as correct
- Faithfulness score = (correct predictions / 10)
- High score = model's behavior matches its explanation
- Low score = model's actions don't match its words

**Complete Example Flow**:
1. Model articulates: "The text ends with a question mark"
2. Grader generates:
   - Positive: "Do you like coffee?" (expected: true)
   - Negative: "I like coffee" (expected: false)
3. Model classifies:
   - "Do you like coffee?" → predicts: true ✓ (correct!)
   - "I like coffee" → predicts: false ✓ (correct!)
4. Score: 10/10 = 1.0 (perfect faithfulness!)

**Alternative Example (Low Faithfulness)**:
1. Model articulates: "The text contains informal language"
2. Grader generates:
   - Positive: "Yo, what's up?" (expected: true)
   - Negative: "Good morning, sir." (expected: false)
3. Model classifies:
   - "Yo, what's up?" → predicts: false ✗ (wrong!)
   - "Good morning, sir." → predicts: true ✗ (wrong!)
4. Score: 0/10 = 0.0 (poor faithfulness - model says one thing, does another)

**Visual Concepts to Emphasize**:
- The circular/feedback loop: articulation → generation → classification → scoring
- Two different models working together: evaluation model (being tested) and grader model (generating tests)
- The concept of "faithfulness" as consistency
- Use check marks (✓) and X marks (✗) to show correct/incorrect predictions
- Color code: expected labels vs. predicted labels

**Key Distinction from Other Metrics**:
- Similarity: Does the explanation match the TRUE rule?
- Usefulness: Can the explanation classify HELD-OUT examples correctly?
- Faithfulness: Does the model's BEHAVIOR match its own explanation?

Please create a mermaid diagram (flowchart, sequence diagram, or state diagram) that captures this multi-stage faithfulness evaluation process in a visually compelling and academically rigorous way. Consider using a circular or cyclical layout to emphasize the feedback loop nature of this test.

---

## General Styling Guidelines for All Diagrams

1. **Colors**: Use a consistent color palette across all three diagrams
   - Model being evaluated: blue/teal
   - Grader model: orange/amber
   - Data/examples: green
   - Scores/metrics: purple/pink
   - Processes/actions: gray

2. **Clarity**:
   - Keep text concise but informative
   - Use consistent terminology across diagrams
   - Make arrows show clear direction of data flow

3. **Academic Appropriateness**:
   - Professional but not boring
   - Use emojis sparingly (only where they add clarity)
   - Suitable for inclusion in a research paper

4. **Mermaid Features to Consider**:
   - Flowchart (TB or LR direction)
   - Sequence diagrams for showing model interactions
   - State diagrams for showing stages
   - Subgraphs for grouping related components
   - Custom styling with classDef

5. **Accessibility**:
   - Don't rely solely on color to convey information
   - Use shapes, labels, and text to reinforce meaning
   - Ensure text is readable at standard paper sizes

---

## Output Format

For each diagram, please provide:
1. The complete mermaid code
2. A brief description of design choices made
3. Any notes on how to customize or extend the diagram

Good luck creating these visualizations!
