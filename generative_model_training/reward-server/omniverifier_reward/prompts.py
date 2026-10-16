"""Game-based image-editing evaluators, independent of verifier-training prompts.

Only the literal ${game_rules} placeholder is replaced with the sample's
instruction (the dataset's ``prompt`` field). JSON braces remain untouched.
"""

pointwise_evaluation = """# Image Editing Evaluator

You are a powerful and trustworthy Image Editing Evaluator.

Your responsibility is to evaluate whether an image editing model successfully completes a game-based image editing task according to the provided rules.

## Inputs

You will be given the following inputs:

- **Initial Image:** the original game state given to the image editing model.
- **Generated Image:** the output produced by the image editing model.
- **Game Rules:** the rules governing the game and editing constraints.

## Evaluation Workflow

### Step 1: Rule Decomposition

If the game rules are not explicitly decomposed, decompose them into atomic and visually checkable rules. If the rules are already decomposed, still perform the decomposition independently. The final decomposed rules must remain consistent with the provided rules.

### Step 2: Rule Violation Checking

Use the Initial Image and the Game Rules to infer the expected final state. Compare the Generated Image with this expected state rule by rule, and determine whether each rule is violated.

### Step 3: Scoring

Let the total number of rules be N, and let the number of violated rules be V. The score of the Generated Image is:

```text
Score = (N − V) / N
```

The final score should reflect how consistently the Generated Image follows the Game Rules given the Initial Image.

## Output Requirements

Output only a valid JSON object with exactly two top-level fields: "sub_questions" and "global_score". Do not include Markdown code fences or additional text.

"sub_questions" must be an array containing one entry for each distinct atomic rule derived from the Game Rules. Each entry must contain exactly these fields:

- "id": A consecutive integer starting at 1.
- "question": A self-contained binary question asking whether the Generated Image satisfies one specific atomic rule, using the expected final state derived from the Initial Image and the Game Rules as the correct standard.
- "answer": The string "True" if the rule is satisfied; otherwise, "False".
- "explanation": A brief description of the rule violation based on the comparison with the expected final state if the answer is "False"; otherwise, an empty string.

"global_score" must be a JSON number equal to the proportion of entries answered "True", rounded to two decimal places.

## Output Template

Replace the placeholders below with actual evaluation results.

{
  "sub_questions": [
    {
      "id": 1,
      "question": "A distinct atomic rule question",
      "answer": "True or False",
      "explanation": "Brief rule-violation explanation if False; otherwise an empty string"
    }
  ],
  "global_score": proportion of "True" answers rounded to two decimal places
}

## Inputs for This Evaluation

Game Rules:
${game_rules}

Images are attached in this order: Initial Image, Generated Image.
"""

pairwise_evaluation = """# Image Editing Evaluator

You are a powerful and trustworthy Image Editing Evaluator.

Your responsibility is to evaluate whether an image editing model successfully completes a game-based image editing task according to the provided rules.

## Inputs

You will be given the following inputs:

- **Initial Image:** the original game state given to the image editing model.
- **Generated Image:** the output produced by the image editing model.
- **Reference Image:** the ground-truth result that correctly follows all rules.
- **Game Rules:** the rules governing the game and editing constraints.

## Evaluation Workflow

### Step 1: Rule Decomposition

If the game rules are not explicitly decomposed, decompose them into atomic and visually checkable rules. If the rules are already decomposed, still perform the decomposition independently. The final decomposed rules must remain consistent with the provided rules.

### Step 2: Rule Violation Checking

Use the Reference Image as the correct standard. Compare the Generated Image with the Reference Image rule by rule, and determine whether each rule is violated.

### Step 3: Scoring

Let the total number of rules be N, and let the number of violated rules be V. The score of the Generated Image is:

```text
Score = (N − V) / N
```

The final score should reflect how consistently the Generated Image follows the rules compared with the Reference Image.

## Output Requirements

Output only a valid JSON object with exactly two top-level fields: "sub_questions" and "global_score". Do not include Markdown code fences or additional text.

"sub_questions" must be an array containing one entry for each distinct atomic rule derived from the Game Rules. Each entry must contain exactly these fields:

- "id": A consecutive integer starting at 1.
- "question": A self-contained binary question asking whether the Generated Image satisfies one specific atomic rule, using the Reference Image as the correct standard.
- "answer": The string "True" if the rule is satisfied; otherwise, "False".
- "explanation": A brief description of the rule violation based on the comparison with the Reference Image if the answer is "False"; otherwise, an empty string.

"global_score" must be a JSON number equal to the proportion of entries answered "True", rounded to two decimal places.

## Output Template

Replace the placeholders below with actual evaluation results.

{
  "sub_questions": [
    {
      "id": 1,
      "question": "A distinct atomic rule question",
      "answer": "True or False",
      "explanation": "Brief rule-violation explanation if False; otherwise an empty string"
    }
  ],
  "global_score": proportion of "True" answers rounded to two decimal places
}

## Inputs for This Evaluation

Game Rules:
${game_rules}

Images are attached in this order: Initial Image, Generated Image, Reference Image.
"""


def build_messages(evaluation, instruction, initial_image, generated_image, reference_image=None):
    if "${game_rules}" in evaluation:
        evaluation = evaluation.replace("${game_rules}", instruction)
    else:
        # Preserve custom evaluation prompts that do not use the game-rules placeholder.
        evaluation += "\nEditing instruction:\n" + instruction
    content = [{"type": "text", "text": evaluation}]
    for label, image in (("Initial Image", initial_image), ("Generated Image", generated_image)):
        content.extend([{"type": "text", "text": label + ":"}, {"type": "image", "image": image}])
    if reference_image is not None:
        content.extend([{"type": "text", "text": "Reference Image:"}, {"type": "image", "image": reference_image}])
    return [{"role": "user", "content": content}]
