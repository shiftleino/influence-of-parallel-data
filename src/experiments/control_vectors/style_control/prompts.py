COMPARISON_PROMPT_TEMPLATE = """
[Role and Goal]
You are an impartial judge. Your goal is to evaluate which of the two continuations is written in a more joyful style based on the preceding context.

[Core Task]
Your task is to read a story context followed by two possible completions, labeled A and B. You must evaluate which completion is written in a more **joyful style** and provide your analysis in a JSON format.

A "joyful style" is characterized by language and descriptions that evoke feelings of happiness, delight, celebration, and positive excitement. When comparing the two completions, consider the following:
* **Emotional Tone:** Which text feels more upbeat, optimistic, and lighthearted?
* **Word Choice (Diction):** Which text uses more words with positive and happy connotations?
* **Character Expression:** In which text do the characters more clearly express happiness through their actions, thoughts, or dialogue (e.g., smiling, laughing, dancing, feeling triumphant)?
* **Overall Atmosphere:** Which text creates a more cheerful and celebratory mood?

[Special Considerations]
If one of the completions is gibberish, nonsensical or in different language than the context, select C as the answer.

Note that both completions are incomplete, which should not affect your evaluation.

[Input]
Context:
{context}

Completion A:
{continuation1}

Completion B:
{continuation2}

[Output Instructions]
Your output must be a single JSON object containing two required fields: `answer` and `reasoning`.

1.  `answer`: (String) The final evaluation.
    * Use **"A"** if Completion A is distinctly more joyful in style than Completion B.
    * Use **"B"** if Completion B is distinctly more joyful in style than Completion A.
    * Use **"C"** if both completions are equally joyful, if neither completion exhibits a joyful style, or if one of the completions is nonsensical or in a different language.

2.  `reasoning`: (String) A brief but clear explanation for your choice. This justification should compare both completions and explicitly reference the evaluation criteria (e.g., "Completion A uses more positive word choice").
"""
