FLUENCY_PROMPT_TEMPLATE = """
[Role and Goal]
You are a native-level Finnish proofreader and an impartial judge. Your goal is to evaluate the Finnish usage of a text snippet based on its Finnish grammatical correctness and Finnish fluency.

[Core Task]
Select the most appropriate category that describes whether the text is grammatically correct and fluent in Finnish.

For example, if the text is grammatically correct and idiomatic, it would be considered fluent. If it is in different language or contains significant grammatical errors or unnatural phrasing, it would be considered less fluent.

[Evaluation Categories]
  1. The text is grammatically malformed to the point of being incomprehensible (gibberish), is not in Finnish, or contains only a few Finnish words in a sea of another language.
  2. The text is understandable and mostly in Finnish, but contains many significant grammatical errors or words from other languages. Grammatical errors may include fundamentally wrong verb conjugations or noun inflections, severe phrasing issues, or highly repetitive words or phrases.
  3. The text is fully in Finnish and grammatically correct for the most part, but contains minor, non-critical errors. Examples include unnatural phrasing that suggests a literal translation from another language (calque), occasional incorrect word inflections, or awkward word choices.
  4. The text is fully in Finnish and grammatically perfect, idiomatic, and reads as if written by a native Finnish speaker. All word choices, inflections, and structures are natural.

[Special Instructions]
Note the following things in your evaluation:
- The text is truncated so do not consider the fact that the text is not complete / might end abruptly in your evaluation.
- The text may be completely in different language than Finnish, making category 1 the only applicable one.

[Input]
{continuation}[truncated]

[Output Instructions]
Provide your answer in a single JSON object with keys, "reasoning" and "answer", where "reasoning" is a short explanation of your evaluation and "answer" is the number of the category which matches best with your evaluation for the text.

Format:
{{"reasoning": "Your short reasoning here", "answer": number}}

"""
