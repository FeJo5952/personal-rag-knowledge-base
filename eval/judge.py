"""LLM-as-judge: a local model grades a generated answer against the expected one."""

import json

import ollama

from app import config

GRADE_PROMPT = """You are grading an AI assistant's answer to a question about someone's notes.

Question: {question}
Expected answer: {expected}
Assistant's answer: {answer}

Grade the assistant's answer strictly:
- "correct": contains ALL key facts of the expected answer and nothing that contradicts it
- "partial": some key facts are right, but at least one is missing or wrong
- "wrong": the key facts are missing or wrong
Rules:
- If the assistant says it couldn't find the answer, the grade is "wrong".
- Any number, name or formula that differs from the expected answer means at most "partial".
- Judge only against the expected answer, not your own knowledge.

Reply with JSON only: {{"grade": "correct" | "partial" | "wrong"}}"""

REFUSAL_PROMPT = """The question below is NOT answered anywhere in the user's notes.
A good assistant should say it couldn't find the answer instead of making one up.

Question: {question}
Assistant's answer: {answer}

Did the assistant say it couldn't find the answer (and not invent one)?
Reply with JSON only: {{"grade": "correct" | "wrong"}}"""

SCORES = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def grade(question, expected, answer, not_in_notes=False):
    """Return 1.0, 0.5 or 0.0."""
    if not_in_notes:
        prompt = REFUSAL_PROMPT.format(question=question, answer=answer)
    else:
        prompt = GRADE_PROMPT.format(question=question, expected=expected, answer=answer)
    response = ollama.chat(
        model=config.JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",                 # forces the model to reply with valid JSON
        options={"temperature": 0},    # same input -> same grade
    )
    try:
        return SCORES[json.loads(response["message"]["content"])["grade"]]
    except (KeyError, json.JSONDecodeError):
        return 0.0
