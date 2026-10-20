"""Exact numeric rewards for GSM8K. Model outputs are never executed."""
import re
from fractions import Fraction

NUMBER = r"[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?"
FINAL = re.compile(r"<answer>\s*(" + NUMBER + r")\s*</answer>\s*\Z")


def number_value(text):
    text = str(text).strip()
    if len(text) > 80 or not re.fullmatch(NUMBER, text):
        return None
    try:
        return Fraction(text.replace(',', ''))
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(text):
    # Accept exactly one terminal answer, not a list of guesses or an earlier number.
    if text.count('<answer>') != 1 or text.count('</answer>') != 1:
        return None
    match = FINAL.search(text)
    return number_value(match.group(1)) if match else None


def score(text, solution):
    predicted, target = extract_answer(text), number_value(solution)
    return float(predicted is not None and target is not None and predicted == target)


def correctness_reward(completions, solution, **kwargs):
    texts = [c if isinstance(c, str) else c[0]['content'] for c in completions]
    return [score(text, target) for text, target in zip(texts, solution, strict=True)]
