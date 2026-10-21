"""Answer scoring: token F1 for short free-form answers and accuracy for multiple-choice answers."""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Optional, Sequence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = frozenset(string.punctuation) | {"‘", "’", "“", "”"}


def normalize_answer(text: str) -> str:
    """Lowercase, remove punctuation and the articles a, an and the, and collapse whitespace."""
    text = "".join(ch for ch in text.lower() if ch not in _PUNCT)
    return " ".join(_ARTICLES.sub(" ", text).split())


def first_line(text: str) -> str:
    """First nonempty line of a reply, stripped."""
    for line in text.strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_answer(prediction).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def f1_score(reply: str, golds: Sequence[str]) -> float:
    """Highest token F1 of the reply's first line against the reference answers."""
    line = first_line(reply)
    return max((token_f1(line, g) for g in golds), default=0.0)


def exact_match(reply: str, golds: Sequence[str]) -> float:
    line = normalize_answer(first_line(reply))
    return float(any(line == normalize_answer(g) for g in golds))


def parse_letter(text: str, n_options: int) -> Optional[str]:
    """Option letter of a multiple-choice reply, or None.

    The letter is looked for at the start of the reply, then after "answer", "option" or "choice", and finally in
    parentheses or as an option label at the start of a line.
    """
    valid = string.ascii_uppercase[:n_options]
    t = text.strip()
    m = re.match(r"^\W*\(?([A-Z])\)?(?:[\.\):,\s]|$)", t)
    if m and m.group(1) in valid:
        return m.group(1)
    m = re.search(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([A-Z])\b", t, re.I)
    if m and m.group(1).upper() in valid:
        return m.group(1).upper()
    for m in re.finditer(r"\(([A-Z])\)|(?:^|\n)\s*([A-Z])[\.\)]\s", t):
        letter = m.group(1) or m.group(2)
        if letter in valid:
            return letter
    return None


def score(reply: str, example) -> float:
    """Score in [0, 1] of a reply: accuracy for a multiple-choice question, token F1 otherwise."""
    if example.is_multiple_choice:
        return float(parse_letter(reply, len(example.options)) == example.gold_letter)
    return f1_score(reply, example.golds)
