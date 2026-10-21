"""The support test and the memory veto.

Both read surface forms only, so they need no model. A short free-form answer is supported when its normalized form
occurs in the normalized passage as a contiguous sequence of whole tokens. A multiple-choice answer is supported when the
text of its option is the only option text that occurs in the passage. The memory veto compares the closed-book answer
with the passage and decides whether repairs that repeat it are rejected.
"""

from __future__ import annotations

import re
import string
import unicodedata
from typing import Dict, FrozenSet, Optional

from .example import LETTERS, Example
from .scoring import parse_letter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_ASCII_PUNCT = frozenset(string.punctuation)

#: Words that do not count when the share of an option's words found in the passage is measured.
FUNCTION_WORDS: FrozenSet[str] = frozenset(
    "a an the of and or to in on at for by with from as is are was were be been it its this that these those "
    "into than then which who whom what when where how".split()
)


def _split_mark(ch: str) -> str:
    if unicodedata.category(ch)[0] not in "PS":
        return ch
    # ASCII punctuation separates words; any other mark or symbol is a token of its own, so it must match itself
    return " " if ch in _ASCII_PUNCT else f" {ch} "


def normalize(text: str) -> str:
    """Lowercase, replace ASCII punctuation with spaces, keep other punctuation and symbols as separate tokens,
    remove the articles a, an and the, and collapse whitespace."""
    text = "".join(_split_mark(ch) for ch in text.lower())
    return " ".join(_ARTICLES.sub(" ", text).split())


def answer_line(reply: str) -> str:
    """The first line of a reply, which is the part that the support test reads."""
    return reply.strip().split("\n")[0]


def occurs(text: str, normalized_passage: str) -> bool:
    """True when ``text`` normalizes to a nonempty token sequence that occurs in the passage as whole tokens."""
    t = normalize(text)
    return bool(t) and f" {t} " in f" {normalized_passage} "


def supported(answer: str, passage: str) -> bool:
    """Support test of a short free-form answer against a passage (yes and no excluded, see `PassageCheck`)."""
    return occurs(answer_line(answer), normalize(passage))


def content_recall(text: str, normalized_passage: str) -> float:
    """Share of the normalized words of ``text``, function words left out, that occur in the passage."""
    words = [w for w in normalize(text).split() if w not in FUNCTION_WORDS]
    if not words:
        return 0.0
    vocab = set(normalized_passage.split())
    return sum(w in vocab for w in words) / len(words)


class PassageCheck:
    """Support test and memory veto for one example, with the passage normalized once."""

    def __init__(self, example: Example):
        self.example = example
        self.passage = normalize(example.passage)
        self.multiple_choice = example.is_multiple_choice
        if self.multiple_choice:
            #: letters of the options whose text occurs in the passage
            self.occurring = frozenset(c for c, t in zip(LETTERS, example.options) if occurs(t, self.passage))
            #: letters a supported answer can name: the only occurring option, or none
            self.acceptable = self.occurring if len(self.occurring) == 1 else frozenset()

    def key(self, reply: str) -> str:
        """What two replies must share to give the same answer: the option letter, or the normalized first line.
        An empty key means that the reply states no answer."""
        if self.multiple_choice:
            return parse_letter(reply, len(self.example.options)) or ""
        return normalize(answer_line(reply))

    def supports(self, reply: str) -> bool:
        """The support test. A free-form answer that normalizes to yes or no passes it, since a passage rarely
        contains it as a span."""
        if self.multiple_choice:
            return self.key(reply) in self.acceptable
        a = self.key(reply)
        return bool(a) and (f" {a} " in f" {self.passage} " or a in ("yes", "no"))

    def option_scores(self) -> Dict[str, float]:
        """Score of each option: 2 when its text occurs in the passage, otherwise the share of its words, function
        words left out, that occur in the passage."""
        return {c: 2.0 if c in self.occurring else content_recall(t, self.passage)
                for c, t in zip(LETTERS, self.example.options)}

    def vetoes(self, closed_book_reply: str) -> bool:
        """Whether the memory veto holds for the closed-book answer m.

        Free-form: m states an answer and fails the support test. Multiple choice: m names an option and another
        option scores higher than it, so the passage favors another answer.
        """
        m = self.key(closed_book_reply)
        if not m:
            return False
        if not self.multiple_choice:
            return not self.supports(closed_book_reply)
        scores = self.option_scores()
        return any(s > scores[m] for c, s in scores.items() if c != m)

    def masked_letter(self, closed_book_reply: str) -> Optional[str]:
        """The letter that repair decoding masks: the closed-book option when the veto holds on a multiple-choice
        question, else None."""
        if self.multiple_choice and self.vetoes(closed_book_reply):
            return self.key(closed_book_reply)
        return None
