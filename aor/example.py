"""One question over a passage, the unit that every decoder and the procedure work on."""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Optional, Tuple

LETTERS = string.ascii_uppercase


@dataclass(frozen=True)
class Example:
    """A short-answer or multiple-choice question about a passage.

    ``options`` holds the option texts of a multiple-choice question, lettered A, B, C, ... in order; it is empty for a
    free-form question. ``multi_passage`` marks an input made of several paragraphs, such as HotpotQA, for which the
    prompts say "passages". ``uid`` enters the key of the sampling noise, so two decoders that run on the same example
    with the same seed share their noise. ``partner_passage`` is the passage of another example, which RPCD contrasts
    with. ``golds`` and ``gold_letter`` are only needed for scoring.
    """

    question: str
    passage: str
    options: Tuple[str, ...] = ()
    multi_passage: bool = False
    uid: int = 0
    partner_passage: Optional[str] = None
    golds: Tuple[str, ...] = ()
    gold_letter: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.options) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options are supported")
        if self.gold_letter is not None and self.gold_letter not in LETTERS[: len(self.options)]:
            raise ValueError(f"gold_letter {self.gold_letter!r} does not name one of the options")

    @property
    def is_multiple_choice(self) -> bool:
        return bool(self.options)

    @property
    def letters(self) -> str:
        return LETTERS[: len(self.options)]

    def option_text(self, letter: str) -> Optional[str]:
        """Text of the option with this letter, or None for a letter that names no option."""
        k = LETTERS.find(letter) if letter else -1
        return self.options[k] if 0 <= k < len(self.options) else None

    def options_block(self) -> str:
        """The options as they appear in a prompt, one per line in the form "A. text"."""
        return "\n".join(f"{c}. {t}" for c, t in zip(LETTERS, self.options))
