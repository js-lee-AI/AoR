"""Prompts of the paper (Appendix A). Each is a single user turn, rendered later with the model's chat template.

Kinds:
    standard       the task prompt, used by sampling, AoR and every contrastive decoder except Strict CAD
    closed_book    the question without the passage: the closed-book answer of AoR and the contrast of CAD, AdaCAD,
                   COIECD, NWCAD and Strict CAD
    other_passage  the task prompt with the passage of another example: the contrast of RPCD
    opposite       the passage with an instruction to ignore it: the contrast of ID
    strict         an instruction to answer only from the passage: Strict prompt and Strict CAD
"""

from __future__ import annotations

from .example import Example

KINDS = ("standard", "closed_book", "other_passage", "opposite", "strict")

_FORMAT = "Reply with only the answer, a few words at most."
_FORMAT_MC = "Reply with only the letter of the correct option."


def _instruction(kind: str, noun: str, mc: bool) -> str:
    task = "Answer the multiple-choice question" if mc else "Answer the question"
    fmt = _FORMAT_MC if mc else _FORMAT
    if kind in ("standard", "other_passage"):
        return f"{task} according to the {noun}. {fmt}" if mc else f"{task} using the {noun}. {fmt}"
    if kind == "opposite":
        return f"{task} using your own knowledge and ignore the {noun}. {fmt}"
    if kind == "strict":
        verb = "contradicts" if noun == "passage" else "contradict"
        return f"{task} using only the {noun}, even if the {noun} {verb} what you know. {fmt}"
    if kind == "closed_book":
        return f"{task}. {fmt}"
    raise ValueError(f"unknown prompt kind {kind!r}; expected one of {KINDS}")


def build_prompt(example: Example, kind: str) -> str:
    """Prompt text, a single user turn, for one example and one prompt kind."""
    mc = example.is_multiple_choice
    noun = "passages" if example.multi_passage else "passage"
    question = f"Question: {example.question}"
    if mc:
        question += f"\n\n{example.options_block()}"
    instruction = _instruction(kind, noun, mc)
    if kind == "closed_book":
        return f"{instruction}\n\n{question}"
    if kind == "other_passage":
        if example.partner_passage is None:
            raise ValueError("the other_passage prompt needs Example.partner_passage")
        passage = example.partner_passage
    else:
        passage = example.passage
    return f"{instruction}\n\n{noun.capitalize()}: {passage}\n\n{question}"
