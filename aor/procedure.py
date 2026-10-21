"""Accept-or-Repair as a procedure over decoders.

The procedure decides which answers to keep and which decoders to call; it does not decode. Decoders are passed in as
batch functions, so the same control flow runs over a Transformers model (`aor.hf.AcceptOrRepair`), a scripted stand-in
(`examples/quickstart.py`) or any other backend.

For each example:

1. Draw a plain answer a. If the passage supports a, return it (accept).
2. Otherwise draw a closed-book answer m from the prompt without the passage. The memory veto holds when m states an
   answer and the passage favors another answer (`PassageCheck.vetoes`).
3. Draw repairs from the repair decoders in order. On a multiple-choice question under the veto, the letter of m is
   masked during repair decoding. Return the first repair that passes the support test and does not repeat m under the
   veto (repair).
4. If no repair passes, return the first repair that states an answer and does not repeat m under the veto (fallback),
   or a when there is none. On a multiple-choice question where no option except a vetoed m can pass the support test,
   the first such repair is returned at once and no further repair is drawn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

from .example import Example
from .support import PassageCheck

#: sample(indices) -> one reply for each example index
BatchDraw = Callable[[List[int]], List[str]]
#: repair(indices, masked_letters) -> one reply for each example index; masked_letters[j] is the option letter that
#: decoding must not produce for indices[j], or None
BatchRepair = Callable[[List[int], List[Optional[str]]], List[str]]


@dataclass
class Draw:
    decoder: str
    text: str
    supported: bool
    rejected: bool          # repeats the closed-book answer under the memory veto


@dataclass
class Trace:
    """What the procedure did for one example.

    ``action`` is "accept" (the plain answer passed the support test), "repair" (a repair passed it), "fallback" (no
    repair passed, an unsupported repair is returned) or "initial" (no repair states a usable answer, the plain answer is
    returned). ``decoder`` names the decoder of the returned answer.
    """

    answer: str
    action: str
    decoder: str
    initial: str
    closed_book: Optional[str] = None
    veto: bool = False
    masked_letter: Optional[str] = None
    repairs: List[Draw] = field(default_factory=list)


def accept_or_repair(examples: Sequence[Example], sample: BatchDraw, closed_book: BatchDraw,
                     repairs: Sequence[Tuple[str, BatchRepair]]) -> List[Trace]:
    """Run Accept-or-Repair over a batch of examples.

    ``sample`` draws the plain answers, ``closed_book`` the answers from the prompt without the passage, and
    ``repairs`` is the ordered list of (name, decoder) pairs, (("cad", ...), ("id", ...)) in the paper. Every decoder is
    called once per stage with the indices of the examples that still need it.
    """
    n = len(examples)
    checks = [PassageCheck(ex) for ex in examples]
    initial = sample(list(range(n)))
    traces: List[Optional[Trace]] = [None] * n
    pending = []
    for i in range(n):
        if checks[i].supports(initial[i]):
            traces[i] = Trace(initial[i], "accept", "sampling", initial[i])
        else:
            pending.append(i)
    if not pending:
        return traces  # type: ignore[return-value]

    memory = dict(zip(pending, closed_book(pending)))
    veto = {i: checks[i].vetoes(memory[i]) for i in pending}
    masked = {i: checks[i].masked_letter(memory[i]) for i in pending}
    m_key = {i: checks[i].key(memory[i]) for i in pending}
    drawn = {i: [] for i in pending}
    first_usable = {}
    # on multiple choice the acceptable options are known in advance: without one that the veto leaves, the support
    # test can never pass and the first usable repair is the answer
    can_accept = {i: (not checks[i].multiple_choice) or bool(checks[i].acceptable - {m_key[i] if veto[i] else ""})
                  for i in pending}

    def rejected(i: int, text: str) -> bool:
        return veto[i] and checks[i].key(text) == m_key[i]

    def finish(i: int, action: str, decoder: str, text: str) -> None:
        traces[i] = Trace(text, action, decoder, initial[i], memory[i], veto[i], masked[i], drawn[i])

    active = list(pending)
    for name, repair in repairs:
        if not active:
            break
        texts = repair(active, [masked[i] for i in active])
        still = []
        for i, text in zip(active, texts):
            sup, rej = checks[i].supports(text), rejected(i, text)
            drawn[i].append(Draw(name, text, sup, rej))
            if can_accept[i] and sup and not rej:
                finish(i, "repair", name, text)
                continue
            if i not in first_usable and checks[i].key(text) and not rej:
                first_usable[i] = (name, text)
                if not can_accept[i]:
                    finish(i, "fallback", name, text)
                    continue
            still.append(i)
        active = still
    for i in active:
        if i in first_usable:
            finish(i, "fallback", *first_usable[i])
        else:
            finish(i, "initial", "sampling", initial[i])
    return traces  # type: ignore[return-value]
