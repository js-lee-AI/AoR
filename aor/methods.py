"""The decoders of the main comparison (paper Table 1), with the settings selected on the development split.

A decoder samples from a score computed at every step from the next-token log-probabilities under its prompts: the
first prompt is the one with the passage (x), the second, if any, the contrasted prompt (x'). With gamma the contrast
strength, the CAD rule is

    (1 + gamma) log p(y | x) - gamma log p(y | x'),

which CAD, RPCD, ID and Strict CAD share and which differ only in their prompts. AdaCAD, COIECD and NWCAD change how the
contrast is applied. Temperature and truncation are applied to the score afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class Method:
    name: str
    label: str
    prompt: str                     # prompt kind of x, see aor.prompts
    contrast: Optional[str] = None  # prompt kind of x', None for plain sampling
    rule: str = "plain"             # plain, cad, adacad, coiecd or nwcad
    strength: float = 0.0           # gamma
    params: Tuple[Tuple[str, float], ...] = ()

    def param(self, key: str) -> float:
        return dict(self.params)[key]

    def with_strength(self, strength: float) -> "Method":
        return replace(self, strength=float(strength))

    @property
    def streams(self) -> Tuple[str, ...]:
        return (self.prompt,) if self.contrast is None else (self.prompt, self.contrast)


METHODS: Dict[str, Method] = {m.name: m for m in [
    Method("sampling", "Sampling", "standard"),
    Method("cad", "CAD", "standard", "closed_book", "cad", 1.0),
    # gamma is scaled at every step by the Jensen-Shannon divergence of the two distributions, in bits
    Method("adacad", "AdaCAD", "standard", "closed_book", "adacad", 1.0),
    # strength 2 and constraint scale 0.25
    Method("coiecd", "COIECD", "standard", "closed_book", "coiecd", 2.0, (("scale", 0.25),)),
    # divergence threshold 0.1, with the margins of 0.30 (closed-book) and 0.05 (with the passage) of the NWCAD paper
    Method("nwcad", "NWCAD", "standard", "closed_book", "nwcad", 1.0,
           (("threshold", 0.1), ("closed_book_margin", 0.30), ("passage_margin", 0.05))),
    Method("rpcd", "RPCD", "standard", "other_passage", "cad", 1.0),
    Method("id", "ID", "standard", "opposite", "cad", 2.0),
    Method("strict", "Strict prompt", "strict"),
    Method("strict_cad", "Strict CAD", "strict", "closed_book", "cad", 1.0),
    # not a baseline: the answer without the passage, which AoR's memory veto compares with the passage
    Method("closed_book", "Closed-book", "closed_book"),
]}

#: rows of paper Table 1, in order
TABLE1 = ("sampling", "cad", "adacad", "coiecd", "nwcad", "rpcd", "id", "strict", "strict_cad", "aor")
LABELS = {**{k: m.label for k, m in METHODS.items()}, "aor": "AoR"}
#: repair decoders of AoR, in the order they are tried
AOR_REPAIRS = ("cad", "id")


def get_method(name: str, strength: Optional[float] = None) -> Method:
    try:
        m = METHODS[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(METHODS)}") from None
    return m if strength is None else m.with_strength(strength)
