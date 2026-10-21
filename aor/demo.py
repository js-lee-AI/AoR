"""The four test questions that the paper shows as examples, with the answers the models gave (seed 0), replayed
through the procedure without a model. The NQ passages are complete; the FaithEval passage is shortened to three of its
sentences, which keeps every support and veto decision of the full passage.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .example import Example
from .methods import LABELS
from .procedure import Trace, accept_or_repair

CASES = [
    {
        "title": "NQ-Original, Qwen3-8B",
        "example": Example(
            question="how many games in a row have the uconn women 's basketball team won",
            passage="UConn owns the two longest winning streaks , any gender , in college basketball history . The "
                    "longest streak , 111 straight wins , started with a win against Creighton on November 23 , 2014 , "
                    "and ended on March 31 , 2017 when a buzzer - beater at the end of the overtime caused a loss in "
                    "the 2017 NCAA Final Four by Mississippi State . The second streak counts 90 consecutive wins , "
                    "including two undefeated seasons ( 2008 -- 09 and 2009 -- 10 ) , and was delimited by two losses "
                    "against Stanford , the first on April 6 , 2008 in the National Semifinals of the NCAA Tournament "
                    ", and the second -- three seasons later -- on December 19 , 2010 .",
            uid=308, golds=("111",)),
        "answers": {"sampling": "111", "closed_book": "111 games", "cad": "two", "id": "111"},
    },
    {
        "title": "NQ-Swap, Qwen3-8B",
        "example": Example(
            question="when did ireland gain its independence from england",
            passage="In 1971 , after the Irish War of Independence and the Anglo - Irish Treaty , most of Ireland "
                    "seceded from the United Kingdom to become the independent Irish Free State , which after the 1937 "
                    "constitution , began to call itself Ireland . The six northeastern counties , known as Northern "
                    "Ireland , remained within the United Kingdom . The Irish Civil War followed soon after the War of "
                    "Independence . The history of Northern Ireland has since been dominated by sporadic sectarian "
                    "conflict between ( mainly Catholic ) Irish nationalists and ( mainly Protestant ) unionists . "
                    "This conflict erupted into the Troubles in the late 1960s , until peace was achieved with the "
                    "Belfast Agreement thirty years later .",
            uid=912, golds=("1971",)),
        "answers": {"sampling": "1922", "closed_book": "1922", "cad": "1971", "id": "1971"},
    },
    {
        "title": "NQ-Swap, Llama-3.1-8B",
        "example": Example(
            question="who voices hiccup in how to train your dragon 2",
            passage="Neal Dahlen as Hiccup Horrendous Haddock III , the son of the Viking chief Stoick the Vast and "
                    "Valka .",
            uid=3674, golds=("Neal Dahlen",)),
        "answers": {"sampling": "Jay Baruchel.", "closed_book": "Jay Baruchel.", "cad": "Jay Baruchel.",
                    "id": "Neal Dahlen"},
    },
    {
        "title": "FaithEval, Qwen3-14B",
        "example": Example(
            question="Which of the following would most likely form a fossil?",
            passage="Typically, the formation starts when an organism gets buried quickly after its death, usually in "
                    "sedimentary rock formations. Historically, bones and shells are what most people think of when "
                    "they conjure images of fossils, but recent studies have shown that minerals hold a pivotal role "
                    "in the fossilization process. Specifically, it has been observed that the infiltration of "
                    "mineral-rich waters can cause the organic material to undergo a series of transformations.",
            options=("a rock", "a crystal", "a mineral", "a bone"), uid=948, gold_letter="C"),
        # with the letter D masked, as the memory veto requires, both repairs answer C
        "answers": {"sampling": "D. a bone", "closed_book": "D", "cad": "D. a bone", "id": "D. a bone",
                    "cad|D": "C. a mineral", "id|D": "C. a mineral"},
    },
]


def scripted(cases: Sequence[dict], decoder: str):
    """A batch decoder that returns the recorded answer of each case, the masked-letter answer when asked for one."""
    def draw(idx: List[int], masked: Optional[List[Optional[str]]] = None) -> List[str]:
        masked = masked or [None] * len(idx)
        out = []
        for i, letter in zip(idx, masked):
            answers: Dict[str, str] = cases[i]["answers"]
            out.append(answers.get(f"{decoder}|{letter}", answers[decoder]) if letter else answers[decoder])
        return out
    return draw


def replay(cases: Sequence[dict] = CASES) -> List[Trace]:
    examples = [c["example"] for c in cases]
    return accept_or_repair(examples, scripted(cases, "sampling"), scripted(cases, "closed_book"),
                            [("cad", scripted(cases, "cad")), ("id", scripted(cases, "id"))])


def headline(trace: Trace) -> str:
    """The action in the words of the paper: accepted, a supported CAD or ID repair, or a fallback, which returns the
    first repair that is not vetoed when none passes the support test. A vetoed option letter comes first."""
    if trace.action == "accept":
        return "accepted"
    if trace.action == "initial":
        return "initial answer kept"
    label = LABELS.get(trace.decoder, trace.decoder)
    kind = "repair" if trace.action == "repair" else "fallback"
    return f"{trace.masked_letter} vetoed, {label} {kind}" if trace.masked_letter else f"{label} {kind}"


def steps(trace: Trace) -> str:
    """The closed-book answer and the repairs that were drawn, with what the checks said about each."""
    if trace.action == "accept":
        return "the passage supports the initial answer"
    m = f"closed-book {trace.closed_book!r}" + (" vetoed" if trace.veto else "")
    if trace.masked_letter:
        m += f", letter {trace.masked_letter} masked"
    tried = [f"{LABELS.get(d.decoder, d.decoder)} {d.text!r} "
             + ("rejected" if d.rejected else "supported" if d.supported else "unsupported") for d in trace.repairs]
    if trace.action == "fallback":
        tried.append("no repair passes the support test, so the first one that is not vetoed is returned")
    return "; ".join([m] + tried)


def main() -> None:
    for case, trace in zip(CASES, replay()):
        a = case["answers"]
        print(case["title"])
        print(f"  question   {case['example'].question}")
        print(f"  sampling   {a['sampling']!r}, CAD {a['cad']!r}")
        print(f"  AoR        {trace.answer!r}, {headline(trace)}")
        print(f"             {steps(trace)}")


if __name__ == "__main__":
    main()
