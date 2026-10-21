"""Write the answers and decisions of the four paper examples into the step-by-step replay page, demo/replay.html.

    python demo/make_replay.py

The page needs no server or install: open demo/replay.html in a browser. It animates what `aor demo` prints. Every
answer, check result and score on it comes from `aor.demo.replay()` and `aor.scoring`; this script only lays them out.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from aor import PassageCheck
from aor.demo import CASES, headline, replay
from aor.example import Example
from aor.methods import LABELS
from aor.procedure import Trace
from aor.scoring import score
from aor.support import answer_line, normalize

PAGE = Path(__file__).with_name("replay.html")
#: sentences of each passage that the page shows, None for all of them; the checks always read the whole passage
SHOWN: Sequence[Optional[Tuple[int, ...]]] = ((0, 1), (0,), None, (0, 2))
#: the pill of the replay that each repair decoder lights up
PILL = {"cad": 2, "id": 3}
DATA_BLOCK = re.compile(r'(<script type="application/json" id="replay-data">\n)(.*?)(\n</script>)', re.S)


def shown_passage(passage: str, keep: Optional[Tuple[int, ...]]) -> str:
    """The sentences ``keep`` of a passage, with [...] where sentences are left out."""
    if keep is None:
        return passage
    sentences = re.split(r"(?<=\.)\s+", passage.strip())
    parts, last = [], -1
    for k in keep:
        if k != last + 1:
            parts.append("[...]")
        parts.append(sentences[k])
        last = k
    if last != len(sentences) - 1:
        parts.append("[...]")
    return " ".join(parts)


def spans(passage: str, text: str) -> List[Tuple[int, int]]:
    """Where the normalized words of ``text`` occur in ``passage`` as whole words, as the support test reads them."""
    tokens = normalize(text).split()
    if not tokens:
        return []
    pattern = re.compile(r"(?<!\w)" + r"\W*".join(map(re.escape, tokens)) + r"(?!\w)", re.IGNORECASE)
    return [m.span() for m in pattern.finditer(passage)]


def marked(passage: str, marks: Sequence[Tuple[int, int, str]]) -> str:
    """The passage as escaped HTML, with each (start, end, kind) range marked and each [...] set apart."""
    cuts = sorted([(s, e, f'<mark class="{kind}">', "</mark>") for s, e, kind in marks]
                  + [(m.start(), m.end(), '<span class="elide">', "</span>")
                     for m in re.finditer(r"\[\.\.\.\]", passage)])
    out, last = [], 0
    for s, e, opening, closing in cuts:
        if s < last:
            continue
        out += [html.escape(passage[last:s]), opening, html.escape(passage[s:e]), closing]
        last = e
    out.append(html.escape(passage[last:]))
    return "".join(out)


def joined(letters: Sequence[str]) -> str:
    return " and ".join(letters) if len(letters) <= 2 else ", ".join(letters[:-1]) + " and " + letters[-1]


def result(example: Example, reply: str) -> Dict[str, str]:
    """How the replay shows one final answer: the answer line, whether it is right, and its token F1."""
    s = score(reply, example)
    if s >= 1:
        grade, kind = "✓ correct", "ok"
    elif s == 0:
        grade, kind = "× wrong", "bad"
    else:
        grade, kind = "partly right", "partial"
    return {"reply": reply, "shown": answer_line(reply), "grade": grade, "kind": kind,
            "stat": "" if example.is_multiple_choice else f"token F1 {s:.2f}"}


def walkthrough(example: Example, trace: Trace, passage: str) -> Tuple[List[dict], List[str]]:
    """The steps the procedure took on one example, as lines of the replay, and the passage as each check left it."""
    check = PassageCheck(example)
    mc = check.multiple_choice
    views = [marked(passage, [])]

    def view(texts: Sequence[str], kind: str) -> int:
        views.append(marked(passage, [(s, e, kind) for t in texts for s, e in spans(passage, t)]))
        return len(views) - 1

    def answer_text(reply: str) -> str:
        return (example.option_text(check.key(reply)) or "") if mc else answer_line(reply)

    def step(pill, label, reply, outcome, kind, doing, checking="support test…", shown=None, closed_book=False):
        return {"pill": pill, "label": label, "reply": reply, "outcome": outcome, "kind": kind, "doing": doing,
                "checking": checking, "view": shown, "closed_book": closed_book}

    occurring = sorted(check.occurring) if mc else []
    found = view([example.option_text(c) for c in occurring], "warn") if occurring else None

    def unsupported(reply: str) -> str:
        if not mc:
            return "not in the passage"
        if len(occurring) > 1:
            return f"options {joined(occurring)} {'both' if len(occurring) == 2 else 'all'} occur, so none passes"
        if not occurring:
            return "no option occurs in the passage, so none passes"
        return f"only option {occurring[0]} occurs in the passage"

    supported = "the only option in the passage" if mc else "in the passage"
    if trace.action == "accept":
        return [step(0, "sample", trace.initial, f"{supported}: accepted", "ok", "sampling…",
                     shown=view([answer_text(trace.initial)], "ok"))], views
    steps = [step(0, "sample", trace.initial, unsupported(trace.initial), "fail", "sampling…", shown=found)]

    m = trace.closed_book or ""
    m_key = check.key(m)
    if trace.veto and mc:
        scores = check.option_scores()
        higher = [c for c, s in scores.items() if c != m_key and s > scores[m_key]]
        say, kind = f"the passage favors {joined(higher)}: {m_key} is vetoed and masked", "veto"
    elif trace.veto:
        say, kind = "not in the passage: repeats of it are vetoed", "veto"
    else:
        say, kind = ("states no answer" if not m_key else f"{supported}") + ": no veto", "info"
    steps.append(step(1, "closed-book", m, say, kind, "closed-book answer…", "memory veto…",
                      closed_book=True))

    # on multiple choice, when no option that the veto leaves can pass, the first usable repair is returned at once
    at_once = mc and not (check.acceptable - ({m_key} if trace.veto else set()))
    last = len(trace.repairs) - 1
    for j, d in enumerate(trace.repairs):
        name = LABELS.get(d.decoder, d.decoder)
        label = f"{name}, {trace.masked_letter} masked" if trace.masked_letter else f"{name} repair"
        shown = None
        if d.rejected:
            say, kind = "repeats the closed-book answer: rejected", "fail"
        elif j == last and trace.action == "repair":
            say, kind, shown = f"{supported}: returned", "ok", view([answer_text(d.text)], "ok")
        elif j == last and trace.action == "fallback" and at_once:
            say, kind, shown = "no option can pass, so this repair is returned", "fallback", found
        else:
            say, kind = unsupported(d.text), "fail"
        steps.append(step(PILL.get(d.decoder), label, d.text, say, kind, f"{name} repair…", shown=shown))
    if trace.action == "fallback" and not at_once:
        steps.append(step(None, "fallback", trace.answer, "no repair passes: the first one not vetoed is returned",
                          "fallback", "fallback…", "deciding…"))
    elif trace.action == "initial":
        steps.append(step(None, "kept", trace.initial, "no repair states a usable answer: the initial answer stays",
                          "info", "fallback…", "deciding…"))
    return steps, views


def build() -> dict:
    cases = []
    for case, trace, keep in zip(CASES, replay(), SHOWN):
        ex = case["example"]
        task, model = case["title"].split(", ")
        passage = shown_passage(ex.passage, keep)
        steps, views = walkthrough(ex, trace, passage)
        legend = []
        if any('class="ok"' in v for v in views):
            legend.append("green marks what the support test found in the passage")
        if any('class="warn"' in v for v in views):
            legend.append("amber marks the option texts that occur in the passage")
        action = headline(trace)
        reference = f"{ex.gold_letter}. {ex.option_text(ex.gold_letter)}" if ex.gold_letter else " / ".join(ex.golds)
        cases.append({
            "tab": action[0].upper() + action[1:],
            "caption": f"{model} on {task}, seed 0, answers as recorded" + "".join(f"; {x}" for x in legend),
            "passage": views,
            "question": ex.question,
            "options": [f"{c}. {t}" for c, t in zip(ex.letters, ex.options)],
            "reference": reference,
            "aor": dict(result(ex, trace.answer), headline=action, steps=steps),
            "baselines": [dict(result(ex, case["answers"]["cad"]), name="CAD", note="alone", pill="CAD"),
                          dict(result(ex, case["answers"]["sampling"]), name="Sampling", note="plain decoding",
                               pill="sample")],
        })
    return {"cases": cases}


def render() -> str:
    """The data block of the page."""
    return json.dumps(build(), indent=1).replace("</", "<\\/")


def main() -> None:
    page = PAGE.read_text(encoding="utf-8")
    new, n = DATA_BLOCK.subn(lambda m: m.group(1) + render() + m.group(3), page, count=1)
    if n != 1:
        raise SystemExit(f"no data block in {PAGE}")
    PAGE.write_text(new, encoding="utf-8")
    print(f"wrote {PAGE}")


if __name__ == "__main__":
    main()
