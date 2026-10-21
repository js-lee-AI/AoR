"""A browser demo of Accept-or-Repair.

    pip install -e ".[demo]"                          # gradio
    python demo/app.py                                # the paper's four examples, replayed without a model
    python demo/app.py --model qwen3-8b --device 0    # also answers your own questions with a model

The first tab plays the four test questions that the paper shows as examples step by step, like demo/replay.html. The
second shows each step of them in a table, with the passage. The third, available with --model, runs AoR and CAD on a
passage and question you enter.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

import gradio as gr

from aor import Example, PassageCheck
from aor.demo import CASES, headline, replay
from aor.methods import LABELS
from aor.procedure import Trace
from aor.support import normalize

TRACES = replay()
REPLAY = Path(__file__).with_name("replay.html")
# Gradio 6 takes the page style in launch() instead of the Blocks constructor
GRADIO_6 = int(gr.__version__.split(".")[0]) >= 6


CSS = """
.passage {font-size: 15px; line-height: 1.65; padding: 14px 16px; border: 1px solid var(--border-color-primary);
          border-radius: 8px; background: var(--block-background-fill); max-height: 420px; overflow-y: auto}
.passage mark {background: #c8ecd2; color: #0f5132; padding: 1px 4px; border-radius: 4px; font-weight: 600}
"""


def highlight(passage: str, answer: str) -> str:
    """The passage as HTML with the occurrences of the answer marked."""
    tokens = normalize(answer).split()
    pieces, last = [], 0
    if tokens:
        pattern = re.compile(r"(?<!\w)" + r"\W*".join(map(re.escape, tokens)) + r"(?!\w)", re.IGNORECASE)
        for m in pattern.finditer(passage):
            pieces += [html.escape(passage[last:m.start()]), f"<mark>{html.escape(m.group(0))}</mark>"]
            last = m.end()
    pieces.append(html.escape(passage[last:]))
    return '<div class="passage">' + "".join(pieces).replace("\n", "<br>") + "</div>"


def code(text) -> str:
    """A reply as inline code on one line, safe inside a Markdown table cell."""
    t = " ".join(str(text or "").split()).replace("|", "\\|")
    if not t:
        return "*(empty)*"
    return f"`` {t} ``" if "`" in t else f"`{t}`"


def answer_text(example: Example, reply: str) -> str:
    """The span to look for in the passage: the option text of a letter, or the first line of a free-form reply."""
    if example.is_multiple_choice:
        letter = PassageCheck(example).key(reply)
        return example.option_text(letter) or ""
    return reply.strip().split("\n")[0]


def steps_table(example: Example, trace: Trace) -> str:
    check = PassageCheck(example)
    rows = ["| Step | Answer | Check |", "|---|---|---|"]
    if trace.action == "accept":
        rows.append(f"| Initial answer (sampling) | {code(trace.initial)} | supported by the passage: **accepted** |")
        return "\n".join(rows)
    rows.append(f"| Initial answer (sampling) | {code(trace.initial)} | not supported by the passage |")
    if trace.veto:
        veto = "memory veto: the passage favors another answer"
        if trace.masked_letter:
            veto += f", letter {trace.masked_letter} is masked during repair"
    else:
        veto = "no veto"
    rows.append(f"| Closed-book answer *m* | {code(trace.closed_book)} | {veto} |")
    for d in trace.repairs:
        label = LABELS.get(d.decoder, d.decoder)
        if d.rejected:
            res = "repeats *m* under the veto: rejected"
        elif d.supported:
            res = "supported"
        elif check.multiple_choice and not check.acceptable:
            occurring = sorted(check.occurring)
            res = ("no option text occurs in the passage" if not occurring else
                   f"options {' and '.join(occurring)} both occur in the passage" if len(occurring) == 2 else
                   f"options {', '.join(occurring)} occur in the passage") + ", so no option passes the support test"
        else:
            res = "not supported"
        if trace.decoder == d.decoder and d.text == trace.answer:
            res += ": **returned**" if trace.action == "repair" else ": **returned** as the first repair not vetoed"
        rows.append(f"| {label} repair | {code(d.text)} | {res} |")
    if trace.action == "initial":
        rows.append(f"| Fallback | {code(trace.initial)} | no repair states a usable answer: "
                    "the initial answer stays |")
    return "\n".join(rows)


def show_case(title: str):
    i = [c["title"] for c in CASES].index(title)
    case, trace = CASES[i], TRACES[i]
    ex = case["example"]
    q = ex.question + (f"\n\n{ex.options_block()}" if ex.is_multiple_choice else "")
    cad = case["answers"]["cad"]
    summary = (f"**AoR:** {code(trace.answer)} ({headline(trace)})  \n"
               f"**Sampling:** {code(case['answers']['sampling'])}  \n"
               f"**CAD alone:** {code(cad)}  \n"
               f"**Reference answer:** {code(ex.gold_letter or ', '.join(ex.golds))}")
    return q, highlight(ex.passage, answer_text(ex, trace.answer)), summary, steps_table(ex, trace)


def build(aor=None) -> gr.Blocks:
    with gr.Blocks(title="Accept-or-Repair", **({} if GRADIO_6 else {"css": CSS})) as app:
        gr.Markdown("# Accept-or-Repair\nKeep answers the passage supports, and repair the others with contrastive "
                    "decoding. A repair that repeats the model's closed-book answer is rejected when the passage "
                    "favors another answer (the memory veto).")
        with gr.Tab("Step by step"):
            gr.HTML(f'<iframe title="Accept-or-Repair, step by step" srcdoc="{html.escape(REPLAY.read_text("utf-8"))}" '
                    'style="width: 100%; height: 660px; border: 0"></iframe>')
        with gr.Tab("Paper examples"):
            pick = gr.Radio([c["title"] for c in CASES], value=CASES[1]["title"], label="Test question")
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("**Passage**, with AoR's answer marked")
                    passage = gr.HTML()
                with gr.Column(scale=2):
                    question = gr.Textbox(label="Question", lines=4, interactive=False)
                    summary = gr.Markdown()
            steps = gr.Markdown()
            pick.change(show_case, pick, [question, passage, summary, steps])
            app.load(show_case, pick, [question, passage, summary, steps])
        if aor is not None:
            with gr.Tab("Your question"):
                with gr.Row():
                    with gr.Column(scale=3):
                        my_passage = gr.Textbox(label="Passage", lines=8, value=CASES[1]["example"].passage)
                        my_question = gr.Textbox(label="Question", value=CASES[1]["example"].question)
                        my_options = gr.Textbox(label="Options of a multiple-choice question, one per line "
                                                      "(leave empty for a short answer)", lines=4)
                        seed = gr.Number(label="Seed", value=0, precision=0)
                        go = gr.Button("Answer", variant="primary")
                    with gr.Column(scale=2):
                        my_summary = gr.Markdown()
                        my_highlight = gr.HTML()
                my_steps = gr.Markdown()

                def ask(p, q, opts, s):
                    options = tuple(o.strip() for o in (opts or "").splitlines() if o.strip())
                    ex = Example(question=q.strip(), passage=p.strip(), options=options)
                    trace = aor.run([ex], seed=int(s)).traces[0]
                    cad = aor.decoder.generate([ex], "cad", seed=int(s)).texts[0]
                    text = f"**AoR:** {code(trace.answer)} ({headline(trace)})  \n**CAD alone:** {code(cad)}"
                    return text, highlight(ex.passage, answer_text(ex, trace.answer)), steps_table(ex, trace)

                go.click(ask, [my_passage, my_question, my_options, seed], [my_summary, my_highlight, my_steps])
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a short name such as qwen3-8b, or a Hugging Face id (needs a GPU)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    aor = None
    if a.model:
        from aor.hf import AcceptOrRepair

        aor = AcceptOrRepair.from_pretrained(a.model, device=int(a.device) if a.device.isdigit() else a.device)
    build(aor).launch(server_name=a.host, server_port=a.port, share=a.share, **({"css": CSS} if GRADIO_6 else {}))


if __name__ == "__main__":
    main()
