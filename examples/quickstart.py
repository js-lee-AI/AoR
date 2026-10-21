"""Accept-or-Repair on the paper's four example questions.

    python examples/quickstart.py                    # replays the answers the models gave (CPU, no model)
    python examples/quickstart.py --model qwen3-8b   # decodes the same questions with a model (GPU)

The replay prints:

    NQ-Original, Qwen3-8B: '111' (accepted)
    NQ-Swap, Qwen3-8B: '1971' (CAD repair)
    NQ-Swap, Llama-3.1-8B: 'Neal Dahlen' (ID repair)
    FaithEval, Qwen3-14B: 'C. a mineral' (D vetoed, CAD fallback)
"""

import argparse

from aor import PassageCheck
from aor.demo import CASES, headline, replay, steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="a short name such as qwen3-8b, or a Hugging Face id; omit to replay")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # The support test and the memory veto read surface forms only, so they need no model. The closed-book answer is
    # only checked when the initial answer fails the support test.
    print("Checks on the recorded answers")
    for case in CASES:
        check = PassageCheck(case["example"])
        a, m = case["answers"]["sampling"], case["answers"]["closed_book"]
        line = f"  {case['title']}: initial {a!r} supported={check.supports(a)}"
        if not check.supports(a):
            line += f", closed-book {m!r} vetoed={check.vetoes(m)}"
        print(line)

    # The procedure calls the decoders only where it needs them: a closed-book answer and repairs for unsupported
    # initial answers, the letter of a vetoed closed-book option masked on multiple choice.
    examples = [case["example"] for case in CASES]
    if args.model:
        from aor.hf import AcceptOrRepair

        traces = AcceptOrRepair.from_pretrained(args.model).run(examples, seed=args.seed).traces
        print(f"\nAoR with {args.model}, seed {args.seed}")
    else:
        traces = replay()
        print("\nAoR on the recorded answers")
    for case, trace in zip(CASES, traces):
        print(f"  {case['title']}: {trace.answer!r} ({headline(trace)})")
        print(f"      {steps(trace)}")


if __name__ == "__main__":
    main()
