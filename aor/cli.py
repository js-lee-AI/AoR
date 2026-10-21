"""The `aor` command. `python -m aor` runs the same thing."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .data import TASKS
from .methods import METHODS, TABLE1
from .models import MODELS


def _demo(args) -> int:
    from .demo import main

    main()
    return 0


def _table(args) -> int:
    from pathlib import Path

    from . import table

    models = args.models or list(MODELS)
    if not (Path(args.records) / "splits.json").exists():
        raise SystemExit(f"no splits.json in {args.records}; pass --records with the records/ directory of a clone "
                         "of the repository")
    if args.runs:
        try:
            rows = table.from_runs(args.runs, models, args.methods, args.seeds,
                                   splits=Path(args.records) / "splits.json", n_boot=args.boot)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(f"{e}\nPass --models, --methods and --seeds to match the runs in {args.runs}; "
                             "sampling is always needed, since differences and costs are taken relative to it.")
    else:
        rows = table.from_records(args.records, models, args.methods, n_boot=args.boot)
    print(table.markdown(rows))
    print("\nScores averaged over models: F1 for free-form tasks, accuracy for FaithEval. Parentheses give the "
          "difference from sampling, * when its 95% interval excludes zero. Cost is tokens processed relative to "
          "sampling.")
    if args.per_model:
        print()
        print(table.markdown_per_model(rows, models))
    if args.intervals:
        print("\nDifference from sampling with 95% paired cluster-bootstrap intervals:")
        print("\n".join(table.intervals(rows)))
    return 0


def _run(args) -> int:
    from .run import run_model

    run_model(args.model, tasks=args.tasks, methods=args.methods, seeds=args.seeds, out=args.out, split=args.split,
              device=args.device, limit=args.limit, max_batch=args.max_batch)
    return 0


def _ask(args) -> int:
    from .demo import headline, steps
    from .example import Example
    from .hf import AcceptOrRepair

    ex = Example(question=args.question, passage=args.passage, options=tuple(args.option or ()))
    aor = AcceptOrRepair.from_pretrained(args.model, device=args.device)
    trace = aor(ex, seed=args.seed)
    print(f"answer   {trace.answer!r}")
    print(f"action   {headline(trace)}")
    print(f"steps    {steps(trace)}")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    parser = argparse.ArgumentParser(
        prog="aor", description="Accept-or-Repair: keep passage-supported answers and repair the others with "
                                "contrastive decoding.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="replay the paper's four example questions without a model")
    demo.set_defaults(func=_demo)

    tab = sub.add_parser("table", help="Table 1 from the shipped records or from your runs (CPU)")
    tab.add_argument("--records", default=None, help="records directory (default: records/ of the repository)")
    tab.add_argument("--runs", default=None, help="a directory written by `aor run` instead of the records")
    tab.add_argument("--models", nargs="+", default=None, help="models to average over (default: all six)")
    tab.add_argument("--methods", nargs="+", default=list(TABLE1), choices=list(TABLE1))
    tab.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="seeds of the runs")
    tab.add_argument("--boot", type=int, default=2000, help="bootstrap resamples")
    tab.add_argument("--per-model", action="store_true", help="also print the Mean of each model")
    tab.add_argument("--intervals", action="store_true", help="also print the 95%% intervals")
    tab.set_defaults(func=_table)

    run = sub.add_parser("run", help="run decoders and AoR over the five tasks with one model (GPU)")
    run.add_argument("--model", required=True, help=f"a short name ({', '.join(MODELS)}) or a Hugging Face id")
    run.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    run.add_argument("--methods", nargs="+", default=list(TABLE1),
                     choices=sorted(set(TABLE1) | set(METHODS)), metavar="METHOD",
                     help=f"default: the rows of Table 1 ({' '.join(TABLE1)})")
    run.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    run.add_argument("--split", default="test", choices=["test", "dev"])
    run.add_argument("--out", default="runs")
    run.add_argument("--device", default=0, help="CUDA device index or name")
    run.add_argument("--limit", type=int, default=None, help="only the first N examples of each split")
    run.add_argument("--max-batch", type=int, default=64)
    run.set_defaults(func=_run)

    ask = sub.add_parser("ask", help="answer one question about a passage with AoR (GPU)")
    ask.add_argument("--model", default="qwen3-8b")
    ask.add_argument("--passage", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--option", action="append", help="an option of a multiple-choice question, repeated in order")
    ask.add_argument("--seed", type=int, default=0)
    ask.add_argument("--device", default=0)
    ask.set_defaults(func=_ask)

    args = parser.parse_args(argv)
    if args.command == "table" and args.records is None:
        from .table import RECORDS

        args.records = RECORDS
    if getattr(args, "device", None) is not None and str(args.device).isdigit():
        args.device = int(args.device)
    return args.func(args)
