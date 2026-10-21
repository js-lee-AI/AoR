"""Every run behind Table 1: sampling, the eight baselines and AoR on the five tasks with three seeds, for each
model (GPU).

    python experiments/run_main.py --models qwen3-8b --out runs
    python experiments/run_main.py --out runs                       # all six models, one after another
    python experiments/table1_main.py --runs runs --models qwen3-8b

A run is skipped when its summary exists, so the script resumes after an interruption. On one RTX A6000, the ten
methods took 53 minutes per seed with Qwen3-1.7B; larger models take longer.
"""

import argparse

from aor.data import TASKS
from aor.methods import TABLE1
from aor.models import MODELS
from aor.run import run_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=list(TABLE1), choices=list(TABLE1))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    device = int(a.device) if a.device.isdigit() else a.device
    for model in a.models:
        run_model(model, tasks=a.tasks, methods=a.methods, seeds=a.seeds, out=a.out, device=device)


if __name__ == "__main__":
    main()
