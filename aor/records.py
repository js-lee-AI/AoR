"""Reading and writing per-example results.

Two sources share one layout in memory, ``scores[method][model][task]`` of shape [seeds, scored examples]:

* the paper's records in ``records/``: ``splits.json`` lists the development, test and scored uids of every task with
  the cluster label of each scored example, ``table1/<model>/<task>.csv.gz`` holds one column of per-example scores
  for each method and seed, and ``table1/cost.json`` the mean number of tokens processed per example in
  every run;
* runs written by ``aor run``: ``<root>/<model>/<task>/<split>/<method>__s<seed>.jsonl`` with one record per example,
  and a ``.json`` summary next to it.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def load_splits(path) -> dict:
    """``splits.json``: for each task, the uids of the development, test and scored examples and the cluster label of
    each scored example."""
    return json.loads(Path(path).read_text(encoding="utf-8"))["tasks"]


def read_table_csv(path) -> Tuple[List[int], Dict[str, Dict[int, np.ndarray]]]:
    """(uids, {method: {seed: scores}}) from one ``<task>.csv.gz`` of the paper's records."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    uids = [int(r[0]) for r in body]
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for j, col in enumerate(header[1:], start=1):
        method, seed = col.rsplit("@s", 1)
        out.setdefault(method, {})[int(seed)] = np.array([float(r[j]) for r in body])
    return uids, out


def write_table_csv(path, uids: Sequence[int], columns: Dict[str, Sequence[float]]) -> None:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["uid", *columns])
    for i, u in enumerate(uids):
        w.writerow([u, *(_fmt(columns[c][i]) for c in columns)])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        f.write(buf.getvalue().encode())


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else repr(float(x))


def load_paper_records(root, models: Sequence[str], tasks: Sequence[str], methods: Sequence[str],
                       seeds: Sequence[int] = (0, 1, 2)):
    """(scores, clusters, costs) of the paper's main comparison from ``records/``."""
    root = Path(root)
    splits = load_splits(root / "splits.json")
    scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {m: {} for m in methods}
    for model in models:
        for task in tasks:
            uids, cols = read_table_csv(root / "table1" / model / f"{task}.csv.gz")
            if uids != splits[task]["scored"]:
                raise ValueError(f"{model}/{task}: the rows do not follow the scored uids of splits.json")
            for m in methods:
                scores[m].setdefault(model, {})[task] = np.stack([cols[m][s] for s in seeds])
    clusters = {t: splits[t]["clusters"] for t in tasks}
    costs = json.loads((root / "table1" / "cost.json").read_text(encoding="utf-8"))
    return scores, clusters, costs


# --------------------------------------------------------------------------- runs

def run_path(root, model: str, task: str, split: str, method: str, seed: int, ext: str = ".jsonl") -> Path:
    return Path(root) / model / task / split / f"{method}__s{seed}{ext}"


def write_run(root, model: str, task: str, split: str, method: str, seed: int, records: List[dict],
              summary: dict) -> None:
    p = run_path(root, model, task, split, method, seed)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(p)
    s = run_path(root, model, task, split, method, seed, ".json")
    tmp = s.with_name(s.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    tmp.replace(s)


def run_done(root, model: str, task: str, split: str, method: str, seed: int) -> bool:
    return run_path(root, model, task, split, method, seed, ".json").exists()


def load_runs(root, models: Sequence[str], tasks: Sequence[str], methods: Sequence[str], seeds: Sequence[int],
              splits: dict, split: str = "test"):
    """(scores, clusters, costs) from runs, restricted to the scored examples listed in ``splits``. Raises
    FileNotFoundError naming the first missing run."""
    scores: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {m: {} for m in methods}
    costs: Dict[str, Dict[str, Dict[str, List[float]]]] = {m: {} for m in methods}
    for method in methods:
        for model in models:
            for task in tasks:
                scored = splits[task]["scored"]
                rows, cost = [], []
                for seed in seeds:
                    p = run_path(root, model, task, split, method, seed)
                    if not p.exists():
                        raise FileNotFoundError(f"no run at {p}")
                    by_uid = {}
                    with open(p, encoding="utf-8") as f:
                        for line in f:
                            r = json.loads(line)
                            by_uid[r["uid"]] = r["score"]
                    missing = [u for u in scored if u not in by_uid]
                    if missing:
                        raise ValueError(f"{p}: {len(missing)} scored examples missing, e.g. uid {missing[0]}")
                    rows.append([by_uid[u] for u in scored])
                    summary = json.loads(run_path(root, model, task, split, method, seed, ".json")
                                         .read_text(encoding="utf-8"))
                    cost.append(summary["forward_tokens"] / summary["n"])
                scores[method].setdefault(model, {})[task] = np.asarray(rows, dtype=float)
                costs[method].setdefault(model, {})[task] = cost
    clusters = {t: splits[t]["clusters"] for t in tasks}
    return scores, clusters, costs

