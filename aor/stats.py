"""Aggregation of the main comparison and paired cluster-bootstrap intervals (numpy only).

Scores are held as ``scores[method][model][task]``, an array of shape [seeds, examples] with values in [0, 1] over the
scored test examples of the task, in a fixed order. A task score is the mean over seeds and examples, averaged over
models; a group score averages its tasks, and Mean averages the two groups.

Intervals follow Appendix C of the paper. The per-example difference between two methods is averaged over seeds and
models. Clusters of examples (a question, or a passage for SQuAD) are resampled 2,000 times with replacement; the same
resampling indices serve every model and seed, and tasks with the same clusters (NQ-Swap and NQ-Original) share them.
A group difference averages its tasks within each resample, Mean averages the two groups, and the 95% interval takes the
2.5 and 97.5 percentiles.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence

import numpy as np

GROUPS: Dict[str, Sequence[str]] = {"counterfactual": ("nqswap", "faitheval"),
                                    "ordinary": ("nqorig", "squad", "hotpotqa")}

Scores = Mapping[str, Mapping[str, Mapping[str, np.ndarray]]]


def _mean(values) -> float:
    return float(np.mean(list(values)))


def summarize(scores: Scores, method: str, models: Sequence[str], groups: Mapping[str, Sequence[str]] = GROUPS
              ) -> dict:
    """Task, group and Mean scores of one method in points, and the Mean of each model."""
    tasks = [t for ts in groups.values() for t in ts]
    cell = {(m, t): 100.0 * float(np.asarray(scores[method][m][t]).mean()) for m in models for t in tasks}
    per_task = {t: _mean(cell[(m, t)] for m in models) for t in tasks}
    per_group = {g: _mean(per_task[t] for t in ts) for g, ts in groups.items()}
    per_model = {m: _mean(_mean(cell[(m, t)] for t in ts) for ts in groups.values()) for m in models}
    return {"tasks": per_task, "groups": per_group, "mean": _mean(per_group.values()), "per_model": per_model}


def _interval(point: float, boots: np.ndarray) -> dict:
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(point), "lo": float(lo), "hi": float(hi), "significant": bool(lo > 0 or hi < 0)}


def paired_bootstrap(scores: Scores, a: str, b: str, models: Sequence[str], clusters: Mapping[str, Sequence[str]],
                     groups: Mapping[str, Sequence[str]] = GROUPS, n_boot: int = 2000, seed: int = 0) -> dict:
    """Difference a - b in points for every group and for Mean, with 95% intervals."""
    rng = np.random.default_rng(seed)
    draws: Dict[tuple, np.ndarray] = {}
    per_task = {}
    for t in [t for ts in groups.values() for t in ts]:
        diffs = [np.asarray(scores[a][m][t]).mean(0) - np.asarray(scores[b][m][t]).mean(0) for m in models]
        d = np.mean(diffs, axis=0) * 100.0
        uniq, cid = np.unique([str(c) for c in clusters[t]], return_inverse=True)
        k = len(uniq)
        if d.shape[0] != len(cid):
            raise ValueError(f"{t}: {d.shape[0]} scores but {len(cid)} cluster labels")
        key = tuple(uniq.tolist())
        if key not in draws:
            draws[key] = rng.integers(0, k, size=(n_boot, k))
        idx = draws[key]
        total = np.bincount(cid, weights=d, minlength=k)
        count = np.bincount(cid, minlength=k).astype(float)
        per_task[t] = (float(d.mean()), total[idx].sum(1) / count[idx].sum(1))
    out, parts = {}, {}
    for g, ts in groups.items():
        parts[g] = (_mean(per_task[t][0] for t in ts), np.mean([per_task[t][1] for t in ts], axis=0))
        out[g] = _interval(*parts[g])
    out["mean"] = _interval(_mean(p[0] for p in parts.values()), np.mean([p[1] for p in parts.values()], axis=0))
    return out


def cost_ratios(costs: Mapping[str, Mapping[str, Mapping[str, Sequence[float]]]], method: str, models: Sequence[str],
                tasks: Sequence[str], base: str = "sampling") -> Optional[float]:
    """Tokens processed per example relative to sampling, averaged over models, tasks and seeds.
    ``costs[method][model][task]`` lists the tokens per example of each seed."""
    try:
        ratios = [c / s for m in models for t in tasks
                  for c, s in zip(costs[method][m][t], costs[base][m][t])]
    except KeyError:
        return None
    return float(np.mean(ratios)) if ratios else None
