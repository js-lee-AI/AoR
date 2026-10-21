"""Runs of the main comparison: decoders and AoR over the test split of the five tasks, one file per example set.

    python -m aor run --model qwen3-8b --out runs
    python -m aor table --runs runs --models qwen3-8b

Each run writes ``<out>/<model>/<task>/<split>/<method>__s<seed>.jsonl`` with one record per example (uid, reply,
score, and the number of generated tokens, or for AoR what the procedure did) and a ``.json`` summary with the score
over the scored examples and the tokens processed. Finished runs are skipped, so an interrupted job resumes where it
stopped.
"""

from __future__ import annotations

import platform
import time
from typing import Callable, Optional, Sequence

from . import __version__
from .data import TASKS, load_split
from .methods import TABLE1
from .models import short_name
from .procedure import Trace
from .records import run_done, write_run
from .scoring import score


def _versions() -> dict:
    import torch
    import transformers

    return {"aor": __version__, "python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__}


def trace_record(trace: Trace) -> dict:
    return {"action": trace.action, "decoder": trace.decoder, "initial": trace.initial,
            "closed_book": trace.closed_book, "veto": trace.veto, "masked_letter": trace.masked_letter,
            "repairs": [{"decoder": d.decoder, "reply": d.text, "supported": d.supported, "rejected": d.rejected}
                        for d in trace.repairs]}


def run_model(model: str, *, tasks: Sequence[str] = TASKS, methods: Sequence[str] = TABLE1,
              seeds: Sequence[int] = (0, 1, 2), out: str = "runs", split: str = "test", device=0,
              limit: Optional[int] = None, max_batch: int = 64, log: Callable[[str], None] = print) -> None:
    """Run every method and seed of one model over the tasks. ``limit`` keeps the first examples of each split, for a
    quick check; such runs are written under ``<split>-first<limit>``, so they never take the place of full runs."""
    from .hf import AcceptOrRepair, Decoder

    name = short_name(model)
    where = f"{split}-first{limit}" if limit else split
    decoder = None
    for task in tasks:
        pending = [(m, s) for m in methods for s in seeds if not run_done(out, name, task, where, m, s)]
        if not pending:
            log(f"{name}/{task}: done")
            continue
        if decoder is None:
            decoder = Decoder.from_pretrained(model, device=device, max_batch=max_batch)
            versions = _versions()
        ts = load_split(task)
        examples = ts.test if split == "test" else ts.dev
        if limit:
            examples = examples[:limit]
        scored = {e.uid for e in ts.scored} if split == "test" else {e.uid for e in examples}
        for method, seed in pending:
            t0 = time.time()
            if method == "aor":
                res = AcceptOrRepair(decoder).run(examples, seed=seed)
                replies, fwd = [t.answer for t in res.traces], res.forward_tokens
                extra = [trace_record(t) for t in res.traces]
            else:
                o = decoder.generate(examples, method, seed=seed)
                replies, fwd, extra = o.texts, o.forward_tokens, [{"n_tokens": n} for n in o.n_generated]
            records = [{"uid": ex.uid, "reply": r, "score": score(r, ex), **x}
                       for ex, r, x in zip(examples, replies, extra)]
            kept = [r["score"] for r in records if r["uid"] in scored]
            summary = {"model": name, "hf_id": decoder.model.config.name_or_path,
                       "revision": getattr(decoder.model.config, "_commit_hash", None), "task": task, "split": split,
                       "limit": limit, "method": method, "seed": seed, "n": len(records), "n_scored": len(kept),
                       "score": sum(kept) / max(len(kept), 1), "forward_tokens": fwd,
                       "seconds": round(time.time() - t0, 1), "sampling": vars(decoder.sampling), "versions": versions}
            write_run(out, name, task, where, method, seed, records, summary)
            log(f"{name}/{task}/{where} {method} s{seed}: {100 * summary['score']:.1f} on {len(kept)} scored examples, "
                f"{summary['seconds']:.0f}s")
        decoder.clear_cache()
