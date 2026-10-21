"""The five tasks of the main comparison, their development and test splits, and the examples that are scored.

Each task draws a fixed random pool of 1,000 examples (seed 1234) from one split of a Hugging Face dataset, pinned to
the revision the paper used: the first 200 form the development split and the next 800 the test split. Test examples
whose question, or for SQuAD whose passage, also occurs in the development split are left out of the scores, which leaves
724 of the 800 for NQ-Swap, NQ-Original and SQuAD. RPCD contrasts with the passage of the example
n // 3 + 1 places later in the pool of n examples.

Loading needs the ``datasets`` package (``pip install -e ".[run]"`` in a clone of the repository).
"""

from __future__ import annotations

import random
import re
import string
from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Sequence, Tuple

from .example import Example

_TAG = re.compile(r"<[^>]+>")

#: task groups of the main comparison
GROUPS: Dict[str, Tuple[str, ...]] = {"counterfactual": ("nqswap", "faitheval"),
                                      "ordinary": ("nqorig", "squad", "hotpotqa")}
TASKS: Tuple[str, ...] = GROUPS["counterfactual"] + GROUPS["ordinary"]
LABELS = {"nqswap": "NQ-Swap", "faitheval": "FaithEval", "nqorig": "NQ-Orig.", "squad": "SQuAD",
          "hotpotqa": "HotpotQA"}

#: Hugging Face datasets and the revisions the paper used
SOURCES = {
    "nqswap": ("pminervini/NQ-Swap", None, "dev", "3801dbf037c6f58b0e79246f2a8073ebe5656dad"),
    "nqorig": ("pminervini/NQ-Swap", None, "dev", "3801dbf037c6f58b0e79246f2a8073ebe5656dad"),
    "faitheval": ("Salesforce/FaithEval-counterfactual-v1.0", None, "test", "e655f7c8750aabe431eeeda63fa5c019a2567b18"),
    "squad": ("rajpurkar/squad", None, "validation", "7b6d24c440a36b6815f21b70d25016731768db1f"),
    "hotpotqa": ("hotpotqa/hotpot_qa", "distractor", "validation", "1908d6afbbead072334abe2965f91bd2709910ab"),
}


def _clean_html(text: str) -> str:
    return " ".join(_TAG.sub(" ", text).split())


def _nq(rows, original: bool) -> List[Example]:
    ctx, ans = ("org_context", "org_answer") if original else ("sub_context", "sub_answer")
    return [Example(r["question"], _clean_html(r[ctx]), uid=i, golds=tuple(r[ans])) for i, r in enumerate(rows)]


def _faitheval(rows) -> List[Example]:
    out = []
    letters = string.ascii_uppercase
    for i, r in enumerate(rows):
        labels, texts = list(r["choices"]["label"]), list(r["choices"]["text"])
        # options are lettered by position; a few items label them 1-4 while the key is a letter
        key = r["answerKey"] if r["answerKey"] in letters[:len(texts)] else letters[labels.index(r["answerKey"])]
        out.append(Example(r["question"], r["context"], options=tuple(texts), uid=i, gold_letter=key))
    return out


def _squad(rows) -> List[Example]:
    return [Example(r["question"], r["context"], uid=i, golds=tuple(dict.fromkeys(r["answers"]["text"])))
            for i, r in enumerate(rows)]


def _hotpotqa(rows) -> List[Example]:
    out = []
    for i, r in enumerate(rows):
        paras = [f"{t}: {' '.join(s.strip() for s in sents)}"
                 for t, sents in zip(r["context"]["title"], r["context"]["sentences"])]
        out.append(Example(r["question"], "\n\n".join(paras), multi_passage=True, uid=i, golds=(r["answer"],)))
    return out


_BUILD: Dict[str, Callable] = {"nqswap": lambda rows: _nq(rows, False), "nqorig": lambda rows: _nq(rows, True),
                               "faitheval": _faitheval, "squad": _squad, "hotpotqa": _hotpotqa}

#: what makes two examples one cluster: they are resampled together and test examples that share it with the
#: development split are not scored
CLUSTER_KEY: Dict[str, Callable[[Example], str]] = {
    "nqswap": lambda e: e.question, "nqorig": lambda e: e.question, "hotpotqa": lambda e: e.question,
    "squad": lambda e: e.passage, "faitheval": lambda e: str(e.uid),
}


def load_task(task: str) -> List[Example]:
    """Every example of the task's source split, with uid = row index."""
    from datasets import load_dataset

    name, config, split, revision = SOURCES[task]
    rows = load_dataset(name, config, split=split, revision=revision) if config else \
        load_dataset(name, split=split, revision=revision)
    return _BUILD[task](rows)


@dataclass
class TaskSplit:
    task: str
    dev: List[Example]
    test: List[Example]
    scored: List[Example]          # the test examples that enter the scores

    def clusters(self, examples: Sequence[Example] = ()) -> List[str]:
        return [CLUSTER_KEY[self.task](e) for e in (examples or self.scored)]


def make_split(task: str, examples: Sequence[Example], n_dev: int = 200, n_test: int = 800,
               seed: int = 1234) -> TaskSplit:
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    pool = [examples[i] for i in order[:n_dev + n_test]]
    n = len(pool)
    shift = n // 3 + 1
    pool = [replace(ex, partner_passage=pool[(pos + shift) % n].passage) for pos, ex in enumerate(pool)]
    dev, test = pool[:n_dev], pool[n_dev:]
    key = CLUSTER_KEY[task]
    dev_keys = {key(e) for e in dev}
    return TaskSplit(task, dev, test, [e for e in test if key(e) not in dev_keys])


def load_split(task: str, **kwargs) -> TaskSplit:
    return make_split(task, load_task(task), **kwargs)
