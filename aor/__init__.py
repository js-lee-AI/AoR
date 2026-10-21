"""aor: Accept-or-Repair, a training-free decoding procedure for questions about a passage.

    from aor import Example, PassageCheck

    ex = Example(question="when did ireland gain its independence from england",
                 passage="In 1971 , after the Irish War of Independence ...")
    PassageCheck(ex).supports("1922")          # False: the passage does not contain the answer

    from aor.hf import AcceptOrRepair           # needs torch and transformers
    aor = AcceptOrRepair.from_pretrained("qwen3-8b")
    trace = aor(ex, seed=0)                     # trace.answer, trace.action, trace.repairs

The support test, the memory veto and the procedure need only the standard library. Decoding needs torch and
transformers (aor.hf, aor.decoding), the task loaders need datasets (aor.data), and the tables need numpy (aor.stats,
aor.table); these modules are imported on first use.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .example import Example
from .methods import AOR_REPAIRS, METHODS, TABLE1, Method, get_method
from .procedure import Draw, Trace, accept_or_repair
from .prompts import build_prompt
from .scoring import f1_score, parse_letter, score
from .support import PassageCheck, normalize, supported

# names that need torch and transformers, imported on first attribute access (PEP 562)
_LAZY = {"AcceptOrRepair": "hf", "Decoder": "hf"}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "Example", "PassageCheck", "normalize", "supported",
    "accept_or_repair", "Trace", "Draw",
    "Method", "METHODS", "TABLE1", "AOR_REPAIRS", "get_method", "build_prompt",
    "score", "f1_score", "parse_letter",
    "AcceptOrRepair", "Decoder",
]
