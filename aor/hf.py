"""Decoders and Accept-or-Repair on a Hugging Face Transformers model.

    from aor import Example
    from aor.hf import AcceptOrRepair

    aor = AcceptOrRepair.from_pretrained("Qwen/Qwen3-8B")
    trace = aor(Example(question="...", passage="..."), seed=0)
    print(trace.answer, trace.action)
"""

from __future__ import annotations

import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from .decoding import Generation, generate, make_rule
from .example import Example
from .methods import AOR_REPAIRS, Method, get_method
from .models import REVISIONS, Sampling, resolve_model
from .procedure import Trace, accept_or_repair
from .prompts import build_prompt

#: maximum new tokens for short free-form and for multiple-choice answers
MAX_NEW_FREE, MAX_NEW_MC = 48, 32
_LETTER_STRIP = " ().:\t\n"


@dataclass
class Output:
    texts: List[str]
    forward_tokens: int
    n_generated: List[int]


class Decoder:
    """A model, its tokenizer and its sampling settings, with every decoder of `aor.methods.METHODS`."""

    def __init__(self, model, tokenizer, sampling: Sampling, *, max_batch: int = 64, kv_fraction: float = 0.55):
        self.model, self.tokenizer, self.sampling = model, tokenizer, sampling
        self.max_batch, self.kv_fraction = max_batch, kv_fraction
        self._thinking_off = "qwen3" in (getattr(tokenizer, "name_or_path", "") or "").lower()
        self._encoded: Dict[Tuple[Example, str], List[int]] = {}
        self._letters: Optional[Dict[str, List[int]]] = None
        self.stop_ids = self._stop_ids()
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    @classmethod
    def from_pretrained(cls, name: str, *, device: Union[str, int] = 0, sampling: Optional[Sampling] = None,
                        revision: Optional[str] = None, attn_implementation: str = "sdpa", **kwargs) -> "Decoder":
        """Load a model in bfloat16. ``revision`` defaults to the revision of the paper's runs for its six models,
        and to the latest one for any other model."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_id, default = resolve_model(name)
        revision = revision or REVISIONS.get(hf_id)
        tok = AutoTokenizer.from_pretrained(hf_id, revision=revision)
        model = AutoModelForCausalLM.from_pretrained(hf_id, revision=revision, dtype=torch.bfloat16,
                                                     device_map={"": device},
                                                     attn_implementation=attn_implementation).eval()
        return cls(model, tok, sampling or default, **kwargs)

    # ------------------------------------------------------------------ prompts

    def render(self, text: str) -> str:
        kw = {"enable_thinking": False} if self._thinking_off else {}
        return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                  add_generation_prompt=True, **kw)

    def encode(self, example: Example, kind: str) -> List[int]:
        key = (example, kind)
        if key not in self._encoded:
            text = self.render(build_prompt(example, kind))
            self._encoded[key] = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        return self._encoded[key]

    def clear_cache(self) -> None:
        self._encoded.clear()

    def _stop_ids(self) -> List[int]:
        ids = set()
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if isinstance(eos, int):
            ids.add(eos)
        elif eos:
            ids.update(int(e) for e in eos)
        if self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        return sorted(ids)

    def letter_ids(self, letter: str) -> List[int]:
        """Token ids whose text, stripped of spaces, tabs, line breaks, parentheses, periods and colons, is the
        letter. Masking them keeps decoding from naming that option."""
        if self._letters is None:
            table: Dict[str, List[int]] = {c: [] for c in string.ascii_uppercase}
            n = min(len(self.tokenizer), self.model.config.vocab_size)
            for i in range(n):
                s = self.tokenizer.decode([i]).strip(_LETTER_STRIP)
                if len(s) == 1 and s in table:
                    table[s].append(i)
            self._letters = table
        return self._letters.get(letter, [])

    # ------------------------------------------------------------------ decoding

    def generate(self, examples: Sequence[Example], method: Union[str, Method], *, seed: int = 0,
                 masked_letters: Optional[Sequence[Optional[str]]] = None, max_new_tokens: Optional[int] = None,
                 log=None) -> Output:
        """Decode the examples with one method. ``masked_letters[i]`` is an option letter that example i may not
        produce, or None."""
        m = get_method(method) if isinstance(method, str) else method
        if not examples:
            return Output([], 0, [])
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_MC if all(ex.is_multiple_choice for ex in examples) else MAX_NEW_FREE
        streams = [[self.encode(ex, kind) for ex in examples] for kind in m.streams]
        masks = None
        if masked_letters is not None and any(masked_letters):
            masks = [self.letter_ids(c) if c else [] for c in masked_letters]
        gen: Generation = generate(self.model, streams, [ex.uid for ex in examples], rule=make_rule(m),
                                   sampling=self.sampling, seed=seed, max_new_tokens=max_new_tokens,
                                   stop_ids=self.stop_ids, pad_id=self.pad_id, masks=masks, max_batch=self.max_batch,
                                   kv_fraction=self.kv_fraction, log=log)
        texts = [self.tokenizer.decode(t, skip_special_tokens=True) for t in gen.tokens]
        return Output(texts, gen.forward_tokens, [len(t) for t in gen.tokens])


@dataclass
class AoRResult:
    traces: List[Trace]
    forward_tokens: int


class AcceptOrRepair:
    """Accept-or-Repair over a `Decoder`.

    ``repairs`` names the repair decoders in order, CAD then ID by default, at the strengths of `aor.methods`
    unless ``strengths`` overrides them. The closed-book answer is sampled with ``closed_book_seed`` whatever the run's
    seed, so the runs of several seeds share it, as in the paper.
    """

    def __init__(self, decoder: Decoder, repairs: Sequence[str] = AOR_REPAIRS,
                 strengths: Optional[Dict[str, float]] = None, closed_book_seed: Optional[int] = 0):
        self.decoder = decoder
        strengths = strengths or {}
        self.repairs = [get_method(r, strengths.get(r)) for r in repairs]
        self.closed_book_seed = closed_book_seed

    @classmethod
    def from_pretrained(cls, name: str, *, device: Union[str, int] = 0, sampling: Optional[Sampling] = None,
                        revision: Optional[str] = None, **kwargs) -> "AcceptOrRepair":
        return cls(Decoder.from_pretrained(name, device=device, sampling=sampling, revision=revision), **kwargs)

    def run(self, examples: Sequence[Example], seed: int = 0, log=None) -> AoRResult:
        examples = list(examples)
        fwd = [0]
        cb_seed = seed if self.closed_book_seed is None else self.closed_book_seed

        def draw(method, s):
            def f(idx, masked=None):
                out = self.decoder.generate([examples[i] for i in idx], method, seed=s, masked_letters=masked, log=log)
                fwd[0] += out.forward_tokens
                return out.texts
            return f

        repairs = [(m.name, draw(m, seed)) for m in self.repairs]
        traces = accept_or_repair(examples, draw("sampling", seed), draw("closed_book", cb_seed), repairs)
        return AoRResult(traces, fwd[0])

    def __call__(self, examples: Union[Example, Sequence[Example]], seed: int = 0):
        if isinstance(examples, Example):
            return self.run([examples], seed).traces[0]
        return self.run(examples, seed).traces
