"""Batched generation over several prompts at once, with shared sampling noise.

An example has one prompt for plain sampling and two for a contrastive decoder: the prompt with the passage (stream 0)
and the contrasted prompt (stream 1). All streams of a batch run in one forward pass with a KV cache, and a rule turns
their next-token log-probabilities into the score that is sampled from. Plain sampling is the one-stream case, so every
decoder uses the same loop.

Sampling noise is counter based: the Gumbel noise of a token is a hash of (seed, example uid, decoding step, token id).
A token is drawn by adding this noise to the filtered score and taking the argmax, which samples from the decoder's
distribution while giving decoders with the same distribution at a step the same token. Up to floating-point rounding,
scores do not depend on which other examples share a batch, so runs of different decoders are paired example by example.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .methods import Method
from .models import Sampling

Rule = Callable[[torch.Tensor], torch.Tensor]

_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------- noise

def _mix32(x: torch.Tensor) -> torch.Tensor:
    """lowbias32 integer hash on an int64 tensor holding uint32 values."""
    x = x & _M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & _M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & _M32
    x = x ^ (x >> 16)
    return x


def step_keys(seed: int, uids: torch.Tensor, step: int) -> torch.Tensor:
    """One 32-bit key for each row, from (seed, example uid, step)."""
    base = _mix32(uids.to(torch.int64) * 0x9E3779B1 + (seed * 0x85EBCA6B + 0x165667B1))
    return _mix32(base + (step * 0xC2B2AE35 + 0x27D4EB2F))


def uniform_noise(keys: torch.Tensor, vocab: int) -> torch.Tensor:
    """Uniform noise in (0, 1) of shape [len(keys), vocab]. It uses 23 random bits, so (k + 0.5) / 2**23 is exact in
    float32 and never reaches 0 or 1, and the Gumbel transform stays finite."""
    idx = torch.arange(vocab, device=keys.device, dtype=torch.int64)
    h = _mix32(keys[:, None] + _mix32(idx[None, :] + 0x6A09E667))
    h = _mix32(h ^ keys[:, None])
    return ((h >> 9).to(torch.float32) + 0.5) / 8388608.0


def gumbel_noise(keys: torch.Tensor, vocab: int) -> torch.Tensor:
    return -torch.log(-torch.log(uniform_noise(keys, vocab)))


# --------------------------------------------------------------------------- filtering

def filter_scores(scores: torch.Tensor, sp: Sampling) -> torch.Tensor:
    """Temperature, then top-k, then top-p, in the order of the Transformers logits warpers."""
    x = scores / sp.temperature
    vocab = x.size(-1)
    if sp.top_k and sp.top_k < vocab:
        kth = torch.topk(x, sp.top_k, dim=-1).values[:, -1:]
        x = x.masked_fill(x < kth, float("-inf"))
    if sp.top_p < 1.0:
        sx, si = torch.sort(x, descending=True, dim=-1)
        p = torch.softmax(sx, dim=-1)
        drop_sorted = (p.cumsum(-1) - p) >= sp.top_p
        drop = torch.zeros_like(drop_sorted).scatter(1, si, drop_sorted)
        x = x.masked_fill(drop, float("-inf"))
    return x


def _pick(scores: torch.Tensor, sp: Sampling, keys: torch.Tensor) -> torch.Tensor:
    x = filter_scores(scores, sp)
    return (x + gumbel_noise(keys, x.size(-1))).argmax(-1)


# --------------------------------------------------------------------------- decoding rules

def make_rule(method: Method) -> Rule:
    """Score function of a decoder. It takes log-probabilities of shape [S, B, V] (stream, row, token) and returns
    scores of shape [B, V]."""
    g = method.strength
    if method.rule == "plain":
        return lambda lp: lp[0]
    if method.rule == "cad":
        return lambda lp: lp[0] + g * (lp[0] - lp[1])
    if method.rule == "adacad":
        def adacad(lp: torch.Tensor) -> torch.Tensor:
            p, v = lp[0], lp[1]
            pp, pv = p.exp(), v.exp()
            lm = (0.5 * (pp + pv)).clamp_min(1e-30).log()
            js = 0.5 * (pp * (p - lm)).sum(-1) + 0.5 * (pv * (v - lm)).sum(-1)      # nats
            return p + (g * js / math.log(2)).unsqueeze(-1) * (p - v)
        return adacad
    if method.rule == "coiecd":
        log_ratio = -math.log(method.param("scale"))

        def coiecd(lp: torch.Tensor) -> torch.Tensor:
            # The shift of a token is its information content under x minus the entropy under x'; a softmax of the
            # shifts gives the bounded distribution. Tokens above `scale` times its maximum, and, when more than one
            # token lies below its minimum divided by `scale`, those tokens too, violate the entropy constraint and are
            # scored log p_x + gamma (log p_x - log p_x'); every other token log p_x' + gamma (log p_x - log p_x').
            pc, p0 = lp[0], lp[1]
            ent0 = -(p0.exp() * p0).nansum(-1, keepdim=True)
            lpd = (-pc - ent0).log_softmax(dim=-1)
            low = lpd < lpd.min(-1, keepdim=True).values + log_ratio
            high = lpd > lpd.max(-1, keepdim=True).values - log_ratio
            lone = low.sum(-1, keepdim=True) == 1
            violate = high | (low & ~lone)
            return torch.where(violate, pc, p0) + g * (pc - p0)
        return coiecd
    if method.rule == "nwcad":
        tau = method.param("threshold")
        k_closed, k_passage = method.param("closed_book_margin"), method.param("passage_margin")

        def nwcad(lp: torch.Tensor) -> torch.Tensor:
            # Stage 1: the two distributions agree (JS divergence in bits at most tau) and the closed-book one is
            # confident (top-1 margin at least k_closed): sample from the closed-book distribution. Stage 2: sample
            # from the distribution with the passage when its margin is at least k_passage, and else from the CAD score.
            pc, p0 = lp[0], lp[1]
            qc, q0 = pc.exp(), p0.exp()
            lm = (0.5 * (qc + q0)).clamp_min(1e-30).log()
            js = (0.5 * (qc * (pc - lm)).sum(-1) + 0.5 * (q0 * (p0 - lm)).sum(-1)) / math.log(2)
            tc, t0 = qc.topk(2, dim=-1).values, q0.topk(2, dim=-1).values
            stage1 = (js <= tau) & ((t0[:, 0] - t0[:, 1]) >= k_closed)
            use_passage = (~stage1) & ((tc[:, 0] - tc[:, 1]) >= k_passage)
            cad = pc + g * (pc - p0)
            return torch.where(stage1[:, None], p0, torch.where(use_passage[:, None], pc, cad))
        return nwcad
    raise ValueError(f"unknown rule {method.rule!r}")


# --------------------------------------------------------------------------- generation

@dataclass
class Generation:
    tokens: List[List[int]]        # generated ids without the stop token, in input order
    forward_tokens: int            # tokens processed over all streams: prompts and every decoding step
    seconds: float


def kv_bytes_per_token(model) -> int:
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * kv_heads * head_dim * 2      # keys and values in bfloat16


def plan_batches(lengths: Sequence[int], n_streams: int, max_new: int, kv_bytes: int, budget_bytes: int,
                 max_batch: int) -> List[List[int]]:
    """Batches of example indices, longest prompts first, sized so that the KV cache fits the memory budget."""
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, i = [], 0
    while i < len(order):
        per_row = n_streams * (lengths[order[i]] + max_new) * kv_bytes
        b = int(max(1, min(max_batch, budget_bytes // max(per_row, 1))))
        batches.append(order[i:i + b])
        i += b
    return batches


@torch.no_grad()
def _run_batch(model, streams, idx, uids, rule, sp, seed, max_new, stops, pad_id, masks, compact):
    from transformers import DynamicCache

    dev = next(model.parameters()).device
    S, B = len(streams), len(idx)
    L = max(len(streams[s][i]) for s in range(S) for i in idx)
    ids = torch.full((S * B, L), pad_id, dtype=torch.long)
    attn = torch.zeros((S * B, L), dtype=torch.long)
    for s in range(S):
        for b, i in enumerate(idx):
            p = streams[s][i]
            ids[s * B + b, L - len(p):] = torch.tensor(p, dtype=torch.long)
            attn[s * B + b, L - len(p):] = 1
    ids, attn = ids.to(dev), attn.to(dev)
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    cache = DynamicCache()
    out = model(input_ids=ids, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=True,
                logits_to_keep=1)
    logits = out.logits[:, -1, :]
    del out, ids, pos
    n_fwd = int(attn.sum())

    ban = None
    if masks is not None and any(masks[i] for i in idx):
        vocab = logits.size(-1)
        ban = torch.zeros((B, vocab), dtype=torch.bool, device=dev)
        for b, i in enumerate(idx):
            if masks[i]:
                ban[b, [t for t in masks[i] if t < vocab]] = True

    stop_t = torch.tensor(stops, device=dev)
    gen = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    n_gen = torch.zeros(B, dtype=torch.long, device=dev)
    rows = torch.arange(B, device=dev)               # current row -> position in the batch
    fin = torch.zeros(B, dtype=torch.bool, device=dev)
    uid_t = torch.tensor([uids[i] for i in idx], dtype=torch.long, device=dev)
    Bc = B
    for t in range(max_new):
        lp = torch.log_softmax(logits.float(), dim=-1).view(S, Bc, -1)
        scores = rule(lp)
        if ban is not None:
            scores = scores.masked_fill(ban, float("-inf"))
        nxt = _pick(scores, sp, step_keys(seed, uid_t, t))
        is_stop = torch.isin(nxt, stop_t)
        gen[rows, t] = torch.where(fin, torch.full_like(nxt, pad_id), nxt)
        n_gen[rows] += ((~fin) & (~is_stop)).long()
        fin = fin | is_stop
        if bool(fin.all()) or t == max_new - 1:
            break
        if compact and Bc >= 8 and int(fin.sum()) >= max(4, Bc // 4) and t % 8 == 0:
            # drop finished rows from the cache and the batch
            keep = (~fin).nonzero().squeeze(1)
            sel = torch.cat([s * Bc + keep for s in range(S)])
            cache.batch_select_indices(sel)
            attn = attn[sel]
            nxt, rows, fin, uid_t = nxt[keep], rows[keep], fin[keep], uid_t[keep]
            if ban is not None:
                ban = ban[keep]
            Bc = int(keep.numel())
        tok_in = nxt.repeat(S)[:, None]
        pos = attn.sum(-1, keepdim=True)
        attn = torch.cat([attn, torch.ones((S * Bc, 1), dtype=attn.dtype, device=dev)], dim=1)
        out = model(input_ids=tok_in, attention_mask=attn, position_ids=pos, past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1, :]
        n_fwd += S * Bc
        del out
    gen_l, n_l = gen.cpu().tolist(), n_gen.cpu().tolist()
    return [gen_l[b][:n_l[b]] for b in range(B)], n_fwd


def generate(model, streams: Sequence[Sequence[Sequence[int]]], uids: Sequence[int], *, rule: Rule,
             sampling: Sampling, seed: int, max_new_tokens: int, stop_ids: Sequence[int], pad_id: int,
             masks: Optional[Sequence[Sequence[int]]] = None, max_batch: int = 64, kv_fraction: float = 0.55,
             compact: bool = True, log: Optional[Callable[[str], None]] = None) -> Generation:
    """Decode N examples. ``streams[s][i]`` is the token ids of prompt s of example i. ``masks[i]``, if given, lists
    token ids that example i may not produce at any step."""
    S, N = len(streams), len(streams[0])
    if not all(len(s) == N for s in streams) or len(uids) != N:
        raise ValueError("every stream and the uids need one entry per example")
    dev = next(model.parameters()).device
    free = torch.cuda.mem_get_info(dev)[0] if dev.type == "cuda" else 8 * 2 ** 30
    lengths = [max(len(streams[s][i]) for s in range(S)) for i in range(N)]
    pending = plan_batches(lengths, S, max_new_tokens, kv_bytes_per_token(model), int(free * kv_fraction), max_batch)
    tokens: List[Optional[List[int]]] = [None] * N
    n_fwd, t0, done = 0, time.time(), 0
    while pending:
        idx = pending.pop(0)
        try:
            tk, nf = _run_batch(model, streams, idx, uids, rule, sampling, seed, max_new_tokens, list(stop_ids),
                                pad_id, masks, compact)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(idx) == 1:
                raise
            h = len(idx) // 2
            pending[:0] = [idx[:h], idx[h:]]
            if log:
                log(f"out of memory at a batch of {len(idx)}; retrying as {h} + {len(idx) - h}")
            continue
        for b, i in enumerate(idx):
            tokens[i] = tk[b]
        n_fwd += nf
        done += len(idx)
        if log:
            log(f"  {done}/{N} examples, {time.time() - t0:.0f}s")
    return Generation(tokens=tokens, forward_tokens=n_fwd, seconds=time.time() - t0)  # type: ignore[arg-type]
