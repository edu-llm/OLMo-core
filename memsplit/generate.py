"""Greedy decoding with store interception. Length-grouped, never padded.

## Why this file exists

The previous generation left-padded every prompt in a batch to the batch maximum
with `EOT` and ran prefill with **no attention mask**, justified in a comment as
"the pad tokens are ordinary EOTs the model has seen as separators". `EOT` is the
document separator, so a prompt sitting behind enough pads reads as the start of
a fresh document and the model invents its premises. The measured effect was
**3/64 = 4.7% batched against 60/64 = 93.8% one prompt at a time** on the same
checkpoint and the same items, monotone in pad length -- exact at 0-16 pads,
broken by 32. Every generative accuracy the project reported went through that
path and was withdrawn.

The fix is structural rather than a correction: decode in groups of **exactly
equal prompt length**, so no padding exists and no mask is required. Grouping
costs one extra forward pass per distinct length and nothing else.

`test_generate.py` asserts that batching never changes a generation. That
assertion is the point of this module; do not weaken it, and do not add a
padding path "for throughput".

## Store protocol

A two-state machine per sequence, run only on model-chosen tokens:

    FREE      --<|db_start|>-->    IN_QUERY   (begin recording the key)
    IN_QUERY  --<|db_retrieve|>--> resolve, then FREE
    IN_QUERY  --any other-->       accumulate into the key

On a hit the resolved value is *force-appended* (` value` + `<|db_end|>`). Forced
tokens still pass through `forward_step`, so the KV cache stays consistent, and
they count against `max_new`. On a miss the sequence returns to FREE having
emitted nothing -- the model then continues freely, which is what produces the
observed "re-address or trail off" behaviour. A key longer than
`QUERY_TOKEN_CAP` is malformed: return to FREE and record it.

`organizer=None` disables the whole path. That is the closed-book condition.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

QUERY_TOKEN_CAP = 32

_FREE = "free"
_IN_QUERY = "in_query"


@dataclass
class _Seq:
    state: str = _FREE
    query_ids: list[int] = field(default_factory=list)
    force: list[int] = field(default_factory=list)
    out: list[int] = field(default_factory=list)
    done: bool = False


def _new_stats() -> dict:
    """Counters, split so the store-detached condition stays measurable.

    `n_query_spans` counts complete `<|db_start|> key <|db_retrieve|>` spans the
    model emitted, **whether or not a store was attached**. `n_lookups` counts
    resolutions actually attempted against a store, so it is 0 when
    `organizer is None`.

    Keeping these separate is not bookkeeping pedantry. With the store detached,
    `n_query_spans` is the split arm's *addressing* rate, which is the quantity
    that distinguishes "the weights hold no value" from "the model has forgotten
    how to ask" -- and H1 needs both readings. Collapsing them into one counter
    makes the closed-book condition look like a store full of misses.
    """
    return {
        "n_query_spans": 0,
        "n_lookups": 0,
        "n_hits": 0,
        "n_misses": 0,
        "n_malformed": 0,
    }


def _advance(seq: _Seq, tid: int, tok, organizer, stats: dict) -> None:
    """Update one sequence's store state after it *chose* token `tid`."""
    if seq.state == _FREE:
        if tid == tok.DB_START:
            seq.state = _IN_QUERY
            seq.query_ids = []
        return

    # _IN_QUERY
    if tid == tok.DB_RETRIEVE:
        stats["n_query_spans"] += 1
        seq.state = _FREE
        key = tok.decode(seq.query_ids)
        seq.query_ids = []
        if organizer is None:
            # Closed book: the span was emitted and is counted, but nothing is
            # resolved and nothing is injected. The model continues freely.
            return
        stats["n_lookups"] += 1
        value = organizer.lookup(key)
        if value is None:
            stats["n_misses"] += 1
            logger.debug("store miss for key %r", key)
        else:
            stats["n_hits"] += 1
            seq.force.extend(tok.encode(" " + value) + [tok.DB_END])
        return

    seq.query_ids.append(tid)
    if len(seq.query_ids) > QUERY_TOKEN_CAP:
        stats["n_malformed"] += 1
        logger.debug("malformed query, %d tokens without retrieve", len(seq.query_ids))
        seq.state = _FREE
        seq.query_ids = []


def _decode_group(
    model, tok, prompt_ids: list[list[int]], max_new: int, organizer, device,
    stop_at_eot: bool,
) -> tuple[list[list[int]], dict]:
    """Decode a group of equal-length prompts. No padding, so no mask."""
    lengths = {len(p) for p in prompt_ids}
    if len(lengths) != 1:
        raise AssertionError(f"group is not length-homogeneous: {sorted(lengths)}")

    seqs = [_Seq() for _ in prompt_ids]
    stats = _new_stats()
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)

    cache = None
    with torch.no_grad():
        for step in range(max_new):
            logits, cache = model.forward_step(x, cache)
            nxt = torch.argmax(logits[:, -1, :], dim=-1).tolist()

            step_ids: list[int] = []
            for seq, chosen in zip(seqs, nxt):
                if seq.done:
                    step_ids.append(tok.EOT)
                    continue
                if seq.force:
                    # A forced token: emitted, fed back, but not re-interpreted
                    # by the state machine (it is the store's output, not the
                    # model's choice).
                    tid = seq.force.pop(0)
                    seq.out.append(tid)
                    step_ids.append(tid)
                    continue
                tid = chosen
                if stop_at_eot and tid == tok.EOT:
                    seq.done = True
                    step_ids.append(tok.EOT)
                    continue
                seq.out.append(tid)
                _advance(seq, tid, tok, organizer, stats)
                step_ids.append(tid)

            if all(s.done for s in seqs):
                break
            x = torch.tensor([[t] for t in step_ids], dtype=torch.long, device=device)

    return [s.out for s in seqs], stats


def generate_batch_with_stats(
    model,
    tok,
    prompts: list[str],
    max_new: int,
    organizer,
    device,
    stop_at_eot: bool = True,
    max_group_size: int | None = None,
) -> tuple[list[str], dict]:
    """Greedy-decode continuations for `prompts`, grouped by exact length.

    Returns `(texts, stats)`. Texts exclude the prompt and the stopping EOT but
    include any forced lookup values. `max_new` counts every appended token,
    forced ones included.

    Prompts are grouped by token length; each group is decoded with no padding.
    Results are reassembled in input order, so the return value does not depend
    on input ordering or on `max_group_size` -- which is what
    `test_generate.py::test_batching_never_changes_a_generation` checks.
    """
    if not prompts:
        return [], _new_stats()

    encoded = [tok.encode(p) for p in prompts]
    by_len: dict[int, list[int]] = defaultdict(list)
    for i, ids in enumerate(encoded):
        by_len[len(ids)].append(i)

    out: list[list[int] | None] = [None] * len(prompts)
    stats = _new_stats()

    for length in sorted(by_len):
        idxs = by_len[length]
        chunks = (
            [idxs]
            if max_group_size is None
            else [idxs[i : i + max_group_size] for i in range(0, len(idxs), max_group_size)]
        )
        for chunk in chunks:
            got, s = _decode_group(
                model, tok, [encoded[i] for i in chunk], max_new, organizer, device,
                stop_at_eot,
            )
            for i, ids in zip(chunk, got):
                out[i] = ids
            for k in stats:
                stats[k] += s[k]

    assert all(o is not None for o in out)
    if stats["n_lookups"] != stats["n_hits"] + stats["n_misses"]:
        raise AssertionError(
            f"lookup accounting broken: {stats}"
        )
    return [tok.decode(o) for o in out], stats


def generate_batch(model, tok, prompts, max_new, organizer, device, **kw) -> list[str]:
    texts, _ = generate_batch_with_stats(
        model, tok, prompts, max_new, organizer, device, **kw
    )
    return texts
