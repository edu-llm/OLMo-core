"""
B3 — per-position drop accounting for capacity-based MoE dispatch.

WHY THIS IS A SEPARATE FILE. Three lanes wanted ``router.py::compute_metrics``, and the ruling in
``agents/contracts/file-ownership.md`` is that L5 owns the emitting method while L3 owns drop
accounting "wherever it lives in dispatch". A new module is the only place that satisfies both: the
computation lives here, next to the dispatch that produces the inputs, and the emission stays L5's.
It also means this can be unit-tested without constructing a router.

WHAT GETS DROPPED, TRACED RATHER THAN ASSUMED
---------------------------------------------
``ParallelMLP`` pads every expert to ``expert_capacity`` slots and silently discards assignments
beyond it. The discard is not a mask or a zero-fill -- it is an *absence*. ``binned_gather``
allocates ``(num_experts, expert_capacity, d_model)`` and launches one Triton program per
``(expert, slot)``; ``_binned_copy`` early-returns when ``entry_idx >= num_tokens``, and reads
``indices + start + entry_idx`` otherwise. An assignment sitting at bin offset ``>= capacity`` is
never addressed by any program, so it never enters the buffer and ``binned_scatter`` cannot return
it. Nothing in the loss says why it got slightly worse.

So: **the survivors of expert e are the first ``expert_capacity`` entries of bin e, in sorted
order.** ``indices_and_bins`` sorts by expert id *only* -- it carries an upstream ``TODO`` saying a
secondary sort by expert weight is absent -- so within a bin the order is the assignment order, and
truncation keeps the lowest assignment indices.

THE AXIS IS THE WHOLE POINT, AND THE OBVIOUS CHOICE IS WRONG
------------------------------------------------------------
It is tempting to histogram the normalized *flattened* assignment index. That would be wrong in a
way that looks right, so it is worth writing down.

Both dispatch call sites reduce a ``(batch_size, seq_len, d_model)`` activation with
``x.view(-1, d_model)``, so the flattened index is row-major over ``(batch, seq)``::

    j -> token t = j // top_k -> document d = t // seq_len, position p = t % seq_len

Each document therefore owns a *contiguous* run of ``seq_len * top_k`` flattened indices. Since
truncation keeps the lowest indices, what overflow actually discards is **the last documents of the
microbatch across all of their positions** -- not the late positions of every sequence. At the
default 16384-token microbatch with ``seq_len=2048`` that is 8 documents, so a 16-bucket histogram
over flattened index bins roughly one document per bucket, ramps steeply, and reports a **document
ordering artifact** under a metric named "position". A reader concludes that SWA plus long context
is compromised. The metric would be worse than absent.

Hence two series on two axes, and neither pretends to be the other:

``drop_by_position``
    binned on ``p = (j // top_k) % seq_len``, folded across documents. This is the series that
    speaks to the SWA / long-context claim.
``drop_by_doc_index``
    binned on ``d = (j // top_k) // seq_len``. This is where the truncation bias actually lands.

EVERY BUCKET IS A RATE, NOT A COUNT. A count histogram confounds drop bias with bucket population:
buckets hold different numbers of assignments whenever ``seq_len`` does not divide evenly by the
bucket count, and then a *uniform* drop rate reads as a sloped line. Dividing by each bucket's own
population makes the uniform-drop null **flat**, which is the only way a ramp is legible. Buckets
are relative (fixed count, normalized axis) so the series is comparable across sequence lengths and
across expert counts, per ``telemetry-schema.md`` rule 2.

THE NULL IS NOT ZERO, WHICH WAS PREDICTED WRONG AND THEN MEASURED. BOTH SERIES ARE UNGATED.
--------------------------------------------------------------------------------------------
The prediction written here first was that ``drop_by_position`` would be **flat** while
``drop_by_doc_index`` ramped. Measured on a forced overflow (L40S sm_89, 512 assignments per bucket,
binomial sd 0.0197):

    drop_by_doc_index   spread 0.363   =  18.5 sd   -- doc 0 loses nothing, docs 2-7 lose 33-36%
    drop_by_position    spread 0.107   =   5.46 sd  -- NOT noise: +0.047 tilt into the last quarter

So the positional axis carries **real signal**, and "flat" was too strong. The direction of the
framing above survives -- the document effect is **7.7x** larger, so binning the flattened order
would still be reporting the wrong axis -- but the positional null is a *tilt*, not a flat line.

The tilt has a mechanical cause rather than being noise or a second phenomenon: the last surviving
document is truncated **mid-document**, so it contributes its early positions and not its late ones,
which leaks a tail-ward tilt into the position-folded histogram. It is the same truncation seen
through a different projection.

**CONSEQUENCE, AND THE REASON IT IS WRITTEN IN THE SOURCE RATHER THAN ONLY IN A CONTRACT: both
series are deliberately UNGATED.** An assertion of the form "the positional axis is flat" fires on
every healthy run -- it would have been added in good faith on the strength of the original
prediction, and it would have produced a false alarm in production. If a gate is ever wanted here,
the discriminating condition is a positional ramp **large relative to** ``drop_by_doc_index``, not
non-flatness. The gated quantity is ``drop_frac`` (ceiling 1%), which is scalar and has a real null.

Generalizing -- and this is now a project-wide precedent rather than a note about this file, because
two lanes reached it independently within hours (here on a positional histogram, and on a raw CV in
the telemetry lane): **the null for a balance or drop metric is almost never zero. Measure the band
under a known-good condition before asserting against it, or do not assert.**

The corollary that came out of the capacity metric below is the same shape: assert
``effective >= configured``, never ``==``, because rounding legitimately moves the realized value in
one direction. A gate written against the value you *intended* rather than the one the code
*allocates* is a gate that fires on healthy runs.

ON SORT STABILITY, WHICH THIS FILE DELIBERATELY DOES NOT ASSUME
---------------------------------------------------------------
``indices_and_bins`` calls ``torch.sort`` without ``stable=True``, and every assignment routed to
one expert is an equivalent element under its key. PyTorch does not promise a tie order there. So
"survivors are earliest-by-position" is the upstream *intent* and the overwhelmingly likely
behaviour, but it is not an API guarantee, and a test asserting the dropped set is *exactly* the
positional tail could pass locally and fail intermittently -- the worst failure shape available.

This module therefore **measures** rather than assumes: it derives each assignment's bin offset from
the permutation actually produced, so it reports the truth for whatever tie order the sort chose.
:func:`drop_position_summary` additionally returns mean dropped/kept positions, which supports a
one-sided distributional assertion that is robust to tie order.

NO HOST SYNC. Every operation here is a device-side tensor op. There is no ``.item()``, no
``.cpu()``, and no data-dependent control flow, so this is safe to call every step in a training
loop -- unlike the dropless fallback in ``mlp.py``, which synchronizes once per expert.
"""

from typing import NamedTuple, Optional

import torch

__all__ = ["DropAccounting", "compute_drop_accounting", "drop_position_summary"]


#: Number of relative buckets in both histograms. Relative rather than absolute so the series is
#: comparable across sequence lengths and expert counts.
DEFAULT_NUM_BUCKETS = 16


class DropAccounting(NamedTuple):
    """
    Per-step drop accounting for one MoE block. All fields are device tensors; nothing here has
    been synchronized to the host.

    :param drop_frac: Scalar. Dropped assignments / total assignments. This is an **exact** count
        for the step, not the accumulated upper bound that ``router.compute_metrics`` reports.
    :param drop_by_position: Shape ``(num_buckets,)``. Dropped / total assignments per bucket of
        **within-sequence position**, folded across documents. ``nan`` for empty buckets, so an
        empty bucket is not reported as a measured zero.
    :param drop_by_doc_index: Shape ``(num_buckets,)``. Same, bucketed on **document index**.
    :param dropped_count: Scalar, exact number of dropped assignments.
    :param total_count: Scalar, total assignments considered.
    """

    drop_frac: torch.Tensor
    drop_by_position: torch.Tensor
    drop_by_doc_index: torch.Tensor
    dropped_count: torch.Tensor
    total_count: torch.Tensor


def _bin_offsets(indices: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """
    Recover each assignment's offset *within its expert's bin*, in the sorted order that
    ``indices_and_bins`` produced.

    ``bins`` is the inclusive cumulative sum of ``batch_size_per_expert``, so bin ``e`` occupies
    sorted slots ``[bins[e-1], bins[e])``. A slot's offset is therefore its sorted rank minus its
    bin's start, which is what ``_binned_copy`` compares against ``expert_capacity`` when it
    early-returns.

    Computed from ``bins`` by a searchsorted rather than by expanding ``bin_ids``, so it costs
    ``O(n log E)`` with no large intermediate and does not require the caller to have kept
    ``bin_ids`` -- ``ParallelMLP.forward_once`` deletes it.
    """
    num_slots = indices.numel()
    sorted_rank = torch.arange(num_slots, device=indices.device)

    # Which bin each sorted slot belongs to: the first bin whose (inclusive) end exceeds the rank.
    # `bins` is non-decreasing, so this is exact even for empty bins.
    bin_of_slot = torch.searchsorted(bins.to(torch.int64).contiguous(), sorted_rank, right=True)

    # Start of each slot's bin: the previous bin's cumulative end, or 0 for bin 0.
    bin_start = torch.cat(
        [bins.new_zeros(1, dtype=torch.int64), bins.to(torch.int64)[:-1]],
    )[bin_of_slot.clamp_max(bins.numel() - 1)]

    return sorted_rank - bin_start


def _bucketed_rate(
    values: torch.Tensor,
    dropped: torch.Tensor,
    *,
    num_bins: int,
    num_buckets: int,
) -> torch.Tensor:
    """
    Per-bucket dropped/total rate over an integer axis ``values`` in ``[0, num_bins)``.

    Returns ``nan`` for buckets with no assignments. A measured zero and an empty bucket must not
    look the same -- the same argument the router makes for omitting its drop metric entirely on the
    dropless path.
    """
    if num_bins <= 0:
        return torch.full((num_buckets,), float("nan"), device=values.device, dtype=torch.float32)

    # Relative bucketing: scale the axis onto [0, num_buckets). clamp guards the exact endpoint.
    bucket = (values.to(torch.float32) * (num_buckets / num_bins)).to(torch.int64)
    bucket = bucket.clamp_(0, num_buckets - 1)

    total = torch.zeros(num_buckets, device=values.device, dtype=torch.float32)
    total.scatter_add_(0, bucket, torch.ones_like(bucket, dtype=torch.float32))

    dropped_per_bucket = torch.zeros(num_buckets, device=values.device, dtype=torch.float32)
    dropped_per_bucket.scatter_add_(0, bucket, dropped.to(torch.float32))

    # nan rather than 0 where a bucket is empty.
    return torch.where(
        total > 0,
        dropped_per_bucket / total.clamp_min(1.0),
        torch.full_like(total, float("nan")),
    )


def compute_drop_accounting(
    *,
    indices: torch.Tensor,
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
    seq_len: Optional[int],
    num_buckets: int = DEFAULT_NUM_BUCKETS,
) -> DropAccounting:
    """
    Compute exact per-step drop accounting for a capacity-based dispatch.

    Call this with the *same* ``indices``, ``bins`` and ``expert_capacity`` that reach
    ``ops.binned_gather``, or the accounting describes a dispatch that did not happen.

    :param indices: The permutation from ``indices_and_bins``, shape ``(num_tokens * top_k,)``.
        Position ``r`` holds the flattened assignment index occupying sorted slot ``r``.
    :param bins: Inclusive cumulative sum of ``batch_size_per_expert``, shape ``(num_experts,)``.
    :param expert_capacity: Slots per expert. Assignments at bin offset ``>= expert_capacity`` are
        dropped.
    :param top_k: Experts per token, needed to map an assignment index back to a token.
    :param seq_len: Sequence length, needed to split a token index into document and position. If
        ``None`` the positional axes cannot be formed and both histograms are returned as ``nan``
        -- reported as unavailable rather than silently binned on the wrong axis.
    """
    if indices.numel() == 0:
        nan = torch.full((num_buckets,), float("nan"), device=indices.device, dtype=torch.float32)
        zero = torch.zeros((), device=indices.device, dtype=torch.float32)
        return DropAccounting(zero, nan, nan.clone(), zero.clone(), zero.clone())

    offsets = _bin_offsets(indices, bins)
    dropped = offsets >= expert_capacity

    total_count = torch.full((), float(indices.numel()), device=indices.device, dtype=torch.float32)
    dropped_count = dropped.sum(dtype=torch.float32)
    drop_frac = dropped_count / total_count

    if seq_len is None or seq_len <= 0:
        nan = torch.full((num_buckets,), float("nan"), device=indices.device, dtype=torch.float32)
        return DropAccounting(drop_frac, nan, nan.clone(), dropped_count, total_count)

    # `indices` holds flattened assignment indices; recover the token, then split it. Note this
    # indexes by assignment, so a token appears up to `top_k` times and is counted once per
    # assignment -- which is the right denominator, because capacity drops assignments.
    token = indices.to(torch.int64) // top_k
    position = token % seq_len
    doc_index = token // seq_len

    # THE DOCUMENT COUNT COMES FROM THE SHAPE, NOT FROM THE DATA. `doc_index.max().item()` is the
    # obvious spelling and it is a host sync; this function promises none, because it runs every
    # step. The microbatch holds `numel // top_k` tokens, so the count is `ceil(tokens / seq_len)`
    # -- known from shapes alone.
    num_tokens = indices.numel() // top_k
    num_docs_static = max(1, -(-num_tokens // seq_len))

    # THE DOCUMENT HISTOGRAM USES AT MOST ONE BUCKET PER DOCUMENT. Asking for 16 buckets over 8
    # documents leaves every other bucket empty, and empty buckets are reported as `nan` -- so the
    # series comes out as a comb of alternating values and holes, which is legible to nobody and
    # invites a reader to mistake the holes for missing data. Capping at the document count keeps
    # every bucket populated. `drop_by_position` needs no such cap: seq_len is far larger than the
    # bucket count in every configuration this project runs.
    return DropAccounting(
        drop_frac=drop_frac,
        drop_by_position=_bucketed_rate(
            position, dropped, num_bins=seq_len, num_buckets=num_buckets
        ),
        drop_by_doc_index=_bucketed_rate(
            doc_index,
            dropped,
            num_bins=num_docs_static,
            num_buckets=min(num_buckets, num_docs_static),
        ),
        dropped_count=dropped_count,
        total_count=total_count,
    )


def drop_position_summary(
    *,
    indices: torch.Tensor,
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
    seq_len: int,
) -> dict:
    """
    Mean within-sequence position of dropped vs kept assignments.

    This exists for the assertion, not the dashboard. The exact claim "the dropped set is the
    positional tail" rests on the tie order of a ``torch.sort`` call that does not pass
    ``stable=True``, so asserting it exactly risks an intermittent failure. Comparing *mean*
    positions is one-sided and robust to tie order: under overflow, dropped assignments should sit
    at a materially higher mean position than kept ones, whatever the sort did with ties.

    Returns ``nan`` means when either set is empty, so "no drops" is distinguishable from
    "drops at position 0".
    """
    offsets = _bin_offsets(indices, bins)
    dropped = offsets >= expert_capacity
    position = (indices.to(torch.int64) // top_k) % seq_len

    n_dropped = dropped.sum(dtype=torch.float32)
    n_kept = (~dropped).sum(dtype=torch.float32)
    pos_f = position.to(torch.float32)
    nan = torch.full((), float("nan"), device=indices.device, dtype=torch.float32)

    return {
        "mean_position_dropped": torch.where(
            n_dropped > 0, (pos_f * dropped).sum() / n_dropped.clamp_min(1.0), nan
        ),
        "mean_position_kept": torch.where(
            n_kept > 0, (pos_f * (~dropped)).sum() / n_kept.clamp_min(1.0), nan
        ),
        "num_dropped": n_dropped,
        "num_kept": n_kept,
    }
