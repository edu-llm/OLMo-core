"""
Chunked linear-plus-cross-entropy, for when the logits tensor is the memory constraint.

WHY THIS FILE EXISTS RATHER THAN A CONFIG FLAG.

``LMLossImplementation.fused_linear`` already solves this problem, and better than this file
does, by calling Liger-Kernel's fused triton kernel (``functional/cross_entropy_loss.py:53-64``).
**Liger is not in the platform image.** ``src/Dockerfile:93-95`` installs it; that is the AI2
Beaker image. ``.edullm/Dockerfile`` -- the one the eduLLM platform actually builds and the one a
platform training run executes inside -- installs torch, the project, boto3, botocore[crt] and
edullm-data, and never mentions liger-kernel. So on the image we run,
``loss_implementation=fused_linear`` does not degrade: it raises ``RuntimeError`` from
``cross_entropy_loss.py:104-105`` at the first micro-step, after the queue wait and the GPU
allocation have been paid for.

Adding liger-kernel to ``.edullm/Dockerfile`` would be the smaller change and it is the right
long-term answer. It is not this file's call to make: that Dockerfile is L3's under
``contracts/file-ownership.md``, liger builds triton kernels at install time (new build surface in
an image whose scan gate has ``exceptions: []``), and the platform's image-scan review is an admin
block when it trips. This file is the no-new-dependency path, so that the flagship is not blocked
on a dependency negotiation. See ``agents/lanes/L6-memory-ce/STATUS.md``.

WHAT IT SAVES, AT R3 / mb=8192 / V=100,352 (numbers derived in
``agents/lanes/L6-memory-ce/evidence/E2-memory-budget-of-record.md``).

The stock path (``lm_head.py:253-262``) holds **four** ``(N, V)`` tensors at once -- the bf16
``w_out`` output saved for its weight gradient, the fp32 upcast at
``cross_entropy_loss.py:35``, the fp32 ``log_softmax`` output that ``F.cross_entropy`` saves for
backward, and the fp32 incoming gradient. At N=8192, V=100,352 that is
1.53 + 3.06 + 3.06 + 3.06 = **10.72 GiB**, which is larger than the entire per-GPU persistent
state (5.71 GiB) on the A100-40GB the platform's ``gpu-8xa100`` actually provides.

Note that the frequently-quoted ``8192 x 100352 x 4 = 3.06 GiB`` is *one* of those four. The
constraint is 3.5x that figure.

With ``chunk_size`` tokens per chunk the peak becomes ``4 * chunk_size * V * 4 B``-ish plus the
returned per-token losses (``N`` floats, negligible). At ``chunk_size=1024`` that is ~1.34 GiB.

HOW CORRECTNESS IS OBTAINED -- THIS IS THE LOAD-BEARING DESIGN DECISION.

The obvious implementation is a custom ``autograd.Function`` that computes ``softmax - onehot``
by hand and accumulates ``grad_weight`` itself. That is what Liger does, and it is the fastest.
It also means hand-written gradient math for the one term that is 43.7 % of this model's counted
FLOPs, landing on a $21.96/hr shape where OOM and wrong answers alike get no retry.

So this uses ``torch.utils.checkpoint`` on a per-chunk closure instead. **Autograd computes every
gradient**; the only thing this file decides is *when* the logits are allowed to exist. Each
chunk's projection and loss are recomputed in backward rather than stored. The consequences,
stated rather than buried:

* **Per-token loss values are bitwise identical to the unchunked path.** Each chunk calls the
  same ``cross_entropy_loss`` on the same rows with ``reduction="none"``, so no token's loss
  depends on which chunk it landed in. This is the strong equivalence claim and it is testable.
* **The reduced scalar is not bitwise identical**, and cannot be: summing per-token losses in a
  different order than ``F.cross_entropy(reduction="sum")`` sums them internally changes float
  rounding. Agreement is to summation-order tolerance, ~1e-6 relative at fp32 and N=8192. Any
  implementation that claims bitwise equality of the reduced scalar is not chunking.
* **Cost: one extra forward projection per chunk.** The lm_head is 43.7 % of counted FLOPs/token
  at R3, forward is a third of that, so recompute adds ~14.6 % of forward = **~4.9 % of counted
  step FLOPs**. That is a real, predictable throughput cost and it must appear in any MFU
  comparison between this path and the stock one -- an MFU that improves because a recompute was
  added to the denominator-free numerator is a measurement error.

``z_loss`` is supported and is chunked identically: ``logsumexp(-1).pow(2)`` is per-token, so it
partitions exactly the same way. It is not optional to support -- ``train_on_corpus.py:1486``
defaults ``--z-loss-multiplier`` to 1e-5, so z-loss is **on** by default on this launcher.
"""

import logging
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .cross_entropy_loss import cross_entropy_loss

__all__ = ["chunked_linear_cross_entropy_loss", "DEFAULT_CE_CHUNK_SIZE"]

log = logging.getLogger(__name__)

#: Tokens per chunk. 1,024 keeps the live logits family near 1.3 GiB at V=100,352 while leaving
#: chunks large enough that the ``(chunk, d) @ (d, V)`` GEMM is not launch-bound: at V=100,352 the
#: N dimension of that GEMM is 100,352 regardless, so M=1,024 is already a well-shaped matmul.
#: Smaller chunks buy little memory and start costing kernel launches.
DEFAULT_CE_CHUNK_SIZE = 1024


def chunked_linear_cross_entropy_loss(
    _input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    chunk_size: int = DEFAULT_CE_CHUNK_SIZE,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross-entropy fused with the logits projection by *chunking* it, so that no ``(N, V)`` tensor
    is ever materialised. A drop-in replacement for ``cross_entropy_loss(linear(x), ...)`` that
    trades ~4.9 % of step FLOPs for ~9.4 GiB at R3/mb=8192.

    Unlike :func:`fused_linear_cross_entropy_loss` this needs **no** extra dependency -- it is
    plain torch plus ``torch.utils.checkpoint`` -- which is why it exists; see the module
    docstring for why Liger is unavailable on the platform image.

    :param _input: Hidden states of shape ``(N, D)``.
    :param weight: The projection weight of shape ``(V, D)``.
    :param labels: Target class indices of shape ``(N,)``.
    :param bias: Optional projection bias of shape ``(V,)``.
    :param ignore_index: Target value that does not contribute to the loss or its gradient.
    :param reduction: ``"mean"``, ``"sum"`` or ``"none"``.
    :param compute_z_loss: Also return the softmax auxiliary loss.
    :param z_loss_multiplier: Coefficient on the z-loss.
    :param chunk_size: Tokens per chunk. Peak logits memory is proportional to this.

    :returns: ``(ce_loss, z_loss)``, with shapes matching :func:`cross_entropy_loss` for the same
        ``reduction``.
    """
    if _input.ndim != 2:
        raise RuntimeError(f"expected '_input' to be 2D (N, D), found shape {tuple(_input.shape)}")
    if labels.ndim != 1:
        raise RuntimeError(f"expected 'labels' to be 1D (N,), found shape {tuple(labels.shape)}")
    if _input.shape[0] != labels.shape[0]:
        raise RuntimeError(
            f"'_input' has {_input.shape[0]} rows but 'labels' has {labels.shape[0]} entries"
        )
    if chunk_size < 1:
        raise RuntimeError(f"'chunk_size' must be positive, got {chunk_size}")

    N = _input.shape[0]

    # Per-token losses, computed chunk by chunk and concatenated. Reducing here rather than
    # inside the loop is deliberate: it keeps every token's loss bitwise identical to the
    # unchunked path, so the reduction is the ONLY place chunking is observable.
    ce_parts = []
    z_parts = []

    for start in range(0, N, chunk_size):
        stop = min(start + chunk_size, N)

        # `use_reentrant=False` is required, not stylistic: the reentrant implementation does not
        # support a closure over tensors that need grad and silently drops `weight`'s gradient in
        # some configurations. The non-reentrant version also composes with torch.compile.
        ce_chunk, z_chunk = checkpoint(
            _chunk_loss,
            _input[start:stop],
            weight,
            bias,
            labels[start:stop],
            ignore_index,
            compute_z_loss,
            z_loss_multiplier,
            use_reentrant=False,
        )
        ce_parts.append(ce_chunk)
        if compute_z_loss:
            z_parts.append(z_chunk)

    # shape: (N,)
    ce_per_token = torch.cat(ce_parts)
    z_per_token = torch.cat(z_parts) if compute_z_loss else None

    if reduction == "none":
        return ce_per_token, z_per_token

    # `cross_entropy_loss(reduction=...)` divides a "mean" by the number of NON-ignored tokens,
    # matching F.cross_entropy. Reproduce that exactly rather than dividing by N: at
    # `loss_reduction="sum"` (what the train module uses, train_module.py:406) it does not matter,
    # but at "mean" dividing by N instead of the unignored count is a silent scale error that
    # grows with the padding fraction.
    mask = labels != ignore_index

    if reduction == "sum":
        ce_loss = ce_per_token.sum()
        z_loss = z_per_token.sum() if z_per_token is not None else None
    elif reduction == "mean":
        denom = mask.sum().clamp(min=1)
        ce_loss = ce_per_token.sum() / denom
        z_loss = (z_per_token.sum() / denom) if z_per_token is not None else None
    else:
        raise NotImplementedError(f"unsupported reduction '{reduction}'")

    return ce_loss, z_loss


def _chunk_loss(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    labels: torch.Tensor,
    ignore_index: int,
    compute_z_loss: bool,
    z_loss_multiplier: float,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    One chunk: project, then per-token CE. Everything here is recomputed in backward, so the
    ``(chunk, V)`` logits exist only for the duration of one chunk's forward and one chunk's
    recompute.

    ``reduction="none"`` is what makes the per-token equivalence claim hold -- see the module
    docstring. A per-chunk ``"sum"`` would be cheaper by one ``cat`` and would make the result
    depend on the chunk boundaries.
    """
    logits = F.linear(x, weight, bias)

    ce, z = cross_entropy_loss(
        logits,
        labels,
        ignore_index=ignore_index,
        reduction="none",
        compute_z_loss=compute_z_loss,
        z_loss_multiplier=z_loss_multiplier,
    )

    if not compute_z_loss:
        # `checkpoint` cannot return None alongside a tensor in all torch versions; return a
        # zero-element tensor and let the caller drop it.
        return ce, ce.new_zeros(())

    assert z is not None
    # `cross_entropy_loss` with reduction="none" leaves z per-token but does NOT mask it
    # (functional/cross_entropy_loss.py:41-48 only masks for sum/mean). Mask here so the
    # caller's reduction over ignored positions matches the unchunked path.
    z = z * (labels != ignore_index)
    return ce, z
