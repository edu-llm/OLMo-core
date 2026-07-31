"""
Pure-PyTorch, **fully differentiable** implementation of the KDA + ``R``-Householder recurrence.

This is the *reference* counterpart of the fused Triton kernel in
:mod:`olmo_core.nn.attention.kda_householder` (which implements both a forward and a backward, and
is the backend you want for training). It is built exclusively out of ordinary
differentiable ``torch`` ops -- a Python loop over time and over the ``R`` Householder factors,
with every state update performed *out of place* -- so autograd derives the backward pass. There
is no custom :class:`torch.autograd.Function` and no hand-derived gradient anywhere in this file,
which is precisely the point: the backward pass cannot be subtly wrong in a way the forward pass
is not.

The recurrence is a transcription of ``probes/naive_kda_householder.naive_recurrent_kda_householder``
(the validated oracle for this op), using the *same* ``einsum`` / broadcast-and-reduce calls in the
same order, so the two agree to floating-point round-off rather than merely to a tolerance.

Per token ``t``, with ``R = num_householder`` and state ``S`` of shape ``[K, V]`` per
(batch, head)::

    S <- S * exp(g[:, t])[..., None]              # KDA per-channel decay, ONCE per token
    for r in 0 .. R - 1:                          # R delta / Householder updates
        kr, vr, br = k[:, t * R + r], v[:, t * R + r], beta[:, t * R + r]
        u = br * (vr - kr @ S)                    # reads the CURRENT S
        S <- S + outer(kr, u)
    o[:, t] = q[:, t] @ S                         # readout after all R updates

.. warning::
    This implementation is **slow and memory-hungry** by design. The Python loop runs
    ``T * (R + 1)`` autograd nodes per (batch, head) and every intermediate state is retained for
    backward, i.e. ``O(T * R * B * H * K * V)`` saved activations. It is the ground truth against
    which the fused Triton backward is validated, and it remains the only path that runs on CPU,
    supports ``float64`` (hence ``torch.autograd.gradcheck``), and is twice differentiable.
    Prefer the Triton backend for everything else -- it is ~400x faster at ``B2/T8192/R4``.

Numerical policy
----------------
All arithmetic happens in a *compute dtype* obtained by promoting the input dtypes against
``float32``: ``bfloat16``/``float16``/``float32`` inputs accumulate in ``float32`` (matching the
Triton kernel and the naive oracle), while ``float64`` inputs accumulate in ``float64``. The latter
is what makes ``torch.autograd.gradcheck`` in ``float64`` meaningful -- nothing silently truncates
the perturbations to ``float32``.
"""

from typing import List, Optional, Tuple

import torch

__all__ = ["kda_householder_torch"]


def _compute_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Promote the input dtypes against ``float32``.

    ``bfloat16``/``float16``/``float32`` all promote to ``float32``; ``float64`` wins over
    ``float32``. This keeps low-precision training numerics identical to the Triton kernel while
    letting a ``float64`` :func:`torch.autograd.gradcheck` run end-to-end in ``float64``.

    :param tensors: The input tensors whose dtypes should be promoted.

    :returns: The floating dtype to use for all internal arithmetic.
    """
    dtype = torch.float32
    for t in tensors:
        dtype = torch.promote_types(dtype, t.dtype)
    return dtype


def _recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int,
    initial_state: Optional[torch.Tensor],
    compute_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run the recurrence for a single (already ``scale``-multiplied, already cast) segment.

    Every operation is out of place, so this function is differentiable with respect to all five
    inputs and to ``initial_state``.

    :param q: Queries, ``[B, T, H, K]``, already multiplied by ``scale``.
    :param k: Keys, ``[B, T * R, H, K]``, interleaved along time.
    :param v: Values, ``[B, T * R, H, V]``, interleaved along time.
    :param g: Raw log-space per-channel decay, ``[B, T, H, K]``. ``exp`` is applied here.
    :param beta: Delta-rule step sizes, ``[B, T * R, H]``, interleaved along time.
    :param num_householder: Number of Householder / delta factors ``R`` per token.
    :param initial_state: Optional initial state, ``[B, H, K, V]``.
    :param compute_dtype: Floating dtype of all inputs, and of the returned tensors.

    :returns: ``(o, S)`` with shapes ``[B, T, H, V]`` and ``[B, H, K, V]``.
    """
    R = num_householder
    B, T, H, K = q.shape
    V = v.shape[-1]

    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=compute_dtype, device=q.device)
    else:
        S = initial_state

    outputs: List[torch.Tensor] = []
    for i in range(T):
        # 1. KDA: a single per-channel decay of the state for this token.
        S = S * g[:, i][..., None].exp()
        # 2. DeltaProduct: R successive rank-1 delta updates, each reading the *current* S.
        for j in range(R):
            k_ij, v_ij, b_ij = k[:, i * R + j], v[:, i * R + j], beta[:, i * R + j]
            S = S + torch.einsum(
                "b h k, b h v -> b h k v",
                b_ij[..., None] * k_ij,
                v_ij - (k_ij[..., None] * S).sum(-2),
            )
        # 3. Readout.
        outputs.append(torch.einsum("b h k, b h k v -> b h v", q[:, i], S))

    if outputs:
        o = torch.stack(outputs, dim=1)
    else:
        o = q.new_zeros(B, 0, H, V)
    return o, S


def kda_householder_torch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    num_householder: int = 1,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Differentiable KDA per-channel gated delta rule with ``R`` Householder factors per token.

    Drop-in, autograd-capable replacement for
    :func:`olmo_core.nn.attention.kda_householder.chunk_kda_householder` with the same argument
    semantics (including ``cu_seqlens`` in **token** units) and the same output shapes. Unlike the
    Triton path, gradients flow to ``q``, ``k``, ``v``, ``g``, ``beta`` and ``initial_state``.

    :param q: Queries of shape ``[B, T, H, K]``.
    :param k: Keys of shape ``[B, T * R, H, K]``, interleaved along the time axis.
    :param v: Values of shape ``[B, T * R, H, V]``, interleaved along the time axis.
    :param g: Log-space **per-channel** forget gate of shape ``[B, T, H, K]``, one entry per token
        (*not* interleaved). This is the raw log-decay: ``exp`` is applied internally, and no
        cumsum is taken anywhere.
    :param beta: Delta-rule step sizes of shape ``[B, T * R, H]``, interleaved along time.
    :param num_householder: Number of Householder / delta factors ``R`` applied per token.
    :param scale: Query scaling factor. Defaults to ``K ** -0.5``.
    :param initial_state: Optional initial state of shape ``[N, H, K, V]``.
    :param output_final_state: Whether to also return the final state.
    :param cu_seqlens: Optional cumulative sequence lengths of shape ``[N + 1]`` in **token**
        units, for variable-length batches. Requires ``q.shape[0] == 1``. The state is reset at
        every sequence boundary, exactly as in the Triton kernel.

    :returns: ``(o, final_state)`` where ``o`` has shape ``[B, T, H, V]`` and dtype ``q.dtype``,
        and ``final_state`` has shape ``[N, H, K, V]`` in the internal compute dtype
        (``float32`` for low-precision inputs, ``float64`` for ``float64`` inputs) if
        ``output_final_state`` else ``None``.

    :raises AssertionError: on a dtype or shape violation.
    :raises ValueError: if ``cu_seqlens`` is given with a batch size other than 1, or if the
        number of initial states disagrees with the number of sequences.
    """
    R = num_householder
    assert R >= 1, f"num_householder must be >= 1, got {R}"
    assert q.ndim == 4, f"expected q to be 4-D [B, T, H, K], got {tuple(q.shape)}"
    B, T, H, K = q.shape
    V = v.shape[-1]
    assert k.shape == (B, T * R, H, K), f"expected k {(B, T * R, H, K)}, got {tuple(k.shape)}"
    assert v.shape == (B, T * R, H, V), f"expected v {(B, T * R, H, V)}, got {tuple(v.shape)}"
    assert beta.shape == (B, T * R, H), f"expected beta {(B, T * R, H)}, got {tuple(beta.shape)}"
    assert g.shape == (B, T, H, K), f"expected g {(B, T, H, K)}, got {tuple(g.shape)}"

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        assert cu_seqlens.ndim == 1, f"expected cu_seqlens to be 1-D, got {tuple(cu_seqlens.shape)}"

    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    if initial_state is not None:
        if initial_state.shape[0] != N:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input "
                f"sequences, i.e., {N} rather than {initial_state.shape[0]}."
            )
        assert initial_state.shape == (
            N,
            H,
            K,
            V,
        ), f"expected initial_state {(N, H, K, V)}, got {tuple(initial_state.shape)}"

    if scale is None:
        scale = K**-0.5

    out_dtype = q.dtype
    dtype = _compute_dtype(q, k, v, g, beta)
    q, k, v, g, beta = (x.to(dtype) for x in (q, k, v, g, beta))
    q = q * scale
    h0 = None if initial_state is None else initial_state.to(dtype)

    if cu_seqlens is None:
        o, final_state = _recurrence(q, k, v, g, beta, R, h0, dtype)
    else:
        bounds = [int(x) for x in cu_seqlens.tolist()]
        o_parts: List[torch.Tensor] = []
        state_parts: List[torch.Tensor] = []
        for n in range(len(bounds) - 1):
            bos, eos = bounds[n], bounds[n + 1]
            o_n, s_n = _recurrence(
                q[:, bos:eos],
                k[:, bos * R : eos * R],
                v[:, bos * R : eos * R],
                g[:, bos:eos],
                beta[:, bos * R : eos * R],
                R,
                None if h0 is None else h0[n : n + 1],
                dtype,
            )
            o_parts.append(o_n)
            state_parts.append(s_n)
        o = torch.cat(o_parts, dim=1)
        final_state = torch.cat(state_parts, dim=0)

    return o.to(out_dtype), final_state if output_final_state else None
