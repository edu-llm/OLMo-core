"""
Correctness tests for the KDA + ``R``-Householder Triton kernel (forward and backward).

The oracle is ``naive_recurrent_kda_householder`` from ``probes/naive_kda_householder.py``, which
lives *outside* this repository (sibling directory of the repo root). It is **imported**, not
copied: :func:`_load_oracle` locates it via ``$KDA_PROBES_DIR`` or the default sibling layout and
appends that directory to ``sys.path``. If neither path exists the GPU tests skip with a message
naming the env var, so nothing silently passes.

``triton`` is imported lazily inside each test (via ``pytest.importorskip``) so that collecting
this file never fails on a machine without triton installed. The differentiable backend lives in
:mod:`olmo_core.nn.attention.kda_householder_torch`, which imports no triton at all, so the
gradient tests below run unconditionally on CPU.
"""

import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import pytest
import torch
import torch.nn.functional as F

from olmo_core.testing import requires_gpu
from olmo_core.testing.utils import requires_fla

# bf16 inputs with float32 accumulation: |o| ~ 1 in these tests, and a single bf16 round-trip of
# the output is already ~4e-3, so 2e-2 is a few output-quantisation steps of slack.
ATOL = 2e-2
RTOL = 2e-2


def _load_oracle() -> Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Import ``naive_recurrent_kda_householder`` from the out-of-repo ``probes`` directory.

    :returns: the oracle function.
    """
    candidates = []
    env_dir = os.environ.get("KDA_PROBES_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    # src/test/nn/attention/<this file> -> parents[4] is the repo root, [5] its parent.
    candidates.append(Path(__file__).resolve().parents[5] / "probes")

    found = next((c for c in candidates if (c / "naive_kda_householder.py").is_file()), None)
    if found is None:
        pytest.skip(
            "Could not locate probes/naive_kda_householder.py (the correctness oracle). "
            f"Tried: {[str(c) for c in candidates]}. Set $KDA_PROBES_DIR to its directory."
        )
    if str(found) not in sys.path:
        sys.path.append(str(found))
    from naive_kda_householder import (  # type: ignore[import-not-found]
        naive_recurrent_kda_householder,
    )

    return naive_recurrent_kda_householder


def _load_oracle_recurrence() -> Callable[..., Tuple[torch.Tensor, torch.Tensor]]:
    """Import the oracle's dtype-parameterised ``_recurrence`` body from ``probes``.

    The oracle's *public* entry point hard-codes ``float32`` accumulation (matching the fla naive
    kernel convention), which floors any comparison at ``~5e-8`` even with ``float64`` inputs. Its
    ``_recurrence`` body takes an explicit ``compute_dtype``, documented for exactly this purpose,
    so running both sides in ``float64`` is what makes a ``1e-12`` tolerance meaningful.

    :returns: the oracle's ``_recurrence(q, k, v, g, beta, R, scale, initial_state, dtype)``.
    """
    _load_oracle()  # populates sys.path (and skips with a helpful message if absent)
    from naive_kda_householder import _recurrence  # type: ignore[import-not-found]

    return _recurrence


def _kda_householder_torch() -> Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Import the differentiable backend. Needs no triton, so this never skips.

    :returns: :func:`olmo_core.nn.attention.kda_householder_torch.kda_householder_torch`.
    """
    from olmo_core.nn.attention.kda_householder_torch import kda_householder_torch

    return kda_householder_torch


def _make_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    R: int,
    device: torch.device,
    seed: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Build a well-conditioned bf16 input set.

    ``k`` is L2-normalized (in float32, then cast) so the delta rule stays contractive, and ``g``
    is ``logsigmoid``-shaped so it is strictly negative, as in real KDA.

    :returns: ``(q, k, v, g, beta)``.
    """
    gen = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float32)

    q = rnd(B, T, H, K)
    k = rnd(B, T * R, H, K)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    g = F.logsigmoid(rnd(B, T, H, K))

    q = F.normalize(q, p=2, dim=-1).to(torch.bfloat16)
    k = F.normalize(k, p=2, dim=-1).to(torch.bfloat16)
    v = v.to(torch.bfloat16)
    beta = beta.to(torch.bfloat16)
    # `g` stays float32: the kernel and the oracle both consume it in float32.
    return q, k, v, g, beta


def _make_inputs_f64(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    R: int,
    seed: int = 0,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Build a well-conditioned ``float64`` CPU input set for the oracle / gradient tests.

    Same conditioning as :func:`_make_inputs` (L2-normalized ``q``/``k``, ``logsigmoid`` gate) but
    kept in ``float64`` throughout, which is what makes a finite-difference Jacobian meaningful.

    :returns: ``(q, k, v, g, beta)``.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    g = F.logsigmoid(rnd(B, T, H, K))

    tensors: Tuple[torch.Tensor, ...] = (q, k, v, g, beta)
    if requires_grad:
        tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    return tensors


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """Compare two tensors in float32 and report the max absolute difference on failure."""
    a, e = actual.float(), expected.float()
    max_diff = (a - e).abs().max().item()
    torch.testing.assert_close(
        a, e, atol=ATOL, rtol=RTOL, msg=lambda m: f"{what}: max|diff| = {max_diff:.3e}\n{m}"
    )


# --------------------------------------------------------------------------------------------
# GPU: Triton kernel vs the validated naive oracle
# --------------------------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_matches_oracle(R: int):
    """Triton forward must match ``naive_recurrent_kda_householder`` for ``R`` in ``{1, 2, 3}``."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    oracle = _load_oracle()
    device = torch.device("cuda")
    B, T, H, K, V = 2, 64, 3, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=R)

    o_tri, s_tri = chunk_kda_householder(
        q=q, k=k, v=v, g=g, beta=beta, num_householder=R, output_final_state=True
    )
    o_ref, s_ref = oracle(q, k, v, g, beta, num_householder=R, output_final_state=True)

    assert o_tri.shape == (B, T, H, V)
    assert o_tri.dtype == q.dtype
    assert s_tri is not None and s_tri.shape == (B, H, K, V)
    assert s_tri.dtype == torch.float32
    _assert_close(o_tri, o_ref, f"o (R={R})")
    assert s_ref is not None
    _assert_close(s_tri, s_ref, f"final_state (R={R})")


@requires_gpu
def test_kda_householder_initial_state():
    """A non-zero ``initial_state`` must be threaded through identically to the oracle."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    oracle = _load_oracle()
    device = torch.device("cuda")
    B, T, H, K, V, R = 2, 32, 2, 64, 64, 2
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=11)
    h0 = torch.randn(B, H, K, V, device=device, dtype=torch.float32) * 0.1

    o_tri, s_tri = chunk_kda_householder(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        num_householder=R,
        initial_state=h0,
        output_final_state=True,
    )
    o_ref, s_ref = oracle(
        q, k, v, g, beta, num_householder=R, initial_state=h0, output_final_state=True
    )
    _assert_close(o_tri, o_ref, "o (initial_state)")
    assert s_tri is not None and s_ref is not None
    _assert_close(s_tri, s_ref, "final_state (initial_state)")


@requires_gpu
def test_kda_householder_ragged_head_dims():
    """Exercise the ``K``/``V`` masking paths with dims that are not multiples of the tile sizes."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    oracle = _load_oracle()
    device = torch.device("cuda")
    # K = 48 -> BK = 64 (masked); V = 60 -> BV = 8, so the last V tile is partial.
    B, T, H, K, V, R = 1, 33, 2, 48, 60, 3
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=7)

    o_tri, _ = chunk_kda_householder(q=q, k=k, v=v, g=g, beta=beta, num_householder=R)
    o_ref, _ = oracle(q, k, v, g, beta, num_householder=R)
    _assert_close(o_tri, o_ref, "o (ragged K/V)")


@requires_gpu
def test_kda_householder_varlen():
    """``cu_seqlens`` (token units) must reproduce per-sequence oracle calls."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    oracle = _load_oracle()
    device = torch.device("cuda")
    H, K, V, R = 2, 64, 64, 2
    lengths = [16, 24, 8]
    T = sum(lengths)
    q, k, v, g, beta = _make_inputs(1, T, H, K, V, R, device, seed=5)
    cu_seqlens = torch.tensor([0, 16, 40, 48], device=device, dtype=torch.long)

    o_tri, _ = chunk_kda_householder(
        q=q, k=k, v=v, g=g, beta=beta, num_householder=R, cu_seqlens=cu_seqlens
    )

    start = 0
    for length in lengths:
        end = start + length
        o_ref, _ = oracle(
            q[:, start:end],
            k[:, start * R : end * R],
            v[:, start * R : end * R],
            g[:, start:end],
            beta[:, start * R : end * R],
            num_householder=R,
        )
        _assert_close(o_tri[:, start:end], o_ref, f"o (varlen seq [{start}, {end}))")
        start = end


@requires_gpu
@requires_fla
def test_kda_householder_backward_runs() -> None:
    """The triton backward produces finite gradients for every differentiable input.

    Previously this asserted ``NotImplementedError``; the backward kernel is now implemented
    and validated against the ``torch`` backend (see ``probes/gpu_bwd_accept.py``).
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    B, T, H, K, V, R = 2, 32, 2, 64, 64, 2
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device="cuda")
    for t_ in (q, k, v, g, beta):
        t_.requires_grad_(True)
    o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R)
    o.sum().backward()
    for name, t_ in zip(("q", "k", "v", "g", "beta"), (q, k, v, g, beta)):
        assert t_.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite gradient for {name}"


def test_kda_householder_r1_matches_fla_chunk_kda():
    """At ``R == 1`` the kernel must agree with ``fla.ops.kda.chunk_kda``.

    Convention matching (any of these being wrong causes a spurious failure):

    * ``use_gate_in_kernel=False`` -- ``g`` is the *precomputed* per-channel log decay, exactly
      what this kernel consumes. ``chunk_kda`` takes the within-chunk cumsum itself.
    * ``use_qk_l2norm_in_kernel=False`` on both sides, with ``q`` / ``k`` L2-normalized by the
      test *before* the bf16 cast, so both kernels see bit-identical inputs.
    * the same explicit ``scale``, rather than relying on both defaults being ``K ** -0.5``.
    """
    pytest.importorskip("triton")
    from fla.ops.kda import chunk_kda  # type: ignore[import-not-found]

    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    device = torch.device("cuda")
    B, T, H, K, V = 2, 128, 3, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, 1, device, seed=13)
    scale = K**-0.5

    o_mine, _ = chunk_kda_householder(q=q, k=k, v=v, g=g, beta=beta, num_householder=1, scale=scale)
    o_fla, _ = chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        use_qk_l2norm_in_kernel=False,
        use_gate_in_kernel=False,
    )
    _assert_close(o_mine, o_fla, "o (R=1 vs fla.chunk_kda)")


# --------------------------------------------------------------------------------------------
# CPU: argument-validation guards. No kernel launch happens, so no GPU is required -- but the
# module import still pulls in triton, hence the importorskip.
# --------------------------------------------------------------------------------------------


def _cpu_inputs(R: int = 2) -> dict:
    """Small CPU input set for the guard tests (never reaches a kernel launch)."""
    B, T, H, K, V = 1, 4, 2, 8, 8
    return dict(
        q=torch.randn(B, T, H, K, dtype=torch.bfloat16),
        k=torch.randn(B, T * R, H, K, dtype=torch.bfloat16),
        v=torch.randn(B, T * R, H, V, dtype=torch.bfloat16),
        g=torch.randn(B, T, H, K, dtype=torch.float32),
        beta=torch.randn(B, T * R, H, dtype=torch.bfloat16),
        num_householder=R,
    )


def _chunk_kda_householder() -> Any:
    """Lazily import the entry point, skipping if triton is unavailable."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    return chunk_kda_householder


def test_kda_householder_rejects_float32():
    """float32 ``q`` must be rejected, mirroring fla's gated-delta-product guard."""
    fn = _chunk_kda_householder()
    kwargs = _cpu_inputs()
    kwargs["q"] = kwargs["q"].float()
    with pytest.raises(AssertionError, match="does not support float32"):
        fn(**kwargs)


@pytest.mark.parametrize("bad", ["k", "v", "beta", "g"])
def test_kda_householder_rejects_bad_shapes(bad: str):
    """Each of the four interleaved/gate shape asserts must fire."""
    fn = _chunk_kda_householder()
    kwargs = _cpu_inputs()
    # Drop one time step, which breaks the T*R (or T) relationship for that tensor only.
    kwargs[bad] = kwargs[bad][:, :-1].contiguous()
    with pytest.raises(AssertionError, match=f"expected {bad} "):
        fn(**kwargs)


def test_kda_householder_rejects_bad_num_householder():
    """``num_householder`` must be at least 1."""
    fn = _chunk_kda_householder()
    kwargs = _cpu_inputs()
    kwargs["num_householder"] = 0
    with pytest.raises(AssertionError, match="num_householder must be >= 1"):
        fn(**kwargs)


def test_kda_householder_rejects_batched_cu_seqlens():
    """``cu_seqlens`` requires a batch size of exactly 1."""
    fn = _chunk_kda_householder()
    R = 2
    B, T, H, K, V = 2, 4, 2, 8, 8
    with pytest.raises(ValueError, match="batch size is expected to be 1"):
        fn(
            q=torch.randn(B, T, H, K, dtype=torch.bfloat16),
            k=torch.randn(B, T * R, H, K, dtype=torch.bfloat16),
            v=torch.randn(B, T * R, H, V, dtype=torch.bfloat16),
            g=torch.randn(B, T, H, K, dtype=torch.float32),
            beta=torch.randn(B, T * R, H, dtype=torch.bfloat16),
            num_householder=R,
            cu_seqlens=torch.tensor([0, 4, 8], dtype=torch.long),
        )


def test_kda_householder_rejects_mismatched_initial_state_count():
    """The number of initial states must match the number of sequences in ``cu_seqlens``."""
    fn = _chunk_kda_householder()
    kwargs = _cpu_inputs()
    kwargs["cu_seqlens"] = torch.tensor([0, 2, 4], dtype=torch.long)
    kwargs["initial_state"] = torch.zeros(1, 2, 8, 8, dtype=torch.float32)
    with pytest.raises(ValueError, match="number of initial states"):
        fn(**kwargs)


def test_kda_householder_rejects_unknown_backend():
    """An unrecognised ``backend`` must be rejected before any work happens."""
    fn = _chunk_kda_householder()
    kwargs = _cpu_inputs()
    with pytest.raises(ValueError, match="backend must be 'triton' or 'torch'"):
        fn(**kwargs, backend="cuda")


# --------------------------------------------------------------------------------------------
# CPU: the differentiable ("torch") backend. No triton, no GPU -- these always run.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_torch_matches_oracle_f64(R: int):
    """The torch backend's forward must match the oracle in ``float64`` for ``R`` in ``{1, 2, 3}``.

    Both sides run the recurrence in ``float64``, so the tolerance is genuine round-off (in
    practice this is bit-exact: the two share the same ``einsum`` call order).
    """
    kda_torch = _kda_householder_torch()
    oracle_recurrence = _load_oracle_recurrence()

    B, T, H, K, V = 2, 8, 2, 4, 4
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R, seed=R)
    scale = K**-0.5

    o, s = kda_torch(q, k, v, g, beta, num_householder=R, output_final_state=True)
    o_ref, s_ref = oracle_recurrence(q, k, v, g, beta, R, scale, None, torch.float64)

    assert o.shape == (B, T, H, V)
    assert o.dtype == torch.float64
    assert s is not None and s.shape == (B, H, K, V)
    assert (o - o_ref).abs().max().item() < 1e-12, f"o (R={R})"
    assert (s - s_ref).abs().max().item() < 1e-12, f"final_state (R={R})"


def test_kda_householder_torch_matches_oracle_initial_state():
    """A non-zero ``initial_state`` must be threaded through identically to the oracle."""
    kda_torch = _kda_householder_torch()
    oracle_recurrence = _load_oracle_recurrence()

    B, T, H, K, V, R = 2, 8, 2, 4, 4, 2
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R, seed=101)
    h0 = torch.randn(B, H, K, V, dtype=torch.float64) * 0.1

    o, s = kda_torch(q, k, v, g, beta, num_householder=R, initial_state=h0, output_final_state=True)
    o_ref, s_ref = oracle_recurrence(q, k, v, g, beta, R, K**-0.5, h0, torch.float64)
    assert (o - o_ref).abs().max().item() < 1e-12
    assert s is not None and (s - s_ref).abs().max().item() < 1e-12


@pytest.mark.parametrize("R", [1, 2])
def test_kda_householder_torch_gradcheck_f64(R: int):
    """``torch.autograd.gradcheck`` in ``float64`` w.r.t. all of ``q, k, v, g, beta`` jointly.

    This is the real correctness guarantee for the backward pass: a numerical Jacobian check
    against the autograd-derived analytic Jacobian, so no gradient math is taken on trust.
    """
    kda_torch = _kda_householder_torch()
    inputs = _make_inputs_f64(1, 4, 1, 3, 3, R, seed=1000 + R, requires_grad=True)

    def fn(*args: torch.Tensor) -> torch.Tensor:
        o, _ = kda_torch(*args, num_householder=R)
        return o

    assert torch.autograd.gradcheck(fn, inputs, eps=1e-6, atol=1e-8, rtol=1e-5)


def test_kda_householder_torch_gradcheck_initial_state_f64():
    """``gradcheck`` must also pass w.r.t. ``initial_state``."""
    kda_torch = _kda_householder_torch()
    inputs = _make_inputs_f64(1, 4, 1, 3, 3, 2, seed=2002, requires_grad=True)
    h0 = (torch.randn(1, 1, 3, 3, dtype=torch.float64) * 0.1).requires_grad_(True)

    def fn(*args: torch.Tensor) -> torch.Tensor:
        o, _ = kda_torch(*args[:5], num_householder=2, initial_state=args[5])
        return o

    assert torch.autograd.gradcheck(fn, (*inputs, h0), eps=1e-6, atol=1e-8, rtol=1e-5)


@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_torch_grads_finite_and_nonzero(R: int):
    """Every input must receive a finite, non-zero gradient.

    Guards against the failure mode ``gradcheck`` cannot see: a gradient that is *consistently*
    zero because an input was detached, cast, or dropped on the way into the recurrence.
    """
    kda_torch = _kda_householder_torch()
    names = ("q", "k", "v", "g", "beta")
    tensors = _make_inputs_f64(2, 6, 2, 4, 4, R, seed=R, requires_grad=True)

    o, _ = kda_torch(*tensors, num_householder=R)
    o.square().mean().backward()

    for name, t in zip(names, tensors):
        assert t.grad is not None, f"grad w.r.t. {name} is None (R={R})"
        assert torch.isfinite(t.grad).all(), f"grad w.r.t. {name} has non-finite entries (R={R})"
        assert t.grad.abs().max().item() > 0.0, f"grad w.r.t. {name} is all zero (R={R})"


def test_kda_householder_torch_varlen_matches_per_sequence():
    """``cu_seqlens`` must reproduce independent per-sequence calls, forward *and* backward."""
    kda_torch = _kda_householder_torch()
    R = 2
    lengths = [3, 5, 2]
    tensors = _make_inputs_f64(1, sum(lengths), 2, 4, 4, R, seed=31, requires_grad=True)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.long)

    o_var, s_var = kda_torch(
        *tensors, num_householder=R, cu_seqlens=cu_seqlens, output_final_state=True
    )
    o_var.square().sum().backward()

    ref = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    o_parts, s_parts = [], []
    start = 0
    for length in lengths:
        end = start + length
        o_n, s_n = kda_torch(
            ref[0][:, start:end],
            ref[1][:, start * R : end * R],
            ref[2][:, start * R : end * R],
            ref[3][:, start:end],
            ref[4][:, start * R : end * R],
            num_householder=R,
            output_final_state=True,
        )
        o_parts.append(o_n)
        s_parts.append(s_n)
        start = end
    torch.cat(o_parts, dim=1).square().sum().backward()

    assert (o_var - torch.cat(o_parts, dim=1)).abs().max().item() < 1e-12
    assert s_var is not None and (s_var - torch.cat(s_parts, dim=0)).abs().max().item() < 1e-12
    for a, b in zip(tensors, ref):
        assert a.grad is not None and b.grad is not None
        assert (a.grad - b.grad).abs().max().item() < 1e-12


# --------------------------------------------------------------------------------------------
# GPU: the two backends must not drift apart
# --------------------------------------------------------------------------------------------


@requires_gpu
@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_triton_matches_torch_backend(R: int):
    """The Triton and torch backends must agree on GPU in bf16.

    This is the tie between the fast forward-only inference path and the differentiable training
    path: if a future change to either drifts, this fails.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    device = torch.device("cuda")
    B, T, H, K, V = 2, 64, 3, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=R)

    o_tri, s_tri = chunk_kda_householder(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        num_householder=R,
        output_final_state=True,
        backend="triton",
    )
    o_torch, s_torch = chunk_kda_householder(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        num_householder=R,
        output_final_state=True,
        backend="torch",
    )

    assert o_torch.shape == o_tri.shape
    assert o_torch.dtype == q.dtype
    _assert_close(o_tri, o_torch, f"o (triton vs torch, R={R})")
    assert s_tri is not None and s_torch is not None
    _assert_close(s_tri, s_torch, f"final_state (triton vs torch, R={R})")


@requires_gpu
def test_kda_householder_torch_backend_trains_on_gpu():
    """The ``backend="torch"`` path must actually backprop on GPU in bf16."""
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    device = torch.device("cuda")
    B, T, H, K, V, R = 1, 16, 2, 32, 32, 2
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=3)
    tensors = tuple(t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta))

    o, _ = chunk_kda_householder(*tensors, num_householder=R, backend="torch")
    o.float().square().mean().backward()
    for name, t in zip(("q", "k", "v", "g", "beta"), tensors):
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"
        assert t.grad.abs().max().item() > 0.0, f"all-zero grad for {name}"


@requires_gpu
@requires_fla
@pytest.mark.parametrize("neg_eigval", [False, True])
@pytest.mark.parametrize("a_log_scale", [1.0, 16.0])
@pytest.mark.parametrize("R", [1, 2, 3])
def test_kda_householder_backward_relative_error(
    R: int, a_log_scale: float, neg_eigval: bool
) -> None:
    """Triton backward vs the torch backend using PER-GRADIENT RELATIVE error.

    A flat ``atol`` cannot test ``dg``. With the production gate
    (``g = -exp(A_log) * softplus(...)``, ``A_log = log U(1, 16)``) the decay is ~8-68x
    stronger than a naive ``-rand()`` gate, so ``|dg|`` collapses to ~1.6e-3 -- far below any
    absolute tolerance of 2e-2. A ``dg`` that is identically zero, or wrong by 98%, passes a
    flat-atol check. Storing ``dg`` after the decay hand-off instead of before yields exactly
    ``dg * exp(g)``, whose relative error grows as the decay strengthens.

    ``a_log_scale=16.0`` is the band where ~75% of production heads initialise;
    ``neg_eigval=True`` pushes ``beta`` into ``(0, 2)`` where the update is a true reflection.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    B, T, H, K, V = 2, 64, 2, 64, 64
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(R)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    a = torch.rand(H, generator=gen, device=dev) * (a_log_scale - 1.0) + 1.0
    g = (-a.view(1, 1, H, 1) * F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_(True)
    beta = rnd(B, T * R, H).sigmoid() * (2.0 if neg_eigval else 1.0)
    beta = beta.to(torch.bfloat16).requires_grad_(True)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    v = rnd(B, T * R, H, V).to(torch.bfloat16).requires_grad_(True)
    do = rnd(B, T, H, V).to(torch.bfloat16)

    leaves = [q, k, v, g, beta]
    grads = {}
    for backend in ("torch", "triton"):
        o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R, backend=backend)
        grads[backend] = torch.autograd.grad(o, leaves, grad_outputs=do)

    for name, got, ref in zip(("dq", "dk", "dv", "dg", "dbeta"), grads["triton"], grads["torch"]):
        got_f, ref_f = got.float(), ref.float()
        scale = ref_f.abs().max()
        # Guard against a vacuous pass if both sides are identically zero.
        assert scale > 0, f"{name} reference is all-zero -- the comparison would be vacuous"
        rel = ((got_f - ref_f).abs().max() / scale).item()
        assert rel < 2e-2, (
            f"{name}: relative error {rel:.3e} (|ref|max={scale:.3e}, R={R}, "
            f"a_log_scale={a_log_scale}, neg_eigval={neg_eigval})"
        )


@requires_gpu
@requires_fla
@pytest.mark.parametrize("R", [1, 2])
def test_kda_householder_varlen_backward_matches_torch(R: int) -> None:
    """The triton backward on the VARLEN path, against the torch backend.

    The existing varlen test is forward-only, and the only ``cu_seqlens``-plus-backward coverage
    is on the torch backend. The kernel's varlen backward is real code -- ``bos``/``eos`` are
    reloaded in both passes and the ``hs`` workspace is indexed by the flat token index so that
    sequences cannot alias -- and an off-by-one there would corrupt one sequence's gradients
    while leaving the others correct, which no forward test would catch.

    Sequence lengths are deliberately unequal and not powers of two.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    lens = [3, 11, 7]
    T = sum(lens)
    H, K, V = 2, 64, 64
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(1000 + R)
    cu = torch.tensor([0, 3, 14, 21], dtype=torch.int32, device=dev)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    a = torch.rand(H, generator=gen, device=dev) * 15.0 + 1.0
    g = (-a.view(1, 1, H, 1) * F.softplus(rnd(1, T, H, K) * 0.02)).requires_grad_(True)
    beta = (rnd(1, T * R, H).sigmoid() * 2.0).to(torch.bfloat16).requires_grad_(True)
    q = F.normalize(rnd(1, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    k = F.normalize(rnd(1, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    v = rnd(1, T * R, H, V).to(torch.bfloat16).requires_grad_(True)
    do = rnd(1, T, H, V).to(torch.bfloat16)

    leaves = [q, k, v, g, beta]
    grads = {}
    for backend in ("torch", "triton"):
        o, _ = chunk_kda_householder(
            q, k, v, g, beta, num_householder=R, cu_seqlens=cu, backend=backend
        )
        grads[backend] = torch.autograd.grad(o, leaves, grad_outputs=do)

    for name, got, ref in zip(("dq", "dk", "dv", "dg", "dbeta"), grads["triton"], grads["torch"]):
        got_f, ref_f = got.float(), ref.float()
        scale = ref_f.abs().max()
        assert scale > 0, f"{name} reference is all-zero -- the comparison would be vacuous"
        rel = ((got_f - ref_f).abs().max() / scale).item()
        assert rel < 2e-2, f"{name}: relative error {rel:.3e} (R={R}, lens={lens})"

    # A per-sequence check: a bug that corrupted exactly one sequence would still be caught by the
    # global max above, but this localises it.
    for i, (lo, hi) in enumerate(zip([0, 3, 14], [3, 14, 21])):
        gt, gr = grads["triton"][3][:, lo:hi].float(), grads["torch"][3][:, lo:hi].float()
        s = gr.abs().max()
        if s > 0:
            assert ((gt - gr).abs().max() / s).item() < 2e-2, f"dg wrong in sequence {i}"


@requires_gpu
def test_kda_householder_final_state_is_non_differentiable() -> None:
    """The kernel cannot propagate a cotangent on the final state, so it must say so structurally.

    This was previously a value test in ``backward`` (``if dht is not None and dht.abs().any()``),
    which passes silently whenever the cotangent happens to be all-zero even though the final
    state is genuinely in the graph -- e.g. a masked TBPTT carry-over with an all-zero mask --
    and then returns gradients that quietly omit that path. ``mark_non_differentiable`` is
    checked by autograd when the graph is built, so the failure is loud and value-independent.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    B, T, H, K, V = 1, 8, 2, 64, 64
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(7)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    g = (-F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_(True)
    beta = rnd(B, T, H).sigmoid().to(torch.bfloat16).requires_grad_(True)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    k = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    v = rnd(B, T, H, V).to(torch.bfloat16).requires_grad_(True)

    o, ht = chunk_kda_householder(
        q, k, v, g, beta, num_householder=1, output_final_state=True, backend="triton"
    )
    assert ht is not None
    assert not ht.requires_grad, "final_state must be marked non-differentiable"

    # The output path still trains.
    torch.autograd.grad(o.sum(), [q], retain_graph=True)

    # Differentiating the final state must raise, not silently drop the term. The all-zero mask
    # is the case the old value-based guard let through.
    with pytest.raises(RuntimeError):
        torch.autograd.grad((ht * torch.zeros_like(ht)).sum(), [q])


@requires_gpu
def test_kda_householder_backward_is_once_differentiable() -> None:
    """Double-backward must raise rather than silently return zero second-order terms.

    The gradients returned by the triton backward are freshly-allocated kernel outputs that live
    outside the autograd graph (``requires_grad=False``, ``grad_fn=None``). Without
    ``@once_differentiable`` a ``create_graph=True`` consumer -- a gradient penalty, an
    HVP, a MAML inner loop -- would receive zero for every second-order term through this op with
    no error at all. ``backend="torch"`` remains available when a real double-backward is needed.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    B, T, H, K, V = 1, 8, 2, 64, 64
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(11)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    g = (-F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_(True)
    beta = rnd(B, T, H).sigmoid().to(torch.bfloat16).requires_grad_(True)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    k = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    v = rnd(B, T, H, V).to(torch.bfloat16).requires_grad_(True)

    o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=1, backend="triton")
    (gq,) = torch.autograd.grad(o.sum(), [q], create_graph=True)
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gq.sum(), [q])


@requires_gpu
@requires_fla
@pytest.mark.parametrize("K", [128, 256])
def test_kda_householder_backward_multiwarp_head_dim(K: int) -> None:
    """Backward correctness at ``head_dim >= 128``, where the kernel uses more than one warp.

    ``num_warps`` is ``1`` for ``BK <= 64`` and ``2``/``4`` above it, and *every* other backward
    test runs at ``K = 64`` -- i.e. entirely in the single-warp regime, where a warp executes in
    lockstep and cross-lane hazards cannot manifest. The backward writes the ``hs`` workspace in
    pass 1 and reads it back in pass 2, and triton may pick different register layouts for the
    store and the load, so the writing lane need not be the reading lane. That race is only
    reachable at multi-warp head dims, which production KDA configurations use.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    B, T, H, V, R = 1, 32, 2, 64, 2
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(K)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=dev, dtype=torch.float32)

    a = torch.rand(H, generator=gen, device=dev) * 15.0 + 1.0
    g = (-a.view(1, 1, H, 1) * F.softplus(rnd(B, T, H, K) * 0.02)).requires_grad_(True)
    beta = (rnd(B, T * R, H).sigmoid() * 2.0).to(torch.bfloat16).requires_grad_(True)
    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1).to(torch.bfloat16).requires_grad_(True)
    v = rnd(B, T * R, H, V).to(torch.bfloat16).requires_grad_(True)
    do = rnd(B, T, H, V).to(torch.bfloat16)

    leaves = [q, k, v, g, beta]
    grads = {}
    for backend in ("torch", "triton"):
        o, _ = chunk_kda_householder(q, k, v, g, beta, num_householder=R, backend=backend)
        grads[backend] = torch.autograd.grad(o, leaves, grad_outputs=do)

    for name, got, ref in zip(("dq", "dk", "dv", "dg", "dbeta"), grads["triton"], grads["torch"]):
        got_f, ref_f = got.float(), ref.float()
        scale = ref_f.abs().max()
        assert scale > 0, f"{name} reference is all-zero -- the comparison would be vacuous"
        rel = ((got_f - ref_f).abs().max() / scale).item()
        assert rel < 2e-2, f"{name}: relative error {rel:.3e} at K={K}"
