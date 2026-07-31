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


# =============================================================================================
# Runbook section 4.7 -- the named Phase-0 gate tests.
#
# These implement the test IDs named in 'docs/dp2-kda/phase-0-1-runbook.md' section 4.7. The
# names are part of the deliverable: a reviewer must be able to find each required check by
# name in the collected-test report, without reverse-engineering a prose description.
#
# Tolerance policy is split by backend, per section 4.3, and the split is mandatory:
#
#   * float64 against ``backend="torch"`` gets a tight bar (``rtol=0``, atol ~1e-11). The Triton
#     path rejects float32 outright (``kda_householder.py:737-739``), so a float64 oracle can only
#     ever validate the reference implementation.
#   * kernel-versus-reference is only comparable at bf16, where the file-level ``ATOL``/``RTOL``
#     of 2e-2 is a *smoke* bound rather than a semantic gate -- a seeded dropped-cross-term bug
#     lands at ~3.5e-3 median relative error and slides under 2e-2 for most seeds. That is why
#     ``test_r2_bf16_check_fails_on_seeded_cross_term_bug`` exists: it falsifies the bf16 check.
#
# Negative controls assert a *separation floor*, never merely "the outputs differ". Both controls
# here have measured regimes where the difference is exactly 0.0 (orthogonal keys for the
# factor-order swap; a zero incoming state for the zero-value control), so an unfloored control
# passes while proving nothing.
# =============================================================================================

# Tight bar for float64 semantic tests. `rtol=0` is deliberate: these comparisons are against an
# independent float64 computation of the same quantity, so the residual is genuine round-off and
# does not scale with the magnitude of the value being compared.
F64_ATOL = 1e-11

# Section 4.4's factor-order-swap floor, as a fraction of the output scale. A tolerance-multiplied
# floor does not work in either regime: 1e3 x 1e-11 is inert, 1e3 x 2e-2 is unsatisfiable.
SWAP_SEPARATION_FLOOR = 5e-2

# Minimum |k1 . k2| for the swap control. The separation scales with the key overlap and is
# exactly 0.0 at orthogonal keys, so the keys must be constructed non-orthogonal by hand.
SWAP_MIN_KEY_OVERLAP = 0.3


def _tied_angle_key(k1: torch.Tensor, cos: float, seed: int = 99) -> torch.Tensor:
    """Build a unit key whose inner product with ``k1`` is exactly ``cos``.

    ``k2 = cos * k1 + sqrt(1 - cos^2) * perp`` with ``perp`` a unit vector orthogonal to ``k1``,
    so ``k1 . k2 == cos`` to float64 round-off regardless of the random draw. Perturbing ``k1``
    by a random vector and renormalizing does *not* do this -- the realized overlap then varies
    per (batch, head, time) slot and cannot be asserted.

    :param k1: Unit-norm keys of shape ``[..., K]``.
    :param cos: Target inner product, in ``(0, 1]``.
    :param seed: Seed for the random orthogonal direction.

    :returns: Unit-norm keys of the same shape as ``k1``.
    """
    gen = torch.Generator().manual_seed(seed)
    r = torch.randn(k1.shape, generator=gen, dtype=k1.dtype)
    perp = F.normalize(r - (r * k1).sum(-1, keepdim=True) * k1, dim=-1)
    return cos * k1 + (1.0 - cos**2) ** 0.5 * perp


def _rank_two_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    scale: float,
    corrupt: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Section 4.5's independent rank-two algebraic form of the ``R = 2`` update.

    Instead of applying the two delta-rule factors sequentially, this composes them into a single
    rank-two write against the *undecayed* incoming state::

        u_j = beta_j k_j,   rho = k_2^T u_1
        U = [u_1 - u_2 rho, u_2],  Vm = [D_t k_1, D_t k_2],  Rm = [v_1, v_2]
        S_t = D_t S_{t-1} + U (Rm^T - Vm^T S_{t-1})

    ``Vm`` uses the *decayed* keys because it is contracted against the undecayed ``S_{t-1}``, and
    ``D_t`` is diagonal hence symmetric, so ``(D_t k)^T S_{t-1} = k^T (D_t S_{t-1})`` -- the decay
    appears exactly once overall. Every term is load-bearing; see ``corrupt`` and the mutation
    cases in :func:`test_r2_matches_rank_two_float64_oracle`.

    :param q: Queries ``[B, T, H, K]``.
    :param k: Keys ``[B, 2T, H, K]``, interleaved along time.
    :param v: Values ``[B, 2T, H, V]``, interleaved along time.
    :param g: Log-decay ``[B, T, H, K]``, one entry per real token.
    :param beta: Step sizes ``[B, 2T, H]``, interleaved along time.
    :param initial_state: Incoming state ``[B, H, K, V]``.
    :param scale: Query scaling factor.
    :param corrupt: If given, deliberately break one term. One of ``"drop_rho"`` (omit the
        cross-term entirely), ``"rho_on_u2"`` (subtract the cross-term from the wrong column), or
        ``"plain_k"`` (contract against undecayed keys). Used only by the mutation check.

    :returns: ``(o, S)`` with ``o`` of shape ``[B, T, H, V]`` and ``S`` of shape ``[B, H, K, V]``.

    :raises ValueError: if ``corrupt`` is not a recognized mutation name.
    """
    if corrupt not in (None, "drop_rho", "rho_on_u2", "plain_k"):
        raise ValueError(f"unknown corruption {corrupt!r}")

    S = initial_state.clone()
    outputs = []
    for t in range(q.shape[1]):
        decay = g[:, t].exp()
        k1, k2 = k[:, 2 * t], k[:, 2 * t + 1]
        v1, v2 = v[:, 2 * t], v[:, 2 * t + 1]
        b1, b2 = beta[:, 2 * t], beta[:, 2 * t + 1]

        u1, u2 = b1[..., None] * k1, b2[..., None] * k2
        rho = (k2 * u1).sum(-1)

        if corrupt == "drop_rho":
            col1 = u1
        elif corrupt == "rho_on_u2":
            col1, u2 = u1, u2 - u1 * rho[..., None]
        else:
            col1 = u1 - u2 * rho[..., None]

        dk1, dk2 = (k1, k2) if corrupt == "plain_k" else (decay * k1, decay * k2)

        r1 = v1 - torch.einsum("bhk,bhkv->bhv", dk1, S)
        r2 = v2 - torch.einsum("bhk,bhkv->bhv", dk2, S)
        S = (
            decay[..., None] * S
            + torch.einsum("bhk,bhv->bhkv", col1, r1)
            + torch.einsum("bhk,bhv->bhkv", u2, r2)
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, t] * scale, S))

    return torch.stack(outputs, dim=1), S


def _virtual_token_inputs(q: torch.Tensor, g: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the section 4.4 doubled-length query and log-decay for the virtual-token oracle.

    Native ``R = 2`` on ``T`` tokens is re-expressed as ``R = 1`` on ``2T`` virtual positions: the
    first virtual position of each real token carries the real decay and a discarded readout, the
    second carries zero log-decay (so the state decays exactly once per real token) and the real
    query, whose output is the one retained.

    The dummy query is a **unit** vector, not zero. ``olmo_core.nn.functional.l2_normalize``
    (``functional/__init__.py:16-18``) is a bare ``x / ||x||`` with no epsilon and returns NaN on a
    zero vector -- unlike ``F.normalize``, which returns zeros -- and the module applies it at
    ``recurrent.py:1299``, *after* the point where a zero query would be injected. A zero query
    therefore NaNs the entire doubled sequence at module level.

    .. note::
        Verified: at the **operator** level used by these tests this hazard does not fire, because
        the operator is downstream of ``l2_normalize`` and a zero query merely yields a zero
        output (which is discarded). Substituting a zero dummy here does *not* fail the tests. The
        unit dummy is kept anyway so this helper stays correct if it is ever reused against the
        module, where the NaN is real -- but do not mistake it for a load-bearing assertion at
        this level.

    :param q: Real queries ``[B, T, H, K]``.
    :param g: Real log-decay ``[B, T, H, K]``.

    :returns: ``(q_virtual, g_virtual)``, both ``[B, 2T, H, K]``.
    """
    B, T, H, K = q.shape
    q_virtual = torch.zeros(B, 2 * T, H, K, dtype=q.dtype, device=q.device)
    q_virtual[:, 0::2] = F.normalize(torch.ones_like(q), p=2, dim=-1)
    q_virtual[:, 1::2] = q

    g_virtual = torch.zeros(B, 2 * T, H, K, dtype=g.dtype, device=g.device)
    g_virtual[:, 0::2] = g
    # Second virtual position: zero log-decay, i.e. exp(0) = 1, no second decay.
    g_virtual[:, 1::2] = 0.0
    return q_virtual, g_virtual


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_r2_matches_rank_two_float64_oracle(with_initial_state: bool):
    """P0.3: the sequential ``R = 2`` recurrence equals section 4.5's rank-two algebraic form.

    This is an *independent algebraic* oracle, not a second transcription: it composes the two
    delta-rule factors into one rank-two write rather than applying them in sequence. Agreement to
    float64 round-off is therefore evidence about the mathematics, not about the code path.

    The mutation half of this test is what makes it load-bearing. Each of the three plausible
    corruptions of the formula is checked to fail by a large *ratio* to the correct residual --
    gating on a ratio rather than an absolute constant because the corruption magnitudes scale
    with ``||S_{t-1}|| ||v||`` and are not reproducible numbers.
    """
    B, T, H, K, V = 2, 6, 2, 8, 8
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=7)
    initial_state = (
        torch.randn(B, H, K, V, generator=torch.Generator().manual_seed(8), dtype=torch.float64)
        if with_initial_state
        else torch.zeros(B, H, K, V, dtype=torch.float64)
    )
    scale = K**-0.5

    o_ref, s_ref = _kda_householder_torch()(
        q,
        k,
        v,
        g,
        beta,
        num_householder=2,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_oracle, s_oracle = _rank_two_oracle(q, k, v, g, beta, initial_state, scale)

    residual = max((o_ref - o_oracle).abs().max().item(), (s_ref - s_oracle).abs().max().item())
    assert residual < F64_ATOL, f"rank-two oracle residual {residual:.3e} exceeds {F64_ATOL:.0e}"

    # Every term is load-bearing: each corruption must fail by orders of magnitude. The measured
    # ratios are ~1e15; 1e6 is a deliberately conservative floor that still cannot be met by
    # round-off.
    for corruption in ("drop_rho", "rho_on_u2", "plain_k"):
        o_bad, s_bad = _rank_two_oracle(q, k, v, g, beta, initial_state, scale, corrupt=corruption)
        bad = max((o_ref - o_bad).abs().max().item(), (s_ref - s_bad).abs().max().item())
        assert bad / max(residual, 1e-300) > 1e6, (
            f"corruption {corruption!r} was not caught: residual {bad:.3e} vs correct "
            f"{residual:.3e}. Every term in the section 4.5 form is load-bearing."
        )


@pytest.mark.parametrize("with_initial_state", [False, True])
def test_r2_matches_post_shortconv_virtual_token_oracle(with_initial_state: bool):
    """P0.2: native ``R = 2`` equals two ordinary KDA microsteps with only one real-token decay.

    Compares on the operator output ``o`` (pre-``o_norm``) and on the final state. The output
    comparison is the load-bearing half: ``final_state`` is invariant to which microstep the
    readout happens at, so a state-only comparison passes an implementation that reads out after
    factor 1 instead of factor 2 -- exactly the bug this oracle claims to exclude.
    """
    B, T, H, K, V = 1, 5, 2, 8, 8
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=11)
    initial_state = (
        torch.randn(B, H, K, V, generator=torch.Generator().manual_seed(12), dtype=torch.float64)
        if with_initial_state
        else None
    )
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    q_virtual, g_virtual = _virtual_token_inputs(q, g)
    o_virtual, s_virtual = kda_torch(
        q_virtual,
        k,
        v,
        g_virtual,
        beta,
        num_householder=1,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_native, s_native = kda_torch(
        q,
        k,
        v,
        g,
        beta,
        num_householder=2,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )

    # Retain only the second virtual position's output -- the first is the discarded dummy read.
    assert s_virtual is not None and s_native is not None
    o_diff = (o_virtual[:, 1::2] - o_native).abs().max().item()
    s_diff = (s_virtual - s_native).abs().max().item()
    assert o_diff < F64_ATOL, f"virtual-token oracle output mismatch: {o_diff:.3e}"
    assert s_diff < F64_ATOL, f"virtual-token oracle state mismatch: {s_diff:.3e}"


def test_r2_virtual_oracle_gradients_and_initial_state():
    """P0.2/P0.4: every differentiable input and the incoming state get matching gradients.

    Gradients are compared between the native ``R = 2`` path and the doubled virtual-token path,
    so this checks that the two formulations agree in the backward as well as the forward.
    ``initial_state`` is threaded explicitly and its gradient is compared too -- the torch
    backend's final state is differentiable, unlike the Triton path, which marks it
    non-differentiable (``kda_householder.py:639-640``).
    """
    B, T, H, K, V = 1, 4, 2, 8, 8
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    def run(virtual: bool):
        q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=21, requires_grad=True)
        s0 = torch.randn(
            B, H, K, V, generator=torch.Generator().manual_seed(22), dtype=torch.float64
        ).requires_grad_(True)
        if virtual:
            q_in, g_in = _virtual_token_inputs(q, g)
            o, _ = kda_torch(
                q_in,
                k,
                v,
                g_in,
                beta,
                num_householder=1,
                scale=scale,
                initial_state=s0,
                output_final_state=True,
            )
            o = o[:, 1::2]
        else:
            o, _ = kda_torch(
                q,
                k,
                v,
                g,
                beta,
                num_householder=2,
                scale=scale,
                initial_state=s0,
                output_final_state=True,
            )
        o.square().sum().backward()
        return (q, k, v, g, beta, s0)

    native, virtual = run(virtual=False), run(virtual=True)
    names = ("q", "k", "v", "g", "beta", "initial_state")
    for name, a, b in zip(names, native, virtual):
        assert a.grad is not None and b.grad is not None, f"{name} has no gradient"
        diff = (a.grad - b.grad).abs().max().item()
        assert diff < F64_ATOL, f"grad mismatch for {name}: {diff:.3e}"
        # A gradient that is identically zero would make the comparison vacuous.
        assert a.grad.abs().max().item() > 0, f"{name} gradient is identically zero"


def test_r2_virtual_oracle_varlen_doubled_offsets():
    """P0.2: packed boundaries are correct and virtual offsets are exactly double the real ones.

    The interleaved key/value/beta tensors live at ``2x`` the real token rate, so a ``cu_seqlens``
    given in token units must map to doubled offsets in the virtual formulation. Getting this
    wrong silently mixes state across document boundaries.
    """
    H, K, V = 2, 8, 8
    lengths = [3, 5, 2]
    total = sum(lengths)
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    q, k, v, g, beta = _make_inputs_f64(1, total, H, K, V, R=2, seed=31)
    cu_seqlens = torch.tensor([0, 3, 8, 10], dtype=torch.long)
    doubled = cu_seqlens * 2
    assert doubled.tolist() == [0, 6, 16, 20], "virtual offsets must be exactly 2x the real ones"

    o_native, s_native = kda_torch(
        q,
        k,
        v,
        g,
        beta,
        num_householder=2,
        scale=scale,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
    )
    q_virtual, g_virtual = _virtual_token_inputs(q, g)
    o_virtual, s_virtual = kda_torch(
        q_virtual,
        k,
        v,
        g_virtual,
        beta,
        num_householder=1,
        scale=scale,
        cu_seqlens=doubled,
        output_final_state=True,
    )

    assert s_virtual is not None and s_native is not None
    o_diff = (o_virtual[:, 1::2] - o_native).abs().max().item()
    s_diff = (s_virtual - s_native).abs().max().item()
    assert o_diff < F64_ATOL, f"varlen virtual-token output mismatch: {o_diff:.3e}"
    assert s_diff < F64_ATOL, f"varlen virtual-token state mismatch: {s_diff:.3e}"
    assert s_native.shape[0] == len(lengths), "one final state per packed document"


def test_factor_two_zero_beta_is_identity():
    """P0.4: masking ``beta_2`` to zero leaves ``R = 1`` behavior *exactly* unchanged.

    Asserted with zero tolerance, not a tolerance band. With ``beta_2 = 0`` the second factor's
    write is identically zero, so this is a true algebraic identity rather than a numerical
    approximation, and the measured difference is exactly 0.0.
    """
    B, T, H, K, V = 1, 5, 2, 8, 8
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=41)
    initial_state = torch.randn(
        B, H, K, V, generator=torch.Generator().manual_seed(42), dtype=torch.float64
    )
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    beta_masked = beta.clone()
    beta_masked[:, 1::2] = 0.0
    o_masked, s_masked = kda_torch(
        q,
        k,
        v,
        g,
        beta_masked,
        num_householder=2,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_r1, s_r1 = kda_torch(
        q,
        k[:, 0::2],
        v[:, 0::2],
        g,
        beta[:, 0::2],
        num_householder=1,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )

    assert s_masked is not None and s_r1 is not None
    assert (o_masked - o_r1).abs().max().item() == 0.0, "beta_2 = 0 must be an exact identity"
    assert (s_masked - s_r1).abs().max().item() == 0.0, "beta_2 = 0 must be an exact identity"


def test_zero_value_is_not_factor_identity():
    """P0.4 negative control: ``v_2 = 0`` with ``beta_2 != 0`` still changes the state.

    The erase term ``-beta_2 k_2 (k_2^T S)`` survives a zero value, so zeroing ``v_2`` is *not* a
    way to disable the second factor. This control needs a non-degenerate setup and an asserted
    floor: with ``S_{t-1} = 0`` and ``beta_1 v_1 = 0`` the difference is exactly 0.000e+00 and an
    unfloored control silently "passes" while proving nothing. Both regimes are checked here.
    """
    B, T, H, K, V = 1, 5, 2, 8, 8
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=51)
    initial_state = torch.randn(
        B, H, K, V, generator=torch.Generator().manual_seed(52), dtype=torch.float64
    )
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    v_zeroed = v.clone()
    v_zeroed[:, 1::2] = 0.0
    o_zeroed, _ = kda_torch(
        q,
        k,
        v_zeroed,
        g,
        beta,
        num_householder=2,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_r1, _ = kda_torch(
        q,
        k[:, 0::2],
        v[:, 0::2],
        g,
        beta[:, 0::2],
        num_householder=1,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )

    separation = ((o_zeroed - o_r1).abs().max() / o_r1.abs().max()).item()
    assert separation > SWAP_SEPARATION_FLOOR, (
        f"zero-value control separation {separation:.3e} did not clear "
        f"{SWAP_SEPARATION_FLOOR:.0e}; v_2 = 0 must not behave as an identity"
    )

    # The degenerate regime the floor exists to exclude: zero incoming state and zero values make
    # the difference exactly 0.0, so this control is only meaningful when seeded as above.
    v_all_zero = torch.zeros_like(v)
    o_a, _ = kda_torch(
        q, k, v_all_zero, g, beta, num_householder=2, scale=scale, output_final_state=True
    )
    o_b, _ = kda_torch(
        q,
        k[:, 0::2],
        v_all_zero[:, 0::2],
        g,
        beta[:, 0::2],
        num_householder=1,
        scale=scale,
        output_final_state=True,
    )
    assert (o_a - o_b).abs().max().item() == 0.0, (
        "the degenerate case is expected to be exactly identical -- if this ever fails, the "
        "documented rationale for requiring a non-degenerate control has changed"
    )


def test_r2_virtual_oracle_factor_order_swap_separates():
    """P0.2 negative control: swapping the two factors changes the output, by a stated floor.

    "Must fail" is half a spec. The swap difference scales with ``|k_1 . k_2|``, which is
    ``O(K^{-1/2})`` for random keys and **exactly 0.0** for orthogonal ones -- so with random keys
    this control can be satisfied by round-off. The keys are therefore constructed with an exact
    inner product via :func:`_tied_angle_key`, the realized overlap is asserted, and the
    separation floor is stated as a fraction of the output scale rather than as a multiple of the
    comparison tolerance.
    """
    B, T, H, K, V = 1, 5, 2, 8, 8
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=61)
    initial_state = torch.randn(
        B, H, K, V, generator=torch.Generator().manual_seed(62), dtype=torch.float64
    )
    scale = K**-0.5
    kda_torch = _kda_householder_torch()

    k_tied = k.clone()
    k_tied[:, 1::2] = _tied_angle_key(k[:, 0::2], SWAP_MIN_KEY_OVERLAP)
    overlap = (k_tied[:, 0::2] * k_tied[:, 1::2]).sum(-1).abs()
    assert overlap.min().item() >= SWAP_MIN_KEY_OVERLAP - 1e-9, (
        f"key overlap {overlap.min().item():.4f} below the required "
        f"{SWAP_MIN_KEY_OVERLAP}; the control would be testing round-off"
    )

    def run(kk, vv, bb):
        o, _ = kda_torch(
            q,
            kk,
            vv,
            g,
            bb,
            num_householder=2,
            scale=scale,
            initial_state=initial_state,
            output_final_state=True,
        )
        return o

    o_ordered = run(k_tied, v, beta)
    k_s, v_s, b_s = k_tied.clone(), v.clone(), beta.clone()
    k_s[:, 0::2], k_s[:, 1::2] = k_tied[:, 1::2], k_tied[:, 0::2]
    v_s[:, 0::2], v_s[:, 1::2] = v[:, 1::2], v[:, 0::2]
    b_s[:, 0::2], b_s[:, 1::2] = beta[:, 1::2], beta[:, 0::2]
    o_swapped = run(k_s, v_s, b_s)

    separation = ((o_swapped - o_ordered).abs().max() / o_ordered.abs().max()).item()
    assert separation >= SWAP_SEPARATION_FLOOR, (
        f"factor-order swap separation {separation:.3e} did not clear the floor "
        f"{SWAP_SEPARATION_FLOOR:.0e} at |k1.k2| = {overlap.min().item():.4f}. The two factors "
        f"are supposed to be order-dependent."
    )


def test_r2_bf16_long_rollout_is_finite():
    """P0.4: a bf16 rollout with float32 accumulation stays finite and grows no worse than linear.

    Error growth is operationalized rather than eyeballed: measure ``max|o - o_fp64|`` over a
    length ladder, fit ``log e = a + p log T``, and require the exponent ``p`` to be at most
    linear with margin. Without a reference, a metric, an exponent and a ladder, "no superlinear
    growth" is unfalsifiable. Measured baseline for these shapes is ``p`` well under 1.
    """
    H, K, V = 2, 16, 16
    ladder = [32, 64, 128, 256]
    kda_torch = _kda_householder_torch()
    errors = []

    for T in ladder:
        q, k, v, g, beta = _make_inputs_f64(1, T, H, K, V, R=2, seed=71)
        o_f64, _ = kda_torch(
            q, k, v, g, beta, num_householder=2, scale=K**-0.5, output_final_state=True
        )
        o_bf16, _ = kda_torch(
            q.bfloat16(),
            k.bfloat16(),
            v.bfloat16(),
            g.float(),
            beta.bfloat16(),
            num_householder=2,
            scale=K**-0.5,
            output_final_state=True,
        )
        assert torch.isfinite(o_bf16).all(), f"bf16 rollout produced NaN/Inf at T = {T}"
        errors.append((o_bf16.double() - o_f64).abs().max().item())

    assert all(e > 0 for e in errors), "bf16 error is identically zero -- comparison is vacuous"
    log_t = torch.tensor(ladder, dtype=torch.float64).log()
    log_e = torch.tensor(errors, dtype=torch.float64).log()
    # Least-squares slope of log(error) against log(T).
    slope = (
        ((log_t - log_t.mean()) * (log_e - log_e.mean())).sum()
        / ((log_t - log_t.mean()) ** 2).sum()
    ).item()
    assert slope <= 1.25, (
        f"bf16 error growth exponent {slope:.3f} exceeds linear-with-margin (1.25). "
        f"Per-T errors: {[f'{e:.3e}' for e in errors]}"
    )


@requires_gpu
@pytest.mark.parametrize("R", [1, 2])
def test_r2_triton_matches_torch_relative_error_budget(R: int):
    """P0.1: kernel-versus-reference agreement at bf16, as a per-tensor relative-error budget.

    Reports max / median / p99 relative error per compared tensor rather than collapsing the
    comparison into one opaque ``allclose``. The 2e-2 bound is the repo's calibrated bf16 constant
    and is treated here as a **smoke** bound, not as the semantic gate -- see
    :func:`test_r2_bf16_check_fails_on_seeded_cross_term_bug`, which exists precisely because a
    real cross-term bug can slide under 2e-2. The semantic gate is the float64 suite above.
    """
    pytest.importorskip("triton")
    from olmo_core.nn.attention.kda_householder import chunk_kda_householder

    device = torch.device("cuda")
    B, T, H, K, V = 2, 64, 2, 64, 64
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, device, seed=81)
    scale = K**-0.5

    outputs = {}
    for backend in ("torch", "triton"):
        o, s = chunk_kda_householder(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            num_householder=R,
            scale=scale,
            output_final_state=True,
            backend=backend,
        )
        outputs[backend] = (o, s)

    for name, got, ref in zip(("o", "final_state"), outputs["triton"], outputs["torch"]):
        got_f, ref_f = got.float(), ref.float()
        scale_ref = ref_f.abs().max()
        assert scale_ref > 0, f"{name} reference is all-zero -- the comparison would be vacuous"

        rel = (got_f - ref_f).abs() / scale_ref
        rel_max = rel.max().item()
        rel_med = rel.median().item()
        rel_p99 = rel.flatten().quantile(0.99).item()
        print(
            f"[R={R}] {name}: rel_max={rel_max:.3e} rel_median={rel_med:.3e} "
            f"rel_p99={rel_p99:.3e} (|ref|max={scale_ref:.3e})"
        )
        assert rel_max < 2e-2, (
            f"{name}: max relative error {rel_max:.3e} exceeds the bf16 smoke bound 2e-2 "
            f"(median {rel_med:.3e}, p99 {rel_p99:.3e}, R={R})"
        )


@requires_gpu
def test_r2_bf16_check_fails_on_seeded_cross_term_bug():
    """P0.1: the bf16 comparison is falsifiable -- a seeded cross-term bug is actually caught.

    Without this, the bf16 gate is unfalsified. The mutation used is the section 4.5 cross-term
    corruption (``rho`` dropped from the rank-two form), which is equivalent to reordering the two
    factors' interaction. Both the mutated and the correct oracle are evaluated in float64 so the
    check measures the *mutation*, not bf16 noise.

    The assertion is deliberately two-sided: the correct form must pass a tight bar **and** the
    mutated form must fail it. A one-sided check would be satisfied by a gate that rejects
    everything.
    """
    pytest.importorskip("triton")

    B, T, H, K, V = 1, 16, 2, 16, 16
    q, k, v, g, beta = _make_inputs_f64(B, T, H, K, V, R=2, seed=91)
    initial_state = torch.randn(
        B, H, K, V, generator=torch.Generator().manual_seed(92), dtype=torch.float64
    )
    scale = K**-0.5

    o_ref, _ = _kda_householder_torch()(
        q,
        k,
        v,
        g,
        beta,
        num_householder=2,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
    )
    o_good, _ = _rank_two_oracle(q, k, v, g, beta, initial_state, scale)
    o_bad, _ = _rank_two_oracle(q, k, v, g, beta, initial_state, scale, corrupt="drop_rho")

    good = (o_ref - o_good).abs().max().item()
    bad = (o_ref - o_bad).abs().max().item()
    rel_bad = (bad / o_ref.abs().max()).item()
    print(
        f"mutation check: correct residual={good:.3e}  mutated residual={bad:.3e} "
        f"(relative {rel_bad:.3e})"
    )

    assert good < F64_ATOL, f"the correct form must pass the tight bar, got {good:.3e}"
    assert rel_bad > 2e-2, (
        f"the seeded cross-term bug produced only {rel_bad:.3e} relative error, which would "
        f"slide under the 2e-2 bf16 bound. The bf16 check would not catch this mutation."
    )


@requires_gpu
@requires_fla
def test_r2_matches_external_gated_delta_product_naive():
    """P0.3 external anchor: agreement with ``fla.ops.gated_delta_product.naive``.

    P0.2 and P0.3 are independent *implementations* of the same specification, not independent
    specifications -- if the spec itself is wrong, every other Phase-0 test passes. This is the
    one check against a reference maintained outside this repository.

    ``g`` is constructed **constant along K** because fla's reference takes a per-head scalar gate
    of shape ``[B, T, H]`` while this kernel takes a per-channel gate of shape ``[B, T, H, K]``.
    The two are only comparable where the per-channel gate is channel-invariant, which is exactly
    the regime the repo's own docstring documents (``kda_householder.py:689-693``).

    Two conventions of ``naive_recurrent_gated_delta_product`` were measured against fla 0.5.1 and
    both are load-bearing here:

    * **It ignores its own ``scale`` argument.** The parameter is accepted but never applied to
      ``q`` -- passing ``scale=1.0`` and ``scale=K**-0.5`` returns byte-identical output. This
      side must therefore also pass ``scale=1.0``; using ``K**-0.5`` produces a 0.65 relative
      disagreement that looks like a correctness failure and is not one.
    * **It computes in float32.** The first statement of its body is
      ``q, k, v, beta = map(lambda x: x.float(), ...)``, so float64 inputs are downcast and the
      achievable agreement is floored at float32 round-off. The runbook's "float64 ulp" phrasing
      is not attainable against this reference: the measured residual is ~6.4e-8, which matches
      this side's own float32-versus-float64 gap (~1.0e-7) and is therefore explained entirely by
      the downcast rather than by any disagreement about the recurrence.
    """
    pytest.importorskip("triton")
    from fla.ops.gated_delta_product.naive import (  # type: ignore[import-not-found]
        naive_recurrent_gated_delta_product,
    )

    device = torch.device("cuda")
    B, T, H, K, V, R = 1, 24, 2, 32, 32, 2
    gen = torch.Generator(device=device).manual_seed(101)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, device=device, dtype=torch.float64)

    q = F.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = F.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    # Per-head scalar decay, broadcast along K so the two gate conventions coincide.
    g_head = F.logsigmoid(rnd(B, T, H))
    g_channel = g_head[..., None].expand(B, T, H, K).contiguous()
    # Both sides unscaled: see the note above -- fla never applies `scale` to `q`.
    scale = 1.0

    o_mine, _ = _kda_householder_torch()(
        q, k, v, g_channel, beta, num_householder=R, scale=scale, output_final_state=True
    )
    o_fla, _ = naive_recurrent_gated_delta_product(
        q=q,
        k=k,
        v=v,
        g=g_head,
        beta=beta,
        scale=scale,
        cu_seqlens=None,
        num_householder=R,
        output_final_state=True,
    )

    denom = o_fla.double().abs().max().item()
    assert denom > 0, "fla reference is all-zero -- the comparison would be vacuous"
    diff = (o_mine.double() - o_fla.double()).abs().max().item()
    rel = diff / denom

    # This side's own float32-vs-float64 gap, which is the floor imposed by fla's internal
    # downcast. Computing it here rather than hard-coding a constant keeps the bound honest if
    # the shapes or the seed change.
    o_f32, _ = _kda_householder_torch()(
        q.float(),
        k.float(),
        v.float(),
        g_channel.float(),
        beta.float(),
        num_householder=R,
        scale=scale,
        output_final_state=True,
    )
    float32_floor = (o_f32.double() - o_mine.double()).abs().max().item() / denom
    print(
        f"external anchor: max|diff|={diff:.3e} relative={rel:.3e} "
        f"(float32 floor {float32_floor:.3e})"
    )

    # Agreement must be at float32 round-off, with headroom -- not at float64 ulp, which fla's
    # downcast makes unattainable. A real disagreement about the recurrence lands orders above
    # this (the measured scale/convention mismatch was 0.65).
    budget = max(10.0 * float32_floor, 1e-6)
    assert rel < budget, (
        f"disagreement with fla.ops.gated_delta_product.naive: relative {rel:.3e} exceeds the "
        f"float32-round-off budget {budget:.3e} (own float32 floor {float32_floor:.3e}). At g "
        f"constant along K and scale=1.0 these must agree to float32 round-off."
    )
