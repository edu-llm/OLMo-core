"""Differentiable (autograd) KDA + ``R``-Householder recurrence, and its verification harness.

The implementation itself lives in the repository, at
``OLMo-core/src/olmo_core/nn/attention/kda_householder_torch.py``, and is **re-exported** here as
:func:`kda_householder_torch` rather than copied -- a second copy of the recurrence is exactly the
kind of thing that silently drifts out of sync with the one that ships. The module is loaded
straight off disk by file path (see :func:`_load_impl`), so this probe does not need the whole
``olmo_core`` package to import, only ``torch``.

The implementation is built out of ordinary differentiable ``torch`` ops with a Python loop over
``T`` and over ``R``: there is no custom ``autograd.Function`` and no hand-derived gradient, so
autograd supplies the backward pass.

Run directly to execute the verification suite::

    OLMo-core/.venv/bin/python probes/kda_householder_autograd.py

Checks performed:

1. **forward vs oracle** -- against ``naive_recurrent_kda_householder`` from
   ``probes/naive_kda_householder.py`` (already validated bit-exact against fla's KDA at ``R = 1``),
   in ``float64``, for ``R`` in ``{1, 2, 3}``.
2. **gradcheck** -- ``torch.autograd.gradcheck`` in ``float64`` on ``B=1, T=4, H=1, K=3, V=3`` for
   ``R`` in ``{1, 2}``, w.r.t. all of ``q, k, v, g, beta`` jointly. This is a *numerical* Jacobian
   check, so it catches any gradient error without anyone deriving one.
3. **gradgradcheck** -- second-order check, same case, ``R = 1``.
4. **gradient sanity** -- every input receives a finite, non-zero gradient.
5. **varlen** -- ``cu_seqlens`` reproduces independent per-sequence calls (forward and gradient).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch

__all__ = ["kda_householder_torch"]

_HERE = Path(__file__).resolve().parent
_IMPL_PATH = (
    _HERE.parent / "OLMo-core" / "src" / "olmo_core" / "nn" / "attention" / "kda_householder_torch.py"
)


def _load_impl() -> Callable[..., Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """Load ``kda_householder_torch`` from the in-repo module by file path.

    Loading by path (rather than ``import olmo_core...``) keeps this probe runnable with any
    ``torch``-equipped interpreter and avoids importing the rest of ``olmo_core``.

    :returns: the differentiable implementation.

    :raises FileNotFoundError: if the repository module is missing.
    """
    if not _IMPL_PATH.is_file():
        raise FileNotFoundError(f"expected the implementation at {_IMPL_PATH}")
    spec = importlib.util.spec_from_file_location("_kda_householder_torch_impl", _IMPL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.kda_householder_torch


kda_householder_torch = _load_impl()


# ---------------------------------------------------------------------------------------------
# Verification harness
# ---------------------------------------------------------------------------------------------


def _make_inputs(
    B: int,
    T: int,
    H: int,
    K: int,
    V: int,
    R: int,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
    requires_grad: bool = False,
) -> Tuple[torch.Tensor, ...]:
    """Build a well-conditioned input set.

    ``k`` is L2-normalized so the delta rule stays contractive and ``g`` is ``logsigmoid``-shaped
    so the decay is strictly in ``(0, 1)`` -- the regime the real layer operates in, and the one
    where the recurrence is numerically stable enough for a finite-difference Jacobian.

    :returns: ``(q, k, v, g, beta)``.
    """
    gen = torch.Generator().manual_seed(seed)

    def rnd(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=gen, dtype=dtype)

    q = torch.nn.functional.normalize(rnd(B, T, H, K), p=2, dim=-1)
    k = torch.nn.functional.normalize(rnd(B, T * R, H, K), p=2, dim=-1)
    v = rnd(B, T * R, H, V)
    beta = rnd(B, T * R, H).sigmoid()
    g = torch.nn.functional.logsigmoid(rnd(B, T, H, K))

    tensors = (q, k, v, g, beta)
    if requires_grad:
        tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    return tensors


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    """:returns: max absolute difference between ``a`` and ``b``, computed in ``float64``."""
    return (a.double() - b.double()).abs().max().item()


def _main() -> None:  # noqa: C901
    from naive_kda_householder import (  # type: ignore
        _recurrence as oracle_recurrence,
        naive_recurrent_kda_householder,
    )

    failures: list[str] = []

    # -------------------------------------------------------------------- CHECK 1: forward
    # Two tolerances, because the oracle has two entry points:
    #   * its *public* entry point hard-codes float32 accumulation (matching the fla naive-kernel
    #     convention), so even with float64 inputs the comparison floors at float32 round-off.
    #   * its `_recurrence` body takes a `compute_dtype` hook, documented for exactly this purpose,
    #     and running BOTH sides in float64 is what makes a ~1e-12 tolerance meaningful.
    print("CHECK 1  forward vs naive oracle")
    B, T, H, K, V = 2, 8, 2, 4, 4
    atol64, atol32 = 1e-12, 1e-6
    scale = K**-0.5
    for R in (1, 2, 3):
        q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, seed=R)
        o_mine, s_mine = kda_householder_torch(
            q, k, v, g, beta, num_householder=R, output_final_state=True
        )
        # (a) float64 on both sides -- the real equivalence check.
        o_ref64, s_ref64 = oracle_recurrence(
            q, k, v, g, beta, R, scale, None, torch.float64
        )
        do64, ds64 = _maxdiff(o_mine, o_ref64), _maxdiff(s_mine, s_ref64)
        # (b) against the oracle's public (float32-accumulating) entry point.
        o_ref32, _ = naive_recurrent_kda_householder(q, k, v, g, beta, num_householder=R)
        do32 = _maxdiff(o_mine, o_ref32)

        ok = do64 <= atol64 and ds64 <= atol64 and do32 <= atol32
        if not ok:
            failures.append(
                f"CHECK 1 R={R}: fp64 max|do| = {do64:.3e}, max|dS| = {ds64:.3e} "
                f"(atol {atol64:.0e}); vs public fp32 oracle {do32:.3e} (atol {atol32:.0e})"
            )
        print(
            f"  R={R}: fp64 max|diff o| = {do64:.3e}  max|diff S| = {ds64:.3e}   |   "
            f"vs public fp32 oracle: {do32:.3e}   {'PASS' if ok else 'FAIL'}"
        )

    # Also confirm a non-zero initial_state is threaded identically.
    R = 2
    q, k, v, g, beta = _make_inputs(B, T, H, K, V, R, seed=101)
    h0 = torch.randn(B, H, K, V, dtype=torch.float64) * 0.1
    o_mine, s_mine = kda_householder_torch(
        q, k, v, g, beta, num_householder=R, initial_state=h0, output_final_state=True
    )
    o_ref, s_ref = oracle_recurrence(q, k, v, g, beta, R, scale, h0, torch.float64)
    d = max(_maxdiff(o_mine, o_ref), _maxdiff(s_mine, s_ref))
    ok = d <= atol64
    if not ok:
        failures.append(f"CHECK 1 initial_state: max|diff| = {d:.3e} > {atol64:.0e}")
    print(f"  initial_state (R=2, fp64): max|diff| = {d:.3e}   {'PASS' if ok else 'FAIL'}")

    # -------------------------------------------------------------------- CHECK 2: gradcheck
    print()
    print("CHECK 2  torch.autograd.gradcheck, float64, B=1 T=4 H=1 K=3 V=3")
    for R in (1, 2):
        inputs = _make_inputs(1, 4, 1, 3, 3, R, seed=1000 + R, requires_grad=True)

        def fn(*args: torch.Tensor, _R: int = R) -> torch.Tensor:
            o, _ = kda_householder_torch(*args, num_householder=_R)
            return o

        try:
            passed = torch.autograd.gradcheck(
                fn, inputs, eps=1e-6, atol=1e-8, rtol=1e-5, raise_exception=True
            )
            err = _max_jacobian_error(fn, inputs)
            print(f"  R={R}: gradcheck PASS   max|analytic - numeric| = {err:.3e}")
            if not passed:
                failures.append(f"CHECK 2 R={R}: gradcheck returned False")
        except Exception as exc:  # noqa: BLE001 - the message is the finding
            err = _max_jacobian_error(fn, inputs)
            failures.append(f"CHECK 2 R={R}: gradcheck FAILED (max err {err:.3e}): {exc}")
            print(f"  R={R}: gradcheck FAIL   max|analytic - numeric| = {err:.3e}")
            print(f"    {str(exc).splitlines()[0]}")

    # gradcheck with a non-zero initial_state in the input tuple.
    inputs = _make_inputs(1, 4, 1, 3, 3, 2, seed=2002, requires_grad=True)
    h0 = (torch.randn(1, 1, 3, 3, dtype=torch.float64) * 0.1).requires_grad_(True)

    def fn_h0(*args: torch.Tensor) -> torch.Tensor:
        o, _ = kda_householder_torch(*args[:5], num_householder=2, initial_state=args[5])
        return o

    try:
        torch.autograd.gradcheck(fn_h0, (*inputs, h0), eps=1e-6, atol=1e-8, rtol=1e-5)
        print("  R=2 + initial_state: gradcheck PASS")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"CHECK 2 initial_state: gradcheck FAILED: {exc}")
        print(f"  R=2 + initial_state: gradcheck FAIL -- {str(exc).splitlines()[0]}")

    # -------------------------------------------------------------------- CHECK 3: gradgradcheck
    print()
    print("CHECK 3  torch.autograd.gradgradcheck (second order), float64, R=1")
    inputs = _make_inputs(1, 3, 1, 3, 3, 1, seed=77, requires_grad=True)

    def fn1(*args: torch.Tensor) -> torch.Tensor:
        o, _ = kda_householder_torch(*args, num_householder=1)
        return o

    try:
        torch.autograd.gradgradcheck(fn1, inputs, eps=1e-6, atol=1e-6, rtol=1e-4)
        print("  R=1: gradgradcheck PASS")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"CHECK 3: gradgradcheck FAILED: {exc}")
        print(f"  R=1: gradgradcheck FAIL -- {str(exc).splitlines()[0]}")

    # -------------------------------------------------------------------- CHECK 4: grad sanity
    print()
    print("CHECK 4  gradients finite and non-zero for every input (float32, R=1,2,3)")
    names = ("q", "k", "v", "g", "beta")
    for R in (1, 2, 3):
        tensors = _make_inputs(2, 6, 2, 4, 4, R, dtype=torch.float32, seed=R, requires_grad=True)
        o, _ = kda_householder_torch(*tensors, num_householder=R)
        o.square().mean().backward()
        for name, t in zip(names, tensors):
            grad = t.grad
            bad = grad is None or not torch.isfinite(grad).all() or grad.abs().max().item() == 0.0
            if bad:
                failures.append(f"CHECK 4 R={R}: grad w.r.t. {name} is None/non-finite/all-zero")
        summary = "  ".join(
            f"{n}:{(t.grad.abs().max().item() if t.grad is not None else float('nan')):.2e}"
            for n, t in zip(names, tensors)
        )
        print(f"  R={R}: max|grad| -> {summary}")

    # -------------------------------------------------------------------- CHECK 5: varlen
    print()
    print("CHECK 5  cu_seqlens equals independent per-sequence calls (float64, R=2)")
    R = 2
    lengths = [3, 5, 2]
    T_tot = sum(lengths)
    tensors = _make_inputs(1, T_tot, 2, 4, 4, R, seed=31, requires_grad=True)
    cu = torch.tensor([0, 3, 8, 10], dtype=torch.long)
    o_var, s_var = kda_householder_torch(
        *tensors, num_householder=R, cu_seqlens=cu, output_final_state=True
    )
    o_var.square().sum().backward()
    grads_var = [t.grad.clone() for t in tensors]

    ref_tensors = tuple(t.detach().clone().requires_grad_(True) for t in tensors)
    o_ref_parts, s_ref_parts = [], []
    start = 0
    for length in lengths:
        end = start + length
        q_, k_, v_, g_, b_ = ref_tensors
        o_n, s_n = kda_householder_torch(
            q_[:, start:end],
            k_[:, start * R : end * R],
            v_[:, start * R : end * R],
            g_[:, start:end],
            b_[:, start * R : end * R],
            num_householder=R,
            output_final_state=True,
        )
        o_ref_parts.append(o_n)
        s_ref_parts.append(s_n)
        start = end
    o_seq = torch.cat(o_ref_parts, dim=1)
    s_seq = torch.cat(s_ref_parts, dim=0)
    o_seq.square().sum().backward()

    d_fwd = max(_maxdiff(o_var, o_seq), _maxdiff(s_var, s_seq))
    d_bwd = max(_maxdiff(a, b.grad) for a, b in zip(grads_var, ref_tensors))
    ok = d_fwd <= 1e-12 and d_bwd <= 1e-12
    if not ok:
        failures.append(f"CHECK 5: varlen mismatch fwd {d_fwd:.3e} bwd {d_bwd:.3e}")
    print(
        f"  max|diff fwd| = {d_fwd:.3e}   max|diff grads| = {d_bwd:.3e}   "
        f"{'PASS' if ok else 'FAIL'}"
    )

    # -------------------------------------------------------------------- SUMMARY
    print()
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("RESULT: ALL PASS")


def _max_jacobian_error(
    fn: Callable[..., torch.Tensor],
    inputs: Tuple[torch.Tensor, ...],
    eps: float = 1e-6,
) -> float:
    """Max absolute discrepancy between the analytic and a central-difference Jacobian.

    ``gradcheck`` only reports the *first* offending entry, so this recomputes both Jacobians in
    full to give a single headline number for the report. The analytic side uses the public
    :func:`torch.autograd.functional.jacobian`; the numerical side is an explicit central
    difference over every input element.

    :param fn: The function under test.
    :param inputs: Its differentiable inputs.
    :param eps: Central-difference step.

    :returns: ``max |J_analytic - J_numeric|`` over all inputs and all Jacobian entries.
    """
    analytic = torch.autograd.functional.jacobian(fn, inputs, vectorize=True)
    worst = 0.0
    for idx, x in enumerate(inputs):
        base = x.detach()
        flat = base.reshape(-1)
        j_a = analytic[idx].reshape(-1, flat.numel())  # [n_out, n_in]
        for i in range(flat.numel()):
            plus, minus = base.clone(), base.clone()
            plus.reshape(-1)[i] += eps
            minus.reshape(-1)[i] -= eps
            args_p = list(inputs)
            args_m = list(inputs)
            args_p[idx], args_m[idx] = plus, minus
            with torch.no_grad():
                col = (fn(*args_p) - fn(*args_m)).reshape(-1) / (2 * eps)
            worst = max(worst, (j_a[:, i] - col).abs().max().item())
    return worst


if __name__ == "__main__":
    _main()
