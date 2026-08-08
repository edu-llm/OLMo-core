import inspect
from collections.abc import Callable

import pytest
import torch
import olmo_core.nn.flash_pd_ssm.triton_kernel as triton_kernel_module

from olmo_core.nn.flash_pd_ssm import (
    affine_chunkwise_reference,
    affine_recurrent_reference,
    destination_diagonal_to_dense,
    flash_pd_triton_scan,
    sparse_chunkwise_reference,
    sparse_recurrent_reference,
    triton_capability,
)
from olmo_core.testing import requires_gpu


def _run_with_gradients(
    scan: Callable[..., torch.Tensor],
    *,
    dtype: torch.dtype,
    chunk_size: int | None = None,
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    torch.manual_seed(10)
    batch, heads, time, state = 2, 2, 7, 4
    destination = torch.randint(state, (batch, heads, time, state))
    values = [
        torch.randn(batch, heads, time, state, dtype=dtype, requires_grad=True) for _ in range(4)
    ]
    initial_values = [
        torch.randn(batch, heads, state, dtype=dtype, requires_grad=True) for _ in range(2)
    ]
    diagonal = torch.complex(values[0], values[1])
    bias = torch.complex(values[2], values[3])
    initial = torch.complex(initial_values[0], initial_values[1])
    transition = destination_diagonal_to_dense(destination, diagonal)

    kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
    output = scan(transition, bias, initial=initial, **kwargs)
    loss = output.real.square().mean() + output.imag.square().mean()
    gradients = torch.autograd.grad(loss, (*values, *initial_values))
    return output, gradients


@pytest.mark.parametrize(
    "dtype, rtol, atol",
    [
        pytest.param(torch.float64, 1e-10, 1e-10, id="float64"),
        pytest.param(torch.float32, 2e-5, 2e-5, id="float32"),
    ],
)
def test_recurrent_and_chunkwise_outputs_and_gradients_match(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
):
    recurrent_output, recurrent_gradients = _run_with_gradients(
        affine_recurrent_reference,
        dtype=dtype,
    )
    chunkwise_output, chunkwise_gradients = _run_with_gradients(
        affine_chunkwise_reference,
        dtype=dtype,
        chunk_size=3,
    )

    torch.testing.assert_close(
        chunkwise_output,
        recurrent_output,
        rtol=rtol,
        atol=atol,
    )
    for chunkwise_gradient, recurrent_gradient in zip(
        chunkwise_gradients,
        recurrent_gradients,
    ):
        torch.testing.assert_close(
            chunkwise_gradient,
            recurrent_gradient,
            rtol=rtol,
            atol=atol,
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sparse_recurrence_matches_dense_reference(dtype: torch.dtype):
    torch.manual_seed(11)
    batch, heads, time, state = 2, 3, 9, 5
    destination = torch.randint(state, (batch, heads, time, state))
    diagonal = torch.complex(
        torch.randn(batch, heads, time, state, dtype=dtype),
        torch.randn(batch, heads, time, state, dtype=dtype),
    )
    bias = torch.complex(
        torch.randn(batch, heads, time, state, dtype=dtype),
        torch.randn(batch, heads, time, state, dtype=dtype),
    )
    initial = torch.complex(
        torch.randn(batch, heads, state, dtype=dtype),
        torch.randn(batch, heads, state, dtype=dtype),
    )

    expected = affine_recurrent_reference(
        destination_diagonal_to_dense(destination, diagonal),
        bias,
        initial=initial,
    )
    actual = sparse_recurrent_reference(destination, diagonal, bias, initial=initial)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_sparse_chunkwise_outputs_and_gradients_match_recurrent(dtype: torch.dtype):
    torch.manual_seed(13)
    batch, heads, time, state = 2, 2, 11, 5
    destination = torch.randint(state, (batch, heads, time, state))
    destination[..., :2] = 0
    leaves = [
        torch.randn(batch, heads, time, state, dtype=dtype, requires_grad=True) for _ in range(4)
    ]
    initial_leaves = [
        torch.randn(batch, heads, state, dtype=dtype, requires_grad=True) for _ in range(2)
    ]

    def run(scan, chunk_size=None):
        diagonal = torch.complex(leaves[0], leaves[1])
        bias = torch.complex(leaves[2], leaves[3])
        initial = torch.complex(initial_leaves[0], initial_leaves[1])
        kwargs = {} if chunk_size is None else {"chunk_size": chunk_size}
        output = scan(destination, diagonal, bias, initial=initial, **kwargs)
        loss = output.real.square().mean() + output.imag.square().mean()
        gradients = torch.autograd.grad(
            loss,
            (*leaves, *initial_leaves),
            retain_graph=True,
        )
        return output, gradients

    recurrent_output, recurrent_gradients = run(sparse_recurrent_reference)
    chunkwise_output, chunkwise_gradients = run(
        sparse_chunkwise_reference,
        chunk_size=4,
    )
    tolerance = 1e-10 if dtype == torch.float64 else 2e-5
    torch.testing.assert_close(
        chunkwise_output,
        recurrent_output,
        rtol=tolerance,
        atol=tolerance,
    )
    for actual, expected in zip(chunkwise_gradients, recurrent_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def test_triton_capability_probe_explains_cpu_fallback():
    destination = torch.zeros(1, 1, 2, 4, dtype=torch.long)
    values = torch.zeros(1, 1, 2, 4, dtype=torch.complex64)
    capability = triton_capability(destination, values, values)

    assert not capability.available
    assert "CUDA" in capability.reason or "Triton" in capability.reason


def test_triton_capability_rejects_zero_state_before_environment_probe():
    destination = torch.zeros(1, 1, 2, 0, dtype=torch.long)
    values = torch.zeros(1, 1, 2, 0, dtype=torch.complex64)

    capability = triton_capability(destination, values, values)

    assert not capability.available
    assert "state_size" in capability.reason


def test_triton_chunk_loops_are_not_statically_unrolled():
    source = inspect.getsource(triton_kernel_module)

    assert "tl.static_range" not in source


@pytest.mark.gpu
@requires_gpu
def test_triton_three_phase_scan_matches_sparse_reference():
    torch.manual_seed(12)
    batch, heads, time, state = 2, 2, 17, 16
    destination = torch.randint(state, (batch, heads, time, state), device="cuda")
    diagonal = torch.complex(
        torch.randn(batch, heads, time, state, device="cuda"),
        torch.randn(batch, heads, time, state, device="cuda"),
    )
    bias = torch.complex(
        torch.randn(batch, heads, time, state, device="cuda"),
        torch.randn(batch, heads, time, state, device="cuda"),
    )

    capability = triton_capability(destination, diagonal, bias)
    assert capability.available, capability.reason
    expected = sparse_recurrent_reference(destination, diagonal, bias)
    actual = flash_pd_triton_scan(destination, diagonal, bias, chunk_size=8)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.gpu
@requires_gpu
def test_triton_preserves_non_bijective_collision_sums():
    destination = torch.tensor(
        [[[[0, 0, 2, 3], [1, 1, 1, 1], [3, 0, 3, 0]]]],
        dtype=torch.int64,
        device="cuda",
    )
    diagonal = torch.ones_like(destination, dtype=torch.complex64)
    bias = torch.zeros_like(diagonal)

    capability = triton_capability(
        destination,
        diagonal,
        bias,
        chunk_size=2,
    )
    assert capability.available, capability.reason
    expected = sparse_recurrent_reference(destination, diagonal, bias)
    actual = flash_pd_triton_scan(destination, diagonal, bias, chunk_size=2)

    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)
