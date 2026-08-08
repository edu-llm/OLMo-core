import pytest
import torch

import olmo_core.nn.flash_pd_ssm.autograd as autograd_module
from olmo_core.nn.flash_pd_ssm import (
    affine_recurrent_reference,
    selected_transition_matrix,
    sparse_recurrent_reference,
    sparse_ste_scan,
)


def _run_dense_or_sparse(*, sparse: bool, dtype: torch.dtype):
    torch.manual_seed(20)
    batch, heads, time, state, dictionary_size = 2, 2, 5, 4, 3
    temperature = 0.7
    dictionary = torch.randn(
        heads,
        dictionary_size,
        state,
        state,
        dtype=dtype,
        requires_grad=True,
    )
    selector = torch.randn(
        batch,
        time,
        heads,
        dictionary_size,
        dtype=dtype,
        requires_grad=True,
    )
    leaves = [
        torch.randn(
            batch,
            heads,
            time,
            state,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(4)
    ]
    diagonal = torch.complex(leaves[0], leaves[1])
    bias = torch.complex(leaves[2], leaves[3])

    if sparse:
        output = sparse_ste_scan(
            dictionary,
            selector,
            diagonal,
            bias,
            temperature=temperature,
            use_triton=False,
        )
    else:
        transition_selector = selected_transition_matrix(
            dictionary,
            selector,
            temperature=temperature,
        )
        diagonal_bt = diagonal.permute(0, 2, 1, 3)
        transition = transition_selector.to(diagonal.dtype) * diagonal_bt.unsqueeze(-2)
        output = affine_recurrent_reference(
            transition.permute(0, 2, 1, 3, 4),
            bias,
        )

    weight_real = torch.randn(output.shape, dtype=dtype)
    weight_imag = torch.randn(output.shape, dtype=dtype)
    loss = (output.real * weight_real + output.imag * weight_imag).sum()
    gradients = torch.autograd.grad(loss, (dictionary, selector, *leaves))
    return output, gradients


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 1e-9), (torch.float32, 3e-5)],
)
def test_sparse_ste_scan_matches_dense_forward_and_all_gradients(
    dtype: torch.dtype,
    tolerance: float,
):
    expected_output, expected_gradients = _run_dense_or_sparse(
        sparse=False,
        dtype=dtype,
    )
    actual_output, actual_gradients = _run_dense_or_sparse(
        sparse=True,
        dtype=dtype,
    )

    torch.testing.assert_close(
        actual_output,
        expected_output,
        rtol=tolerance,
        atol=tolerance,
    )
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=tolerance,
            atol=tolerance,
        )


def test_sparse_autograd_detaches_inputs_for_forward_only_triton(monkeypatch):
    torch.manual_seed(21)
    dictionary = torch.randn(1, 2, 3, 3, requires_grad=True)
    selector = torch.randn(1, 4, 1, 2, requires_grad=True)
    diagonal = torch.complex(
        torch.randn(1, 1, 4, 3, requires_grad=True),
        torch.randn(1, 1, 4, 3, requires_grad=True),
    )
    bias = torch.complex(
        torch.randn(1, 1, 4, 3, requires_grad=True),
        torch.randn(1, 1, 4, 3, requires_grad=True),
    )

    def forward_only_kernel(destination, kernel_diagonal, kernel_bias, **kwargs):
        del kwargs
        assert not kernel_diagonal.requires_grad
        assert not kernel_bias.requires_grad
        return sparse_recurrent_reference(destination, kernel_diagonal, kernel_bias)

    monkeypatch.setattr(
        autograd_module,
        "flash_pd_triton_scan",
        forward_only_kernel,
    )
    output = sparse_ste_scan(
        dictionary,
        selector,
        diagonal,
        bias,
        temperature=1.0,
        use_triton=True,
    )
    output.abs().sum().backward()

    assert dictionary.grad is not None
    assert selector.grad is not None
