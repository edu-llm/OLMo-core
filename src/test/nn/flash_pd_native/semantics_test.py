import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    NativePDMode,
    compact_hard_selection,
    dense_scan_oracle,
    flash_pd_scan,
    prove_selected_maps_bijective,
)


def _split_inputs(dtype: torch.dtype = torch.float64):
    diagonal_real = torch.tensor([[[[0.5, 0.25], [0.75, 0.5]]]], dtype=dtype)
    diagonal_imag = torch.tensor([[[[0.1, -0.2], [0.0, 0.25]]]], dtype=dtype)
    bias_real = torch.tensor([[[[1.0, 2.0], [-0.5, 0.25]]]], dtype=dtype)
    bias_imag = torch.tensor([[[[0.0, 0.5], [0.75, -0.25]]]], dtype=dtype)
    return diagonal_real, diagonal_imag, bias_real, bias_imag


def test_general_scatter_preserves_two_source_one_destination_collision():
    destination = torch.tensor([[[0, 0]]], dtype=torch.int16)
    routes = torch.zeros((1, 1, 2), dtype=torch.int16)
    diagonal_real, diagonal_imag, bias_real, bias_imag = _split_inputs()

    expected_real, expected_imag = dense_scan_oracle(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        mode=NativePDMode.GENERAL_SCATTER,
    )
    actual_real, actual_imag = flash_pd_scan(
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        mode=NativePDMode.GENERAL_SCATTER,
        backend="reference",
    )

    torch.testing.assert_close(actual_real, expected_real)
    torch.testing.assert_close(actual_imag, expected_imag)
    # At t=1 both source lanes contribute to destination zero.
    assert actual_real[0, 0, 1, 0] != bias_real[0, 0, 1, 0]
    torch.testing.assert_close(actual_real[0, 0, 1, 1], bias_real[0, 0, 1, 1])


def test_permutation_gather_requires_and_uses_bijection_proof():
    permutation = torch.tensor([[[1, 2, 0], [2, 0, 1]]], dtype=torch.int16)
    routes = torch.tensor([[[0, 1, 0, 1]]], dtype=torch.int16)
    colliding = permutation.clone()
    colliding[0, 1] = torch.tensor([0, 0, 2], dtype=torch.int16)
    used_collision = routes.clone()
    used_collision[..., 1] = 1

    proof = prove_selected_maps_bijective(permutation, routes)
    assert proof.proven
    assert proof.inverse_destination is not None
    assert proof.inverse_destination.dtype == torch.int16

    failed = prove_selected_maps_bijective(colliding, used_collision)
    assert not failed.proven
    assert failed.failing_head == 0
    assert failed.failing_dictionary == 1

    diagonal_real = torch.ones((1, 1, 4, 3))
    diagonal_imag = torch.zeros_like(diagonal_real)
    bias_real = torch.randn_like(diagonal_real)
    bias_imag = torch.randn_like(diagonal_real)
    with pytest.raises(ValueError, match="bijective"):
        flash_pd_scan(
            colliding,
            used_collision,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            mode=NativePDMode.PERMUTATION_GATHER,
            backend="reference",
        )


def test_compact_hard_selection_is_int16_and_never_materializes_token_matrices():
    dictionary_logits = torch.full((2, 3, 4, 4), -5.0)
    for head in range(2):
        for dictionary_idx in range(3):
            destination = torch.roll(torch.arange(4), dictionary_idx + head)
            dictionary_logits[head, dictionary_idx, destination, torch.arange(4)] = 5.0
    selector_logits = torch.randn(2, 7, 2, 3)

    selected = compact_hard_selection(dictionary_logits, selector_logits)

    assert selected.destination.shape == (2, 3, 4)
    assert selected.routes.shape == (2, 2, 7)
    assert selected.destination.dtype == torch.int16
    assert selected.routes.dtype == torch.int16
    assert selected.destination.numel() == 2 * 3 * 4
    assert selected.routes.numel() == 2 * 2 * 7


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_reference_matches_dense_oracle_outputs_and_every_input_gradient(dtype: torch.dtype):
    torch.manual_seed(9)
    destination = torch.tensor([[[1, 0, 2], [0, 0, 2]]], dtype=torch.int16)
    routes = torch.tensor([[[0, 1, 0, 1, 0]]], dtype=torch.int16)
    leaves = [torch.randn(1, 1, 5, 3, dtype=dtype, requires_grad=True) for _ in range(4)]

    expected = dense_scan_oracle(destination, routes, *leaves)
    expected_loss = sum(value.square().sum() for value in expected)
    expected_gradients = torch.autograd.grad(expected_loss, leaves, retain_graph=True)

    actual = flash_pd_scan(destination, routes, *leaves, backend="reference")
    actual_loss = sum(value.square().sum() for value in actual)
    actual_gradients = torch.autograd.grad(actual_loss, leaves)

    tolerance = 1e-10 if dtype == torch.float64 else 2e-5
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, atol=tolerance, rtol=tolerance)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )
