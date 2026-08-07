from __future__ import annotations

from collections.abc import Sequence

import pytest
import torch

from olmo_core.nn.memory.counterfactual import (
    CounterfactualLookupFunction,
    counterfactual_lookup,
)


def _logits_for_codes(codes: torch.Tensor, bits_per_route: int) -> torch.Tensor:
    bit_ids = torch.arange(bits_per_route, device=codes.device)
    bits = ((codes.unsqueeze(-1) >> bit_ids) & 1).bool()
    return torch.where(bits, 1.0, -1.0).reshape(*codes.shape[:2], -1)


def _pack_codes(z: torch.Tensor, bits_per_route: int) -> torch.Tensor:
    batch_size, seq_len, channels = z.shape
    num_routes = channels // bits_per_route
    bits = (z > 0).reshape(batch_size, seq_len, num_routes, bits_per_route)
    weights = 1 << torch.arange(bits_per_route, device=z.device)
    return (bits.to(torch.long) * weights).sum(dim=-1)


def _hard_address(
    codes: torch.Tensor,
    batch_idx: int,
    window_start: int,
    route_idx: int,
    order: int,
    alphabet_size: int,
) -> int:
    address = route_idx * alphabet_size**order
    for offset in range(order):
        address += int(codes[batch_idx, window_start + offset, route_idx]) * alphabet_size**offset
    return address


def test_public_api_documents_the_lngram_paper():
    assert "arXiv:2605.24869" in (CounterfactualLookupFunction.__doc__ or "")
    assert "arXiv:2605.24869" in (counterfactual_lookup.__doc__ or "")


def test_hard_forward_uses_little_endian_addresses_and_causal_zeros():
    # Route codes by token are [[1, 2], [3, 0], [0, 1]].
    codes = torch.tensor([[[1, 2], [3, 0], [0, 1]]])
    z = _logits_for_codes(codes, bits_per_route=2)
    unigram_table = torch.stack(
        (torch.arange(8, dtype=torch.float64), -torch.arange(8, dtype=torch.float64)),
        dim=-1,
    )
    bigram_table = torch.stack(
        (torch.arange(32, dtype=torch.float64), -torch.arange(32, dtype=torch.float64)),
        dim=-1,
    )

    # Deliberately request orders out of numerical order: outputs must follow `orders`.
    bigrams, unigrams = counterfactual_lookup(
        z.to(torch.float64),
        (bigram_table, unigram_table),
        (2, 1),
        bits_per_route=2,
    )

    expected_bigrams = torch.tensor(
        [
            [
                [0, 0, 0, 0],
                [13, -13, 18, -18],
                [3, -3, 20, -20],
            ]
        ],
        dtype=torch.float64,
    )
    expected_unigrams = torch.tensor(
        [
            [
                [1, -1, 6, -6],
                [3, -3, 4, -4],
                [0, 0, 5, -5],
            ]
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(bigrams, expected_bigrams)
    torch.testing.assert_close(unigrams, expected_unigrams)
    assert torch.count_nonzero(bigrams[:, :1]) == 0


def test_exact_table_gradients_scatter_add_hard_lookup_counts():
    bits_per_route = 1
    alphabet_size = 2
    codes = torch.tensor(
        [
            [[0, 1], [1, 0], [0, 1], [1, 0]],
            [[0, 1], [0, 1], [1, 0], [1, 0]],
        ]
    )
    z = _logits_for_codes(codes, bits_per_route).requires_grad_(True)
    orders = (1, 2)
    tables = tuple(
        torch.zeros(
            codes.shape[-1] * alphabet_size**order,
            2,
            dtype=torch.float64,
            requires_grad=True,
        )
        for order in orders
    )

    outputs = counterfactual_lookup(
        z.to(torch.float64),
        tables,
        orders,
        bits_per_route=bits_per_route,
    )
    sum(output.sum() for output in outputs).backward()

    for table, order in zip(tables, orders):
        expected = torch.zeros_like(table)
        for batch_idx in range(codes.shape[0]):
            for output_idx in range(order - 1, codes.shape[1]):
                window_start = output_idx - order + 1
                for route_idx in range(codes.shape[2]):
                    row = _hard_address(
                        codes,
                        batch_idx,
                        window_start,
                        route_idx,
                        order,
                        alphabet_size,
                    )
                    expected[row] += 1
        torch.testing.assert_close(table.grad, expected)


@pytest.mark.parametrize("seq_len", [0, 1])
def test_empty_and_short_sequences_return_zeros_and_zero_gradients(seq_len: int):
    z = torch.empty(2, seq_len, 2, dtype=torch.float64, requires_grad=True)
    table = torch.randn(64, 3, dtype=torch.float64, requires_grad=True)

    (output,) = counterfactual_lookup(z, (table,), (3,), bits_per_route=2)

    assert output.shape == (2, seq_len, 3)
    assert torch.count_nonzero(output) == 0
    output.sum().backward()
    assert z.grad is not None
    assert z.grad.shape == z.shape
    assert table.grad is not None
    torch.testing.assert_close(table.grad, torch.zeros_like(table))


def test_preserves_table_dtype_and_device_with_mixed_floating_inputs():
    z = torch.tensor([[[-0.5, 0.25]]], dtype=torch.float32, requires_grad=True)
    table = torch.arange(8, dtype=torch.float64).reshape(4, 2).requires_grad_(True)

    (output,) = counterfactual_lookup(z, (table,), (1,), bits_per_route=2)
    output.sum().backward()

    assert output.dtype is table.dtype
    assert output.device == table.device
    assert z.grad is not None and z.grad.dtype is z.dtype
    assert table.grad is not None and table.grad.dtype is table.dtype


def test_counterfactual_gradient_does_not_leak_into_table_gradients():
    z = torch.tensor([[[-0.4], [-0.8]]], dtype=torch.float64, requires_grad=True)
    table = torch.tensor([[1.0], [5.0], [9.0], [13.0]], requires_grad=True)
    temperature = 1.3
    scale = 0.7

    (output,) = counterfactual_lookup(
        z,
        (table,),
        (2,),
        bits_per_route=1,
        temperature=temperature,
        scale=scale,
    )
    upstream = torch.tensor([[[7.0], [3.0]]])
    output.backward(upstream)

    # The hard address is row 0. Rows 1 and 2 are consulted only by the
    # counterfactual z gradient and must receive no gradient.
    torch.testing.assert_close(
        table.grad,
        torch.tensor([[3.0], [0.0], [0.0], [0.0]]),
    )
    probabilities = torch.sigmoid(temperature * z.detach())
    slopes = scale * temperature * probabilities * (1 - probabilities)
    expected_z_grad = slopes * torch.tensor([[[3.0 * 4.0], [3.0 * 8.0]]])
    torch.testing.assert_close(z.grad, expected_z_grad)


def _local_smooth_objective(
    value: torch.Tensor,
    coordinate: tuple[int, int, int],
    *,
    hard_codes: torch.Tensor,
    tables: Sequence[torch.Tensor],
    orders: Sequence[int],
    upstreams: Sequence[torch.Tensor],
    bits_per_route: int,
    temperature: float,
    scale: float,
) -> torch.Tensor:
    batch_idx, token_idx, channel_idx = coordinate
    route_idx, bit_idx = divmod(channel_idx, bits_per_route)
    alphabet_size = 2**bits_per_route
    probability = torch.sigmoid(temperature * value)
    objective = value.new_zeros(())

    for table, order, upstream in zip(tables, orders, upstreams):
        memory_dim = table.shape[1]
        first_window = max(0, token_idx - order + 1)
        last_window = min(token_idx, hard_codes.shape[1] - order)
        for window_start in range(first_window, last_window + 1):
            offset = token_idx - window_start
            hard_address = _hard_address(
                hard_codes,
                batch_idx,
                window_start,
                route_idx,
                order,
                alphabet_size,
            )
            current_bit = int((hard_codes[batch_idx, token_idx, route_idx] >> bit_idx) & 1)
            delta = (1 << bit_idx) * alphabet_size**offset
            address_zero = hard_address - current_bit * delta
            address_one = address_zero + delta
            output_idx = window_start + order - 1
            grad = upstream[
                batch_idx,
                output_idx,
                route_idx * memory_dim : (route_idx + 1) * memory_dim,
            ]
            score_zero = torch.dot(grad, table[address_zero])
            score_one = torch.dot(grad, table[address_one])
            objective = objective + scale * (
                probability * score_one + (1 - probability) * score_zero
            )
    return objective


def test_z_gradient_matches_local_smooth_one_bit_finite_differences():
    torch.manual_seed(19)
    bits_per_route = 2
    temperature = 0.8
    scale = 1.4
    orders = (1, 2, 3)
    z = torch.tensor(
        [
            [
                [0.4, -0.7, 1.1, -0.3],
                [-0.8, 0.2, 0.6, 1.2],
                [0.9, 0.5, -1.0, 0.3],
                [-0.2, -1.3, 0.7, -0.6],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    num_routes = z.shape[-1] // bits_per_route
    alphabet_size = 2**bits_per_route
    tables = tuple(
        torch.randn(
            num_routes * alphabet_size**order,
            2,
            dtype=torch.float64,
        )
        for order in orders
    )
    upstreams = tuple(
        torch.randn(1, z.shape[1], num_routes * 2, dtype=torch.float64) for _ in orders
    )

    outputs = counterfactual_lookup(
        z,
        tables,
        orders,
        bits_per_route=bits_per_route,
        temperature=temperature,
        scale=scale,
    )
    torch.autograd.backward(outputs, upstreams)

    hard_codes = _pack_codes(z.detach(), bits_per_route)
    expected = torch.empty_like(z)
    epsilon = 1e-6
    for batch_idx in range(z.shape[0]):
        for token_idx in range(z.shape[1]):
            for channel_idx in range(z.shape[2]):
                coordinate = (batch_idx, token_idx, channel_idx)
                center = z.detach()[coordinate]
                plus = _local_smooth_objective(
                    center + epsilon,
                    coordinate,
                    hard_codes=hard_codes,
                    tables=tables,
                    orders=orders,
                    upstreams=upstreams,
                    bits_per_route=bits_per_route,
                    temperature=temperature,
                    scale=scale,
                )
                minus = _local_smooth_objective(
                    center - epsilon,
                    coordinate,
                    hard_codes=hard_codes,
                    tables=tables,
                    orders=orders,
                    upstreams=upstreams,
                    bits_per_route=bits_per_route,
                    temperature=temperature,
                    scale=scale,
                )
                expected[coordinate] = (plus - minus) / (2 * epsilon)

    torch.testing.assert_close(z.grad, expected, rtol=2e-6, atol=2e-8)


def test_validates_input_shapes_counts_and_types():
    valid_z = torch.randn(1, 3, 2)
    valid_table = torch.randn(4, 2)

    with pytest.raises(ValueError, match="3 dimensions"):
        counterfactual_lookup(valid_z[0], (valid_table,), (1,), bits_per_route=2)
    with pytest.raises(TypeError, match="floating point"):
        counterfactual_lookup(valid_z.long(), (valid_table,), (1,), bits_per_route=2)
    with pytest.raises(ValueError, match="divisible"):
        counterfactual_lookup(torch.randn(1, 2, 3), (valid_table,), (1,), bits_per_route=2)
    with pytest.raises(ValueError, match="at least one route"):
        counterfactual_lookup(torch.empty(1, 2, 0), (torch.empty(0, 2),), (1,))
    with pytest.raises(ValueError, match="same length"):
        counterfactual_lookup(valid_z, (valid_table,), (1, 2), bits_per_route=2)
    with pytest.raises(ValueError, match="positive integer"):
        counterfactual_lookup(valid_z, (valid_table,), (0,), bits_per_route=2)
    with pytest.raises(ValueError, match="2 dimensions"):
        counterfactual_lookup(valid_z, (valid_table.unsqueeze(0),), (1,), bits_per_route=2)
    with pytest.raises(ValueError, match="rows"):
        counterfactual_lookup(valid_z, (torch.randn(5, 2),), (1,), bits_per_route=2)
    with pytest.raises(ValueError, match="memory dimension"):
        counterfactual_lookup(
            valid_z,
            (valid_table, torch.randn(16, 3)),
            (1, 2),
            bits_per_route=2,
        )
    with pytest.raises(TypeError, match="floating point"):
        counterfactual_lookup(valid_z, (valid_table.long(),), (1,), bits_per_route=2)
    with pytest.raises(ValueError, match="same device"):
        counterfactual_lookup(
            valid_z,
            (torch.empty(4, 2, device="meta"),),
            (1,),
            bits_per_route=2,
        )


@pytest.mark.parametrize("bits_per_route", [True, 0, -1, 1.5])
def test_rejects_invalid_bits_per_route(bits_per_route):
    with pytest.raises((TypeError, ValueError), match="bits_per_route"):
        counterfactual_lookup(
            torch.randn(1, 2, 2),
            (torch.randn(4, 1),),
            (1,),
            bits_per_route=bits_per_route,
        )


@pytest.mark.parametrize("temperature", [True, 0.0, -0.5, float("nan"), float("inf")])
def test_rejects_invalid_temperature(temperature):
    with pytest.raises((TypeError, ValueError), match="temperature"):
        counterfactual_lookup(
            torch.randn(1, 2, 2),
            (torch.randn(4, 1),),
            (1,),
            bits_per_route=2,
            temperature=temperature,
        )


@pytest.mark.parametrize(
    "scale",
    [True, -0.5, float("nan"), float("inf"), -float("inf")],
)
def test_rejects_invalid_scale(scale):
    with pytest.raises((TypeError, ValueError), match="scale"):
        counterfactual_lookup(
            torch.randn(1, 2, 2),
            (torch.randn(4, 1),),
            (1,),
            bits_per_route=2,
            scale=scale,
        )
