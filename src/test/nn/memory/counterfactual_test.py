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


def test_route_code_packing_preserves_routes_wider_than_eight_bits() -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    z = torch.ones(1, 1, 9)
    codes = counterfactual_module._pack_route_codes(z, 9)

    assert codes.dtype is torch.int64
    assert codes.item() == 2**9 - 1


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


def test_optimized_grad_z_preserves_exact_hard_table_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    z = torch.full((1, 3, 4), -1.0, requires_grad=True)
    tables = (
        torch.zeros(16**2, 1, requires_grad=True),
        torch.zeros(16**3, 1, requires_grad=True),
    )
    sentinel = torch.full_like(z, 7.0)
    calls = []

    def fake_optimized(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        fake_optimized,
    )
    outputs = counterfactual_lookup(z, tables, (2, 3), bits_per_route=4)
    sum(output.sum() for output in outputs).backward()

    assert len(calls) == 1
    torch.testing.assert_close(z.grad, sentinel)
    expected_order_2 = torch.zeros_like(tables[0])
    expected_order_2[0] = 2
    expected_order_3 = torch.zeros_like(tables[1])
    expected_order_3[0] = 1
    torch.testing.assert_close(tables[0].grad, expected_order_2)
    torch.testing.assert_close(tables[1].grad, expected_order_3)


def test_scale_zero_skips_counterfactual_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        lambda *args, **kwargs: pytest.fail("optimized path should not run"),
    )
    z = torch.randn(1, 3, 4, requires_grad=True)
    tables = (
        torch.randn(16**2, 1, requires_grad=True),
        torch.randn(16**3, 1, requires_grad=True),
    )
    outputs = counterfactual_lookup(
        z,
        tables,
        (2, 3),
        bits_per_route=4,
        scale=0.0,
    )
    sum(output.sum() for output in outputs).backward()

    torch.testing.assert_close(z.grad, torch.zeros_like(z))


def test_required_triton_rejects_reference_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        lambda *args, **kwargs: None,
    )
    z = torch.randn(1, 3, 4, requires_grad=True)
    tables = (torch.randn(16**2, 1), torch.randn(16**3, 1))
    outputs = counterfactual_lookup(
        z,
        tables,
        (2, 3),
        bits_per_route=4,
        require_triton=True,
    )

    with pytest.raises(RuntimeError, match="requires Triton acceleration"):
        sum(output.sum() for output in outputs).backward()


def test_higher_order_gradients_use_reference_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        lambda *args, **kwargs: pytest.fail("Triton path is not higher-order differentiable"),
    )
    z = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    tables = (
        torch.randn(16**2, 1, dtype=torch.float64),
        torch.randn(16**3, 1, dtype=torch.float64),
    )
    outputs = counterfactual_lookup(z, tables, (2, 3), bits_per_route=4)
    first_grad = torch.autograd.grad(
        sum(output.sum() for output in outputs),
        z,
        create_graph=True,
    )[0]
    second_grad = torch.autograd.grad(first_grad.sum(), z)[0]

    assert torch.isfinite(first_grad).all()
    assert torch.isfinite(second_grad).all()


def test_scale_zero_preserves_higher_order_gradients() -> None:
    z = torch.randn(1, 3, 4, dtype=torch.float64, requires_grad=True)
    tables = (
        torch.randn(16**2, 1, dtype=torch.float64),
        torch.randn(16**3, 1, dtype=torch.float64),
    )
    outputs = counterfactual_lookup(
        z,
        tables,
        (2, 3),
        bits_per_route=4,
        scale=0.0,
    )
    first_grad = torch.autograd.grad(
        sum(output.sum() for output in outputs),
        z,
        create_graph=True,
    )[0]
    second_grad = torch.autograd.grad(first_grad.sum(), z)[0]

    assert first_grad.requires_grad
    torch.testing.assert_close(first_grad, torch.zeros_like(first_grad))
    torch.testing.assert_close(second_grad, torch.zeros_like(second_grad))


def test_aot_autograd_uses_required_optimized_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import olmo_core.nn.memory.counterfactual as counterfactual_module

    calls: list[bool] = []

    def fake_optimized(z, *args, **kwargs):
        del args, kwargs
        calls.append(torch.is_grad_enabled())
        return torch.zeros_like(z)

    monkeypatch.setattr(
        counterfactual_module,
        "try_counterfactual_grad_z",
        fake_optimized,
    )
    tables = (torch.randn(16**2, 2), torch.randn(16**3, 2))
    upstreams = (torch.randn(1, 3, 2), torch.randn(1, 3, 2))

    def loss(z):
        outputs = counterfactual_lookup(
            z,
            tables,
            (2, 3),
            bits_per_route=4,
            require_triton=True,
        )
        return sum((output * upstream).sum() for output, upstream in zip(outputs, upstreams))

    compiled_loss = torch.compile(loss, backend="aot_eager")
    z = torch.randn(1, 3, 4, requires_grad=True)
    compiled_loss(z).backward()

    assert calls == [False]
    assert z.grad is not None


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
