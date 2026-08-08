"""Hard latent n-gram lookup with a local counterfactual gradient."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from numbers import Integral, Real

import torch

from ...ops.lngram import _try_counterfactual_grad_z as try_counterfactual_grad_z

__all__ = ["CounterfactualLookupFunction", "counterfactual_lookup"]


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _pack_route_codes(z: torch.Tensor, bits_per_route: int) -> torch.Tensor:
    batch_size, seq_len, channels = z.shape
    num_routes = channels // bits_per_route
    bits = (z > 0).reshape(batch_size, seq_len, num_routes, bits_per_route)
    dtype = torch.uint8 if bits_per_route <= 8 else torch.long
    weights = 1 << torch.arange(bits_per_route, device=z.device, dtype=dtype)
    return (bits.to(dtype) * weights).sum(dim=-1, dtype=dtype)


def _addresses(codes: torch.Tensor, order: int, alphabet_size: int) -> torch.Tensor:
    """Return little-endian route-local addresses for all complete windows."""
    windows = codes.unfold(1, order, 1)
    powers = alphabet_size ** torch.arange(order, device=codes.device, dtype=torch.long)
    route_offsets = (
        torch.arange(codes.shape[-1], device=codes.device, dtype=torch.long)
        * alphabet_size**order
    )
    return route_offsets.view(1, 1, -1) + (windows * powers.view(1, 1, 1, order)).sum(dim=-1)


class CounterfactualLookupFunction(torch.autograd.Function):
    """Exact hard lookup with the one-bit surrogate from arXiv:2605.24869.

    Forward values and table gradients use only hard route addresses. For
    every route logit, backward compares the rows produced by forcing that bit
    to zero or one while all other bits stay fixed. Contributions accumulate
    from every n-gram window containing the bit, without exponential state
    enumeration.
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, *args):
        orders = tuple(args[-5])
        bits_per_route = int(args[-4])
        temperature = float(args[-3])
        scale = float(args[-2])
        require_triton = bool(args[-1])
        tables = tuple(args[:-5])

        codes = _pack_route_codes(z, bits_per_route)
        batch_size, seq_len, num_routes = codes.shape
        alphabet_size = 2**bits_per_route
        outputs: list[torch.Tensor] = []
        for table, order in zip(tables, orders):
            memory_dim = table.shape[1]
            output = table.new_zeros((batch_size, seq_len, num_routes * memory_dim))
            if seq_len >= order:
                addresses = _addresses(codes, order, alphabet_size)
                output[:, order - 1 :] = table[addresses].reshape(
                    batch_size,
                    seq_len - order + 1,
                    num_routes * memory_dim,
                )
            outputs.append(output)

        ctx.orders = orders
        ctx.bits_per_route = bits_per_route
        ctx.temperature = temperature
        ctx.scale = scale
        ctx.require_triton = require_triton
        ctx.save_for_backward(z, codes, *tables)
        return tuple(outputs)

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor | None):
        z, codes, *tables = ctx.saved_tensors
        orders: tuple[int, ...] = ctx.orders
        bits_per_route: int = ctx.bits_per_route
        temperature: float = ctx.temperature
        scale: float = ctx.scale
        require_triton: bool = ctx.require_triton

        batch_size, seq_len, num_routes = codes.shape
        alphabet_size = 2**bits_per_route
        grad_z: torch.Tensor | None = None
        if ctx.needs_input_grad[0]:
            if scale == 0 and not torch.is_grad_enabled():
                grad_z = torch.zeros_like(z)
            elif not torch.is_grad_enabled():
                grad_z = try_counterfactual_grad_z(
                    z,
                    codes,
                    tables,
                    orders,
                    grad_outputs,
                    bits_per_route=bits_per_route,
                    temperature=temperature,
                    scale=scale,
                )
                if grad_z is None and require_triton:
                    raise RuntimeError(
                        "Lngram requires Triton acceleration, but this backward "
                        "does not satisfy the fused kernel contract"
                    )
        bit_scores = (
            z.new_zeros((batch_size, seq_len, num_routes, bits_per_route))
            if ctx.needs_input_grad[0] and grad_z is None
            else None
        )
        table_grads: list[torch.Tensor | None] = []

        for table_idx, (table, order, grad_output) in enumerate(zip(tables, orders, grad_outputs)):
            needs_table_grad = ctx.needs_input_grad[table_idx + 1]
            table_grad: torch.Tensor | None = None
            window_count = seq_len - order + 1
            if grad_output is not None and window_count > 0:
                memory_dim = table.shape[1]
                addresses = _addresses(codes, order, alphabet_size)
                valid_grad = grad_output[:, order - 1 :].reshape(
                    batch_size, window_count, num_routes, memory_dim
                )
                if needs_table_grad:
                    table_grad = torch.ops.aten.embedding_dense_backward(
                        valid_grad.reshape(-1, memory_dim).to(table.dtype),
                        addresses.reshape(-1),
                        table.shape[0],
                        -1,
                        False,
                    )

                if bit_scores is not None:
                    windows = codes.unfold(1, order, 1)
                    for offset in range(order):
                        place_value = alphabet_size**offset
                        route_codes = windows[..., offset]
                        for bit_idx in range(bits_per_route):
                            delta = (1 << bit_idx) * place_value
                            current_bit = ((route_codes >> bit_idx) & 1).to(addresses.dtype)
                            address_zero = addresses - current_bit * delta
                            address_one = address_zero + delta
                            difference = table[address_one] - table[address_zero]
                            score = (valid_grad * difference).sum(dim=-1)
                            bit_scores[:, offset : offset + window_count, :, bit_idx].add_(
                                score.to(z.dtype)
                            )
            if needs_table_grad and table_grad is None:
                table_grad = torch.zeros_like(table)
            table_grads.append(table_grad)

        if bit_scores is not None:
            route_logits = z.reshape(batch_size, seq_len, num_routes, bits_per_route)
            probability = torch.sigmoid(temperature * route_logits)
            slope = scale * temperature * probability * (1 - probability)
            grad_z = (slope * bit_scores).reshape_as(z)

        return grad_z, *table_grads, None, None, None, None, None


def _validate_inputs(
    z: torch.Tensor,
    tables: Iterable[torch.Tensor],
    orders: Sequence[int],
    *,
    bits_per_route: int,
    temperature: float,
    scale: float,
) -> tuple[tuple[torch.Tensor, ...], tuple[int, ...]]:
    if not isinstance(z, torch.Tensor):
        raise TypeError("'z' must be a tensor")
    if z.ndim != 3:
        raise ValueError(f"'z' must have 3 dimensions [batch, sequence, channels], got {z.ndim}")
    if not torch.is_floating_point(z):
        raise TypeError("'z' must use a floating point dtype")
    if not _is_integer(bits_per_route) or bits_per_route <= 0:
        raise ValueError(f"'bits_per_route' must be a positive integer (got {bits_per_route!r})")
    if z.shape[-1] == 0:
        raise ValueError("'z' must contain at least one route")
    if z.shape[-1] % bits_per_route != 0:
        raise ValueError(f"'z' channels must be divisible by bits_per_route={bits_per_route}")
    if (
        not isinstance(temperature, Real)
        or isinstance(temperature, bool)
        or not math.isfinite(float(temperature))
        or temperature <= 0
    ):
        raise ValueError(f"'temperature' must be positive and finite (got {temperature!r})")
    if (
        not isinstance(scale, Real)
        or isinstance(scale, bool)
        or not math.isfinite(float(scale))
        or scale < 0
    ):
        raise ValueError(f"'scale' must be finite and nonnegative (got {scale!r})")

    table_tuple = tuple(tables)
    order_tuple = tuple(orders)
    if not table_tuple or len(table_tuple) != len(order_tuple):
        raise ValueError("'tables' and 'orders' must have the same length and be nonzero")
    if any(not _is_integer(order) or order <= 0 for order in order_tuple):
        raise ValueError("'orders' must contain only positive integers")

    num_routes = z.shape[-1] // bits_per_route
    alphabet_size = 2**bits_per_route
    memory_dim: int | None = None
    for table, order in zip(table_tuple, order_tuple):
        if not isinstance(table, torch.Tensor):
            raise TypeError("each memory table must be a tensor")
        if table.ndim != 2:
            raise ValueError(f"each memory table must have 2 dimensions, got {table.ndim}")
        if not torch.is_floating_point(table):
            raise TypeError("memory tables must use a floating point dtype")
        if table.device != z.device:
            raise ValueError("'z' and all memory tables must be on the same device")
        expected_rows = num_routes * alphabet_size**order
        if table.shape[0] != expected_rows:
            raise ValueError(
                f"order-{order} table must have {expected_rows} rows, got {table.shape[0]}"
            )
        if table.shape[1] <= 0:
            raise ValueError("memory table dimension must be positive")
        if memory_dim is None:
            memory_dim = table.shape[1]
        elif table.shape[1] != memory_dim:
            raise ValueError("all memory tables must have the same memory dimension")
    return table_tuple, order_tuple


def counterfactual_lookup(
    z: torch.Tensor,
    tables: Iterable[torch.Tensor],
    orders: Sequence[int],
    *,
    bits_per_route: int = 4,
    temperature: float = 1.0,
    scale: float = 1.0,
    require_triton: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Apply the hard Lngram lookup and local surrogate from arXiv:2605.24869."""
    if not isinstance(require_triton, bool):
        raise TypeError("'require_triton' must be a bool")
    table_tuple, order_tuple = _validate_inputs(
        z,
        tables,
        orders,
        bits_per_route=bits_per_route,
        temperature=temperature,
        scale=scale,
    )
    result = CounterfactualLookupFunction.apply(
        z,
        *table_tuple,
        order_tuple,
        bits_per_route,
        float(temperature),
        float(scale),
        require_triton,
    )
    if isinstance(result, torch.Tensor):
        return (result,)
    return tuple(result)
