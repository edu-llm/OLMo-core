"""Latent n-gram conditional memory.

This module implements the native OLMo-core form of Lngram described in
`arXiv:2605.24869 <https://arxiv.org/abs/2605.24869>`_.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...exceptions import OLMoConfigurationError
from ..config import ModuleConfig
from ..convolution import CausalConv1d
from ..layer_norm import RMSNorm
from .counterfactual import counterfactual_lookup

__all__ = ["LngramConfig", "Lngram"]


def _is_integer(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _validate_d_model(d_model: int) -> None:
    if not _is_integer(d_model) or d_model <= 0:
        raise OLMoConfigurationError(f"'d_model' must be a positive integer (got {d_model!r})")
    if d_model % 4 != 0:
        raise OLMoConfigurationError(
            f"'d_model' must be divisible by 4 for 4-bit Lngram routes (got {d_model})"
        )


@dataclass
class LngramConfig(ModuleConfig):
    """Configuration for latent n-gram memory from arXiv:2605.24869.

    Lngram discretizes normalized hidden states into learned 4-bit route
    symbols, retrieves exact suffix n-grams from per-order tables, and fuses
    those memories with shared key and value readouts.
    """

    orders: tuple[int, ...] = (2, 3)
    bits_per_route: int = 4
    memory_dim: int = 16
    surrogate_temperature: float = 1.0
    surrogate_scale: float = 1.0
    conv_kernel_size: int = 4
    conv_dilation: int = 1
    norm_eps: float = 1e-6
    require_triton: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.orders, (list, tuple)):
            self.orders = tuple(self.orders)
        self.validate()

    def validate(self) -> None:
        if not isinstance(self.orders, tuple) or not self.orders:
            raise OLMoConfigurationError("'orders' must be a non-empty tuple")
        if any(not _is_integer(order) for order in self.orders):
            raise OLMoConfigurationError("'orders' must contain only integers")
        if len(set(self.orders)) != len(self.orders):
            raise OLMoConfigurationError(f"'orders' must be unique (got {self.orders!r})")
        unsupported = tuple(order for order in self.orders if order not in (2, 3))
        if unsupported:
            raise OLMoConfigurationError(
                f"Lngram only supports n-gram orders 2 and 3 (got {unsupported!r})"
            )
        if not _is_integer(self.bits_per_route) or self.bits_per_route != 4:
            raise OLMoConfigurationError(
                "Lngram currently requires exactly 4 bits per route "
                f"(got {self.bits_per_route!r})"
            )
        if not _is_integer(self.memory_dim) or self.memory_dim <= 0:
            raise OLMoConfigurationError(
                f"'memory_dim' must be a positive integer (got {self.memory_dim!r})"
            )
        if (
            not isinstance(self.surrogate_temperature, Real)
            or isinstance(self.surrogate_temperature, bool)
            or not math.isfinite(float(self.surrogate_temperature))
            or self.surrogate_temperature <= 0
        ):
            raise OLMoConfigurationError(
                "'surrogate_temperature' must be positive and finite "
                f"(got {self.surrogate_temperature!r})"
            )
        if (
            not isinstance(self.surrogate_scale, Real)
            or isinstance(self.surrogate_scale, bool)
            or not math.isfinite(float(self.surrogate_scale))
            or self.surrogate_scale < 0
        ):
            raise OLMoConfigurationError(
                "'surrogate_scale' must be finite and nonnegative "
                f"(got {self.surrogate_scale!r})"
            )
        if not _is_integer(self.conv_kernel_size) or self.conv_kernel_size <= 0:
            raise OLMoConfigurationError(
                f"'conv_kernel_size' must be a positive integer (got {self.conv_kernel_size!r})"
            )
        if not _is_integer(self.conv_dilation) or self.conv_dilation <= 0:
            raise OLMoConfigurationError(
                f"'conv_dilation' must be a positive integer (got {self.conv_dilation!r})"
            )
        if (
            not isinstance(self.norm_eps, Real)
            or isinstance(self.norm_eps, bool)
            or not math.isfinite(float(self.norm_eps))
            or self.norm_eps <= 0
        ):
            raise OLMoConfigurationError(
                f"'norm_eps' must be positive and finite (got {self.norm_eps!r})"
            )
        if not isinstance(self.require_triton, bool):
            raise OLMoConfigurationError("'require_triton' must be a bool")

    def _dense_params(self, d_model: int) -> int:
        _validate_d_model(d_model)
        routes = d_model // self.bits_per_route
        readout_dim = routes * self.memory_dim
        return (
            4 * d_model  # routing, query, key, and convolution RMSNorms
            + d_model * d_model  # discretization projection
            + 2 * (d_model * readout_dim + d_model)  # shared W_K and W_V, with bias
            + d_model * self.conv_kernel_size  # bias-free depthwise convolution
        )

    def num_params(self, d_model: int) -> int:
        """Return the exact number of parameters in the built module."""
        self.validate()
        _validate_d_model(d_model)
        routes = d_model // self.bits_per_route
        table_params = sum(
            routes * (2**self.bits_per_route) ** order * self.memory_dim for order in self.orders
        )
        return self._dense_params(d_model) + table_params

    def num_active_params(self, d_model: int) -> int:
        """Return dense parameters plus one active table row per route and order."""
        self.validate()
        _validate_d_model(d_model)
        routes = d_model // self.bits_per_route
        active_table_params = len(self.orders) * routes * self.memory_dim
        return self._dense_params(d_model) + active_table_params

    def num_flops_per_token(self, d_model: int) -> int:
        """Estimate training FLOPs per token using OLMo's six-times-active convention."""
        return 6 * self.num_active_params(d_model)

    def build(
        self,
        d_model: int,
        *,
        init_device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "Lngram":
        """Build an :class:`Lngram` on ``init_device``."""
        self.validate()
        _validate_d_model(d_model)
        return Lngram(
            d_model=d_model,
            orders=self.orders,
            bits_per_route=self.bits_per_route,
            memory_dim=self.memory_dim,
            surrogate_temperature=float(self.surrogate_temperature),
            surrogate_scale=float(self.surrogate_scale),
            conv_kernel_size=self.conv_kernel_size,
            conv_dilation=self.conv_dilation,
            norm_eps=float(self.norm_eps),
            require_triton=self.require_triton,
            init_device=init_device,
            dtype=torch.float32 if dtype is None else dtype,
        )


class Lngram(nn.Module):
    """Latent n-gram conditional memory from arXiv:2605.24869.

    The module learns a bias-free projection into 4-bit route symbols, looks
    up order-specific memories with a counterfactual routing surrogate, and
    applies shared key/value readouts followed by a zero-initialized causal
    convolution branch. The returned tensor is the memory contribution only;
    the input hidden states are not added as a residual here.

    .. note::
        Table TP/EP partitioning and host-memory prefetch are intentionally
        deferred until the distributed table-storage interface is available.
        CP/PP integrations will likewise need explicit sequence-boundary halos.
    """

    def __init__(
        self,
        *,
        d_model: int,
        orders: tuple[int, ...] = (2, 3),
        bits_per_route: int = 4,
        memory_dim: int = 16,
        surrogate_temperature: float = 1.0,
        surrogate_scale: float = 1.0,
        conv_kernel_size: int = 4,
        conv_dilation: int = 1,
        norm_eps: float = 1e-6,
        require_triton: bool = False,
        init_device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        config = LngramConfig(
            orders=orders,
            bits_per_route=bits_per_route,
            memory_dim=memory_dim,
            surrogate_temperature=surrogate_temperature,
            surrogate_scale=surrogate_scale,
            conv_kernel_size=conv_kernel_size,
            conv_dilation=conv_dilation,
            norm_eps=norm_eps,
            require_triton=require_triton,
        )
        _validate_d_model(d_model)

        self.d_model = int(d_model)
        self.orders = config.orders
        self.bits_per_route = config.bits_per_route
        self.memory_dim = config.memory_dim
        self.surrogate_temperature = float(config.surrogate_temperature)
        self.surrogate_scale = float(config.surrogate_scale)
        self.conv_kernel_size = config.conv_kernel_size
        self.conv_dilation = config.conv_dilation
        self.norm_eps = float(config.norm_eps)
        self.require_triton = config.require_triton
        self.num_routes = self.d_model // self.bits_per_route
        self.alphabet_size = 2**self.bits_per_route
        self.readout_dim = self.num_routes * self.memory_dim

        self.input_norm = RMSNorm(
            size=self.d_model,
            eps=self.norm_eps,
            elementwise_affine=True,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.w_q = nn.Linear(
            self.d_model,
            self.d_model,
            bias=False,
            dtype=dtype,
            device=init_device,
        )

        # Each route owns a contiguous K**n-row region in an order's table.
        self.tables = nn.ParameterList(
            [
                nn.Parameter(
                    torch.empty(
                        self.num_routes * self.alphabet_size**order,
                        self.memory_dim,
                        dtype=dtype,
                        device=init_device,
                    )
                )
                for order in self.orders
            ]
        )

        # These projections are intentionally shared across all n-gram orders.
        self.w_k = nn.Linear(
            self.readout_dim,
            self.d_model,
            bias=True,
            dtype=dtype,
            device=init_device,
        )
        self.w_v = nn.Linear(
            self.readout_dim,
            self.d_model,
            bias=True,
            dtype=dtype,
            device=init_device,
        )
        self.query_norm = RMSNorm(
            size=self.d_model,
            eps=self.norm_eps,
            elementwise_affine=True,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.key_norm = RMSNorm(
            size=self.d_model,
            eps=self.norm_eps,
            elementwise_affine=True,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.conv_norm = RMSNorm(
            size=self.d_model,
            eps=self.norm_eps,
            elementwise_affine=True,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.conv = CausalConv1d(
            hidden_size=self.d_model,
            kernel_size=self.conv_kernel_size,
            dilation=self.conv_dilation,
            bias=False,
            activation=None,
            dtype=dtype,
            init_device=init_device,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset parameters with a no-op memory residual."""
        for table in self.tables:
            nn.init.zeros_(table)
        self.w_q.reset_parameters()
        self.w_k.reset_parameters()
        self.w_v.reset_parameters()
        if self.w_k.bias is not None:
            nn.init.zeros_(self.w_k.bias)
        if self.w_v.bias is not None:
            nn.init.zeros_(self.w_v.bias)
        self.input_norm.reset_parameters()
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()
        self.conv_norm.reset_parameters()
        nn.init.zeros_(self.conv.weight)

    def _validate_retrievals(
        self,
        retrievals: Iterable[torch.Tensor] | torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        values: tuple[torch.Tensor, ...]
        if isinstance(retrievals, torch.Tensor):
            values = (retrievals,)
        else:
            values = tuple(retrievals)
        if len(values) != len(self.orders):
            raise RuntimeError(
                "counterfactual_lookup returned "
                f"{len(values)} tensors for {len(self.orders)} n-gram orders"
            )

        expected_shape = (
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.readout_dim,
        )
        for order, value in zip(self.orders, values):
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"counterfactual_lookup output for order {order} is not a tensor")
            if tuple(value.shape) != expected_shape:
                raise RuntimeError(
                    f"counterfactual_lookup output for order {order} has shape "
                    f"{tuple(value.shape)}, expected {expected_shape}"
                )
        return values

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute the Lngram memory contribution for ``[batch, sequence, d_model]`` input."""
        if hidden_states.ndim != 3:
            raise ValueError(
                "Lngram input must have rank 3 [batch, sequence, d_model], "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                "Lngram input last dimension must equal "
                f"d_model={self.d_model}, got {hidden_states.shape[-1]}"
            )
        if not torch.is_floating_point(hidden_states):
            raise ValueError("Lngram input must use a floating point dtype")

        normalized_h = self.input_norm(hidden_states)
        z = self.w_q(normalized_h)
        retrievals = counterfactual_lookup(
            z,
            self.tables,
            self.orders,
            bits_per_route=self.bits_per_route,
            temperature=self.surrogate_temperature,
            scale=self.surrogate_scale,
            require_triton=self.require_triton,
        )
        memories = self._validate_retrievals(retrievals, hidden_states)

        query = self.query_norm(hidden_states)
        gated: torch.Tensor | None = None
        for order, memory in zip(self.orders, memories):
            key = self.w_k(memory)
            value = self.w_v(memory)
            normalized_key = self.key_norm(key)
            with torch.autocast(
                device_type=hidden_states.device.type,
                enabled=False,
            ):
                gate_logits = (query.float() * normalized_key.float()).sum(
                    dim=-1, keepdim=True
                ) / math.sqrt(self.d_model)
                alpha = torch.sigmoid(gate_logits).to(value.dtype)
            contribution = alpha * value
            # The affine readout biases must not turn an incomplete n-gram
            # prefix into a real memory contribution.
            if order > 1:
                contribution = contribution.clone()
                contribution[:, : min(order - 1, hidden_states.shape[1])] = 0
            gated = contribution if gated is None else gated + contribution

        assert gated is not None  # Config validation guarantees at least one order.
        return gated + F.silu(self.conv(self.conv_norm(gated)))

    def num_params(self) -> int:
        """Return the exact number of module parameters."""
        return sum(parameter.numel() for parameter in self.parameters())

    def num_active_params(self) -> int:
        """Return active parameters for one token."""
        table_params = sum(table.numel() for table in self.tables)
        active_table_params = len(self.orders) * self.num_routes * self.memory_dim
        return self.num_params() - table_params + active_table_params

    def num_flops_per_token(self, seq_len: int | None = None) -> int:
        """Estimate training FLOPs per token using active parameters."""
        del seq_len
        return 6 * self.num_active_params()
