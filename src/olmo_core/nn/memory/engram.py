"""Native Engram conditional memory (arXiv:2601.07372)."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from array import array
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from numbers import Integral
from typing import Callable, Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...exceptions import OLMoConfigurationError
from ..config import ModuleConfig
from ..convolution import CausalConv1d
from ..layer_norm import RMSNorm

__all__ = ["EngramConfig", "Engram"]

DOLMA2_COMPRESSION_MAP_NAME = "dolma2_nfkc_v1"
DOLMA2_COMPRESSION_MAP_RESOURCE = "dolma2_compression_map.u32"
DOLMA2_COMPRESSION_METADATA_RESOURCE = "dolma2_compression_map.u32.metadata.json"
DOLMA2_COMPRESSION_MAP_SHA256 = "04d0b8396da748fb67f6b4b8d909c2f1600c2f27c237f1e4a55cab134a7c7f29"


def _positive(name: str, value: object) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
        raise OLMoConfigurationError(f"'{name}' must be a positive integer (got {value!r})")
    return int(value)


def _prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def _compression_map(values: Sequence[int], vocab_size: int) -> tuple[int, ...]:
    values = tuple(values)
    if len(values) != vocab_size:
        raise OLMoConfigurationError(
            f"'compression_map' must have vocab_size={vocab_size} entries (got {len(values)})"
        )
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in values):
        raise OLMoConfigurationError("'compression_map' must contain only integers")
    result = tuple(int(value) for value in values)
    if any(value < 0 or value >= vocab_size for value in result):
        raise OLMoConfigurationError("'compression_map' values must lie in [0, vocab_size)")
    unique = set(result)
    if unique != set(range(max(unique) + 1)):
        raise OLMoConfigurationError(
            "'compression_map' must be surjective onto contiguous IDs starting at zero"
        )
    return result


@lru_cache(maxsize=1)
def _load_dolma2_compression_map(vocab_size: int) -> tuple[int, ...]:
    package = resources.files(__package__)
    payload = package.joinpath(DOLMA2_COMPRESSION_MAP_RESOURCE).read_bytes()
    metadata = json.loads(
        package.joinpath(DOLMA2_COMPRESSION_METADATA_RESOURCE).read_text(encoding="utf-8")
    )
    if metadata["entries"] != vocab_size:
        raise OLMoConfigurationError(
            "Dolma2 compression-map size does not match the configured vocabulary: "
            f"{metadata['entries']} != {vocab_size}"
        )
    expected_size = vocab_size * 4
    if len(payload) != expected_size:
        raise OLMoConfigurationError(
            f"Dolma2 compression-map payload has {len(payload)} bytes, " f"expected {expected_size}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if metadata["sha256"] != DOLMA2_COMPRESSION_MAP_SHA256:
        raise OLMoConfigurationError(
            "Dolma2 compression-map metadata does not match the sealed checksum"
        )
    if digest != DOLMA2_COMPRESSION_MAP_SHA256:
        raise OLMoConfigurationError(
            "Dolma2 compression-map checksum mismatch: "
            f"{digest} != {DOLMA2_COMPRESSION_MAP_SHA256}"
        )
    values = struct.unpack(f"<{vocab_size}I", payload)
    return _compression_map(values, vocab_size)


@dataclass
class EngramConfig(ModuleConfig):
    """Configure Engram conditional memory from arXiv:2601.07372.

    ``compression_map`` accepts a map precomputed from normalized tokenizer
    text (NFKC, accent removal, lowercase, and whitespace normalization). An
    identity fallback keeps local config construction tokenizer-free.
    """

    orders: tuple[int, ...] = (2, 3)
    num_hash_heads: int = 8
    table_sizes: tuple[int, ...] = (131071, 131071)
    embedding_dim: int = 16
    vocab_size: int = 100352
    tokenizer_compression: bool = True
    compression_map: tuple[int, ...] | None = None
    compression_map_name: str | None = None
    conv_kernel_size: int = 4
    conv_dilation: int = 1

    def __post_init__(self) -> None:
        self.orders = tuple(self.orders)
        self.table_sizes = tuple(self.table_sizes)
        if self.compression_map is not None:
            self.compression_map = tuple(self.compression_map)
        self.validate()

    def validate(self) -> None:
        if not self.orders or any(
            not isinstance(order, Integral) or isinstance(order, bool) for order in self.orders
        ):
            raise OLMoConfigurationError("'orders' must be a non-empty sequence of integers")
        if len(set(self.orders)) != len(self.orders):
            raise OLMoConfigurationError("'orders' must be unique")
        if any(order not in (2, 3) for order in self.orders):
            raise OLMoConfigurationError("Engram currently supports only orders 2 and 3")
        if len(self.table_sizes) != len(self.orders):
            raise OLMoConfigurationError("'table_sizes' must contain one size per order")
        if any(not _prime(_positive("table size", size)) for size in self.table_sizes):
            raise OLMoConfigurationError("Engram table sizes must be prime")
        _positive("num_hash_heads", self.num_hash_heads)
        _positive("embedding_dim", self.embedding_dim)
        _positive("vocab_size", self.vocab_size)
        _positive("conv_kernel_size", self.conv_kernel_size)
        _positive("conv_dilation", self.conv_dilation)
        if not isinstance(self.tokenizer_compression, bool):
            raise OLMoConfigurationError("'tokenizer_compression' must be a bool")
        if self.compression_map is not None and self.compression_map_name is not None:
            raise OLMoConfigurationError(
                "specify only one of 'compression_map' and 'compression_map_name'"
            )
        if self.compression_map_name not in (None, DOLMA2_COMPRESSION_MAP_NAME):
            raise OLMoConfigurationError(f"unknown compression map {self.compression_map_name!r}")
        if not self.tokenizer_compression and (
            self.compression_map is not None or self.compression_map_name is not None
        ):
            raise OLMoConfigurationError(
                "compression maps cannot be set when tokenizer compression is disabled"
            )
        if self.compression_map is not None:
            self.compression_map = _compression_map(self.compression_map, self.vocab_size)

    @property
    def retrieval_dim(self) -> int:
        return len(self.orders) * self.num_hash_heads * self.embedding_dim

    def _dense_params(self, d_model: int) -> int:
        d_model = _positive("d_model", d_model)
        return 3 * d_model + 2 * self.retrieval_dim * d_model + (d_model * self.conv_kernel_size)

    def num_params(self, d_model: int) -> int:
        """Return the exact trainable parameter count."""
        self.validate()
        tables = sum(self.table_sizes) * self.num_hash_heads * self.embedding_dim
        return self._dense_params(d_model) + tables

    def num_active_params(self, d_model: int) -> int:
        """Return dense parameters plus the rows touched by one token."""
        self.validate()
        rows = len(self.orders) * self.num_hash_heads * self.embedding_dim
        return self._dense_params(d_model) + rows

    def num_flops_per_token(self, d_model: int) -> int:
        """Estimate training FLOPs using OLMo-core's active-parameter convention."""
        return 6 * self.num_active_params(d_model)

    def build(
        self,
        d_model: int,
        *,
        init_device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        compression_map: Sequence[int] | None = None,
        compression_map_hook: Callable[[int], Sequence[int]] | None = None,
    ) -> "Engram":
        """Build Engram without loading a tokenizer or corpus."""
        self.validate()
        d_model = _positive("d_model", d_model)
        if compression_map is not None and compression_map_hook is not None:
            raise OLMoConfigurationError("provide only one compression-map override")
        if not self.tokenizer_compression:
            if compression_map is not None or compression_map_hook is not None:
                raise OLMoConfigurationError(
                    "compression overrides require tokenizer_compression=True"
                )
            resolved: Sequence[int] = range(self.vocab_size)
        elif compression_map is not None:
            resolved = compression_map
        elif compression_map_hook is not None:
            resolved = compression_map_hook(self.vocab_size)
        elif self.compression_map is not None:
            resolved = self.compression_map
        elif self.compression_map_name == DOLMA2_COMPRESSION_MAP_NAME:
            resolved = _load_dolma2_compression_map(self.vocab_size)
        else:
            resolved = range(self.vocab_size)
        named_map = (
            compression_map is None
            and compression_map_hook is None
            and self.compression_map is None
            and self.compression_map_name is not None
        )
        return Engram(
            d_model=d_model,
            orders=self.orders,
            num_hash_heads=self.num_hash_heads,
            table_sizes=self.table_sizes,
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size,
            compression_map=_compression_map(resolved, self.vocab_size),
            compression_map_persistent=not named_map,
            conv_kernel_size=self.conv_kernel_size,
            conv_dilation=self.conv_dilation,
            init_device=init_device,
            dtype=torch.float32 if dtype is None else dtype,
        )


class Engram(nn.Module):
    """Tokenizer n-gram conditional memory from arXiv:2601.07372.

    The module returns the memory contribution only; the enclosing transformer
    block applies the pre-attention residual.

    .. note::
        Future work includes TP/EP table sharding, inference host prefetch, and
        CP/PP token routing with n-gram and convolution boundary halos.
    """

    def __init__(
        self,
        *,
        d_model: int,
        orders: tuple[int, ...],
        num_hash_heads: int,
        table_sizes: tuple[int, ...],
        embedding_dim: int,
        vocab_size: int,
        compression_map: Sequence[int],
        compression_map_persistent: bool = True,
        conv_kernel_size: int = 4,
        conv_dilation: int = 1,
        init_device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.d_model = d_model
        self.orders = tuple(orders)
        self.num_hash_heads = num_hash_heads
        self.table_sizes = tuple(table_sizes)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.conv_kernel_size = conv_kernel_size
        self.conv_dilation = conv_dilation
        self.retrieval_dim = len(self.orders) * num_hash_heads * embedding_dim
        self._compression_map_values = tuple(int(value) for value in compression_map)
        self.compression_map_persistent = compression_map_persistent
        self.register_buffer(
            "compression_map",
            torch.tensor(
                self._compression_map_values,
                dtype=torch.long,
                device=init_device,
            ),
            persistent=compression_map_persistent,
        )

        max_order = max(self.orders)
        multipliers = torch.zeros(len(self.orders), num_hash_heads, max_order, dtype=torch.long)
        for order_idx, order in enumerate(self.orders):
            for head_idx in range(num_hash_heads):
                for offset in range(order):
                    multipliers[order_idx, head_idx, offset] = (
                        2 * (1 + 131 * order + 37 * order_idx + 17 * head_idx + 29 * offset) + 1
                    )
        self._hash_multiplier_values = tuple(
            tuple(tuple(int(value) for value in head) for head in order)
            for order in multipliers.tolist()
        )
        compression_digest = hashlib.sha256(
            array("q", self._compression_map_values).tobytes()
        ).digest()
        self.hash_signature = (
            self.orders,
            self.num_hash_heads,
            self.table_sizes,
            self.vocab_size,
            compression_digest,
            self._hash_multiplier_values,
        )
        self.register_buffer(
            "hash_multipliers",
            multipliers.to(device=init_device),
            persistent=False,
        )
        self.tables = nn.ModuleList(
            [
                nn.Embedding(
                    table_size * num_hash_heads,
                    embedding_dim,
                    device=init_device,
                    dtype=dtype,
                )
                for table_size in table_sizes
            ]
        )
        self.key_proj = nn.Linear(
            self.retrieval_dim, d_model, bias=False, device=init_device, dtype=dtype
        )
        self.value_proj = nn.Linear(
            self.retrieval_dim, d_model, bias=False, device=init_device, dtype=dtype
        )
        norm_kwargs = dict(
            size=d_model,
            elementwise_affine=True,
            bias=False,
            dtype=dtype,
            init_device=init_device,
        )
        self.query_norm = RMSNorm(**norm_kwargs)
        self.key_norm = RMSNorm(**norm_kwargs)
        self.output_norm = RMSNorm(**norm_kwargs)
        self.conv = CausalConv1d(
            hidden_size=d_model,
            kernel_size=conv_kernel_size,
            dilation=conv_dilation,
            bias=False,
            activation=None,
            dtype=dtype,
            init_device=str(init_device),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize lookups/readouts and close the convolution branch."""
        if not self.compression_map.is_meta:
            self.compression_map.copy_(
                torch.tensor(
                    self._compression_map_values,
                    dtype=self.compression_map.dtype,
                    device=self.compression_map.device,
                )
            )
        if not self.hash_multipliers.is_meta:
            self.hash_multipliers.copy_(
                torch.tensor(
                    self._hash_multiplier_values,
                    dtype=self.hash_multipliers.dtype,
                    device=self.hash_multipliers.device,
                )
            )
        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)
        self.key_proj.reset_parameters()
        self.value_proj.reset_parameters()
        self.query_norm.reset_parameters()
        self.key_norm.reset_parameters()
        self.output_norm.reset_parameters()
        nn.init.zeros_(self.conv.weight)

    def compute_hash_indices(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute all multiplicative-XOR suffix hashes once."""
        if input_ids.ndim != 2:
            raise ValueError(
                f"Engram token IDs must have shape [batch, sequence], got {input_ids.shape}"
            )
        if input_ids.dtype != torch.int64:
            raise TypeError("Engram token IDs must use int64")
        if input_ids.device != self.compression_map.device:
            raise ValueError("token IDs and compression map must be on the same device")
        if input_ids.numel():
            minimum, maximum = torch.aminmax(input_ids)
            if minimum < 0 or maximum >= self.vocab_size:
                raise ValueError(f"Engram token IDs must be in range [0, {self.vocab_size})")

        compressed = self.compression_map[input_ids]
        batch_size, seq_len = compressed.shape
        result = []
        for order_idx, (order, table_size) in enumerate(zip(self.orders, self.table_sizes)):
            mixed = torch.zeros(
                batch_size,
                seq_len,
                self.num_hash_heads,
                dtype=torch.long,
                device=input_ids.device,
            )
            for offset in range(order):
                shifted = torch.zeros_like(compressed)
                if offset == 0:
                    shifted.copy_(compressed)
                elif offset < seq_len:
                    shifted[:, offset:] = compressed[:, :-offset]
                product = shifted.unsqueeze(-1) * self.hash_multipliers[order_idx, :, offset]
                mixed = torch.bitwise_xor(mixed, product)
            indices = torch.remainder(mixed, table_size)
            indices[:, : min(order - 1, seq_len)] = 0
            result.append(indices)
        return tuple(result)

    def _validate_indices(
        self,
        hash_indices: Iterable[torch.Tensor],
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, ...]:
        values = tuple(hash_indices)
        if len(values) != len(self.orders):
            raise ValueError("hash_indices must contain one tensor per n-gram order")
        shape = (batch_size, seq_len, self.num_hash_heads)
        for value, table_size in zip(values, self.table_sizes):
            if tuple(value.shape) != shape:
                raise ValueError(f"hash index shape must be {shape}, got {tuple(value.shape)}")
            if value.dtype != torch.int64:
                raise TypeError("hash indices must use int64")
            if value.device != self.compression_map.device:
                raise ValueError("hash indices and memory tables must share a device")
            if value.numel():
                minimum, maximum = torch.aminmax(value)
                if minimum < 0 or maximum >= table_size:
                    raise ValueError(f"hash index range must lie in [0, {table_size})")
        return values

    def retrieve_embeddings(
        self,
        hash_indices: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Retrieve and concatenate every order/head embedding."""
        values = tuple(hash_indices)
        if not values:
            raise ValueError("hash_indices cannot be empty")
        batch_size, seq_len = values[0].shape[:2]
        values = self._validate_indices(values, batch_size, seq_len)
        memories = []
        for order, size, table, indices in zip(self.orders, self.table_sizes, self.tables, values):
            offsets = (
                torch.arange(self.num_hash_heads, device=indices.device, dtype=torch.long) * size
            )
            memory = table(indices + offsets).flatten(start_dim=-2)
            if order > 1:
                memory = memory.clone()
                memory[:, : min(order - 1, seq_len)] = 0
            memories.append(memory)
        return torch.cat(memories, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        hash_indices: Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """Compute the Engram memory contribution."""
        if hidden_states.ndim != 3:
            raise ValueError("Engram hidden_states shape must be [batch, sequence, d_model]")
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(
                f"Engram hidden_states d_model must be {self.d_model}, "
                f"got {hidden_states.shape[-1]}"
            )
        if not torch.is_floating_point(hidden_states):
            raise TypeError("Engram hidden_states must use a floating point dtype")
        values = self._validate_indices(
            hash_indices, hidden_states.shape[0], hidden_states.shape[1]
        )
        memory = self.retrieve_embeddings(values)
        key = self.key_proj(memory)
        value = self.value_proj(memory)
        alpha = torch.sigmoid(
            (self.query_norm(hidden_states) * self.key_norm(key)).sum(dim=-1, keepdim=True)
            / math.sqrt(self.d_model)
        )
        gated = alpha * value
        return gated + F.silu(self.conv(self.output_norm(gated)))

    def num_active_params(self) -> int:
        """Return dense parameters plus rows touched by one token."""
        dense = (
            3 * self.d_model
            + 2 * self.retrieval_dim * self.d_model
            + self.d_model * self.conv_kernel_size
        )
        return dense + len(self.orders) * self.num_hash_heads * self.embedding_dim

    def num_flops_per_token(self, seq_len: int | None = None) -> int:
        """Estimate training FLOPs using active parameters."""
        del seq_len
        return 6 * self.num_active_params()
