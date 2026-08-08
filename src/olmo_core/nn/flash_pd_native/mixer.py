"""Paper-faithful Flash PD-SSM block over a native vector state."""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

from .api import paper_surrogate_scan
from .contracts import NativePDBackend, NativePDMode, ScanMetadata

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


def _validate_options(
    *,
    d_model: int,
    n_heads: int,
    d_state: int,
    dictionary_size: int,
    chunk_size: int,
    ste_temperature: float,
) -> None:
    if n_heads < 1 or d_state < 1 or dictionary_size < 1:
        raise ValueError("n_heads, d_state, and dictionary_size must be positive")
    if d_state >= 1024:
        raise ValueError(f"native Flash PD state size must be below 1024, got {d_state}")
    if d_model != n_heads * d_state:
        raise ValueError(
            "paper-faithful vector state requires d_model == n_heads * d_state, got "
            f"{d_model} != {n_heads} * {d_state}"
        )
    if chunk_size < 1 or chunk_size > 128:
        raise ValueError(f"chunk_size must be in [1, 128], got {chunk_size}")
    if ste_temperature <= 0:
        raise ValueError(f"ste_temperature must be positive, got {ste_temperature}")


def _hardmax_ste(logits: torch.Tensor, *, dim: int, temperature: float) -> torch.Tensor:
    soft = torch.softmax(logits / temperature, dim=dim)
    index = logits.argmax(dim=dim, keepdim=True)
    hard = torch.zeros_like(logits).scatter_(dim, index, 1)
    return (hard - soft).detach() + soft


def _dense_ste_scan(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    bias_real: torch.Tensor,
    bias_imag: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense differentiable oracle for both slope-annealed hard selections."""
    dictionary = _hardmax_ste(dictionary_logits, dim=-2, temperature=temperature)
    selector = _hardmax_ste(selector_logits, dim=-1, temperature=temperature)
    transition = torch.einsum("bthk,hkiq->bthiq", selector, dictionary)
    transition = transition.permute(0, 2, 1, 3, 4)
    batch, heads, time, state = diagonal_real.shape
    state_real = torch.zeros(
        (batch, heads, state), dtype=diagonal_real.dtype, device=diagonal_real.device
    )
    state_imag = torch.zeros_like(state_real)
    states_real = []
    states_imag = []
    for token in range(time):
        product_real = (
            diagonal_real[:, :, token] * state_real - diagonal_imag[:, :, token] * state_imag
        )
        product_imag = (
            diagonal_real[:, :, token] * state_imag + diagonal_imag[:, :, token] * state_real
        )
        token_transition = transition[:, :, token]
        state_real = (
            torch.einsum("bhiq,bhq->bhi", token_transition, product_real) + bias_real[:, :, token]
        )
        state_imag = (
            torch.einsum("bhiq,bhq->bhi", token_transition, product_imag) + bias_imag[:, :, token]
        )
        states_real.append(state_real)
        states_imag.append(state_imag)
    return torch.stack(states_real, dim=2), torch.stack(states_imag, dim=2)


class NativeFlashPDMixer(SequenceMixer):
    """
    Flash PD-SSM block implementing Equation 1 without a Mamba payload axis.

    The state is ``x[B,H,T,N]``. ``B(u)u`` drives a complex recurrence,
    ``C(u)`` reads it out, and ``D`` is the learned skip from the paper's SSM
    equation. A causal depthwise convolution and gated output follow the
    Mamba-style block pattern described in Section 3.2.
    """

    state_contract = ("batch", "head", "time", "state")

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_state: int,
        dictionary_size: int,
        chunk_size: int = 128,
        ste_temperature: float = 1.0,
        mode: NativePDMode | str = NativePDMode.AUTO,
        backend: NativePDBackend | str = NativePDBackend.AUTO,
        conv_kernel_size: int = 4,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        _validate_options(
            d_model=d_model,
            n_heads=n_heads,
            d_state=d_state,
            dictionary_size=dictionary_size,
            chunk_size=chunk_size,
            ste_temperature=ste_temperature,
        )
        if conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be positive")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.dictionary_size = dictionary_size
        self.chunk_size = chunk_size
        self.ste_temperature = ste_temperature
        self.mode = NativePDMode(mode)
        self.backend = NativePDBackend(backend)
        self.conv_kernel_size = conv_kernel_size
        self.last_metadata: Optional[ScanMetadata] = None

        factory = {"dtype": dtype, "device": init_device}
        self.dictionary_logits = nn.Parameter(
            torch.empty(n_heads, dictionary_size, d_state, d_state, **factory)
        )
        self.in_proj = nn.Linear(d_model, 2 * d_model, bias=False, **factory)
        self.conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=conv_kernel_size,
            groups=d_model,
            bias=True,
            **factory,
        )
        self.B_proj = nn.Linear(d_model, 2 * d_model, bias=False, **factory)
        self.C_proj = nn.Linear(d_model, 2 * d_model, bias=False, **factory)
        self.selector_proj = nn.Linear(d_model, n_heads * dictionary_size, bias=False, **factory)
        self.dt_proj = nn.Linear(d_model, d_model, bias=False, **factory)
        self.phase_proj = nn.Linear(d_model, d_model, bias=False, **factory)
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory)
        self.A_log = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.D = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Mix an input sequence while preserving its shape and dtype."""
        if cu_doc_lens is not None and cu_doc_lens.numel() > 2:
            raise NotImplementedError("packed multi-document state resets are not implemented")
        if kwargs.get("initial_state") is not None:
            raise NotImplementedError("initial_state recurrent caching is not implemented")
        if kwargs.get("decode", False):
            raise NotImplementedError("decode recurrent caching is not implemented")
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, time, {self.d_model}), got {tuple(x.shape)}"
            )

        batch, time, _ = x.shape
        projected, gate = self.in_proj(x).chunk(2, dim=-1)
        convolved = self.conv(F.pad(projected.transpose(1, 2), (self.conv_kernel_size - 1, 0)))[
            ..., :time
        ]
        u = F.silu(convolved.transpose(1, 2))

        bias = self.B_proj(u).view(batch, time, self.n_heads, self.d_state, 2)
        readout = self.C_proj(u).view(batch, time, self.n_heads, self.d_state, 2)
        selector_logits = self.selector_proj(u).view(
            batch, time, self.n_heads, self.dictionary_size
        )
        dt = F.softplus(self.dt_proj(u).float() + self.dt_bias.view(1, 1, self.d_model))
        magnitude = torch.exp(-dt * torch.exp(self.A_log).view(1, 1, self.d_model))
        phase = self.phase_proj(u).float()
        diagonal_real = (magnitude * torch.cos(phase)).view(batch, time, self.n_heads, self.d_state)
        diagonal_imag = (magnitude * torch.sin(phase)).view(batch, time, self.n_heads, self.d_state)

        kernel_dtype = x.dtype if x.dtype in (torch.float32, torch.bfloat16) else torch.float32
        split_values = (
            diagonal_real.permute(0, 2, 1, 3).to(kernel_dtype),
            diagonal_imag.permute(0, 2, 1, 3).to(kernel_dtype),
            bias[..., 0].permute(0, 2, 1, 3).to(kernel_dtype),
            bias[..., 1].permute(0, 2, 1, 3).to(kernel_dtype),
        )
        if self.backend == NativePDBackend.REFERENCE:
            states_real, states_imag = _dense_ste_scan(
                self.dictionary_logits.float(),
                selector_logits.float(),
                *split_values,
                temperature=self.ste_temperature,
            )
            chunks = (time + self.chunk_size - 1) // self.chunk_size
            self.last_metadata = ScanMetadata(
                backend="reference_dense_ste_diagnostic",
                mode=NativePDMode.GENERAL_SCATTER,
                forward_launches=0,
                backward_launches=0,
                state_shape=(batch, self.n_heads, time, self.d_state),
                scratch_elements=2 * batch * self.n_heads * chunks * self.d_state * 5,
                shared_memory_bytes=28 * self.d_state,
            )
        else:
            states_real, states_imag, self.last_metadata = paper_surrogate_scan(
                self.dictionary_logits.float(),
                selector_logits.float(),
                *split_values,
                temperature=self.ste_temperature,
                chunk_size=self.chunk_size,
                mode=self.mode,
                backend=self.backend,
                return_metadata=True,
            )

        states_real = states_real.permute(0, 2, 1, 3)
        states_imag = states_imag.permute(0, 2, 1, 3)
        readout_real = (
            readout[..., 0].float() * states_real.float()
            - readout[..., 1].float() * states_imag.float()
        ).reshape(batch, time, self.d_model)
        y = readout_real + self.D.view(1, 1, -1) * u.float()
        return self.out_proj((y * F.silu(gate).float()).to(x.dtype))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        """Reject tensor parallelism until dictionary sharding is defined."""
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("tensor parallelism is not implemented")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """Allow only a size-one context mesh."""
        del ring, uly
        if cp_mesh.size() != 1:
            raise NotImplementedError("context parallelism is not implemented")

    @torch.no_grad()
    def init_weights(
        self,
        *,
        init_method: "InitMethod",
        d_model: int,
        block_idx: int,
        num_blocks: int,
        std: float = 0.02,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Initialize projections, dictionary, convolution, and stable diagonal."""
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError("fan_in initialization is not implemented")
        if init_method == InitMethod.normalized:
            std = d_model**-0.5
        nn.init.normal_(self.dictionary_logits, std=std, generator=generator)
        for projection in (
            self.in_proj,
            self.B_proj,
            self.C_proj,
            self.selector_proj,
            self.dt_proj,
            self.phase_proj,
        ):
            init_linear(projection, std=std, generator=generator)
        nn.init.normal_(self.conv.weight, std=std, generator=generator)
        nn.init.zeros_(self.conv.bias)
        self.A_log.copy_(
            torch.empty_like(self.A_log).uniform_(
                math.log(0.05), math.log(1.0), generator=generator
            )
        )
        dt = torch.empty_like(self.dt_bias).uniform_(0.001, 0.1, generator=generator)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.D.fill_(1)
        output_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            output_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std = std / (2 * (block_idx + 1)) ** 0.5
        init_linear(self.out_proj, std=output_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """Estimate projection, sparse recurrence, selector, and readout work."""
        projection = 2 * sum(
            layer.weight.numel()
            for layer in (
                self.in_proj,
                self.B_proj,
                self.C_proj,
                self.selector_proj,
                self.dt_proj,
                self.phase_proj,
                self.out_proj,
            )
        )
        recurrence = 16 * self.d_model
        dictionary = (self.n_heads * self.dictionary_size * self.d_state * self.d_state) // max(
            seq_len, 1
        )
        return int(projection + recurrence + dictionary)


@SequenceMixerConfig.register("flash_pd_native")
@dataclass
class NativeFlashPDMixerConfig(SequenceMixerConfig[NativeFlashPDMixer]):
    """Configuration for the native, non-Mamba Flash PD-SSM block."""

    n_heads: int = 8
    d_state: int = 64
    dictionary_size: int = 16
    chunk_size: int = 128
    ste_temperature: float = 1.0
    mode: NativePDMode = NativePDMode.AUTO
    backend: NativePDBackend = NativePDBackend.AUTO
    conv_kernel_size: int = 4
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        """Return the exact number of parameters created by :meth:`build`."""
        _validate_options(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            ste_temperature=self.ste_temperature,
        )
        dictionary = self.n_heads * self.dictionary_size * self.d_state**2
        linear_weights = d_model * (
            2 * d_model
            + 2 * d_model
            + 2 * d_model
            + self.n_heads * self.dictionary_size
            + d_model
            + d_model
            + d_model
        )
        convolution = d_model * self.conv_kernel_size + d_model
        stable = 3 * d_model
        return dictionary + linear_weights + convolution + stable

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> NativeFlashPDMixer:
        """Build the configured native mixer."""
        del layer_idx, n_layers, cache
        return NativeFlashPDMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            ste_temperature=self.ste_temperature,
            mode=self.mode,
            backend=self.backend,
            conv_kernel_size=self.conv_kernel_size,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
