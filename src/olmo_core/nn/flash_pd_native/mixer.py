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
    _UNFUSED_PROJECTIONS = ("B_proj", "C_proj", "selector_proj", "dt_proj", "phase_proj")

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
        fuse_input_projections: bool = True,
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
        self.fuse_input_projections = fuse_input_projections
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
        # Every projection below reads the same convolved activation and carries no bias,
        # so one GEMM over the concatenated output width holds exactly the same weights.
        projection_sizes = self._projection_sizes()
        self.u_proj: Optional[nn.Linear] = None
        for name in self._UNFUSED_PROJECTIONS:
            setattr(self, name, None)
        if fuse_input_projections:
            self.u_proj = nn.Linear(d_model, sum(projection_sizes), bias=False, **factory)
        else:
            for name, size in zip(self._UNFUSED_PROJECTIONS, projection_sizes):
                setattr(self, name, nn.Linear(d_model, size, bias=False, **factory))
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory)
        self.A_log = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.D = nn.Parameter(torch.empty(d_model, dtype=torch.float32, device=init_device))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def _projection_sizes(self) -> tuple[int, ...]:
        """Return the output width of each post-convolution projection, in order."""
        return (
            2 * self.d_model,
            2 * self.d_model,
            self.n_heads * self.dictionary_size,
            self.d_model,
            self.d_model,
        )

    def _project(self, u: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project the convolved activation once, or once per projection."""
        if self.fuse_input_projections:
            assert self.u_proj is not None
            return self.u_proj(u).split(self._projection_sizes(), dim=-1)
        return tuple(getattr(self, name)(u) for name in self._UNFUSED_PROJECTIONS)

    def _convert_projection_state_dict(self, state_dict: dict, prefix: str) -> None:
        """Reshape a checkpoint written in the other projection layout in place."""
        fused = prefix + "u_proj.weight"
        separate = tuple(prefix + name + ".weight" for name in self._UNFUSED_PROJECTIONS)
        if self.fuse_input_projections:
            if fused not in state_dict and all(name in state_dict for name in separate):
                state_dict[fused] = torch.cat([state_dict.pop(name) for name in separate], dim=0)
        elif fused in state_dict and not any(name in state_dict for name in separate):
            pieces = state_dict.pop(fused).split(self._projection_sizes(), dim=0)
            state_dict.update(zip(separate, pieces))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        self._convert_projection_state_dict(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

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

        (
            bias_projection,
            readout_projection,
            selector_projection,
            dt_logits,
            phase_logits,
        ) = self._project(u)
        bias = bias_projection.view(batch, time, self.n_heads, self.d_state, 2)
        readout = readout_projection.view(batch, time, self.n_heads, self.d_state, 2)
        selector_logits = selector_projection.view(batch, time, self.n_heads, self.dictionary_size)
        dt = F.softplus(dt_logits.float() + self.dt_bias.view(1, 1, self.d_model))
        magnitude = torch.exp(-dt * torch.exp(self.A_log).view(1, 1, self.d_model))
        phase = phase_logits.float()
        diagonal_real = (magnitude * torch.cos(phase)).view(batch, time, self.n_heads, self.d_state)
        diagonal_imag = (magnitude * torch.sin(phase)).view(batch, time, self.n_heads, self.d_state)

        # The diagonal crosses into the scan in FP32 while the payload keeps the
        # activation dtype. A near-unit per-token decay such as exp(-5e-4) is not
        # representable in bfloat16 and rounds to exactly 1.0, which would erase the
        # long-horizon decay this recurrence is built on.
        kernel_dtype = x.dtype if x.dtype in (torch.float32, torch.bfloat16) else torch.float32
        # Densified here rather than at the kernel boundary. The scan reaches CUDA through a
        # raw pybind call, so Dynamo breaks the graph in front of it and the `contiguous()`
        # calls inside the autograd function run as eager ATen copies Inductor never sees.
        # Neither cast prevents that: the diagonal is already FP32 and in bfloat16 the payload
        # already carries the activation dtype, so `.float()` and `.to(kernel_dtype)` return
        # the very same permuted view. Those four eager copies measure 0.876 ms together at the
        # production shape, the two bfloat16 ones at 133 GB/s against a 307 GB/s copy rate
        # because the interleaved real/imaginary axis leaves them a stride of two.
        #
        # Asking for the dense layout inside the traced region instead lets the pointwise chain
        # that builds the diagonal store `(batch, head, time, state)` directly. Interleaved
        # against the strided form at the production shape: 0.9804, 0.9827, 0.9848 of its
        # forward-and-backward time, and 0.969 of the forward alone -- the backward gives a
        # little back, which is why densifying only the diagonal pair measured no better.
        #
        # Densifying all four is also what keeps the compiled prologue from returning views
        # that alias its own inputs. Left strided, AOTAutograd has to regenerate those aliases
        # on every call, and that path intermittently replayed corrupt view metadata and
        # aborted with an impossible shape.
        split_values = (
            diagonal_real.permute(0, 2, 1, 3).float().contiguous(),
            diagonal_imag.permute(0, 2, 1, 3).float().contiguous(),
            bias[..., 0].permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
            bias[..., 1].permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
        )
        # The CUDA router gradient indexes the selector logits off a raw pointer under a
        # dense (batch, time, head, dictionary) layout, and a fused projection hands the
        # scan a strided split view. Densifying after the cast keeps this free in bfloat16,
        # where the cast already writes a dense buffer.
        dense_selector_logits = selector_logits.float().contiguous()
        if self.backend == NativePDBackend.REFERENCE:
            states_real, states_imag = _dense_ste_scan(
                self.dictionary_logits.float(),
                dense_selector_logits,
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
                dense_selector_logits,
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

    def _init_fused_projection(
        self,
        weight: torch.Tensor,
        *,
        std: float,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        """Fill the fused weight one projection slice at a time."""
        for chunk in weight.split(self._projection_sizes(), dim=0):
            nn.init.trunc_normal_(
                chunk, mean=0.0, std=std, a=-3 * std, b=3 * std, generator=generator
            )

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
        from olmo_core.nn.transformer.init import InitMethod, _apply_init, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError("fan_in initialization is not implemented")
        if init_method == InitMethod.normalized:
            std = d_model**-0.5
        nn.init.normal_(self.dictionary_logits, std=std, generator=generator)
        init_linear(self.in_proj, std=std, generator=generator)
        if self.fuse_input_projections:
            assert self.u_proj is not None
            # Drawn slice by slice, in the order the separate projections are drawn, so
            # one seed produces the same weights in either layout. `_apply_init` keeps
            # that true once FSDP has turned the fused weight into a sharded `DTensor`:
            # it draws the whole projection locally and copies out this rank's shard,
            # rather than splitting a `DTensor` along the dimension it is sharded on.
            _apply_init(
                self._init_fused_projection,
                self.u_proj.weight,
                std=std,
                generator=generator,
            )
        else:
            for name in self._UNFUSED_PROJECTIONS:
                init_linear(getattr(self, name), std=std, generator=generator)
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
        if self.fuse_input_projections:
            assert self.u_proj is not None
            post_convolution: tuple[nn.Linear, ...] = (self.u_proj,)
        else:
            post_convolution = tuple(getattr(self, name) for name in self._UNFUSED_PROJECTIONS)
        projection = 2 * sum(
            layer.weight.numel() for layer in (self.in_proj, *post_convolution, self.out_proj)
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
    # False, MEASURED, NOT THE True THIS BLOCK WAS FIRST WRITTEN WITH. Fusing the five
    # post-convolution projections saves four kernel launches in the forward and costs far
    # more than that in the backward: `split` is a view, so its gradient is a `cat` across
    # the whole 6400-wide projection, and Inductor lowers that concatenation to a single
    # kernel whose `tl.where` chain evaluates all five gradient expressions for every one of
    # its (batch, time, 6400) elements. Unfused, each projection's gradient lands in its own
    # weight and no concatenation is built at all.
    #
    # Interleaved against the fused layout in one process at the production shape
    # (B=2, T=4096, d_model=1024, bfloat16, forward and backward): 0.9469, 0.9453, 0.9477 of
    # the fused time, and 0.9857 of it with the backward excluded, so seven eighths of the
    # win is the concatenation that is no longer built. Absolute milliseconds on this card
    # drift by tens of percent between processes and are not quotable; the ratio is.
    #
    # This moves no weight and no number. `num_params` is the same in either layout because
    # fusing only concatenates output widths, one seed draws bit-identical weights either way
    # (`test_projection_layouts_initialize_identically_from_one_seed`), and
    # `_convert_projection_state_dict` converts a checkpoint written in either layout exactly.
    fuse_input_projections: bool = False
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
        # Fusing the post-convolution projections concatenates their output widths, so
        # this total is the same in either layout.
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
            fuse_input_projections=self.fuse_input_projections,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
