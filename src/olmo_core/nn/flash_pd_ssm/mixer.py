"""Flash PD-SSM sequence mixer and registered OLMo-core configuration."""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Placement
from torch.nn import functional as F

from olmo_core.config import DType, StrEnum
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.attention.base import SequenceMixer, SequenceMixerConfig
from olmo_core.nn.attention.ring import (
    RingContextParallelStyle,
    UlyssesContextParallelStyle,
)
from olmo_core.nn.buffer_cache import BufferCache

from .autograd import sparse_ste_scan
from .reference import (
    affine_chunkwise_reference,
    affine_recurrent_reference,
    sparse_chunkwise_reference,
)
from .transition import selected_transition_destination, selected_transition_matrix
from .triton_kernel import flash_pd_triton_scan, triton_capability

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod

log = logging.getLogger(__name__)

__all__ = [
    "FlashPDSSMImplementation",
    "FlashPDSSMMixer",
    "FlashPDSSMMixerConfig",
]


class FlashPDSSMImplementation(StrEnum):
    """Available Flash PD-SSM scan implementations."""

    auto = "auto"
    """Use Triton when eligible and expose the reason for any PyTorch fallback."""

    recurrent = "recurrent"
    """Use the sequential correctness-first PyTorch reference."""

    chunkwise = "chunkwise"
    """Use the differentiable three-phase PyTorch chunkwise reference."""

    sparse_autograd = "sparse_autograd"
    """Use compact hard transitions with a linear-memory analytic backward."""

    triton = "triton"
    """Require the forward-only custom Triton prototype; never fall back."""


def _validate_options(
    *,
    n_heads: int,
    d_state: int,
    dictionary_size: int,
    chunk_size: int,
    ste_temperature: float,
    decay_init_min: float,
    decay_init_max: float,
) -> None:
    if n_heads < 1:
        raise OLMoConfigurationError(f"n_heads must be positive, got {n_heads}")
    if d_state < 1:
        raise OLMoConfigurationError(f"d_state must be positive, got {d_state}")
    if dictionary_size < 1:
        raise OLMoConfigurationError(f"dictionary_size must be positive, got {dictionary_size}")
    if chunk_size < 1:
        raise OLMoConfigurationError(f"chunk_size must be positive, got {chunk_size}")
    if ste_temperature <= 0:
        raise OLMoConfigurationError(f"ste_temperature must be positive, got {ste_temperature}")
    if decay_init_min <= 0:
        raise OLMoConfigurationError(f"decay_init_min must be positive, got {decay_init_min}")
    if decay_init_max <= decay_init_min:
        raise OLMoConfigurationError(
            "decay_init_max must be greater than decay_init_min, got "
            f"{decay_init_max} <= {decay_init_min}"
        )


class FlashPDSSMMixer(SequenceMixer):
    """
    A configurable state-tracking mixer based on Flash PD-SSM.

    The mixer implements the discrete two-stage selector from arXiv:2605.19150: each head owns
    ``dictionary_size`` dense trainable matrices, column-wise hardmax turns each into a compact
    column-one-hot transition, and a second per-token argmax chooses one dictionary member.
    Both hard decisions use slope-annealed straight-through gradients.

    Its complex affine recurrence is ``state_t = P_t D_t state_(t-1) + b_t``. ``D_t`` combines
    a stable learned decay with an input-dependent phase. The PyTorch paths retain dense
    straight-through matrices as a correctness oracle; the optional Triton path stores only
    integer source indices and runs the paper's three phases. Triton is currently forward-only
    and explicitly rejects autograd, non-CUDA, non-complex64, nonzero-initial-state, and
    ``d_state > 32`` cases.

    Replacing a :class:`~olmo_core.nn.mamba3.Mamba3Mixer` with this module preserves the outer
    block/config shell, but **does not preserve mixer state-dict compatibility**: projection,
    dictionary, and recurrence parameter names and shapes intentionally differ.

    :param d_model: Model embedding width.
    :param n_heads: Number of independent state-tracking heads.
    :param d_state: Complex state size per head.
    :param dictionary_size: Number of structured transition matrices per head.
    :param chunk_size: Timesteps per chunk for chunkwise paths.
    :param ste_temperature: Backward softmax temperature for both hard selectors.
    :param implementation: Scan implementation or explicit auto-dispatch.
    :param decay_init_min: Minimum initial positive decay rate.
    :param decay_init_max: Maximum initial positive decay rate.
    :param dtype: Parameter dtype for projection weights.
    :param init_device: Parameter initialization device, including ``"meta"``.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int = 8,
        d_state: int = 64,
        dictionary_size: int = 16,
        chunk_size: int = 128,
        ste_temperature: float = 1.0,
        implementation: FlashPDSSMImplementation | str = FlashPDSSMImplementation.auto,
        decay_init_min: float = 0.05,
        decay_init_max: float = 1.0,
        dtype: torch.dtype = torch.float32,
        init_device: str = "cpu",
    ):
        super().__init__()
        _validate_options(
            n_heads=n_heads,
            d_state=d_state,
            dictionary_size=dictionary_size,
            chunk_size=chunk_size,
            ste_temperature=ste_temperature,
            decay_init_min=decay_init_min,
            decay_init_max=decay_init_max,
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.dictionary_size = dictionary_size
        self.chunk_size = chunk_size
        self.ste_temperature = ste_temperature
        self.implementation = FlashPDSSMImplementation(implementation)
        self.decay_init_min = decay_init_min
        self.decay_init_max = decay_init_max
        self.last_backend: Optional[str] = None
        self.last_fallback_reason: Optional[str] = None
        self._reported_auto_fallback = False

        inner = n_heads * d_state
        self.dictionary_logits = nn.Parameter(
            torch.empty(
                n_heads,
                dictionary_size,
                d_state,
                d_state,
                dtype=dtype,
                device=init_device,
            )
        )
        self.in_proj = nn.Linear(
            d_model,
            2 * inner,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.selector_proj = nn.Linear(
            d_model,
            n_heads * dictionary_size,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.dt_proj = nn.Linear(
            d_model,
            inner,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.phase_proj = nn.Linear(
            d_model,
            inner,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.gate_proj = nn.Linear(
            d_model,
            inner,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.out_proj = nn.Linear(
            2 * inner,
            d_model,
            bias=False,
            dtype=dtype,
            device=init_device,
        )
        self.A_log = nn.Parameter(
            torch.empty(n_heads, d_state, dtype=torch.float32, device=init_device)
        )
        self.dt_bias = nn.Parameter(
            torch.empty(n_heads, d_state, dtype=torch.float32, device=init_device)
        )
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]

    def set_ste_temperature(self, temperature: float) -> None:
        """
        Update the backward-pass selector temperature for slope annealing.

        :param temperature: Positive tempered-softmax temperature.
        """
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.ste_temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Mix a sequence with discrete structured state transitions.

        :param x: Input of shape ``(batch, time, d_model)``.
        :param cu_doc_lens: Cumulative packed-document boundaries. A single document is accepted;
            multiple documents are rejected because state-reset masking is not implemented.
        :param kwargs: Extra transformer-block kwargs, accepted for shell compatibility.

        :returns: Output with the same shape and dtype as ``x``.

        :raises NotImplementedError: If packed multi-document masking is requested.
        """
        self.last_backend = None
        self.last_fallback_reason = None
        if kwargs.get("initial_state") is not None:
            raise NotImplementedError(
                "FlashPDSSMMixer does not support nonzero initial_state or recurrent caching"
            )
        if kwargs.get("decode", False):
            raise NotImplementedError(
                "FlashPDSSMMixer does not support decode mode or recurrent caching"
            )
        del kwargs
        if cu_doc_lens is not None and cu_doc_lens.numel() > 2:
            raise NotImplementedError(
                "FlashPDSSMMixer does not support packed-document state resets; "
                f"'cu_doc_lens' describes {cu_doc_lens.numel() - 1} documents"
            )
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, time, {self.d_model}), got {tuple(x.shape)}"
            )

        batch, time, _ = x.shape
        H, N, K = self.n_heads, self.d_state, self.dictionary_size
        drive = self.in_proj(x).view(batch, time, H, N, 2)
        bias = torch.complex(drive[..., 0].float(), drive[..., 1].float())
        selector_logits = self.selector_proj(x).view(batch, time, H, K).float()

        dt = F.softplus(self.dt_proj(x).view(batch, time, H, N).float() + self.dt_bias)
        magnitude = torch.exp(-dt * torch.exp(self.A_log).view(1, 1, H, N))
        phase = self.phase_proj(x).view(batch, time, H, N).float()
        diagonal = torch.polar(magnitude, phase)

        scan_bias = bias.permute(0, 2, 1, 3)

        def dense_scan_transition() -> torch.Tensor:
            transition_selector = selected_transition_matrix(
                self.dictionary_logits.float(),
                selector_logits,
                temperature=self.ste_temperature,
            )
            transition = transition_selector.to(diagonal.dtype) * diagonal.unsqueeze(-2)
            return transition.permute(0, 2, 1, 3, 4)

        if self.implementation == FlashPDSSMImplementation.recurrent:
            dense_transition = dense_scan_transition()
            states = affine_recurrent_reference(dense_transition, scan_bias)
            self.last_backend = "pytorch_recurrent"
            self.last_fallback_reason = None
        elif self.implementation == FlashPDSSMImplementation.chunkwise:
            dense_transition = dense_scan_transition()
            states = affine_chunkwise_reference(
                dense_transition,
                scan_bias,
                chunk_size=self.chunk_size,
            )
            self.last_backend = "pytorch_chunkwise"
            self.last_fallback_reason = None
        elif self.implementation == FlashPDSSMImplementation.sparse_autograd:
            sparse_diagonal = diagonal.permute(0, 2, 1, 3)
            states = sparse_ste_scan(
                self.dictionary_logits.float(),
                selector_logits,
                sparse_diagonal,
                scan_bias,
                temperature=self.ste_temperature,
                use_triton=False,
                chunk_size=self.chunk_size,
            )
            self.last_backend = "pytorch_sparse_autograd"
            self.last_fallback_reason = None
        else:
            destination = (
                selected_transition_destination(
                    self.dictionary_logits,
                    selector_logits,
                )
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            sparse_diagonal = diagonal.permute(0, 2, 1, 3)
            requires_autograd = torch.is_grad_enabled() and (
                self.dictionary_logits.requires_grad
                or selector_logits.requires_grad
                or sparse_diagonal.requires_grad
                or scan_bias.requires_grad
            )
            if self.implementation == FlashPDSSMImplementation.auto and requires_autograd:
                forward_capability = triton_capability(
                    destination,
                    sparse_diagonal.detach(),
                    scan_bias.detach(),
                    chunk_size=self.chunk_size,
                    requires_autograd=False,
                )
                states = sparse_ste_scan(
                    self.dictionary_logits.float(),
                    selector_logits,
                    sparse_diagonal,
                    scan_bias,
                    temperature=self.ste_temperature,
                    use_triton=forward_capability.available,
                    chunk_size=self.chunk_size,
                )
                self.last_backend = (
                    "triton_forward_sparse_autograd"
                    if forward_capability.available
                    else "pytorch_sparse_autograd"
                )
                self.last_fallback_reason = (
                    None if forward_capability.available else forward_capability.reason
                )
                if not forward_capability.available and not self._reported_auto_fallback:
                    log.info(
                        "Flash PD Triton unavailable; using sparse analytic autograd: %s",
                        forward_capability.reason,
                    )
                    self._reported_auto_fallback = True
            else:
                capability = triton_capability(
                    destination,
                    sparse_diagonal,
                    scan_bias,
                    chunk_size=self.chunk_size,
                    requires_autograd=requires_autograd,
                )
                if capability.available:
                    states = flash_pd_triton_scan(
                        destination,
                        sparse_diagonal,
                        scan_bias,
                        chunk_size=self.chunk_size,
                    )
                    self.last_backend = "triton_three_phase"
                    self.last_fallback_reason = None
                elif self.implementation == FlashPDSSMImplementation.triton:
                    self.last_fallback_reason = capability.reason
                    raise RuntimeError(
                        f"Flash PD Triton path required but unavailable: {capability.reason}"
                    )
                else:
                    states = sparse_chunkwise_reference(
                        destination,
                        sparse_diagonal,
                        scan_bias,
                        chunk_size=self.chunk_size,
                    )
                    self.last_backend = "pytorch_sparse_chunkwise"
                    self.last_fallback_reason = capability.reason

        states = states.permute(0, 2, 1, 3)
        gate = F.silu(self.gate_proj(x).view(batch, time, H, N).float())
        features = torch.cat((states.real * gate, states.imag * gate), dim=-1)
        return self.out_proj(features.reshape(batch, time, 2 * H * N).to(x.dtype))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        """Reject tensor parallelism, which the prototype does not implement."""
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled
        raise NotImplementedError("Tensor parallelism is not implemented for FlashPDSSMMixer")

    def apply_cp(
        self,
        cp_mesh: DeviceMesh,
        ring: Optional[RingContextParallelStyle] = None,
        uly: Optional[UlyssesContextParallelStyle] = None,
    ):
        """Reject multi-rank context parallelism; a size-one mesh is a no-op."""
        del ring, uly
        if cp_mesh.size() == 1:
            return
        raise NotImplementedError("Context parallelism is not implemented for FlashPDSSMMixer")

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
        """
        Initialize dictionary, projections, stable decay, and phase parameters.

        :param init_method: OLMo transformer initialization policy.
        :param d_model: Model width.
        :param block_idx: Zero-based block index.
        :param num_blocks: Total block count.
        :param std: Base projection standard deviation.
        :param generator: Optional deterministic random generator.
        """
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        if init_method == InitMethod.fan_in:
            raise NotImplementedError(
                f"init method '{init_method}' is not supported for FlashPDSSMMixer"
            )
        if init_method == InitMethod.normalized:
            std = d_model**-0.5

        nn.init.normal_(self.dictionary_logits, std=std, generator=generator)
        for projection in (
            self.in_proj,
            self.selector_proj,
            self.dt_proj,
            self.gate_proj,
        ):
            init_linear(projection, std=std, generator=generator)
        init_linear(self.phase_proj, std=std * 0.1, generator=generator)

        self.A_log.copy_(
            nn.init.uniform_(
                self.A_log,
                a=self.decay_init_min,
                b=self.decay_init_max,
                generator=generator,
            ).log()
        )
        dt_min, dt_max = 0.001, 0.1
        dt = torch.exp(
            nn.init.uniform_(self.dt_bias, generator=generator)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        output_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            output_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std = std / (2 * (block_idx + 1)) ** 0.5
        init_linear(self.out_proj, std=output_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Estimate model FLOPs per token independently of implementation overhead.

        The count includes projection matmuls, selector comparisons, complex sparse recurrence,
        and dictionary hardmax amortized across the sequence. It excludes dense-oracle and Triton
        padding overhead so a slower implementation cannot report artificially higher MFU.

        :param seq_len: Sequence length used to amortize dictionary hardmax.

        :returns: Approximate FLOPs per token.
        """
        projection_flops = 2 * sum(
            module.weight.numel()
            for module in (
                self.in_proj,
                self.selector_proj,
                self.dt_proj,
                self.phase_proj,
                self.gate_proj,
                self.out_proj,
            )
        )
        inner = self.n_heads * self.d_state
        recurrence_flops = 10 * inner
        selector_flops = self.n_heads * self.dictionary_size
        dictionary_flops = (
            self.n_heads * self.dictionary_size * self.d_state * self.d_state
        ) // max(seq_len, 1)
        return int(projection_flops + recurrence_flops + selector_flops + dictionary_flops)


@SequenceMixerConfig.register("flash_pd")
@dataclass
class FlashPDSSMMixerConfig(SequenceMixerConfig[FlashPDSSMMixer]):
    """
    Registered configuration for :class:`FlashPDSSMMixer`.

    The config is additive and does not alter Mamba-3 registration or defaults. See the mixer
    docstring for the explicit Triton, TP/CP, packed-document, and state-dict limitations.
    """

    n_heads: int = 8
    """Number of independent Flash PD state heads."""
    d_state: int = 64
    """Complex state size per head."""
    dictionary_size: int = 16
    """Structured transition dictionary size per head."""
    chunk_size: int = 128
    """Number of timesteps per three-phase chunk."""
    ste_temperature: float = 1.0
    """Backward softmax temperature; anneal downward to increase the STE slope."""
    implementation: FlashPDSSMImplementation = FlashPDSSMImplementation.auto
    """Reference/Triton implementation selection."""
    decay_init_min: float = 0.05
    """Minimum initial positive diagonal decay rate."""
    decay_init_max: float = 1.0
    """Maximum initial positive diagonal decay rate."""
    dtype: DType = DType.float32
    """Projection and dictionary parameter dtype."""

    def num_params(self, d_model: int) -> int:
        """
        Count parameters exactly as :meth:`build` creates them.

        :param d_model: Model embedding width.

        :returns: Exact parameter count.
        """
        _validate_options(
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            ste_temperature=self.ste_temperature,
            decay_init_min=self.decay_init_min,
            decay_init_max=self.decay_init_max,
        )
        inner = self.n_heads * self.d_state
        dictionary = self.n_heads * self.dictionary_size * self.d_state * self.d_state
        projections = d_model * (7 * inner + self.n_heads * self.dictionary_size)
        stable_parameters = 2 * inner
        return dictionary + projections + stable_parameters

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> FlashPDSSMMixer:
        """
        Build a :class:`FlashPDSSMMixer`.

        :param d_model: Model embedding width.
        :param layer_idx: Layer index, accepted for the sequence-mixer factory contract.
        :param n_layers: Layer count, accepted for the sequence-mixer factory contract.
        :param init_device: Initialization device, including ``"meta"``.
        :param cache: Optional shared buffer cache, currently unused.

        :returns: A configured mixer.
        """
        del layer_idx, n_layers, cache
        return FlashPDSSMMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            ste_temperature=self.ste_temperature,
            implementation=self.implementation,
            decay_init_min=self.decay_init_min,
            decay_init_max=self.decay_init_max,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
