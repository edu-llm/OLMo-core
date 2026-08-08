"""Mamba-3-improved SISO mixer over the native collision-capable PD transition."""

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

from .api import mamba3_siso_surrogate_scan
from .contracts import (
    NativePDBackend,
    NativePDMode,
    ScanMetadata,
    SelectorTelemetry,
    SISOAccounting,
    SISOScanCache,
)
from .mixer import _validate_options
from .reference import trapezoidal_reference_scan
from .routes import compact_hard_selection

if TYPE_CHECKING:
    from olmo_core.nn.transformer.init import InitMethod


def _complex_rms_norm(
    real: torch.Tensor,
    imag: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    variance = (real.float().square() + imag.float().square()).mean(dim=-1, keepdim=True)
    scale = torch.rsqrt(variance + eps) * weight.float()
    return real * scale.to(real.dtype), imag * scale.to(imag.dtype)


class NativeFlashPDMamba3SISOMixer(SequenceMixer):
    """
    Native PD-SSM with Mamba-3 SISO discretization and architecture defaults.

    The state has exactly ``(batch, head, time, state)`` axes. The hard
    transition remains ``P_t diag(d_t)``; complex phase is carried only by
    ``d_t`` and no dense rotation is introduced.
    """

    state_contract = ("batch", "head", "time", "state")
    _UNFUSED_PROJECTIONS = (
        "in_x",
        "in_z",
        "B_proj",
        "C_proj",
        "selector_proj",
        "dt_proj",
        "phase_proj",
        "lambda_proj",
    )

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        d_state: int,
        dictionary_size: int,
        chunk_size: int = 64,
        dictionary_temperature: float = 1.0,
        router_temperature: float = 1.0,
        dictionary_temperature_end: Optional[float] = None,
        router_temperature_end: Optional[float] = None,
        temperature_schedule_steps: int = 0,
        mode: NativePDMode | str = NativePDMode.AUTO,
        backend: NativePDBackend | str = NativePDBackend.AUTO,
        bc_norm: bool = True,
        norm_eps: float = 1e-5,
        output_norm: bool = False,
        fuse_input_projections: bool = True,
        a_log_init_min: float = 0.05,
        a_log_init_max: float = 16.0,
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
            ste_temperature=min(dictionary_temperature, router_temperature),
        )
        if chunk_size not in (32, 64, 128):
            raise ValueError(f"chunk_size must be one of (32, 64, 128), got {chunk_size}")
        if dictionary_temperature <= 0 or router_temperature <= 0:
            raise ValueError("dictionary and router temperatures must be positive")
        if dictionary_temperature_end is not None and dictionary_temperature_end <= 0:
            raise ValueError("dictionary temperature endpoint must be positive")
        if router_temperature_end is not None and router_temperature_end <= 0:
            raise ValueError("router temperature endpoint must be positive")
        if temperature_schedule_steps < 0:
            raise ValueError("temperature_schedule_steps must be non-negative")
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if a_log_init_min <= 0 or a_log_init_min >= a_log_init_max:
            raise ValueError("A-log initialization bounds must satisfy 0 < min < max")

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.dictionary_size = dictionary_size
        self.chunk_size = chunk_size
        self.dictionary_temperature_start = dictionary_temperature
        self.router_temperature_start = router_temperature
        self.dictionary_temperature_end = (
            dictionary_temperature
            if dictionary_temperature_end is None
            else dictionary_temperature_end
        )
        self.router_temperature_end = (
            router_temperature if router_temperature_end is None else router_temperature_end
        )
        self.temperature_schedule_steps = temperature_schedule_steps
        self.mode = NativePDMode(mode)
        self.backend = NativePDBackend(backend)
        self.bc_norm_enabled = bc_norm
        self.norm_eps = norm_eps
        self.output_norm_enabled = output_norm
        self.fuse_input_projections = fuse_input_projections
        self.a_log_init_min = a_log_init_min
        self.a_log_init_max = a_log_init_max
        self.conv_kernel_size = None
        self.last_metadata: Optional[ScanMetadata] = None
        self.register_buffer(
            "_temperature_schedule_step",
            torch.tensor(0, dtype=torch.int64, device=init_device),
            persistent=True,
        )
        self.register_buffer(
            "_dictionary_temperature",
            torch.tensor(dictionary_temperature, dtype=torch.float32, device=init_device),
            persistent=True,
        )
        self.register_buffer(
            "_router_temperature",
            torch.tensor(router_temperature, dtype=torch.float32, device=init_device),
            persistent=True,
        )

        factory = {"dtype": dtype, "device": init_device}
        self.dictionary_logits = nn.Parameter(
            torch.empty(n_heads, dictionary_size, d_state, d_state, **factory)
        )
        projection_sizes = self._projection_sizes()
        self.in_proj: Optional[nn.Linear] = None
        for name in self._UNFUSED_PROJECTIONS:
            setattr(self, name, None)
        if fuse_input_projections:
            self.in_proj = nn.Linear(
                d_model,
                sum(projection_sizes),
                bias=False,
                **factory,
            )
        else:
            for name, size in zip(self._UNFUSED_PROJECTIONS, projection_sizes):
                setattr(self, name, nn.Linear(d_model, size, bias=False, **factory))
        self.out_proj = nn.Linear(d_model, d_model, bias=False, **factory)

        self.B_bias = nn.Parameter(
            torch.ones(n_heads, d_state, dtype=torch.float32, device=init_device)
        )
        self.C_bias = nn.Parameter(
            torch.ones(n_heads, d_state, dtype=torch.float32, device=init_device)
        )
        if bc_norm:
            self.bc_norm_b = nn.Parameter(torch.ones(d_state, **factory))
            self.bc_norm_c = nn.Parameter(torch.ones(d_state, **factory))
        else:
            self.register_parameter("bc_norm_b", None)
            self.register_parameter("bc_norm_c", None)
        if output_norm:
            self.output_norm_weight = nn.Parameter(torch.ones(d_state, **factory))
        else:
            self.register_parameter("output_norm_weight", None)

        self.A_log = nn.Parameter(torch.empty(n_heads, dtype=torch.float32, device=init_device))
        self.dt_bias = nn.Parameter(torch.empty(n_heads, dtype=torch.float32, device=init_device))
        self.D = nn.Parameter(torch.ones(n_heads, dtype=torch.float32, device=init_device))
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]
        self.dt_bias._no_weight_decay = True  # type: ignore[attr-defined]
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def _projection_sizes(self) -> tuple[int, ...]:
        return (
            self.d_model,
            self.d_model,
            2 * self.d_model,
            2 * self.d_model,
            self.n_heads * self.dictionary_size,
            self.n_heads,
            self.d_model,
            self.n_heads,
        )

    @property
    def dictionary_temperature(self) -> float:
        """Current dictionary hardmax-surrogate temperature."""
        return float(self._dictionary_temperature.item())

    @property
    def router_temperature(self) -> float:
        """Current token-router hardmax-surrogate temperature."""
        return float(self._router_temperature.item())

    @torch.no_grad()
    def set_temperature_schedule_step(self, step: int) -> None:
        """Set and persist the explicit annealing step used after resume."""
        if step < 0:
            raise ValueError("temperature schedule step must be non-negative")
        self._temperature_schedule_step.fill_(step)
        if self.temperature_schedule_steps == 0:
            fraction = 1.0 if step > 0 else 0.0
        else:
            fraction = min(step / self.temperature_schedule_steps, 1.0)
        dictionary = self.dictionary_temperature_start + fraction * (
            self.dictionary_temperature_end - self.dictionary_temperature_start
        )
        router = self.router_temperature_start + fraction * (
            self.router_temperature_end - self.router_temperature_start
        )
        self._dictionary_temperature.fill_(dictionary)
        self._router_temperature.fill_(router)

    def temperature_schedule_state(self) -> dict[str, int | float]:
        """Return the checkpointed schedule position and realized temperatures."""
        return {
            "step": int(self._temperature_schedule_step.item()),
            "dictionary_temperature": self.dictionary_temperature,
            "router_temperature": self.router_temperature,
        }

    @torch.no_grad()
    def selector_telemetry(self, selector_logits: torch.Tensor) -> SelectorTelemetry:
        """Measure entropy, dead routes, exact ties, and temporal route churn."""
        if selector_logits.ndim != 4 or selector_logits.shape[-2:] != (
            self.n_heads,
            self.dictionary_size,
        ):
            raise ValueError(
                "selector_logits must have shape "
                f"(batch, time, {self.n_heads}, {self.dictionary_size})"
            )
        probabilities = torch.softmax(
            selector_logits.float() / self.router_temperature,
            dim=-1,
        )
        entropy = -(probabilities * probabilities.clamp_min(1e-30).log()).sum(-1).mean()
        routes = selector_logits.argmax(dim=-1)
        selected_counts = F.one_hot(routes, num_classes=self.dictionary_size).sum(dim=(0, 1))
        dead_entries = (selected_counts == 0).sum()
        maximum = selector_logits.max(dim=-1, keepdim=True).values
        ties = ((selector_logits == maximum).sum(dim=-1) > 1).sum()
        if routes.shape[1] > 1:
            churn = (routes[:, 1:] != routes[:, :-1]).float().mean()
        else:
            churn = selector_logits.new_zeros((), dtype=torch.float32)
        return SelectorTelemetry(
            route_entropy=entropy,
            dead_entries=dead_entries,
            ties=ties,
            route_churn=churn,
        )

    def _project(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.fuse_input_projections:
            assert self.in_proj is not None
            return self.in_proj(x).split(self._projection_sizes(), dim=-1)
        return tuple(getattr(self, name)(x) for name in self._UNFUSED_PROJECTIONS)

    def _prepare_recurrence(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch, time, _ = x.shape
        (
            value_input,
            gate,
            b_projection,
            c_projection,
            selector_logits,
            dt_logits,
            phase_logits,
            lambda_logits,
        ) = self._project(x)
        b_projection = b_projection.view(batch, time, self.n_heads, self.d_state, 2)
        c_projection = c_projection.view(batch, time, self.n_heads, self.d_state, 2)
        b_real, b_imag = b_projection.unbind(dim=-1)
        c_real, c_imag = c_projection.unbind(dim=-1)
        if self.bc_norm_enabled:
            assert self.bc_norm_b is not None and self.bc_norm_c is not None
            b_real, b_imag = _complex_rms_norm(b_real, b_imag, self.bc_norm_b, self.norm_eps)
            c_real, c_imag = _complex_rms_norm(c_real, c_imag, self.bc_norm_c, self.norm_eps)
        b_real = b_real.float() + self.B_bias.view(1, 1, self.n_heads, self.d_state)
        c_real = c_real.float() + self.C_bias.view(1, 1, self.n_heads, self.d_state)
        b_imag = b_imag.float()
        c_imag = c_imag.float()
        value_input = value_input.view(batch, time, self.n_heads, self.d_state).float()
        value_real = b_real * value_input
        value_imag = b_imag * value_input

        dt = F.softplus(dt_logits.float() + self.dt_bias.view(1, 1, self.n_heads))
        lam = torch.sigmoid(lambda_logits.float())
        beta = (1.0 - lam) * dt
        gamma = lam * dt
        magnitude = torch.exp(-dt[..., None] * torch.exp(self.A_log).view(1, 1, self.n_heads, 1))
        theta = math.pi * torch.tanh(
            phase_logits.float().view(batch, time, self.n_heads, self.d_state)
        )
        phase = dt[..., None] * theta
        diagonal_real = magnitude * torch.cos(phase)
        diagonal_imag = magnitude * torch.sin(phase)
        selector_logits = selector_logits.view(batch, time, self.n_heads, self.dictionary_size)
        return (
            value_input,
            gate,
            c_real,
            c_imag,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
        )

    def _readout(
        self,
        x_dtype: torch.dtype,
        value_input: torch.Tensor,
        gate: torch.Tensor,
        c_real: torch.Tensor,
        c_imag: torch.Tensor,
        states_real: torch.Tensor,
        states_imag: torch.Tensor,
    ) -> torch.Tensor:
        batch, time = value_input.shape[:2]
        states_real = states_real.permute(0, 2, 1, 3)
        states_imag = states_imag.permute(0, 2, 1, 3)
        readout = c_real * states_real.float() - c_imag * states_imag.float()
        skip = self.D.view(1, 1, self.n_heads, 1) * value_input
        y = readout + skip
        if self.output_norm_enabled:
            assert self.output_norm_weight is not None
            variance = y.square().mean(dim=-1, keepdim=True)
            y = y * torch.rsqrt(variance + self.norm_eps) * self.output_norm_weight
        y = y * F.silu(gate.view(batch, time, self.n_heads, self.d_state).float())
        return self.out_proj(y.reshape(batch, time, self.d_model).to(x_dtype))

    def _convert_projection_state_dict(self, state_dict: dict, prefix: str) -> None:
        fused = prefix + "in_proj.weight"
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
        """Apply the Mamba-3 SISO PD layer without an external convolution."""
        if cu_doc_lens is not None and cu_doc_lens.numel() > 2:
            raise NotImplementedError("packed multi-document state resets are not implemented")
        if kwargs.get("initial_state") is not None or kwargs.get("decode", False):
            raise NotImplementedError("use forward_with_cache for recurrent decoding")
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, time, {self.d_model}), got {tuple(x.shape)}"
            )
        (
            value_input,
            gate,
            c_real,
            c_imag,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
        ) = self._prepare_recurrence(x)
        batch, time = x.shape[:2]

        kernel_dtype = x.dtype if x.dtype in (torch.float32, torch.bfloat16) else torch.float32
        states_real, states_imag, self.last_metadata = mamba3_siso_surrogate_scan(
            self.dictionary_logits.float(),
            selector_logits.float(),
            diagonal_real.permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
            diagonal_imag.permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
            value_real.permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
            value_imag.permute(0, 2, 1, 3).to(kernel_dtype).contiguous(),
            beta.permute(0, 2, 1).to(kernel_dtype).contiguous(),
            gamma.permute(0, 2, 1).to(kernel_dtype).contiguous(),
            dictionary_temperature=self.dictionary_temperature,
            router_temperature=self.router_temperature,
            chunk_size=self.chunk_size,
            mode=self.mode,
            backend=self.backend,
            return_metadata=True,
        )
        return self._readout(
            x.dtype,
            value_input,
            gate,
            c_real,
            c_imag,
            states_real,
            states_imag,
        )

    def forward_with_cache(
        self,
        x: torch.Tensor,
        cache: Optional[SISOScanCache] = None,
        *,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, SISOScanCache]:
        """Run prefill or decode and return the updated ``(h_t, v_t)`` cache."""
        if cu_doc_lens is not None and cu_doc_lens.numel() > 2:
            raise NotImplementedError("packed multi-document state resets are not implemented")
        if x.ndim != 3 or x.shape[-1] != self.d_model or x.shape[1] < 1:
            raise ValueError(
                f"x must have non-empty shape (batch, time, {self.d_model}), got {tuple(x.shape)}"
            )
        (
            value_input,
            gate,
            c_real,
            c_imag,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
        ) = self._prepare_recurrence(x)
        selection = compact_hard_selection(
            self.dictionary_logits.float(),
            selector_logits.float(),
        )
        states_real, states_imag, next_cache = trapezoidal_reference_scan(
            selection.destination,
            selection.routes,
            diagonal_real.permute(0, 2, 1, 3),
            diagonal_imag.permute(0, 2, 1, 3),
            value_real.permute(0, 2, 1, 3),
            value_imag.permute(0, 2, 1, 3),
            beta.permute(0, 2, 1),
            gamma.permute(0, 2, 1),
            chunk_size=self.chunk_size,
            mode=self.mode,
            initial_cache=cache,
            return_cache=True,
        )
        output = self._readout(
            x.dtype,
            value_input,
            gate,
            c_real,
            c_imag,
            states_real,
            states_imag,
        )
        return output, next_cache

    def decode_step(
        self,
        x: torch.Tensor,
        cache: SISOScanCache,
    ) -> tuple[torch.Tensor, SISOScanCache]:
        """Apply one recurrent token update in ``O(batch * heads * state)`` work."""
        if x.ndim != 3 or x.shape[1] != 1:
            raise ValueError("decode_step requires exactly one token")
        return self.forward_with_cache(x, cache)

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
        """Initialize projections, BC biases/norms, decay, and output."""
        from olmo_core.nn.transformer.init import InitMethod, init_linear

        self.set_temperature_schedule_step(0)
        if init_method == InitMethod.fan_in:
            raise NotImplementedError("fan_in initialization is not implemented")
        if init_method == InitMethod.normalized:
            std = d_model**-0.5
        nn.init.normal_(self.dictionary_logits, std=std, generator=generator)
        if self.fuse_input_projections:
            assert self.in_proj is not None
            for name, weight in zip(
                self._UNFUSED_PROJECTIONS,
                self.in_proj.weight.split(self._projection_sizes(), dim=0),
            ):
                slice_std = std * 0.1 if name == "phase_proj" else std
                nn.init.trunc_normal_(
                    weight,
                    mean=0.0,
                    std=slice_std,
                    a=-3 * slice_std,
                    b=3 * slice_std,
                    generator=generator,
                )
        else:
            for name in self._UNFUSED_PROJECTIONS:
                projection = getattr(self, name)
                init_linear(
                    projection,
                    std=std * 0.1 if name == "phase_proj" else std,
                    generator=generator,
                )
        self.B_bias.fill_(1)
        self.C_bias.fill_(1)
        if self.bc_norm_enabled:
            assert self.bc_norm_b is not None and self.bc_norm_c is not None
            self.bc_norm_b.fill_(1)
            self.bc_norm_c.fill_(1)
        if self.output_norm_enabled:
            assert self.output_norm_weight is not None
            self.output_norm_weight.fill_(1)
        self.A_log.copy_(
            torch.empty_like(self.A_log)
            .uniform_(
                self.a_log_init_min,
                self.a_log_init_max,
                generator=generator,
            )
            .log()
        )
        dt = torch.exp(
            torch.empty_like(self.dt_bias).uniform_(
                math.log(0.001),
                math.log(0.1),
                generator=generator,
            )
        ).clamp(min=1e-4)
        self.dt_bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.D.fill_(1)
        output_std = std
        if init_method in (InitMethod.llama, InitMethod.normalized):
            output_std = std / (2 * num_blocks) ** 0.5
        elif init_method == InitMethod.llama_depth:
            output_std = std / (2 * (block_idx + 1)) ** 0.5
        init_linear(self.out_proj, std=output_std, generator=generator)

    def num_flops_per_token(self, seq_len: int) -> int:
        """
        Return exact algebraic model FLOPs per token under the stack's 2-FLOP FMA convention.

        Transcendental evaluations and hard-selection comparisons are intentionally
        excluded and reported separately by :meth:`accounting`.
        """
        if self.fuse_input_projections:
            assert self.in_proj is not None
            projections = (self.in_proj, self.out_proj)
        else:
            projections = tuple(getattr(self, name) for name in self._UNFUSED_PROJECTIONS) + (
                self.out_proj,
            )
        projection_flops = 2 * sum(module.weight.numel() for module in projections)
        del seq_len
        algebra = 30 * self.d_model + 5 * self.n_heads
        if self.bc_norm_enabled:
            algebra += 14 * self.d_model + 2 * self.n_heads
        if self.mode == NativePDMode.PERMUTATION_GATHER:
            algebra -= 2 * self.d_model
        if self.output_norm_enabled:
            algebra += 3 * self.d_model + self.n_heads
        return int(projection_flops + algebra)

    def accounting(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        element_size: int,
    ) -> SISOAccounting:
        """Return exact parameter, model-work, saved-tensor, and workspace counts."""
        if batch_size < 1 or sequence_length < 1:
            raise ValueError("batch_size and sequence_length must be positive")
        if element_size not in (2, 4, 8):
            raise ValueError("element_size must describe a 2-, 4-, or 8-byte floating dtype")
        rows = batch_size * self.n_heads
        chunks = (sequence_length + self.chunk_size - 1) // self.chunk_size
        chunk_state = rows * chunks * self.d_state
        sequence_state = rows * sequence_length * self.d_state

        # Forward holds the fused complex write plus two compact aggregate/prefix sets:
        # each set is one int16 map and four FP32 arrays.
        forward_workspace = 2 * sequence_state * element_size + 36 * chunk_state
        # Backward holds one int16 reverse map, four FP32 aggregate arrays, two FP32
        # carries, the FP32 active dictionary statistic, and FP32 selector scores.
        backward_workspace = 26 * chunk_state + 4 * (
            self.n_heads * self.dictionary_size * self.d_state + rows * sequence_length
        )
        # Autograd retains complex diagonal/value/output (six sequence-state arrays),
        # beta/gamma, FP32 dictionary/router logits, and compact int16 maps/routes.
        saved_tensors = (
            6 * sequence_state * element_size
            + 2 * rows * sequence_length * element_size
            + 4
            * (
                self.n_heads * self.dictionary_size * self.d_state * self.d_state
                + batch_size * sequence_length * self.n_heads * self.dictionary_size
            )
            + 2 * (self.n_heads * self.dictionary_size * self.d_state + rows * sequence_length)
        )
        nonlinear_per_token = 3 * self.n_heads + 4 * self.d_model
        if self.bc_norm_enabled:
            nonlinear_per_token += 2 * self.n_heads
        if self.output_norm_enabled:
            nonlinear_per_token += self.n_heads
        route_comparisons = self.n_heads * self.dictionary_size * self.d_state * (
            self.d_state - 1
        ) + sequence_length * self.n_heads * (self.dictionary_size - 1)
        flops_per_token = self.num_flops_per_token(sequence_length)
        return SISOAccounting(
            parameters=sum(parameter.numel() for parameter in self.parameters()),
            flops_per_token=flops_per_token,
            model_flops_per_sequence=flops_per_token * sequence_length,
            nonlinear_evaluations_per_sequence=(nonlinear_per_token * sequence_length),
            route_comparisons_per_sequence=route_comparisons,
            saved_tensor_bytes=saved_tensors,
            forward_workspace_bytes=forward_workspace,
            backward_workspace_bytes=backward_workspace,
            peak_workspace_bytes=max(forward_workspace, backward_workspace),
        )


@SequenceMixerConfig.register("flash_pd_native_mamba3_siso")
@dataclass
class NativeFlashPDMamba3SISOMixerConfig(SequenceMixerConfig[NativeFlashPDMamba3SISOMixer]):
    """Stable config for the Mamba-3-improved native SISO PD-SSM."""

    n_heads: int = 8
    d_state: int = 64
    dictionary_size: int = 16
    chunk_size: int = 64
    dictionary_temperature: float = 1.0
    router_temperature: float = 1.0
    dictionary_temperature_end: Optional[float] = None
    router_temperature_end: Optional[float] = None
    temperature_schedule_steps: int = 0
    mode: NativePDMode = NativePDMode.AUTO
    backend: NativePDBackend = NativePDBackend.AUTO
    bc_norm: bool = True
    norm_eps: float = 1e-5
    output_norm: bool = False
    fuse_input_projections: bool = True
    a_log_init_min: float = 0.05
    a_log_init_max: float = 16.0
    dtype: DType = DType.float32

    def num_params(self, d_model: int) -> int:
        """Return the exact parameter count of the configured mixer."""
        _validate_options(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            ste_temperature=min(self.dictionary_temperature, self.router_temperature),
        )
        projection_outputs = 7 * d_model + self.n_heads * self.dictionary_size + 2 * self.n_heads
        projections = d_model * projection_outputs + d_model**2
        dictionary = self.n_heads * self.dictionary_size * self.d_state**2
        bc_parameters = 2 * self.n_heads * self.d_state
        if self.bc_norm:
            bc_parameters += 2 * self.d_state
        if self.output_norm:
            bc_parameters += self.d_state
        stable = 3 * self.n_heads
        return dictionary + projections + bc_parameters + stable

    def build(
        self,
        d_model: int,
        *,
        layer_idx: int,
        n_layers: int,
        init_device: str = "cpu",
        cache: Optional[BufferCache] = None,
    ) -> NativeFlashPDMamba3SISOMixer:
        """Build the configured SISO PD mixer."""
        del layer_idx, n_layers, cache
        return NativeFlashPDMamba3SISOMixer(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            dictionary_size=self.dictionary_size,
            chunk_size=self.chunk_size,
            dictionary_temperature=self.dictionary_temperature,
            router_temperature=self.router_temperature,
            dictionary_temperature_end=self.dictionary_temperature_end,
            router_temperature_end=self.router_temperature_end,
            temperature_schedule_steps=self.temperature_schedule_steps,
            mode=self.mode,
            backend=self.backend,
            bc_norm=self.bc_norm,
            norm_eps=self.norm_eps,
            output_norm=self.output_norm,
            fuse_input_projections=self.fuse_input_projections,
            a_log_init_min=self.a_log_init_min,
            a_log_init_max=self.a_log_init_max,
            dtype=self.dtype.as_pt(),
            init_device=init_device,
        )
