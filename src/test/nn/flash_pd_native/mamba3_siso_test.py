import itertools
import math
from typing import Callable

import pytest
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

import olmo_core.nn.flash_pd_native.mamba3_siso as mamba3_siso_module
from olmo_core.config import DType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.flash_pd_native import (
    NativeFlashPDMamba3SISOMixer,
    NativeFlashPDMamba3SISOMixerConfig,
    NativeFlashPDMixerConfig,
    NativePDMode,
    SISOScanCache,
    trapezoidal_reference_scan,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    InitMethod,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.testing import run_distributed_test


def _selected_map(destination: torch.Tensor, routes: torch.Tensor, token: int) -> torch.Tensor:
    batch, heads, _ = routes.shape
    state = destination.shape[-1]
    dictionary = destination.unsqueeze(0).expand(batch, -1, -1, -1)
    index = routes[:, :, token].long().view(batch, heads, 1, 1).expand(-1, -1, 1, state)
    return torch.gather(dictionary, 2, index).squeeze(2).long()


def _independent_dense_trapezoid_oracle(
    destination: torch.Tensor,
    routes: torch.Tensor,
    diagonal_real: torch.Tensor,
    diagonal_imag: torch.Tensor,
    value_real: torch.Tensor,
    value_imag: torch.Tensor,
    beta: torch.Tensor,
    gamma: torch.Tensor,
    initial_cache: SISOScanCache | None = None,
) -> tuple[torch.Tensor, torch.Tensor, SISOScanCache]:
    diagonal = torch.complex(diagonal_real, diagonal_imag)
    value = torch.complex(value_real, value_imag)
    batch, heads, time, state_size = diagonal.shape
    if initial_cache is None:
        state = torch.zeros(batch, heads, state_size, dtype=diagonal.dtype, device=diagonal.device)
        previous_value = torch.zeros_like(state)
    else:
        state = torch.complex(initial_cache.h_real, initial_cache.h_imag)
        previous_value = torch.complex(initial_cache.v_real, initial_cache.v_imag)
    outputs = []
    for token in range(time):
        selected = _selected_map(destination, routes, token)
        transition = F.one_hot(selected, num_classes=state_size).movedim(-1, -2)
        transition = transition.to(diagonal.dtype) * diagonal[:, :, token].unsqueeze(-2)
        transition_input = state + beta[:, :, token, None] * previous_value
        state = (
            torch.einsum("bhij,bhj->bhi", transition, transition_input)
            + gamma[:, :, token, None] * value[:, :, token]
        )
        previous_value = value[:, :, token]
        outputs.append(state)
    output = torch.stack(outputs, dim=2)
    cache = SISOScanCache(
        h_real=state.real,
        h_imag=state.imag,
        v_real=previous_value.real,
        v_imag=previous_value.imag,
    )
    return output.real, output.imag, cache


def _case(
    *,
    state: int,
    time: int,
    collision: bool,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    torch.manual_seed(101 + state + time + collision)
    identity = torch.arange(state, dtype=torch.int16)
    alternate = torch.roll(identity, 3)
    if collision:
        alternate[1::4] = alternate[0::4]
    destination = torch.stack((identity, alternate)).unsqueeze(0)
    routes = torch.randint(0, 2, (1, 1, time), dtype=torch.int16)
    if collision:
        routes[..., 0] = 1
    values = [
        (torch.randn(1, 1, time, state, dtype=dtype) * 0.1).requires_grad_() for _ in range(4)
    ]
    values[0].data.add_(0.9)
    beta = torch.rand(1, 1, time, dtype=dtype, requires_grad=True)
    gamma = torch.rand(1, 1, time, dtype=dtype, requires_grad=True)
    return destination, routes, [*values, beta, gamma]


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_trapezoidal_reference_matches_independent_oracle_and_every_gradient(
    dtype: torch.dtype,
    collision: bool,
    chunk_size: int,
):
    destination, routes, leaves = _case(
        state=16,
        time=chunk_size + 3,
        collision=collision,
        dtype=dtype,
    )
    mode = NativePDMode.GENERAL_SCATTER if collision else NativePDMode.PERMUTATION_GATHER
    expected_leaves = [leaf.detach().clone().requires_grad_() for leaf in leaves]
    expected = _independent_dense_trapezoid_oracle(destination, routes, *expected_leaves)
    weights = [torch.randn_like(expected[0]), torch.randn_like(expected[1])]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected[:2], weights)),
        expected_leaves,
    )

    actual = trapezoidal_reference_scan(
        destination,
        routes,
        *leaves,
        chunk_size=chunk_size,
        mode=mode,
        return_cache=True,
    )
    actual_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(actual[:2], weights)),
        leaves,
    )

    tolerance = 2e-10 if dtype == torch.float64 else 4e-5
    for actual_output, expected_output in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(actual_output, expected_output, atol=tolerance, rtol=tolerance)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, atol=tolerance, rtol=tolerance
        )
    for actual_cache, expected_cache in zip(actual[2], expected[2]):
        torch.testing.assert_close(actual_cache, expected_cache, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("lam", [0.0, 1.0])
def test_lambda_extremes_and_one_token_cache_match_full_prefill(lam: float):
    destination, routes, leaves = _case(
        state=32,
        time=67,
        collision=True,
        dtype=torch.float64,
    )
    *values, beta, gamma = leaves
    dt = torch.rand_like(beta)
    beta = (1.0 - lam) * dt
    gamma = lam * dt

    full_real, full_imag, full_cache = trapezoidal_reference_scan(
        destination,
        routes,
        *values,
        beta,
        gamma,
        chunk_size=64,
        mode=NativePDMode.GENERAL_SCATTER,
        return_cache=True,
    )

    cache = None
    decoded_real = []
    decoded_imag = []
    for token in range(routes.shape[-1]):
        token_values = [value[:, :, token : token + 1] for value in values]
        out_real, out_imag, cache = trapezoidal_reference_scan(
            destination,
            routes[:, :, token : token + 1],
            *token_values,
            beta[:, :, token : token + 1],
            gamma[:, :, token : token + 1],
            chunk_size=32,
            mode=NativePDMode.GENERAL_SCATTER,
            initial_cache=cache,
            return_cache=True,
        )
        decoded_real.append(out_real)
        decoded_imag.append(out_imag)

    torch.testing.assert_close(torch.cat(decoded_real, dim=2), full_real)
    torch.testing.assert_close(torch.cat(decoded_imag, dim=2), full_imag)
    assert cache is not None
    for decoded_cache, expected_cache in zip(cache, full_cache):
        torch.testing.assert_close(decoded_cache, expected_cache)


@pytest.mark.parametrize(
    ("state", "time", "chunk_size"),
    list(itertools.product((16, 32, 64, 128), (1, 33, 129), (32, 64, 128))),
)
def test_trapezoidal_reference_tails_keep_siso_shape_without_payload_axis(
    state: int,
    time: int,
    chunk_size: int,
):
    destination, routes, leaves = _case(
        state=state,
        time=time,
        collision=True,
        dtype=torch.float32,
    )
    real, imag, cache = trapezoidal_reference_scan(
        destination,
        routes,
        *leaves,
        chunk_size=chunk_size,
        mode=NativePDMode.GENERAL_SCATTER,
        return_cache=True,
    )

    assert real.shape == imag.shape == (1, 1, time, state)
    assert cache.h_real.shape == cache.v_real.shape == (1, 1, state)
    assert real.ndim == 4


def _mixer_config(**kwargs) -> NativeFlashPDMamba3SISOMixerConfig:
    values = dict(
        n_heads=4,
        d_state=8,
        dictionary_size=4,
        chunk_size=32,
        backend="reference",
        dtype=DType.float32,
    )
    values.update(kwargs)
    return NativeFlashPDMamba3SISOMixerConfig(**values)


def _initialized_mixer(
    *, fused: bool, dtype: DType = DType.float32
) -> NativeFlashPDMamba3SISOMixer:
    mixer = _mixer_config(fuse_input_projections=fused, dtype=dtype).build(
        32, layer_idx=1, n_layers=3
    )
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(17),
    )
    return mixer


# Every way a tensor element reaches the host as a Python scalar. Each one drains the
# stream on an accelerator, which is what the temperature tests below forbid.
_SCALAR_READBACKS = ("item", "tolist", "__bool__", "__float__", "__index__", "__int__")


class _ScalarReadbackCounter:
    """Record every device-to-host scalar read made inside the ``with`` block."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._originals: dict[str, Callable] = {}

    def __enter__(self) -> "_ScalarReadbackCounter":
        for name in _SCALAR_READBACKS:
            original = getattr(torch.Tensor, name)
            self._originals[name] = original
            self._install(name, original)
        return self

    def __exit__(self, *exc_info) -> None:
        for name, original in self._originals.items():
            setattr(torch.Tensor, name, original)

    def _install(self, name: str, original: Callable) -> None:
        def wrapper(tensor, *args, **kwargs):
            self.calls.append(name)
            return original(tensor, *args, **kwargs)

        setattr(torch.Tensor, name, wrapper)


def _explicit_mode_mixer(**kwargs) -> NativeFlashPDMamba3SISOMixer:
    """
    Build an initialized mixer whose dispatch mode is pinned.

    ``AUTO`` proves the selected maps bijective on every call and that proof reads a route
    index back to the host, which would mask the temperature reads under measurement.
    """
    mixer = _mixer_config(mode=NativePDMode.GENERAL_SCATTER, **kwargs).build(
        32, layer_idx=0, n_layers=1
    )
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator().manual_seed(23),
    )
    return mixer


def _annealing_mixer_config() -> NativeFlashPDMamba3SISOMixerConfig:
    return _mixer_config(
        mode=NativePDMode.GENERAL_SCATTER,
        dictionary_temperature=2.0,
        router_temperature=1.0,
        dictionary_temperature_end=0.5,
        router_temperature_end=0.25,
        temperature_schedule_steps=100,
    )


def _temperatures_seen_by_scan(mixer: NativeFlashPDMamba3SISOMixer) -> tuple[float, float]:
    """Return the temperatures one forward hands to the scan boundary."""
    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan
    seen: dict[str, float] = {}

    def capture_scan(*args, **kwargs):
        seen["dictionary"] = kwargs["dictionary_temperature"]
        seen["router"] = kwargs["router_temperature"]
        return original_scan(*args, **kwargs)

    mamba3_siso_module.mamba3_siso_surrogate_scan = capture_scan
    try:
        mixer(torch.randn(1, 7, mixer.d_model))
    finally:
        mamba3_siso_module.mamba3_siso_surrogate_scan = original_scan
    return seen["dictionary"], seen["router"]


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
@pytest.mark.parametrize("fused", [True, False])
def test_full_mixer_scan_boundary_receives_contiguous_recurrence_and_selection_tensors(
    monkeypatch,
    fused: bool,
    dtype: DType,
):
    mixer = _initialized_mixer(fused=fused, dtype=dtype)
    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan
    recurrence_tensors = {}
    selection_tensors = {}

    def capture_scan(
        dictionary_logits,
        selector_logits,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        **kwargs,
    ):
        selection_tensors.update(
            dictionary_logits=dictionary_logits,
            selector_logits=selector_logits,
        )
        recurrence_tensors.update(
            diagonal_real=diagonal_real,
            diagonal_imag=diagonal_imag,
            value_real=value_real,
            value_imag=value_imag,
            beta=beta,
            gamma=gamma,
        )
        return original_scan(
            dictionary_logits,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
            **kwargs,
        )

    monkeypatch.setattr(mamba3_siso_module, "mamba3_siso_surrogate_scan", capture_scan)

    x = torch.randn(2, 35, 32, dtype=dtype.as_pt())
    mixer(x)

    assert set(recurrence_tensors) == {
        "diagonal_real",
        "diagonal_imag",
        "value_real",
        "value_imag",
        "beta",
        "gamma",
    }
    for name, tensor in recurrence_tensors.items():
        expected_shape = (2, 4, 35, 8) if tensor.ndim == 4 else (2, 4, 35)
        assert tensor.shape == expected_shape, name
        assert tensor.is_contiguous(), name

    # The CUDA dictionary and router gradients walk both logit tensors by raw pointer
    # under a dense (H,K,N,N) and (B,T,H,K) layout. A fused projection hands the scan a
    # strided split view of one activation, and reading it that way would silently pick
    # up the neighbouring projections instead of the selector.
    assert set(selection_tensors) == {"dictionary_logits", "selector_logits"}
    assert selection_tensors["dictionary_logits"].shape == (4, 4, 8, 8)
    assert selection_tensors["selector_logits"].shape == (2, 35, 4, 4)
    for name, tensor in selection_tensors.items():
        assert tensor.dtype == torch.float32, name
        assert tensor.is_contiguous(), name
        # Read each buffer the way the kernel does, straight off the pointer, and
        # require it to hold the logits rather than whatever else the projection
        # left in between them.
        detached = tensor.detach()
        torch.testing.assert_close(
            torch.as_strided(detached, (detached.numel(),), (1,)),
            detached.flatten(),
            msg=lambda default, name=name: f"pointer walk over {name}: {default}",
        )
    torch.testing.assert_close(
        selection_tensors["selector_logits"],
        mixer._prepare_recurrence(x)[3].float(),
    )


class _NonContiguousCastRecorder(TorchDispatchMode):
    """Record every dtype conversion that materializes a non-contiguous buffer."""

    def __init__(self) -> None:
        super().__init__()
        self.casts: list[tuple[tuple[int, ...], torch.dtype]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten._to_copy.default and isinstance(result, torch.Tensor):
            if not result.is_contiguous():
                self.casts.append((tuple(result.shape), result.dtype))
        return result


def test_bfloat16_scan_boundary_casts_write_contiguous_buffers_once(monkeypatch):
    batch, time, d_model = 2, 33, 32
    mixer = _initialized_mixer(fused=True, dtype=DType.bfloat16)
    x = torch.randn(batch, time, d_model, dtype=torch.bfloat16)
    recorder = _NonContiguousCastRecorder()
    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan
    boundary_casts: list = []

    def capture_scan(*args, **kwargs):
        boundary_casts.extend(recorder.casts)
        return original_scan(*args, **kwargs)

    monkeypatch.setattr(mamba3_siso_module, "mamba3_siso_surrogate_scan", capture_scan)
    with recorder:
        mixer(x)

    # A cast of a permuted operand preserves the permuted strides, so pairing it with a
    # separate contiguous() writes the payload twice and throws the first buffer away.
    # Every conversion feeding the scan has to land in its final layout directly.
    assert boundary_casts == [], f"discarded non-contiguous buffers: {boundary_casts}"


class _CastTargetRecorder(TorchDispatchMode):
    """Record ``(source dtype, target dtype, element count)`` for every conversion."""

    def __init__(self) -> None:
        super().__init__()
        self.casts: list[tuple[torch.dtype, torch.dtype, int]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if func is torch.ops.aten._to_copy.default and isinstance(result, torch.Tensor):
            self.casts.append((args[0].dtype, result.dtype, result.numel()))
        return result


def test_bfloat16_readout_consumes_the_scan_states_without_promoting_them():
    batch, time, d_model = 2, 35, 32
    mixer = _initialized_mixer(fused=True, dtype=DType.bfloat16)
    heads, state = mixer.n_heads, mixer.d_state
    torch.manual_seed(53)
    value_input = torch.randn(batch, time, heads, state)
    gate = torch.randn(batch, time, d_model, dtype=torch.bfloat16)
    c_projection = torch.randn(batch, time, 2 * d_model)
    states_real = torch.randn(batch, heads, time, state, dtype=torch.bfloat16)
    states_imag = torch.randn(batch, heads, time, state, dtype=torch.bfloat16)
    recorder = _CastTargetRecorder()

    with recorder:
        actual = mixer._readout(
            torch.bfloat16, value_input, gate, c_projection, states_real, states_imag
        )

    # The states arrive transposed, so promoting them writes a whole activation in the
    # scan's layout and reads it back in the readout's. Multiplying them against the
    # fp32 C projection promotes them in the same pass and costs nothing.
    expected = mixer._readout(
        torch.bfloat16,
        value_input,
        gate,
        c_projection,
        states_real.float(),
        states_imag.float(),
    )
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    tokens = batch * time * d_model
    # Per-channel norm weights are promoted too, but a `d_state`-wide conversion is not
    # an activation; only conversions the size of a whole activation are the subject
    # here, and exactly two of those are allowed: the gate up and the output back down.
    activation_casts = [cast for cast in recorder.casts if cast[2] == tokens]
    assert activation_casts == [
        (torch.bfloat16, torch.float32, tokens),
        (torch.float32, torch.bfloat16, tokens),
    ], recorder.casts


def test_bfloat16_recurrence_promotes_each_full_size_activation_at_most_once():
    batch, time, d_model = 2, 35, 32
    mixer = _initialized_mixer(fused=True, dtype=DType.bfloat16)
    heads, state = mixer.n_heads, mixer.d_state
    x = torch.randn(batch, time, d_model, dtype=torch.bfloat16)
    activation = batch * time * heads * state

    def promotions(run) -> list:
        recorder = _CastTargetRecorder()
        with recorder:
            run()
        return [
            cast for cast in recorder.casts if cast[1] == torch.float32 and cast[2] == activation
        ]

    value_input, gate, c_projection = mixer._prepare_recurrence(x)[:3]
    states = torch.randn(batch, heads, time, state, dtype=torch.bfloat16)
    prologue = promotions(lambda: mixer._prepare_recurrence(x))
    readout = promotions(
        lambda: mixer._readout(torch.bfloat16, value_input, gate, c_projection, states, states)
    )

    # An operand that meets an fp32 tensor in a product or a sum is promoted by that
    # kernel, so a separate conversion in front of it writes a full activation for
    # nothing. The prologue is left with BCNorm reading B, the value input, and the
    # phase logits behind the diagonal. C is not among them: nothing in the recurrence
    # reads it, so the readout owns it and promotes it there, beside the gate.
    assert len(prologue) == 4, prologue
    assert len(readout) == 4, readout


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_prepared_recurrence_builds_the_complex_diagonal_in_the_scan_layout(dtype: DType):
    batch, time, d_model = 2, 35, 32
    mixer = _initialized_mixer(fused=True, dtype=dtype)
    x = torch.randn(batch, time, d_model, dtype=dtype.as_pt())

    (
        _value_input,
        _gate,
        _c_projection,
        _selector_logits,
        diagonal_real,
        diagonal_imag,
        _value_real,
        _value_imag,
        beta,
        gamma,
    ) = mixer._prepare_recurrence(x)

    # Every factor behind the diagonal is either per-head or broadcast over the state,
    # so the whole chain can be evaluated head-major and land in the layout the scan
    # reads. Transposing the two finished planes instead rewrites them both.
    for name, tensor in (("diagonal_real", diagonal_real), ("diagonal_imag", diagonal_imag)):
        assert tensor.shape == (batch, mixer.n_heads, time, mixer.d_state), name
        assert tensor.dtype == torch.float32, name
        assert tensor.is_contiguous(), name
    for name, tensor in (("beta", beta), ("gamma", gamma)):
        assert tensor.shape == (batch, mixer.n_heads, time), name
        assert tensor.dtype == torch.float32, name
        assert tensor.is_contiguous(), name


def test_float32_forward_hands_the_scan_the_prepared_operands_unchanged(monkeypatch):
    mixer = _initialized_mixer(fused=True)
    x = torch.randn(2, 35, 32)
    prepared: dict[str, torch.Tensor] = {}
    scanned: dict[str, torch.Tensor] = {}
    original_prepare = mixer._prepare_recurrence
    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan

    def capture_prepare(value):
        result = original_prepare(value)
        prepared.update(
            diagonal_real=result[4],
            diagonal_imag=result[5],
            beta=result[8],
            gamma=result[9],
        )
        return result

    def capture_scan(
        dictionary_logits,
        selector_logits,
        diagonal_real,
        diagonal_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        **kwargs,
    ):
        scanned.update(
            diagonal_real=diagonal_real,
            diagonal_imag=diagonal_imag,
            beta=beta,
            gamma=gamma,
        )
        return original_scan(
            dictionary_logits,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
            **kwargs,
        )

    monkeypatch.setattr(mixer, "_prepare_recurrence", capture_prepare)
    monkeypatch.setattr(mamba3_siso_module, "mamba3_siso_surrogate_scan", capture_scan)
    mixer(x)

    # Nothing may stand between where an operand is built and where the kernel reads
    # it. In the payload dtype the recurrence already computes in, a permute paired
    # with a contiguous() there is a second full write of a finished tensor.
    for name, tensor in scanned.items():
        assert tensor is prepared[name], name


def _token_major_mixer_oracle(mixer: NativeFlashPDMamba3SISOMixer, x: torch.Tensor) -> torch.Tensor:
    """Recompute the fused mixer from its parameters, token-major throughout."""
    batch, time, _ = x.shape
    heads, state = mixer.n_heads, mixer.d_state
    assert mixer.in_proj is not None
    (
        value_input,
        gate,
        b_projection,
        c_projection,
        selector_logits,
        dt_logits,
        phase_logits,
        lambda_logits,
    ) = mixer.in_proj(x).split(mixer._projection_sizes(), dim=-1)
    b_real, b_imag = b_projection.view(batch, time, heads, state, 2).unbind(dim=-1)
    c_real, c_imag = c_projection.view(batch, time, heads, state, 2).unbind(dim=-1)
    b_real, b_imag = mamba3_siso_module._complex_rms_norm(
        b_real, b_imag, mixer.bc_norm_b, mixer.norm_eps
    )
    c_real, c_imag = mamba3_siso_module._complex_rms_norm(
        c_real, c_imag, mixer.bc_norm_c, mixer.norm_eps
    )
    b_real = b_real.float() + mixer.B_bias.view(1, 1, heads, state)
    c_real = c_real.float() + mixer.C_bias.view(1, 1, heads, state)
    value_input = value_input.view(batch, time, heads, state).float()
    dt = F.softplus(dt_logits.float() + mixer.dt_bias.view(1, 1, heads))
    lam = torch.sigmoid(lambda_logits.float())
    magnitude = torch.exp(-dt[..., None] * torch.exp(mixer.A_log).view(1, 1, heads, 1))
    theta = math.pi * torch.tanh(phase_logits.float().view(batch, time, heads, state))
    phase = dt[..., None] * theta
    payload = x.dtype if x.dtype in (torch.float32, torch.bfloat16) else torch.float32
    states_real, states_imag = mamba3_siso_module.mamba3_siso_surrogate_scan(
        mixer.dictionary_logits.float(),
        selector_logits.view(batch, time, heads, mixer.dictionary_size).float().contiguous(),
        (magnitude * torch.cos(phase)).permute(0, 2, 1, 3).contiguous(),
        (magnitude * torch.sin(phase)).permute(0, 2, 1, 3).contiguous(),
        (b_real * value_input).permute(0, 2, 1, 3).to(payload).contiguous(),
        (b_imag.float() * value_input).permute(0, 2, 1, 3).to(payload).contiguous(),
        ((1.0 - lam) * dt).permute(0, 2, 1).to(payload).contiguous(),
        (lam * dt).permute(0, 2, 1).to(payload).contiguous(),
        dictionary_temperature=mixer.dictionary_temperature,
        router_temperature=mixer.router_temperature,
        chunk_size=mixer.chunk_size,
        mode=mixer.mode,
        backend=mixer.backend,
    )
    readout = (
        c_real * states_real.permute(0, 2, 1, 3).float()
        - c_imag.float() * states_imag.permute(0, 2, 1, 3).float()
    )
    y = readout + mixer.D.view(1, 1, heads, 1) * value_input
    y = y * F.silu(gate.view(batch, time, heads, state).float())
    return mixer.out_proj(y.reshape(batch, time, mixer.d_model).to(x.dtype))


@pytest.mark.parametrize("dtype", [DType.float32, DType.bfloat16])
def test_mixer_forward_and_every_gradient_match_the_token_major_oracle(dtype: DType):
    mixer = _initialized_mixer(fused=True, dtype=dtype)
    torch.manual_seed(37)
    x = torch.randn(2, 35, 32, dtype=dtype.as_pt(), requires_grad=True)
    weight = torch.randn(2, 35, 32, dtype=dtype.as_pt())
    leaves = (x, *mixer.parameters())
    names = ["input", *(name for name, _ in mixer.named_parameters())]

    actual = mixer(x)
    actual_gradients = torch.autograd.grad((actual * weight).sum(), leaves)
    expected = _token_major_mixer_oracle(mixer, x)
    expected_gradients = torch.autograd.grad((expected * weight).sum(), leaves)

    # The forward is the same arithmetic on the same values and has to agree bit for
    # bit. The gradients cannot: dt, the value input and the projections each feed
    # several consumers, and a re-expressed graph sums those contributions in its own
    # order, so the two answers differ by the rounding of that sum.
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(names) == len(actual_gradients)
    for name, actual_gradient, expected_gradient in zip(
        names, actual_gradients, expected_gradients
    ):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            msg=lambda default, name=name: f"gradient of {name}: {default}",
        )


def test_bfloat16_mixer_keeps_complex_diagonal_fp32_at_scan_boundary(monkeypatch):
    decay_steps = 4096
    time = decay_steps + 1
    state = 8
    mixer = _mixer_config(
        n_heads=1,
        d_state=state,
        dictionary_size=1,
        dtype=DType.bfloat16,
    ).build(state, layer_idx=0, n_layers=1)
    with torch.no_grad():
        mixer.dictionary_logits.fill_(-1)
        index = torch.arange(state)
        mixer.dictionary_logits[0, 0, index, index] = 1

    # The payload stays token-major and is transposed at the boundary; the diagonal and
    # the trapezoidal weights are built head-major and handed over as they are.
    recurrence_shape = (1, time, 1, state)
    scan_shape = (1, 1, time, state)
    payload_dtype = torch.bfloat16
    diagonal_real = torch.full(
        scan_shape,
        math.exp(-5e-4),
        dtype=torch.float32,
    )
    diagonal_imag = torch.zeros_like(diagonal_real)
    value_real = torch.zeros(recurrence_shape, dtype=payload_dtype)
    value_imag = torch.zeros_like(value_real)
    value_real[:, 0] = 1
    beta = torch.zeros((1, 1, time), dtype=payload_dtype)
    gamma = torch.zeros_like(beta)
    gamma[..., 0] = 1
    captured = {}

    def prepare_recurrence(x):
        value_input = torch.zeros_like(value_real)
        gate = x.new_zeros(x.shape)
        c_projection = x.new_zeros((1, time, 2 * state))
        selector_logits = x.new_zeros((1, time, 1, 1))
        return (
            value_input,
            gate,
            c_projection,
            selector_logits,
            diagonal_real,
            diagonal_imag,
            value_real,
            value_imag,
            beta,
            gamma,
        )

    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan

    def capture_scan(
        dictionary_logits,
        selector_logits,
        scan_diagonal_real,
        scan_diagonal_imag,
        scan_value_real,
        scan_value_imag,
        scan_beta,
        scan_gamma,
        **kwargs,
    ):
        captured.update(
            diagonal_real=scan_diagonal_real,
            diagonal_imag=scan_diagonal_imag,
            value_real=scan_value_real,
            value_imag=scan_value_imag,
            beta=scan_beta,
            gamma=scan_gamma,
        )
        result = original_scan(
            dictionary_logits,
            selector_logits,
            scan_diagonal_real,
            scan_diagonal_imag,
            scan_value_real,
            scan_value_imag,
            scan_beta,
            scan_gamma,
            **kwargs,
        )
        captured["output_real"] = result[0]
        captured["output_imag"] = result[1]
        return result

    def readout(
        x_dtype,
        value_input,
        gate,
        c_projection,
        states_real,
        states_imag,
    ):
        del value_input, gate, c_projection, states_imag
        return states_real.permute(0, 2, 1, 3).reshape(1, time, state).to(x_dtype)

    monkeypatch.setattr(mixer, "_prepare_recurrence", prepare_recurrence)
    monkeypatch.setattr(mixer, "_readout", readout)
    monkeypatch.setattr(mamba3_siso_module, "mamba3_siso_surrogate_scan", capture_scan)

    output = mixer(torch.zeros((1, time, state), dtype=payload_dtype))

    assert captured["diagonal_real"].dtype == torch.float32
    assert captured["diagonal_imag"].dtype == torch.float32
    for name in ("value_real", "value_imag", "beta", "gamma", "output_real", "output_imag"):
        assert captured[name].dtype == payload_dtype, name
    expected_decay = math.exp(-5e-4 * decay_steps)
    actual_decay = output[0, -1, 0].float().item()
    assert actual_decay == pytest.approx(expected_decay, rel=5e-3, abs=5e-4)
    assert actual_decay != pytest.approx(1.0, abs=0.1)


def test_transformer_init_weights_restores_configured_siso_temperatures_and_zero_step():
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        d_model=32,
        vocab_size=64,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=_mixer_config(
                dictionary_temperature=2.0,
                router_temperature=0.5,
                dictionary_temperature_end=0.25,
                router_temperature_end=0.125,
                temperature_schedule_steps=100,
            ),
            layer_norm=norm,
            feed_forward=FeedForwardConfig(hidden_size=64, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False),
    )
    model = config.build(init_device="meta")

    model.init_weights(device=torch.device("cpu"))

    mixer = model.blocks["0"].attention
    assert isinstance(mixer, NativeFlashPDMamba3SISOMixer)
    assert mixer.temperature_schedule_state() == {
        "step": 0,
        "dictionary_temperature": 2.0,
        "router_temperature": 0.5,
    }


def test_mamba3_siso_config_is_stable_distinct_and_keeps_baseline_semantics():
    baseline = NativeFlashPDMixerConfig(
        n_heads=4,
        d_state=8,
        dictionary_size=4,
        chunk_size=32,
        backend="reference",
    )
    upgraded = _mixer_config()
    baseline_module = baseline.build(32, layer_idx=0, n_layers=1)
    upgraded_meta = upgraded.build(32, layer_idx=0, n_layers=1, init_device="meta")

    assert baseline.as_config_dict()["type"] == "flash_pd_native"
    assert upgraded.as_config_dict()["type"] == "flash_pd_native_mamba3_siso"
    assert hasattr(baseline_module, "conv")
    assert baseline_module.conv_kernel_size == 4
    assert not hasattr(upgraded_meta, "conv")
    assert all(parameter.device.type == "meta" for parameter in upgraded_meta.parameters())
    assert upgraded.num_params(32) == sum(
        parameter.numel() for parameter in upgraded_meta.parameters()
    )
    assert upgraded_meta.state_contract == ("batch", "head", "time", "state")
    for forbidden in ("mimo_rank", "mimo_x", "mimo_o", "payload_rank"):
        assert not hasattr(upgraded_meta, forbidden)


def test_mamba3_siso_faithful_defaults_forward_backward_and_bias_initialization():
    mixer = _initialized_mixer(fused=True)
    x = torch.randn(2, 35, 32, requires_grad=True)

    output = mixer(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert mixer.conv_kernel_size is None
    assert mixer.bc_norm_enabled is True
    assert mixer.output_norm_enabled is False
    torch.testing.assert_close(mixer.B_bias, torch.ones_like(mixer.B_bias))
    torch.testing.assert_close(mixer.C_bias, torch.ones_like(mixer.C_bias))
    for name, parameter in mixer.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


def test_fused_and_unfused_projection_checkpoints_convert_both_directions_exactly():
    unfused = _initialized_mixer(fused=False)
    fused = _mixer_config(fuse_input_projections=True).build(32, layer_idx=1, n_layers=3)
    fused.load_state_dict(unfused.state_dict(), strict=True)
    x = torch.randn(2, 19, 32)

    expected = unfused(x)
    actual = fused(x)
    torch.testing.assert_close(actual, expected)
    assert hasattr(fused, "in_proj")
    assert fused.in_proj is not None
    assert all(
        projection is None
        for projection in (
            fused.in_x,
            fused.in_z,
            fused.B_proj,
            fused.C_proj,
            fused.selector_proj,
            fused.dt_proj,
            fused.phase_proj,
            fused.lambda_proj,
        )
    )

    rebuilt_unfused = _mixer_config(fuse_input_projections=False).build(32, layer_idx=1, n_layers=3)
    rebuilt_unfused.load_state_dict(fused.state_dict(), strict=True)
    torch.testing.assert_close(rebuilt_unfused(x), expected)


def test_fused_and_unfused_projection_layouts_initialize_identically_from_one_seed():
    unfused = _initialized_mixer(fused=False)
    fused = _initialized_mixer(fused=True)
    assert fused.in_proj is not None

    stacked = torch.cat(
        [getattr(unfused, name).weight for name in unfused._UNFUSED_PROJECTIONS], dim=0
    )
    assert torch.equal(fused.in_proj.weight, stacked)


def _fused_projection_slice_stds(
    mixer: NativeFlashPDMamba3SISOMixer, weight: torch.Tensor
) -> dict[str, float]:
    """Return the sample standard deviation of every logical slice of a fused weight."""
    return {
        name: chunk.std().item()
        for name, chunk in zip(
            mixer._UNFUSED_PROJECTIONS,
            weight.split(mixer._projection_sizes(), dim=0),
        )
    }


def _run_sharded_fused_projection_init(world_size: int, reference: dict[str, float]) -> None:
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import DTensor, init_device_mesh

    from olmo_core.distributed.utils import get_full_tensor

    mixer = _mixer_config(fuse_input_projections=True).build(32, layer_idx=1, n_layers=3)
    fully_shard(mixer, mesh=init_device_mesh("cpu", (world_size,)))
    assert mixer.in_proj is not None
    # The defect only exists once the fused weight is sharded on the dimension the
    # logical projection sizes address, which is exactly what training does before it
    # initializes anything.
    assert isinstance(mixer.in_proj.weight, DTensor)
    assert mixer.in_proj.weight.to_local().shape[0] < sum(mixer._projection_sizes())

    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(17),
    )

    observed = _fused_projection_slice_stds(mixer, get_full_tensor(mixer.in_proj.weight.detach()))
    for name, expected in reference.items():
        assert observed[name] == pytest.approx(expected, rel=0.25), (name, observed, reference)
    assert observed["phase_proj"] < 0.2 * min(
        value for name, value in observed.items() if name != "phase_proj"
    )


def test_sharded_fused_projection_gives_every_slice_its_own_standard_deviation():
    reference_mixer = _initialized_mixer(fused=True)
    assert reference_mixer.in_proj is not None
    reference = _fused_projection_slice_stds(
        reference_mixer, reference_mixer.in_proj.weight.detach()
    )

    run_distributed_test(
        _run_sharded_fused_projection_init,
        world_size=2,
        backend="gloo",
        func_args=(2, reference),
    )


def test_mixer_tiny_prefill_then_one_token_decode_matches_full_sequence():
    mixer = _initialized_mixer(fused=True).eval()
    x = torch.randn(2, 11, 32)

    full_output, full_cache = mixer.forward_with_cache(x)
    prefill_output, cache = mixer.forward_with_cache(x[:, :4])
    decoded = [prefill_output]
    for token in range(4, x.shape[1]):
        token_output, cache = mixer.decode_step(x[:, token : token + 1], cache)
        decoded.append(token_output)

    torch.testing.assert_close(torch.cat(decoded, dim=1), full_output)
    torch.testing.assert_close(mixer(x), full_output)
    for actual, expected in zip(cache, full_cache):
        torch.testing.assert_close(actual, expected)
    assert cache.h_real.shape == cache.v_real.shape == (2, 4, 8)
    assert len(tuple(cache)) == 4


def test_mixer_cache_path_keeps_packed_documents_fail_closed():
    mixer = _initialized_mixer(fused=True)
    with pytest.raises(NotImplementedError, match="packed"):
        mixer.forward_with_cache(
            torch.randn(1, 5, 32),
            cu_doc_lens=torch.tensor([0, 2, 5], dtype=torch.int32),
        )


def test_selector_telemetry_reports_entropy_dead_entries_ties_and_churn():
    mixer = _mixer_config(n_heads=1, d_state=32).build(32, layer_idx=0, n_layers=1)
    logits = torch.tensor(
        [
            [
                [[2.0, 1.0, 0.0, 0.0]],
                [[0.0, 2.0, 0.0, 0.0]],
                [[2.0, 2.0, 0.0, 0.0]],
                [[2.0, 1.0, 0.0, 0.0]],
            ]
        ]
    )
    original = logits.clone()

    telemetry = mixer.selector_telemetry(logits)

    torch.testing.assert_close(logits, original)
    assert telemetry.route_entropy.ndim == 0
    assert 0 < telemetry.route_entropy < math.log(4)
    assert telemetry.dead_entries.item() == 2
    assert telemetry.ties.item() == 1
    torch.testing.assert_close(telemetry.route_churn, torch.tensor(2 / 3))
    assert not hasattr(mixer, "load_balance_loss")


def test_separate_temperature_schedule_state_roundtrips_and_resumes_exactly():
    config = _mixer_config(
        dictionary_temperature=2.0,
        router_temperature=1.0,
        dictionary_temperature_end=0.5,
        router_temperature_end=0.25,
        temperature_schedule_steps=100,
    )
    mixer = config.build(32, layer_idx=0, n_layers=1)
    mixer.set_temperature_schedule_step(40)
    state = mixer.temperature_schedule_state()

    assert state == {
        "step": 40,
        "dictionary_temperature": pytest.approx(1.4),
        "router_temperature": pytest.approx(0.7),
    }
    rebuilt = config.build(32, layer_idx=0, n_layers=1)
    rebuilt.load_state_dict(mixer.state_dict(), strict=True)
    assert rebuilt.temperature_schedule_state() == state

    rebuilt.set_temperature_schedule_step(100)
    assert rebuilt.dictionary_temperature == pytest.approx(0.5)
    assert rebuilt.router_temperature == pytest.approx(0.25)


def test_mixer_forward_reads_temperatures_without_any_host_scalar_readback():
    mixer = _explicit_mode_mixer()
    x = torch.randn(2, 9, 32)

    with _ScalarReadbackCounter() as counter:
        output = mixer(x)
        temperatures = (mixer.dictionary_temperature, mixer.router_temperature)

    assert counter.calls == []
    assert temperatures == (1.0, 1.0)
    assert output.shape == x.shape


def test_mixer_forward_compiles_fullgraph_through_the_reference_scan():
    mixer = _explicit_mode_mixer()
    x = torch.randn(1, 9, 32)
    torch._dynamo.reset()

    # Grad-mode tracing of this path is blocked by a separate dynamo limitation on the
    # reference scan's autograd.Function, so inference is what the temperatures gate.
    with torch.no_grad():
        expected = mixer(x)
        actual = torch.compile(mixer, backend="eager", fullgraph=True)(x)

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("step", "dictionary", "router"),
    [(0, 2.0, 1.0), (40, 1.4, 0.7), (100, 0.5, 0.25), (250, 0.5, 0.25)],
)
def test_temperature_schedule_step_moves_buffers_and_hot_path_values_together(
    step: int,
    dictionary: float,
    router: float,
):
    mixer = _annealing_mixer_config().build(32, layer_idx=0, n_layers=1)

    mixer.set_temperature_schedule_step(step)
    with _ScalarReadbackCounter() as counter:
        temperatures = (mixer.dictionary_temperature, mixer.router_temperature)

    assert counter.calls == []
    assert temperatures == (pytest.approx(dictionary), pytest.approx(router))
    assert mixer._dictionary_temperature.item() == temperatures[0]
    assert mixer._router_temperature.item() == temperatures[1]
    assert mixer.temperature_schedule_state() == {
        "step": step,
        "dictionary_temperature": temperatures[0],
        "router_temperature": temperatures[1],
    }
    assert _temperatures_seen_by_scan(mixer) == temperatures


def test_resumed_mixer_takes_hot_path_temperatures_from_the_persisted_buffers():
    config = _annealing_mixer_config()
    mixer = config.build(32, layer_idx=0, n_layers=1)
    mixer.set_temperature_schedule_step(40)
    checkpoint = {name: tensor.clone() for name, tensor in mixer.state_dict().items()}
    resumed = config.build(32, layer_idx=0, n_layers=1)
    assert resumed.dictionary_temperature == pytest.approx(2.0)

    resumed.load_state_dict(checkpoint, strict=True)
    with _ScalarReadbackCounter() as counter:
        temperatures = (resumed.dictionary_temperature, resumed.router_temperature)

    assert counter.calls == []
    assert temperatures == (mixer.dictionary_temperature, mixer.router_temperature)
    assert resumed._dictionary_temperature.item() == temperatures[0]
    assert resumed._router_temperature.item() == temperatures[1]
    assert resumed.temperature_schedule_state() == mixer.temperature_schedule_state()
    assert _temperatures_seen_by_scan(resumed) == _temperatures_seen_by_scan(mixer)

    resumed.set_temperature_schedule_step(100)
    assert _temperatures_seen_by_scan(resumed) == (
        pytest.approx(0.5),
        pytest.approx(0.25),
    )


@pytest.mark.filterwarnings("ignore:.*copying from a non-meta parameter.*:UserWarning")
def test_loading_into_an_unmaterialized_meta_mixer_leaves_hot_path_and_buffers_agreeing():
    config = _annealing_mixer_config()
    mixer = config.build(32, layer_idx=0, n_layers=1)
    mixer.set_temperature_schedule_step(40)
    unmaterialized = config.build(32, layer_idx=0, n_layers=1, init_device="meta")

    # Copying into meta storage is a no-op, so there is nothing to mirror and the
    # configured start temperatures are still what both sides hold.
    unmaterialized.load_state_dict(mixer.state_dict(), strict=True)

    assert unmaterialized.dictionary_temperature == pytest.approx(2.0)
    assert unmaterialized.router_temperature == pytest.approx(1.0)


@pytest.mark.gpu
def test_cuda_mixer_temperature_hot_path_never_drains_the_stream():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    mixer = _annealing_mixer_config().build(32, layer_idx=0, n_layers=1, init_device="cuda")
    mixer.set_temperature_schedule_step(40)

    with _ScalarReadbackCounter() as counter:
        mixer(torch.randn(1, 9, 32, device="cuda"))
    torch.cuda.synchronize()
    previous_mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        temperatures = (mixer.dictionary_temperature, mixer.router_temperature)
    finally:
        torch.cuda.set_sync_debug_mode(previous_mode)

    assert counter.calls == []
    assert temperatures == (pytest.approx(1.4), pytest.approx(0.7))


@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_exact_parameter_flop_saved_tensor_and_workspace_accounting(chunk_size: int):
    mixer = _mixer_config(chunk_size=chunk_size).build(32, layer_idx=0, n_layers=1)
    batch, time, dtype_bytes = 2, 257, 2
    accounting = mixer.accounting(
        batch_size=batch,
        sequence_length=time,
        element_size=dtype_bytes,
    )
    chunks = (time + chunk_size - 1) // chunk_size
    rows = batch * mixer.n_heads
    chunk_state = rows * chunks * mixer.d_state
    sequence_state = rows * time * mixer.d_state

    assert accounting.parameters == sum(parameter.numel() for parameter in mixer.parameters())
    assert accounting.flops_per_token == mixer.num_flops_per_token(time)
    assert accounting.model_flops_per_sequence == accounting.flops_per_token * time
    assert accounting.route_comparisons_per_sequence == (
        mixer.n_heads * mixer.dictionary_size * mixer.d_state * (mixer.d_state - 1)
        + time * mixer.n_heads * (mixer.dictionary_size - 1)
    )
    assert accounting.nonlinear_evaluations_per_sequence == time * (
        5 * mixer.n_heads + 4 * mixer.d_model
    )
    assert accounting.forward_workspace_bytes == (
        2 * sequence_state * dtype_bytes + 36 * chunk_state
    )
    assert accounting.backward_workspace_bytes == (
        26 * chunk_state + 4 * (mixer.n_heads * mixer.dictionary_size * mixer.d_state + rows * time)
    )
    expected_saved = (
        8 * sequence_state
        + 4 * sequence_state * dtype_bytes
        + 2 * rows * time * dtype_bytes
        + 4
        * (
            mixer.n_heads * mixer.dictionary_size * mixer.d_state * mixer.d_state
            + batch * time * mixer.n_heads * mixer.dictionary_size
        )
        + 2 * (mixer.n_heads * mixer.dictionary_size * mixer.d_state + rows * time)
    )
    assert accounting.saved_tensor_bytes == expected_saved
    assert accounting.peak_workspace_bytes == max(
        accounting.forward_workspace_bytes,
        accounting.backward_workspace_bytes,
    )
