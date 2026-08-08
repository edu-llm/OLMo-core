import itertools
import math

import pytest
import torch
from torch.nn import functional as F

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


def _initialized_mixer(*, fused: bool) -> NativeFlashPDMamba3SISOMixer:
    mixer = _mixer_config(fuse_input_projections=fused).build(32, layer_idx=1, n_layers=3)
    mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(17),
    )
    return mixer


def test_full_mixer_scan_boundary_receives_contiguous_bht_recurrence_tensors(monkeypatch):
    mixer = _initialized_mixer(fused=True)
    original_scan = mamba3_siso_module.mamba3_siso_surrogate_scan
    recurrence_tensors = {}

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

    mixer(torch.randn(2, 35, 32))

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
        6 * sequence_state * dtype_bytes
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
