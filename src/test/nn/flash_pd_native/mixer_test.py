import json
import math

import pytest
import torch
import torch.nn as nn

import olmo_core.nn.flash_pd_native.mixer as mixer_module
from olmo_core.config import DType
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.flash_pd_native.contracts import NativePDMode
from olmo_core.nn.flash_pd_native.mixer import (
    NativeFlashPDMixer,
    NativeFlashPDMixerConfig,
)
from olmo_core.nn.layer_norm import LayerNormConfig, LayerNormType
from olmo_core.nn.lm_head import LMHeadConfig
from olmo_core.nn.transformer import (
    InitMethod,
    TransformerBlockConfig,
    TransformerBlockType,
    TransformerConfig,
)

POST_CONVOLUTION_PROJECTIONS = ("B_proj", "C_proj", "selector_proj", "dt_proj", "phase_proj")
SHARED_PARAMETERS = (
    "dictionary_logits",
    "in_proj.weight",
    "conv.weight",
    "conv.bias",
    "out_proj.weight",
    "A_log",
    "dt_bias",
    "D",
)


def _config(**kwargs) -> NativeFlashPDMixerConfig:
    values = dict(
        n_heads=4,
        d_state=8,
        dictionary_size=4,
        chunk_size=8,
        backend="reference",
        dtype=DType.float32,
    )
    values.update(kwargs)
    return NativeFlashPDMixerConfig(**values)


def _projection_slice(mixer: NativeFlashPDMixer, name: str) -> slice:
    """Locate one post-convolution projection's rows inside the fused weight."""
    sizes = mixer._projection_sizes()
    index = mixer._UNFUSED_PROJECTIONS.index(name)
    start = sum(sizes[:index])
    return slice(start, start + sizes[index])


def _projection_weight(mixer: NativeFlashPDMixer, name: str) -> torch.Tensor:
    """Return one post-convolution projection's weight rows in either layout."""
    if mixer.fuse_input_projections:
        assert mixer.u_proj is not None
        return mixer.u_proj.weight[_projection_slice(mixer, name)]
    return getattr(mixer, name).weight


def _projection_grad(mixer: NativeFlashPDMixer, name: str):
    """Return one post-convolution projection's weight gradient in either layout."""
    if mixer.fuse_input_projections:
        assert mixer.u_proj is not None
        gradient = mixer.u_proj.weight.grad
        return None if gradient is None else gradient[_projection_slice(mixer, name)]
    return getattr(mixer, name).weight.grad


def _initialized_mixer(
    *,
    fused: bool,
    d_model: int = 32,
    seed: int = 19,
    **kwargs,
) -> NativeFlashPDMixer:
    module = _config(fuse_input_projections=fused, **kwargs).build(d_model, layer_idx=1, n_layers=3)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=d_model,
        block_idx=1,
        num_blocks=3,
        generator=torch.Generator().manual_seed(seed),
    )
    return module


def _linear_call_names(mixer: NativeFlashPDMixer, x: torch.Tensor) -> list[str]:
    """Record, in order, every ``nn.Linear`` the forward actually invokes."""
    names: list[str] = []
    handles = [
        module.register_forward_hook(lambda _module, _args, _output, name=name: names.append(name))
        for name, module in mixer.named_modules()
        if isinstance(module, nn.Linear)
    ]
    try:
        mixer(x)
    finally:
        for handle in handles:
            handle.remove()
    return names


@pytest.mark.parametrize("fused", [False, True])
def test_config_roundtrip_meta_parameter_count_and_vector_state_contract(fused: bool):
    config = _config(ste_temperature=0.7, fuse_input_projections=fused)
    rebuilt = NativeFlashPDMixerConfig.from_dict(json.loads(json.dumps(config.as_config_dict())))
    module = rebuilt.build(32, layer_idx=0, n_layers=2, init_device="meta")

    assert rebuilt == config
    assert rebuilt.fuse_input_projections is fused
    assert isinstance(module, NativeFlashPDMixer)
    assert all(parameter.device.type == "meta" for parameter in module.parameters())
    assert rebuilt.num_params(32) == sum(parameter.numel() for parameter in module.parameters())
    assert module.state_contract == ("batch", "head", "time", "state")
    assert "payload" not in module.state_contract
    assert module.num_flops_per_token(128) > 0


def test_fusing_the_post_convolution_projections_is_parameter_count_neutral():
    unfused = _config(fuse_input_projections=False)
    fused = _config(fuse_input_projections=True)
    unfused_module = unfused.build(32, layer_idx=0, n_layers=2, init_device="meta")
    fused_module = fused.build(32, layer_idx=0, n_layers=2, init_device="meta")

    # The shipped `native-pd` arm names no flag, so whatever this default produces is what it
    # trains. Pinned here so the default cannot move without someone reading why it is where
    # it is: unfused, because `split`'s gradient is a `cat` over the whole fused projection
    # width and Inductor lowers it to one kernel that evaluates every branch for every
    # element. That is worth 5.3% of a compiled forward and backward at the production shape,
    # and it is free to take -- the two layouts hold the same weights, draw bit-identically
    # from one seed, and convert to each other exactly, which the assertions below and the
    # two tests after this one are what actually hold the parameter count still.
    assert NativeFlashPDMixerConfig().fuse_input_projections is False
    assert unfused.num_params(32) == fused.num_params(32)
    assert sum(parameter.numel() for parameter in fused_module.parameters()) == unfused.num_params(
        32
    )
    assert sum(parameter.numel() for parameter in unfused_module.parameters()) == fused.num_params(
        32
    )
    assert fused_module.u_proj is not None
    assert fused_module.u_proj.weight.shape == (sum(fused_module._projection_sizes()), 32)
    assert fused_module.u_proj.bias is None
    assert fused_module.num_flops_per_token(128) == unfused_module.num_flops_per_token(128)


def test_projection_layouts_initialize_identically_from_one_seed():
    unfused = _initialized_mixer(fused=False)
    fused = _initialized_mixer(fused=True)

    for name in POST_CONVOLUTION_PROJECTIONS:
        assert torch.equal(_projection_weight(fused, name), _projection_weight(unfused, name)), name
    unfused_parameters = dict(unfused.named_parameters())
    fused_parameters = dict(fused.named_parameters())
    for name in SHARED_PARAMETERS:
        assert torch.equal(fused_parameters[name], unfused_parameters[name]), name
    assert set(unfused_parameters) == set(SHARED_PARAMETERS) | {
        f"{name}.weight" for name in POST_CONVOLUTION_PROJECTIONS
    }
    assert set(fused_parameters) == set(SHARED_PARAMETERS) | {"u_proj.weight"}


def test_fused_and_unfused_projection_layouts_match_forward_and_every_gradient():
    unfused = _initialized_mixer(fused=False)
    fused = _initialized_mixer(fused=True)
    x = torch.randn(2, 13, 32)
    unfused_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)

    unfused_output = unfused(unfused_x)
    fused_output = fused(fused_x)
    unfused_output.square().mean().backward()
    fused_output.square().mean().backward()

    torch.testing.assert_close(fused_output, unfused_output, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(fused_x.grad, unfused_x.grad, rtol=1e-6, atol=1e-7)
    for name in POST_CONVOLUTION_PROJECTIONS:
        torch.testing.assert_close(
            _projection_grad(fused, name),
            _projection_grad(unfused, name),
            rtol=1e-6,
            atol=1e-7,
            msg=lambda text, name=name: f"{name}: {text}",
        )
    unfused_parameters = dict(unfused.named_parameters())
    fused_parameters = dict(fused.named_parameters())
    for name in SHARED_PARAMETERS:
        torch.testing.assert_close(
            fused_parameters[name].grad,
            unfused_parameters[name].grad,
            rtol=1e-6,
            atol=1e-7,
            msg=lambda text, name=name: f"{name}: {text}",
        )


def test_fused_and_unfused_projection_checkpoints_convert_both_directions_exactly():
    unfused = _initialized_mixer(fused=False, seed=23)
    fused = _initialized_mixer(fused=True, seed=29)
    unfused_checkpoint = unfused.state_dict()
    fused_checkpoint = fused.state_dict()

    assert "u_proj.weight" in fused_checkpoint
    assert not any(f"{name}.weight" in fused_checkpoint for name in POST_CONVOLUTION_PROJECTIONS)
    assert "u_proj.weight" not in unfused_checkpoint
    assert all(f"{name}.weight" in unfused_checkpoint for name in POST_CONVOLUTION_PROJECTIONS)

    into_fused = _config(fuse_input_projections=True).build(32, layer_idx=1, n_layers=3)
    into_unfused = _config(fuse_input_projections=False).build(32, layer_idx=1, n_layers=3)
    into_fused.load_state_dict(unfused_checkpoint)
    into_unfused.load_state_dict(fused_checkpoint)

    for name in POST_CONVOLUTION_PROJECTIONS:
        converted_to_fused = _projection_weight(into_fused, name)
        converted_to_unfused = _projection_weight(into_unfused, name)
        assert torch.equal(converted_to_fused, _projection_weight(unfused, name)), name
        assert torch.equal(converted_to_unfused, _projection_weight(fused, name)), name

    # A same-layout load stays untouched by the conversion.
    into_fused.load_state_dict(fused_checkpoint)
    into_unfused.load_state_dict(unfused_checkpoint)
    for name in POST_CONVOLUTION_PROJECTIONS:
        reloaded_fused = _projection_weight(into_fused, name)
        reloaded_unfused = _projection_weight(into_unfused, name)
        assert torch.equal(reloaded_fused, _projection_weight(fused, name)), name
        assert torch.equal(reloaded_unfused, _projection_weight(unfused, name)), name


def test_fused_layout_issues_one_post_convolution_matmul_where_unfused_issues_five():
    x = torch.randn(2, 9, 32)
    unfused = _initialized_mixer(fused=False)
    fused = _initialized_mixer(fused=True)

    unfused_calls = _linear_call_names(unfused, x)
    fused_calls = _linear_call_names(fused, x)

    assert unfused_calls == ["in_proj", *POST_CONVOLUTION_PROJECTIONS, "out_proj"]
    assert fused_calls == ["in_proj", "u_proj", "out_proj"]
    assert len(unfused_calls) - len(fused_calls) == 4


@pytest.mark.parametrize("fused", [False, True])
def test_selector_logits_reach_the_scan_dense_for_the_raw_pointer_router_gradient(
    monkeypatch, fused: bool
):
    # The CUDA Appendix-C router gradient reads the selector logits off a raw float
    # pointer under a dense (batch, time, head, dictionary) layout. A fused projection
    # hands out a strided split view, which that kernel would silently read as the
    # neighbouring projections' columns.
    module = _initialized_mixer(fused=fused, backend="auto")
    captured = {}
    original_scan = mixer_module.paper_surrogate_scan

    def capture_scan(dictionary_logits, selector_logits, *args, **kwargs):
        captured["selector_logits"] = selector_logits
        return original_scan(dictionary_logits, selector_logits, *args, **kwargs)

    monkeypatch.setattr(mixer_module, "paper_surrogate_scan", capture_scan)

    module(torch.randn(2, 9, 32))

    selector_logits = captured["selector_logits"]
    assert selector_logits.shape == (2, 9, 4, 4)
    assert selector_logits.dtype == torch.float32
    assert selector_logits.is_contiguous()


@pytest.mark.parametrize("fused", [False, True])
def test_paper_block_reference_forward_backward_initialization_and_readout_parameters(
    fused: bool,
):
    torch.manual_seed(4)
    module = _config(fuse_input_projections=fused).build(32, layer_idx=0, n_layers=2)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(5),
    )
    x = torch.randn(2, 17, 32, requires_grad=True)

    output = module(x)
    output.square().mean().backward()

    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
    assert _projection_weight(module, "B_proj").shape == (2 * 32, 32)
    assert _projection_weight(module, "C_proj").shape == (2 * 32, 32)
    assert hasattr(module, "D")
    assert module.last_metadata is not None
    assert module.last_metadata.state_shape == (2, 4, 17, 8)
    assert module.last_metadata.payload_axes == ()


def test_mixer_rejects_unsupported_state_packing_and_recurrent_cache():
    with pytest.raises(ValueError, match="below 1024"):
        NativeFlashPDMixerConfig(n_heads=1, d_state=1024).build(1024, layer_idx=0, n_layers=1)
    module = _config().build(32, layer_idx=0, n_layers=1)
    x = torch.randn(1, 5, 32)

    with pytest.raises(NotImplementedError, match="packed"):
        module(x, cu_doc_lens=torch.tensor([0, 2, 5], dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="initial_state"):
        module(x, initial_state=torch.zeros(1, 4, 8))


@pytest.mark.parametrize("fused", [False, True])
def test_auto_training_preserves_both_paper_surrogate_gradients(fused: bool):
    module = _config(backend="auto", fuse_input_projections=fused).build(
        32, layer_idx=0, n_layers=1
    )
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
    )

    module(torch.randn(1, 9, 32)).square().mean().backward()

    assert module.dictionary_logits.grad is not None
    assert module.dictionary_logits.grad.abs().sum() > 0
    selector_gradient = _projection_grad(module, "selector_proj")
    assert selector_gradient is not None
    assert selector_gradient.abs().sum() > 0
    assert module.last_metadata is not None
    assert module.last_metadata.backend == "reference_paper_surrogate"


def test_bfloat16_mixer_keeps_complex_diagonal_fp32_at_paper_scan_boundary(monkeypatch):
    torch.manual_seed(23)
    decay_steps = 4096
    time = decay_steps + 1
    state = 8
    per_token_decay = 5e-4
    payload_dtype = torch.bfloat16
    mixer = _config(
        n_heads=1,
        d_state=state,
        dictionary_size=1,
        chunk_size=128,
        backend="auto",
        dtype=DType.bfloat16,
    ).build(state, layer_idx=0, n_layers=1)
    with torch.no_grad():
        # An identity dictionary keeps every token on the same state channel, so the
        # only thing between the first token and the last is 4096 multiplications by
        # the per-token decay.
        mixer.dictionary_logits.fill_(-1)
        channel = torch.arange(state)
        mixer.dictionary_logits[0, 0, channel, channel] = 1
        _projection_weight(mixer, "dt_proj").zero_()
        _projection_weight(mixer, "phase_proj").zero_()
        mixer.conv.weight.zero_()
        mixer.conv.weight[:, 0, -1] = 1
        mixer.conv.bias.zero_()
        mixer.A_log.zero_()
        mixer.dt_bias.fill_(math.log(math.expm1(per_token_decay)))
        mixer.D.zero_()

    captured = {}
    original_scan = mixer_module.paper_surrogate_scan

    def capture_scan(
        dictionary_logits,
        selector_logits,
        scan_diagonal_real,
        scan_diagonal_imag,
        scan_bias_real,
        scan_bias_imag,
        **kwargs,
    ):
        captured.update(
            diagonal_real=scan_diagonal_real,
            diagonal_imag=scan_diagonal_imag,
            bias_real=scan_bias_real,
            bias_imag=scan_bias_imag,
        )
        result = original_scan(
            dictionary_logits,
            selector_logits,
            scan_diagonal_real,
            scan_diagonal_imag,
            scan_bias_real,
            scan_bias_imag,
            **kwargs,
        )
        captured.update(states_real=result[0], states_imag=result[1])
        return result

    monkeypatch.setattr(mixer_module, "paper_surrogate_scan", capture_scan)

    # A single non-zero token drives the depthwise delta convolution, so the scan sees
    # one impulse at token zero and nothing after it.
    x = torch.zeros((1, time, state), dtype=payload_dtype)
    x[:, 0] = 1
    mixer(x)

    assert captured["diagonal_real"].dtype == torch.float32
    assert captured["diagonal_imag"].dtype == torch.float32
    for name in ("bias_real", "bias_imag", "states_real", "states_imag"):
        assert captured[name].dtype == payload_dtype, name
    diagonal = captured["diagonal_real"].float()
    assert diagonal.max().item() < 1.0
    assert diagonal.min().item() == pytest.approx(math.exp(-per_token_decay), rel=1e-5)

    impulse = captured["states_real"][0, 0, 0].float()
    tail = captured["states_real"][0, 0, -1].float()
    assert impulse.abs().min().item() > 1e-3
    observed_decay = tail / impulse
    expected_decay = math.exp(-per_token_decay * decay_steps)
    torch.testing.assert_close(
        observed_decay,
        torch.full_like(observed_decay, expected_decay),
        rtol=5e-2,
        atol=5e-3,
    )
    assert observed_decay.max().item() < 0.9


@pytest.mark.gpu
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("backend", ["auto", "cuda"])
def test_cuda_mixer_backward_uses_native_paper_training(backend: str, fused: bool):
    module = _config(backend=backend, dtype=DType.bfloat16, fuse_input_projections=fused).build(
        32, layer_idx=0, n_layers=1, init_device="cuda"
    )
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
    )
    output = module(torch.randn(1, 17, 32, device="cuda", dtype=torch.bfloat16))
    output.float().square().mean().backward()

    assert module.last_metadata is not None
    assert module.last_metadata.backend == "cuda_paper_training"
    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


@pytest.mark.gpu
def test_cuda_fused_projection_matches_unfused_including_the_router_gradient():
    # The CUDA router backward reads the selector logits off a raw pointer under a dense
    # (batch, time, head, dictionary) layout, so a fused split view has to be densified
    # before it crosses the scan boundary.
    scatter = NativePDMode.GENERAL_SCATTER
    unfused = _initialized_mixer(fused=False, backend="cuda", mode=scatter).to("cuda")
    fused = _initialized_mixer(fused=True, backend="cuda", mode=scatter).to("cuda")
    x = torch.randn(2, 13, 32, device="cuda")
    unfused_x = x.clone().requires_grad_(True)
    fused_x = x.clone().requires_grad_(True)

    unfused_output = unfused(unfused_x)
    fused_output = fused(fused_x)
    unfused_output.square().mean().backward()
    fused_output.square().mean().backward()

    assert unfused.last_metadata is not None and fused.last_metadata is not None
    assert unfused.last_metadata.backend == "cuda_paper_training"
    assert fused.last_metadata.backend == "cuda_paper_training"
    assert unfused.last_metadata.mode == NativePDMode.GENERAL_SCATTER
    assert fused.last_metadata.mode == NativePDMode.GENERAL_SCATTER
    torch.testing.assert_close(fused_output, unfused_output, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(fused_x.grad, unfused_x.grad, rtol=1e-5, atol=1e-6)
    selector_gradient = _projection_grad(fused, "selector_proj")
    assert selector_gradient is not None and selector_gradient.abs().sum() > 0
    for name in POST_CONVOLUTION_PROJECTIONS:
        torch.testing.assert_close(
            _projection_grad(fused, name),
            _projection_grad(unfused, name),
            rtol=1e-5,
            atol=1e-6,
            msg=lambda text, name=name: f"{name}: {text}",
        )
    unfused_parameters = dict(unfused.named_parameters())
    fused_parameters = dict(fused.named_parameters())
    for name in SHARED_PARAMETERS:
        torch.testing.assert_close(
            fused_parameters[name].grad,
            unfused_parameters[name].grad,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda text, name=name: f"{name}: {text}",
        )


@pytest.mark.parametrize("fused", [False, True])
def test_tiny_transformer_model_uses_native_sequence_mixer(fused: bool):
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        d_model=32,
        vocab_size=64,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=_config(fuse_input_projections=fused),
            layer_norm=norm,
            feed_forward=FeedForwardConfig(hidden_size=64, bias=False),
        ),
        lm_head=LMHeadConfig(layer_norm=norm, bias=False),
    )
    model = config.build()
    model.init_weights(device=torch.device("cpu"))

    input_ids = torch.randint(0, 64, (2, 11))
    labels = torch.randint(0, 64, (2, 11))
    output = model(input_ids, labels=labels)
    assert output.loss is not None
    output.loss.sum().backward()

    assert isinstance(model.blocks["0"].attention, NativeFlashPDMixer)
