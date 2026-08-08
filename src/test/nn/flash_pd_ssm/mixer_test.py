import hashlib
import json

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.attention import AttentionConfig
from olmo_core.nn.flash_pd_ssm import (
    FlashPDSSMImplementation,
    FlashPDSSMMixer,
    FlashPDSSMMixerConfig,
    Mamba3FlashPDSSMMixer,
    Mamba3FlashPDSSMMixerConfig,
    StateTracker,
    TritonCapability,
    mamba3_flash_pd_olmo3_370m,
    mamba3_olmo3_370m_with_state_tracker,
    replace_state_tracker,
)
from olmo_core.nn.mamba3 import Mamba3Config, Mamba3MixerConfig
from olmo_core.nn.transformer.init import InitMethod


def _tiny_mamba_config() -> Mamba3Config:
    return Mamba3Config.mamba3_hybrid_like(
        d_model=32,
        vocab_size=64,
        n_layers=4,
        n_heads=4,
        intermediate_size=64,
        mamba_n_heads=4,
        mamba_head_dim=8,
        d_state=8,
        n_groups=1,
        mimo_rank=1,
    )


def _flash_config(**kwargs) -> FlashPDSSMMixerConfig:
    return FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=3,
        implementation=FlashPDSSMImplementation.chunkwise,
        **kwargs,
    )


def _config_hash(config: Mamba3Config) -> str:
    payload = json.dumps(config.as_config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def test_mixer_config_json_roundtrip_meta_build_and_num_params():
    config = _flash_config(
        dtype=DType.float32,
        ste_temperature=0.7,
    )
    serialized = json.loads(json.dumps(config.as_config_dict()))
    rebuilt = FlashPDSSMMixerConfig.from_dict(serialized)

    assert rebuilt == config
    module = rebuilt.build(32, layer_idx=0, n_layers=4, init_device="meta")
    assert isinstance(module, FlashPDSSMMixer)
    assert all(parameter.device.type == "meta" for parameter in module.parameters())
    assert rebuilt.num_params(32) == sum(parameter.numel() for parameter in module.parameters())
    assert module.num_flops_per_token(seq_len=128) > 0


def test_mixer_forward_backward_init_and_extra_kwargs():
    torch.manual_seed(0)
    config = _flash_config()
    module = config.build(32, layer_idx=1, n_layers=4, init_device="cpu")
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=1,
        num_blocks=4,
        generator=torch.Generator().manual_seed(1),
    )
    x = torch.randn(2, 7, 32, requires_grad=True)

    output = module(x, max_doc_len=7, unused_training_kwarg=True)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"
    assert module.last_backend == "pytorch_chunkwise"


def test_mixer_rejects_packed_documents_and_parallelism():
    module = _flash_config().build(32, layer_idx=0, n_layers=1)
    x = torch.randn(1, 6, 32)
    with pytest.raises(NotImplementedError, match="cu_doc_lens"):
        module(x, cu_doc_lens=torch.tensor([0, 3, 6], dtype=torch.int32))

    class _Mesh:
        def size(self):
            return 2

    with pytest.raises(NotImplementedError, match="Tensor parallelism"):
        module.apply_tp(_Mesh())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Context parallelism"):
        module.apply_cp(_Mesh())  # type: ignore[arg-type]


def test_mixer_rejects_initial_state_and_decode_kwargs():
    module = _flash_config().build(32, layer_idx=0, n_layers=1)
    x = torch.randn(1, 4, 32)

    with pytest.raises(NotImplementedError, match="initial_state"):
        module(x, initial_state=torch.zeros(1, 2, 4, dtype=torch.complex64))
    with pytest.raises(NotImplementedError, match="decode"):
        module(x, decode=True)


def test_auto_falls_back_when_only_router_parameters_require_grad(monkeypatch):
    config = FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=3,
        implementation=FlashPDSSMImplementation.auto,
    )
    module = config.build(32, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
    )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.dictionary_logits.requires_grad_(True)
    module.selector_proj.weight.requires_grad_(True)

    def capability(*args, **kwargs):
        del args, kwargs
        return TritonCapability(False, "Triton unavailable")

    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.triton_capability",
        capability,
    )
    output = module(torch.randn(2, 7, 32))

    assert module.last_backend == "pytorch_sparse_autograd"
    assert module.last_fallback_reason == "Triton unavailable"
    output.square().mean().backward()
    assert module.dictionary_logits.grad is not None
    assert module.selector_proj.weight.grad is not None


def test_auto_training_uses_sparse_autograd_without_dense_transition(monkeypatch):
    config = FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=3,
        implementation=FlashPDSSMImplementation.auto,
    )
    module = config.build(32, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
    )
    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.triton_capability",
        lambda *args, **kwargs: TritonCapability(False, "Triton unavailable"),
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("auto training materialized a dense per-token transition")

    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.selected_transition_matrix",
        forbidden,
    )
    x = torch.randn(2, 7, 32, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    assert module.last_backend == "pytorch_sparse_autograd"
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_auto_passes_chunk_limit_to_capability_and_falls_back(monkeypatch):
    config = FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=257,
        implementation=FlashPDSSMImplementation.auto,
    )
    module = config.build(32, layer_idx=0, n_layers=1)

    def capability(*args, chunk_size=None, **kwargs):
        del args, kwargs
        assert chunk_size == 257
        return TritonCapability(False, "chunk_size exceeds 256")

    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.triton_capability",
        capability,
    )
    with torch.no_grad():
        module(torch.randn(1, 3, 32))

    assert module.last_backend == "pytorch_sparse_chunkwise"
    assert module.last_fallback_reason == "chunk_size exceeds 256"


def test_strict_triton_path_does_not_materialize_per_token_dense_matrices(monkeypatch):
    config = FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=3,
        implementation=FlashPDSSMImplementation.triton,
    )
    module = config.build(32, layer_idx=0, n_layers=1)

    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.triton_capability",
        lambda *args, **kwargs: TritonCapability(True, "supported"),
    )
    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.flash_pd_triton_scan",
        lambda source, diagonal, bias, **kwargs: torch.zeros_like(bias),
    )

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("strict Triton path materialized an O(BTHN^2) transition")

    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.selected_transition_matrix",
        forbidden,
    )
    with torch.no_grad():
        module(torch.randn(1, 3, 32))

    assert module.last_backend == "triton_three_phase"


def test_strict_failure_clears_previous_backend_status(monkeypatch):
    config = FlashPDSSMMixerConfig(
        n_heads=2,
        d_state=4,
        dictionary_size=3,
        chunk_size=3,
        implementation=FlashPDSSMImplementation.triton,
    )
    module = config.build(32, layer_idx=0, n_layers=1)
    capabilities = iter(
        [
            TritonCapability(True, "supported"),
            TritonCapability(False, "autograd is required"),
        ]
    )
    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.triton_capability",
        lambda *args, **kwargs: next(capabilities),
    )
    monkeypatch.setattr(
        "olmo_core.nn.flash_pd_ssm.mixer.flash_pd_triton_scan",
        lambda destination, diagonal, bias, **kwargs: torch.zeros_like(bias),
    )

    with torch.no_grad():
        module(torch.randn(1, 3, 32))
    assert module.last_backend == "triton_three_phase"

    with pytest.raises(RuntimeError, match="autograd is required"):
        module(torch.randn(1, 3, 32, requires_grad=True))
    assert module.last_backend is None
    assert module.last_fallback_reason == "autograd is required"


def test_default_state_tracker_preserves_existing_mamba_config_hash_and_state_contract():
    baseline = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    selected = mamba3_olmo3_370m_with_state_tracker(vocab_size=1024)

    assert selected.as_config_dict() == baseline.as_config_dict()
    assert _config_hash(selected) == _config_hash(baseline)
    assert all(
        isinstance(block.sequence_mixer, (Mamba3MixerConfig, AttentionConfig))
        for block in selected.resolved_block_configs
    )

    tiny = _tiny_mamba_config()
    unchanged = replace_state_tracker(tiny)
    assert unchanged is tiny
    baseline_model = tiny.build(init_device="meta")
    selected_model = unchanged.build(init_device="meta")
    baseline_state = {
        name: (tuple(value.shape), value.dtype)
        for name, value in baseline_model.state_dict().items()
    }
    selected_state = {
        name: (tuple(value.shape), value.dtype)
        for name, value in selected_model.state_dict().items()
    }
    assert selected_state == baseline_state


def test_flash_pd_factory_replaces_only_named_recurrent_slots():
    baseline = _tiny_mamba_config()
    assert isinstance(baseline.block, dict)
    baseline_attention = baseline.block["attn"].as_config_dict()
    flash_config = _flash_config()

    selected = replace_state_tracker(
        baseline,
        state_tracker=StateTracker.flash_pd,
        recurrent_slots=("mamba3",),
        flash_pd_config=flash_config,
    )

    assert selected is not baseline
    assert isinstance(selected.block, dict)
    assert selected.block["attn"].as_config_dict() == baseline_attention
    assert selected.block_pattern == baseline.block_pattern
    assert selected.block["mamba3"].sequence_mixer == flash_config
    assert isinstance(selected.block["attn"].sequence_mixer, AttentionConfig)
    assert isinstance(baseline.block["mamba3"].sequence_mixer, Mamba3MixerConfig)

    model = selected.build(init_device="meta")
    assert selected.num_params == sum(parameter.numel() for parameter in model.parameters())
    assert isinstance(model.blocks["0"].attention, FlashPDSSMMixer)
    assert isinstance(model.blocks["3"].attention, torch.nn.Module)


def test_hybrid_state_tracker_keeps_mamba_and_inserts_flash_pd_slots():
    baseline = _tiny_mamba_config()
    assert isinstance(baseline.block, dict)
    baseline_mamba = baseline.block["mamba3"].as_config_dict()
    flash_config = _flash_config()

    selected = replace_state_tracker(
        baseline,
        state_tracker=StateTracker.hybrid,
        flash_pd_config=flash_config,
    )

    assert isinstance(selected.block, dict)
    assert selected.block["mamba3"].as_config_dict() == baseline_mamba
    assert selected.block["flash_pd"].sequence_mixer == flash_config
    assert selected.block_pattern == ["mamba3", "mamba3", "flash_pd", "attn"]
    assert baseline.block_pattern == ["mamba3", "mamba3", "mamba3", "attn"]

    model = selected.build(init_device="meta")
    assert isinstance(model.blocks["0"].attention, torch.nn.Module)
    assert isinstance(model.blocks["2"].attention, FlashPDSSMMixer)


def test_370m_hybrid_repeats_two_mamba_one_flash_one_attention():
    selected = mamba3_olmo3_370m_with_state_tracker(
        vocab_size=1024,
        state_tracker=StateTracker.hybrid,
    )

    assert selected.block_pattern == ["mamba3", "mamba3", "flash_pd", "attn"]
    resolved = selected.resolved_block_configs
    assert sum(isinstance(block.sequence_mixer, FlashPDSSMMixerConfig) for block in resolved) == 4
    assert sum(isinstance(block.sequence_mixer, Mamba3MixerConfig) for block in resolved) == 8


def test_dedicated_mamba_flash_factory_uses_one_fused_mixer_type():
    selected = mamba3_flash_pd_olmo3_370m(vocab_size=1024)
    assert isinstance(selected.block, dict)
    fused = selected.block["mamba3_flash_pd"].sequence_mixer

    assert selected.block_pattern == [
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "mamba3_flash_pd",
        "attn",
    ]
    assert isinstance(fused, Mamba3FlashPDSSMMixerConfig)
    assert (fused.n_heads, fused.d_state, fused.mimo_rank, fused.dictionary_size) == (
        8,
        20,
        4,
        16,
    )
    baseline = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    assert isinstance(baseline.block, dict)
    mamba = baseline.block["mamba3"].sequence_mixer
    assert isinstance(mamba, Mamba3MixerConfig)
    assert abs(fused.num_params(1024) - mamba.num_params(1024)) / mamba.num_params(1024) < 0.01

    model = selected.build(init_device="meta")
    assert (
        sum(isinstance(block.attention, Mamba3FlashPDSSMMixer) for block in model.blocks.values())
        == 12
    )


def test_mamba_flash_hybrid_forward_and_backward():
    torch.manual_seed(5)
    config = replace_state_tracker(
        _tiny_mamba_config(),
        state_tracker=StateTracker.hybrid,
        flash_pd_config=_flash_config(),
    )
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    mixers = [type(block.attention).__name__ for block in model.blocks.values()]
    assert mixers == ["Mamba3Mixer", "Mamba3Mixer", "FlashPDSSMMixer", "Attention"]

    input_ids = torch.randint(0, 64, (2, 12))
    labels = torch.randint(0, 64, (2, 12))
    output = model(input_ids, labels=labels)
    assert output.loss is not None and torch.isfinite(output.loss).all()
    output.loss.sum().backward()


def test_default_370m_flash_config_uses_kernel_eligible_square_root_scaling():
    selected = mamba3_olmo3_370m_with_state_tracker(
        vocab_size=1024,
        state_tracker=StateTracker.flash_pd,
    )
    assert isinstance(selected.block, dict)
    mixer = selected.block["mamba3"].sequence_mixer

    assert isinstance(mixer, FlashPDSSMMixerConfig)
    assert mixer.n_heads == 32
    assert mixer.d_state == 32
    assert mixer.dictionary_size == 32


def test_flash_pd_state_dict_is_explicitly_incompatible_with_mamba_mixer_weights():
    baseline = _tiny_mamba_config().build(init_device="meta")
    selected = replace_state_tracker(
        _tiny_mamba_config(),
        state_tracker="flash_pd",
        flash_pd_config=_flash_config(),
    ).build(init_device="meta")

    with pytest.raises(RuntimeError, match="Missing key|Unexpected key|size mismatch"):
        selected.load_state_dict(baseline.state_dict(), strict=True)
