import json

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.feed_forward import FeedForwardConfig
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


def test_config_roundtrip_meta_parameter_count_and_vector_state_contract():
    config = _config(ste_temperature=0.7)
    rebuilt = NativeFlashPDMixerConfig.from_dict(json.loads(json.dumps(config.as_config_dict())))
    module = rebuilt.build(32, layer_idx=0, n_layers=2, init_device="meta")

    assert rebuilt == config
    assert isinstance(module, NativeFlashPDMixer)
    assert all(parameter.device.type == "meta" for parameter in module.parameters())
    assert rebuilt.num_params(32) == sum(parameter.numel() for parameter in module.parameters())
    assert module.state_contract == ("batch", "head", "time", "state")
    assert "payload" not in module.state_contract
    assert module.num_flops_per_token(128) > 0


def test_paper_block_reference_forward_backward_initialization_and_readout_parameters():
    torch.manual_seed(4)
    module = _config().build(32, layer_idx=0, n_layers=2)
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
    assert hasattr(module, "B_proj")
    assert hasattr(module, "C_proj")
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


def test_auto_training_preserves_both_paper_surrogate_gradients():
    module = _config(backend="auto").build(32, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
    )

    module(torch.randn(1, 9, 32)).square().mean().backward()

    assert module.dictionary_logits.grad is not None
    assert module.selector_proj.weight.grad is not None
    assert module.last_metadata is not None
    assert module.last_metadata.backend == "reference_paper_surrogate"


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["auto", "cuda"])
def test_cuda_mixer_backward_uses_native_paper_training(backend: str):
    module = _config(backend=backend, dtype=DType.bfloat16).build(
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


def test_tiny_transformer_model_uses_native_sequence_mixer():
    norm = LayerNormConfig(name=LayerNormType.rms, bias=False)
    config = TransformerConfig(
        d_model=32,
        vocab_size=64,
        n_layers=1,
        block=TransformerBlockConfig(
            name=TransformerBlockType.reordered_norm,
            sequence_mixer=_config(),
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
