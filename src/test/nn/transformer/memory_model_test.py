import importlib
from collections import defaultdict
from types import MethodType
from typing import Any

import pytest
import torch
import torch.nn as nn

transformer_model = importlib.import_module("olmo_core.nn.transformer.model")
TransformerConfig = importlib.import_module("olmo_core.nn.transformer.config").TransformerConfig
TransformerBlockType = importlib.import_module(
    "olmo_core.nn.transformer.config"
).TransformerBlockType
EngramConfig = importlib.import_module("olmo_core.nn.memory").EngramConfig
OLMoConfigurationError = importlib.import_module("olmo_core.exceptions").OLMoConfigurationError
MoETransformer = transformer_model.MoETransformer
Transformer = transformer_model.Transformer


class _HashMemory(nn.Module):
    def __init__(self, hash_indices: object):
        super().__init__()
        self.hash_indices = hash_indices
        self.input_ids: list[torch.Tensor] = []

    def compute_hash_indices(self, input_ids: torch.Tensor) -> object:
        self.input_ids.append(input_ids)
        return self.hash_indices


class _RecordingAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.kwargs: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.kwargs.append(kwargs)
        return x


class _RecordingBlock(nn.Module):
    def __init__(self, *, memory: _HashMemory | None = None):
        super().__init__()
        self.attention = _RecordingAttention()
        self.memory = memory
        self.kwargs: list[dict[str, Any]] = []

    def forward(self, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        self.kwargs.append(kwargs)
        return self.attention(x, **kwargs)


class _FakeLngramBlock(_RecordingBlock):
    pass


class _FakeEngramBlock(nn.Module):
    def __init__(self, hash_indices: object):
        super().__init__()
        self.memory = _HashMemory(hash_indices)
        self.attention = _RecordingAttention()
        self.kwargs: list[dict[str, Any]] = []

    def forward(
        self,
        x: torch.Tensor,
        *,
        engram_hash_indices: object,
        **kwargs: Any,
    ) -> torch.Tensor:
        self.kwargs.append({"engram_hash_indices": engram_hash_indices, **kwargs})
        return self.attention(x, **kwargs)


def _build_forward_harness(
    model_cls: type[Transformer],
    blocks: list[nn.Module],
) -> Transformer:
    model = model_cls.__new__(model_cls)
    nn.Module.__init__(model)
    model.d_model = 4
    model.vocab_size = 32
    model.n_layers = len(blocks)
    model.dtype = torch.float32
    model.embed_scale = None
    model.embeddings = nn.Embedding(model.vocab_size, model.d_model)
    model.embedding_norm = None
    model.blocks = nn.ModuleDict({str(idx): block for idx, block in enumerate(blocks)})
    model.lm_head = None
    model._cp_load_balancer = None
    model._compile_enabled = False
    model._device = None
    return model


def _prepare_with(model: Transformer, prepared_input_ids: torch.Tensor) -> None:
    def prepare_inputs(
        self: Transformer,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ):
        del self, input_ids, labels, kwargs
        return prepared_input_ids, None, {}, defaultdict(dict), {}

    model._prepare_inputs = MethodType(prepare_inputs, model)


@pytest.mark.parametrize("model_cls", [Transformer, MoETransformer])
def test_forward_computes_engram_hashes_once_and_routes_only_to_engram_blocks(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[Transformer],
):
    monkeypatch.setattr(
        transformer_model,
        "MoEEngramReorderedNormTransformerBlock",
        _FakeEngramBlock,
        raising=False,
    )
    shared_hash_indices = object()
    ordinary = _RecordingBlock()
    first_engram = _FakeEngramBlock(shared_hash_indices)
    lngram = _FakeLngramBlock(memory=_HashMemory(object()))
    second_engram = _FakeEngramBlock(object())
    model = _build_forward_harness(
        model_cls,
        [ordinary, first_engram, lngram, second_engram],
    )
    raw_input_ids = torch.tensor([[1, 2]])
    prepared_input_ids = torch.tensor([[3, 4]])
    _prepare_with(model, prepared_input_ids)

    output = model(raw_input_ids)

    assert output.shape == (1, 2, model.d_model)
    assert first_engram.memory.input_ids == [prepared_input_ids]
    assert first_engram.memory.input_ids[0] is prepared_input_ids
    assert first_engram.memory.input_ids[0] is not raw_input_ids
    assert second_engram.memory.input_ids == []
    assert first_engram.kwargs[0]["engram_hash_indices"] is shared_hash_indices
    assert second_engram.kwargs[0]["engram_hash_indices"] is shared_hash_indices
    assert ordinary.kwargs == [{}]
    assert lngram.kwargs == [{}]
    for block in (ordinary, first_engram, lngram, second_engram):
        assert block.attention.kwargs == [{}]


def test_forward_does_not_compute_hashes_without_engram_blocks(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        transformer_model,
        "MoEEngramReorderedNormTransformerBlock",
        _FakeEngramBlock,
        raising=False,
    )
    ordinary_memory = _HashMemory(object())
    lngram_memory = _HashMemory(object())
    ordinary = _RecordingBlock(memory=ordinary_memory)
    lngram = _FakeLngramBlock(memory=lngram_memory)
    model = _build_forward_harness(Transformer, [ordinary, lngram])

    model(torch.tensor([[1, 2]]))

    assert ordinary_memory.input_ids == []
    assert lngram_memory.input_ids == []
    assert ordinary.kwargs == [{}]
    assert lngram.kwargs == [{}]


class _ResetTrackingMemory(nn.Module):
    def __init__(self, *, init_device: str):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(4, device=init_device))
        self.conv = nn.Conv1d(
            4,
            4,
            kernel_size=3,
            groups=4,
            bias=False,
            device=init_device,
        )
        self.reset_calls = 0

    def reset_parameters(self) -> None:
        self.reset_calls += 1
        nn.init.constant_(self.weight, 7.0)
        nn.init.zeros_(self.conv.weight)


def test_init_weights_materializes_and_resets_meta_memory():
    model = TransformerConfig.llama_like(
        d_model=64,
        vocab_size=32,
        n_layers=1,
        n_heads=2,
    ).build(init_device="meta")
    memory = _ResetTrackingMemory(init_device="meta")
    model.blocks["0"].add_module("memory", memory)

    model.init_weights(device=torch.device("cpu"))

    assert memory.reset_calls == 1
    assert memory.weight.device == torch.device("cpu")
    torch.testing.assert_close(memory.weight, torch.full_like(memory.weight, 7.0))
    torch.testing.assert_close(memory.conv.weight, torch.zeros_like(memory.conv.weight))


def test_model_rejects_heterogeneous_engram_hash_configs() -> None:
    config = TransformerConfig.smallmoe(
        vocab_size=32,
        d_model=32,
        n_layers=2,
        n_heads=4,
    )
    config.block.feed_forward_moe.num_experts = 4
    first = config.block.copy()
    first.name = TransformerBlockType.moe_engram_reordered_norm
    first.memory = EngramConfig(
        vocab_size=32,
        orders=(2, 3),
        num_hash_heads=1,
        table_sizes=(5, 7),
        embedding_dim=1,
        compression_map=tuple(range(32)),
    )
    second = config.block.copy()
    second.name = TransformerBlockType.moe_engram_reordered_norm
    second.memory = EngramConfig(
        vocab_size=32,
        orders=(2, 3),
        num_hash_heads=1,
        table_sizes=(5, 7),
        embedding_dim=1,
        compression_map=tuple([0, *range(31)]),
    )
    config.block_overrides = {0: first, 1: second}

    with pytest.raises(OLMoConfigurationError, match="identical hash"):
        config.build(init_device="meta")
