import json
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

import olmo_core.nn.transformer.model as model_module
from olmo_core.distributed.parallel import DataParallelType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import MoETransformer, TransformerConfig
from olmo_core.train.train_module.transformer.common import parallelize_model
from olmo_core.train.train_module.transformer.config import TransformerDataParallelConfig


@pytest.mark.parametrize("policy", [True, False])
def test_reshard_policy_preserves_default_and_serializes_override(policy):
    default = TransformerDataParallelConfig(name=DataParallelType.fsdp)
    assert default.reshard_after_forward is None
    assert "reshard_after_forward" not in default.as_config_dict()

    payload = default.as_config_dict()
    payload["reshard_after_forward"] = policy
    restored = TransformerDataParallelConfig.from_dict(json.loads(json.dumps(payload)))
    assert restored.reshard_after_forward is policy


def test_reshard_policy_rejects_ddp():
    with pytest.raises(OLMoConfigurationError, match="reshard_after_forward.*FSDP"):
        TransformerDataParallelConfig(name=DataParallelType.ddp, reshard_after_forward=False)


def _mock_moe_model() -> MoETransformer:
    model = object.__new__(MoETransformer)
    nn.Module.__init__(model)
    model.prepare_experts_for_fsdp = Mock()
    model.apply_fsdp = Mock()
    model.init_weights = Mock()
    return model


@pytest.mark.parametrize("policy", [None, True, False])
def test_training_bridge_forwards_policy_to_model_and_experts(monkeypatch, policy):
    model = _mock_moe_model()
    dp_config = TransformerDataParallelConfig(
        name=DataParallelType.fsdp, reshard_after_forward=policy
    )
    monkeypatch.setattr(
        "olmo_core.train.train_module.transformer.common.get_dp_model_mesh",
        lambda mesh: object(),
    )
    monkeypatch.setattr(
        "olmo_core.train.train_module.transformer.common.get_device_mesh_info",
        lambda mesh: "test mesh",
    )

    parallelize_model(
        model,
        world_mesh=object(),
        device=torch.device("cpu"),
        dp_config=dp_config,
    )
    assert model.prepare_experts_for_fsdp.call_args.kwargs["reshard_after_forward"] is policy
    assert model.apply_fsdp.call_args.kwargs["reshard_after_forward"] is policy


def _install_fsdp_spies(model, monkeypatch):
    block_spies = {}
    for name, block in model.blocks.items():
        block_spies[name] = Mock()
        block.apply_fsdp = block_spies[name]

    names = {
        id(model): "root",
        id(model.embeddings): "embeddings",
        id(model.lm_head): "lm_head",
    }
    calls = {}

    def fake_fully_shard(module, **kwargs):
        calls[names[id(module)]] = kwargs
        if module is model.embeddings:
            module.set_unshard_in_backward = Mock()
        return module

    monkeypatch.setattr(model_module, "fully_shard", fake_fully_shard)
    return block_spies, calls


@pytest.mark.parametrize(
    "policy,pp_enabled,expected_blocks,expected_embeddings,expected_root,expected_head",
    [
        (None, False, True, True, True, False),
        (True, False, True, True, True, True),
        (False, False, False, False, False, False),
        (None, True, False, False, False, False),
    ],
)
def test_dense_fsdp_applies_consistent_reshard_policy(
    monkeypatch,
    policy,
    pp_enabled,
    expected_blocks,
    expected_embeddings,
    expected_root,
    expected_head,
):
    model = TransformerConfig.olmo2_1M(vocab_size=32, n_layers=2).build()
    block_spies, calls = _install_fsdp_spies(model, monkeypatch)
    model.apply_fsdp(pp_enabled=pp_enabled, reshard_after_forward=policy)

    assert {spy.call_args.kwargs["reshard_after_forward"] for spy in block_spies.values()} == {
        expected_blocks
    }
    assert calls["embeddings"]["reshard_after_forward"] is expected_embeddings
    assert calls["root"]["reshard_after_forward"] is expected_root
    assert calls["lm_head"]["reshard_after_forward"] is expected_head


def test_pipeline_parallelism_rejects_explicit_resharding(monkeypatch):
    model = TransformerConfig.olmo2_1M(vocab_size=32, n_layers=1).build()
    _install_fsdp_spies(model, monkeypatch)
    with pytest.raises(OLMoConfigurationError, match="pipeline parallelism.*retain"):
        model.apply_fsdp(pp_enabled=True, reshard_after_forward=True)


class _ExpertBlock(nn.Module):
    def __init__(self, *, ep_enabled=False, tp_enabled=False):
        super().__init__()
        self.is_moe = True
        self.ep_enabled = ep_enabled
        self.tp_enabled = tp_enabled
        self.feed_forward_moe = nn.Module()
        self.feed_forward_moe.prepare_experts_for_fsdp = Mock()


@pytest.mark.parametrize(
    "policy,pp_enabled,ep_enabled,tp_enabled,expected",
    [
        (None, False, False, False, True),
        (False, False, False, False, False),
        (None, True, False, False, False),
        (None, False, True, False, False),
        (None, False, False, True, False),
    ],
)
def test_expert_fsdp_resolves_topology_default(
    policy, pp_enabled, ep_enabled, tp_enabled, expected
):
    model = object.__new__(MoETransformer)
    nn.Module.__init__(model)
    block = _ExpertBlock(ep_enabled=ep_enabled, tp_enabled=tp_enabled)
    model.blocks = nn.ModuleDict({"0": block})
    model.dtype = torch.float32

    model.prepare_experts_for_fsdp(
        world_mesh=object(),
        pp_enabled=pp_enabled,
        reshard_after_forward=policy,
    )
    assert (
        block.feed_forward_moe.prepare_experts_for_fsdp.call_args.kwargs["reshard_after_forward"]
        is expected
    )
