import math

import pytest
import torch

from olmo_core.nn.attention import AttentionBackendName
from olmo_core.nn.memory import EngramConfig, LngramConfig
from olmo_core.nn.transformer.config import TransformerBlockType
from scripts.train.engram_experiment import smoke


@pytest.mark.parametrize("arm", smoke.ARMS)
def test_tiny_config_preserves_arm_architecture_and_cpu_backend(arm: str) -> None:
    config = smoke.build_smoke_config(arm)

    assert config.vocab_size == smoke.VOCAB_SIZE
    assert config.d_model == smoke.MODEL_DIM
    assert config.n_layers == smoke.NUM_LAYERS
    assert config.block.sequence_mixer.backend is AttentionBackendName.torch
    assert config.block.feed_forward_moe is not None
    assert config.block.feed_forward_moe.num_experts == smoke.NUM_EXPERTS
    assert config.block.feed_forward_moe.hidden_size == smoke.EXPERT_HIDDEN_SIZE

    if arm == "base":
        assert config.block_overrides is None
        assert config.block.name is TransformerBlockType.moe_reordered_norm
        assert config.block.memory is None
        return

    assert config.block_overrides is not None
    assert tuple(config.block_overrides) == (2, config.n_layers // 2)
    first, second = config.block_overrides.values()
    assert first is not second
    assert first.memory is not second.memory
    assert config.block.memory is None
    if arm == "engram":
        assert all(
            block.name is TransformerBlockType.moe_engram_reordered_norm
            and isinstance(block.memory, EngramConfig)
            for block in (first, second)
        )
    else:
        assert all(
            block.name is TransformerBlockType.moe_lngram_reordered_norm
            and isinstance(block.memory, LngramConfig)
            for block in (first, second)
        )


@pytest.mark.parametrize("arm", smoke.ARMS)
def test_arm_runs_one_cpu_forward_and_backward(arm: str) -> None:
    result = smoke.run_smoke(arm)

    assert result.arm == arm
    assert result.device == "cpu"
    assert math.isfinite(result.loss)
    assert result.finite_nonzero_gradients > 0


def test_main_selects_one_arm_or_all(monkeypatch, capsys) -> None:
    seen: list[str] = []

    def fake_run(arm: str) -> smoke.SmokeResult:
        seen.append(arm)
        return smoke.SmokeResult(
            arm=arm,
            device="cpu",
            loss=0.25,
            finite_nonzero_gradients=1,
        )

    monkeypatch.setattr(smoke, "run_smoke", fake_run)

    assert smoke.main(["engram"]) == 0
    assert seen == ["engram"]
    assert capsys.readouterr().out == "PASS engram loss=0.250000 gradients=1\n"

    seen.clear()
    assert smoke.main(["all"]) == 0
    assert seen == list(smoke.ARMS)
    assert capsys.readouterr().out.splitlines() == [
        f"PASS {arm} loss=0.250000 gradients=1" for arm in smoke.ARMS
    ]


def test_smoke_is_deterministic() -> None:
    first = smoke.run_smoke("base")
    second = smoke.run_smoke("base")

    assert first.loss == second.loss
    assert first.finite_nonzero_gradients == second.finite_nonzero_gradients
    assert torch.get_num_threads() >= 1
