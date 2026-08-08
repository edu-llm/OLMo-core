"""Tests for the experiment arms, confound assertion, and multi-mode loss (PRD Phase 5)."""

from dataclasses import replace

import pytest
import torch

from olmo_core.latentcot import tokens as T
from olmo_core.latentcot.arms import (
    ARM_WHITELIST,
    ARMS,
    assert_arms_differ_only_in,
    build_arm_config,
)
from olmo_core.latentcot.data.dataset import codi_collate
from olmo_core.latentcot.data.encode import encode_example
from olmo_core.latentcot.data.graph_gen import generate
from olmo_core.latentcot.loss import arm_loss
from olmo_core.latentcot.train_module import CodiTransformerTrainModuleConfig
from olmo_core.nn.transformer import TransformerConfig
from olmo_core.optim import AdamWConfig

D_MODEL = 128


@pytest.fixture(scope="module")
def tok():
    try:
        return T.load_tokenizer()
    except Exception as e:
        pytest.skip(f"dolma2 tokenizer unavailable: {e}")


def _base_config():
    return CodiTransformerTrainModuleConfig(
        rank_microbatch_size=256,
        max_sequence_length=256,
        optim=AdamWConfig(lr=3e-4),
        distill_weight=1.0,
    )


def _build_tiny():
    cfg = TransformerConfig.llama_like(
        d_model=D_MODEL, n_layers=2, n_heads=4, vocab_size=T.PADDED_VOCAB_SIZE
    )
    return cfg.build(init_device="cpu")


def _examples(k, n=2):
    return [
        encode_example(
            generate(num_nodes=12, branching=2, depth=2, seed=s, reachable=bool(s % 2)), k
        )
        for s in range(n)
    ]


def test_all_five_arms_present():
    assert set(ARMS) == {"A0", "A1", "A2", "A3", "A4"}
    modes = {a.arm_mode for a in ARMS.values()}
    assert modes == {"explicit_cot", "no_cot", "codi"}


def test_arms_differ_only_in_whitelist():
    base = _base_config()
    configs = [build_arm_config(base, arm) for arm in ARMS.values()]
    # should NOT raise: arms share everything outside the whitelist
    assert_arms_differ_only_in(configs, ARM_WHITELIST)


def test_entropy_floor_is_whitelisted_and_propagates():
    base = _base_config()
    # A3 with its anti-collapse floor switched on stays confound-clean (floor is whitelisted).
    a3_floored = replace(ARMS["A3"], vocab_reg_entropy_floor=1.0)
    cfg = build_arm_config(base, a3_floored)
    assert cfg.vocab_reg_entropy_floor == 1.0  # build_arm_config carries it through
    configs = [build_arm_config(base, arm) for arm in ARMS.values()] + [cfg]
    assert_arms_differ_only_in(configs, ARM_WHITELIST)  # must NOT raise


def test_assertion_catches_out_of_whitelist_confound():
    base = _base_config()
    configs = [build_arm_config(base, arm) for arm in ARMS.values()]
    # tamper with a NON-whitelisted field (learning rate) on one arm
    configs[2] = replace(configs[2], optim=AdamWConfig(lr=1e-2))
    with pytest.raises(AssertionError):
        assert_arms_differ_only_in(configs, ARM_WHITELIST)


def test_codi_collate_wraps_examples(tok):
    items = _examples(2)
    batch = codi_collate(items)
    assert batch == {"examples": items}


def test_arm_loss_unknown_mode_raises(tok):
    model = _build_tiny()
    with pytest.raises(ValueError):
        arm_loss(model, _examples(2), mode="bogus")


@pytest.mark.parametrize("arm_key", ["A0", "A1", "A2", "A3", "A4"])
def test_each_arm_trains(tok, arm_key):
    """Each arm reduces its primary CE over a short run (mechanism check for done-when)."""
    arm = ARMS[arm_key]
    torch.manual_seed(0)
    model = _build_tiny()
    model.train()
    examples = _examples(arm.num_continuous_thoughts)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def primary(metrics):
        return metrics.get("ce_student") or metrics.get("ce_teacher") or metrics.get("ce_answer")

    first = last = None
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        loss, metrics = arm_loss(
            model,
            examples,
            mode=arm.arm_mode,
            distill_weight=1.0,
            vocab_reg=arm.vocab_reg,
            vocab_reg_weight=arm.vocab_reg_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        first = first if first is not None else primary(metrics)
        last = primary(metrics)

    assert first is not None and last is not None  # the loop ran at all
    assert torch.isfinite(torch.tensor(last))
    assert last < first  # the arm is learning
