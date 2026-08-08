"""The arm table is the experiment, so it gets asserted rather than eyeballed.

Run with ``pytest -v .edullm/test_hyper_connection_arms.py``.
"""

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import hyper_connection_arms as arms  # noqa: E402

from olmo_core.nn.residual_stream import HyperConnectionMode  # noqa: E402
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockType,
    TransformerConfig,
)

VOCAB_SIZE = 100_352
arms.install()


def baseline_370m() -> TransformerConfig:
    return TransformerConfig.olmo3_370M(vocab_size=VOCAB_SIZE)


def arm_370m(name: str) -> TransformerConfig:
    return arms.ARMS[name].apply(baseline_370m())


def test_every_arm_in_the_plan_is_present_and_numbered_once():
    numbers = sorted(arm.number for arm in arms.ARMS.values())
    assert numbers == list(range(1, 12))
    assert all(arm.isolates and arm.summary for arm in arms.ARMS.values())


def test_the_three_seed_arms_are_the_ones_that_get_claimed():
    """
    Baseline, faithful and mHC carry claims, so they get three seeds. Everything else is
    single-seed reconnaissance and has to be reported as such.
    """
    three = {name for name, arm in arms.ARMS.items() if arm.seeds == 3}
    assert three == {"baseline", "faithful", "mhc"}


@pytest.mark.parametrize("name", sorted(arms.ARMS))
def test_arms_stay_iso_parameter_with_the_baseline(name: str):
    """
    Every untied arm has to be iso-parameter with the baseline to within rounding, or the
    comparison is measuring capacity. The tied arms are deliberately not, which is their point.
    """
    base = baseline_370m().num_params
    arm = arms.ARMS[name]
    delta = arm_370m(name).num_params - base

    if arm.reuse_factor is not None:
        assert delta < 0
        return
    assert 0 <= delta / base < 0.001, f"{name} moved parameters by {delta:+,d}"


@pytest.mark.parametrize("name", sorted(arms.ARMS))
def test_arms_stay_iso_flop_with_the_baseline(name: str):
    base = baseline_370m().build(init_device="meta").num_flops_per_token(4096)
    flops = arm_370m(name).build(init_device="meta").num_flops_per_token(4096)
    assert abs(flops - base) / base < 0.005, f"{name} moved FLOPs by {flops - base:+,d}"


def test_each_arm_differs_from_the_faithful_one_in_exactly_what_it_claims():
    """
    Arms 3, 4 and 5 each isolate one documented difference between the two published setups.
    If any of them differs from the faithful arm in a second field it is no longer isolating
    anything.
    """
    faithful = arms.ARMS["faithful"].hyper_connections
    assert faithful is not None

    changed = {
        "output-only": {"mode": HyperConnectionMode.output},
        "no-output-init": {"output_init_exponent": 0.0},
        "n1": {"n_lanes": 1},
        "n2": {"n_lanes": 2},
        "n8": {"n_lanes": 8},
        "mhc": {"doubly_stochastic": True},
    }
    for name, expected in changed.items():
        hc = arms.ARMS[name].hyper_connections
        assert hc is not None
        differing = {
            f: getattr(hc, f)
            for f in faithful.__dataclass_fields__
            if getattr(hc, f) != getattr(faithful, f)
        }
        assert differing == expected, f"{name} differs in {differing}, expected {expected}"

    # decay-everything is identical as a model; the whole difference lives in the optimizer.
    assert arms.ARMS["decay-everything"].hyper_connections == faithful


def test_decay_everything_is_the_only_arm_without_the_weight_decay_split():
    for name, arm in arms.ARMS.items():
        overrides = arm.optim_group_overrides(weight_decay=0.033)
        if arm.hyper_connections is None:
            assert overrides == []
        else:
            assert len(overrides) == 2, name
            assert overrides[0].opts == dict(weight_decay=0.0)
            assert overrides[1].opts == dict(weight_decay=0.033)


def test_tied_arms_actually_tie_at_both_sizes():
    """
    An absolute block count would be a silent no-op at the rehearsal size, and the rehearsal
    would pass without ever running the code these arms depend on.
    """
    for factory in (baseline_370m, lambda: TransformerConfig.hc_rehearsal(vocab_size=VOCAB_SIZE)):
        for name in ("tied-faithful", "tied-baseline"):
            config = arms.ARMS[name].apply(factory())
            model = config.build(init_device="meta")
            assert len(model.blocks) == config.n_layers // arms.REUSE_FACTOR
            assert len(model.block_execution_order) == config.n_layers


def test_arms_refuse_a_base_config_they_were_not_defined_against():
    config = TransformerConfig.olmo2_370M(
        vocab_size=VOCAB_SIZE, block_name=TransformerBlockType.peri_norm
    )
    with pytest.raises(ValueError, match="reordered-norm baseline"):
        arms.ARMS["faithful"].apply(config)


def test_rehearsal_is_small_in_the_blocks_and_shaped_like_the_real_thing():
    rehearsal = TransformerConfig.hc_rehearsal(vocab_size=VOCAB_SIZE)
    real = baseline_370m()

    block_params = rehearsal.num_params - 2 * rehearsal.d_model * VOCAB_SIZE
    assert 15e6 < block_params < 25e6

    assert not isinstance(rehearsal.block, dict) and not isinstance(real.block, dict)
    assert rehearsal.block.name == real.block.name
    assert rehearsal.init_method == real.init_method
    assert rehearsal.lm_head.name == real.lm_head.name


@pytest.mark.parametrize("name", sorted(arms.ARMS))
def test_every_arm_builds_and_runs_a_forward_pass(name: str):
    config = arms.ARMS[name].apply(
        TransformerConfig.hc_rehearsal(
            vocab_size=512, d_model=64, n_layers=4, n_heads=4, attn_backend=None
        )
    )
    model = config.build()
    model.init_weights(device=torch.device("cpu"), max_seq_len=32)
    model.eval()

    with torch.no_grad():
        logits = model(torch.randint(0, 512, (2, 32)))
    assert logits.shape == (2, 32, config.vocab_size)
    assert torch.isfinite(logits).all()


def test_cut_order_names_real_arms_and_spares_the_core():
    assert set(arms.CUT_ORDER) <= set(arms.ARMS)
    core = {"baseline", "faithful", "n1", "mhc"}
    assert not core & set(arms.CUT_ORDER)


def test_describe_lists_every_arm():
    table = arms.describe()
    for name in arms.ARMS:
        assert name in table
