"""The arm table is the experiment, so it gets asserted rather than eyeballed.

Run with ``pytest -v .edullm/test_hyper_connection_arms.py``.
"""

import os
import pathlib
import sys
from typing import Dict, Set

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


def test_the_seed_allocation_is_the_one_the_pre_registration_argues_for():
    """
    Three arms at five seeds and nothing else. The tranche buys H1 (arm 2 against arm 1) and
    H2a (arm 3 against arm 2) at five versus five, and gives up the six partial answers an
    earlier allocation spread the same money over.

    BALANCED, WHICH IS AN ASSERTION AND NOT A COINCIDENCE. The standard error of a contrast is
    sigma*sqrt(1/n_a + 1/n_b), so an unbalanced design pays for the smaller arm twice: a
    three-seed baseline against five-seed treatments is 22% worse on H1 than five against five
    and costs the same. Funding one arm out of step with the others is the shape of mistake
    this asserts against.
    """
    by_seeds: Dict[int, Set[str]] = {}
    for name, arm in arms.ARMS.items():
        by_seeds.setdefault(arm.seeds, set()).add(name)

    assert set(by_seeds) == {0, 5}, "the tranche funds an arm at five seeds or not at all"
    assert by_seeds[5] == {"baseline", "faithful", "output-only"}
    assert sorted(arms.FUNDED) == ["baseline", "faithful", "output-only"]


def test_the_tranche_is_fifteen_runs_and_says_so():
    """
    Fifteen, because that is what the budget buys and nothing else in this repository states
    it. ``edullm check`` prices a ceiling from the workload profile and never reads the arm
    table, so if this number is wrong nothing downstream disagrees with it.

    It was nine when the design assumed seed sigma fell as 1/sqrt(tokens). DataDecide measures
    it falling as D^-0.172, which makes horizon the worse thing to spend a fixed budget on and
    replicates the better one, so the same money buys five seeds at 6,000 steps instead of
    three at 12,715. See the module docstring for the arithmetic.
    """
    assert arms.total_runs() == 15
    assert str(arms.total_runs()) in arms.describe()
    # And the fifteen went out as three five-cell submissions, which the table also has to say
    # -- three separate submissions are the thing that can silently disagree with each other.
    assert sum(stage.cells for stage in arms.STAGE_SPECS.values()) == arms.total_runs()
    for stage in arms.STAGE_SPECS.values():
        assert stage.spec in arms.describe()


def test_mhc_is_deferred_rather_than_dropped_and_the_table_says_which():
    """
    H5 is the best-designed hypothesis in the module. Carrying zero seeds is a budget
    decision; being last in the cut order is what makes it a deferral, because the cut order
    read backwards is the order a second tranche restores arms in.
    """
    assert arms.ARMS["mhc"].seeds == 0
    assert arms.CUT_ORDER[-1] == "mhc"
    assert "DEFERRED" in arms.ARMS["mhc"].isolates
    assert "deferred" in arms.describe().lower() or "DEFERRED" in arms.describe()


def test_the_output_only_arm_records_that_it_was_degenerate_until_it_was_fixed():
    """
    Before b7983ea9 this arm dropped the paper's fixed staggered read along with the learned
    input map, so every lane read the same vector and the arm was the baseline with dead
    parameters. Three seeds of that would have measured nothing, twice: no effect, and no
    signal that there was no effect. The table has to carry the reason it is runnable.
    """
    summary = arms.ARMS["output-only"].summary
    assert "b7983ea9" in summary
    assert arms.ARMS["output-only"].seeds == arms.ARMS["baseline"].seeds == 5


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
    # The core is now the three arms the tranche funds. It was {baseline, faithful, n1, mhc}
    # when the budget stretched to seventeen runs; n1 and mhc were cut when it stopped, and
    # this line moving is the record of that rather than a test being loosened.
    core = {"baseline", "faithful", "output-only"}
    assert not core & set(arms.CUT_ORDER)
    assert set(arms.FUNDED) == core


def test_the_unfunded_arms_are_the_head_of_the_cut_order():
    """
    A budget balanced by cutting something that was never nominated for cutting is a different
    experiment from the one that was pre-registered. Whatever carries zero seeds has to be a
    prefix of the order the plan committed to in advance.
    """
    unfunded = [name for name in arms.ARMS if arms.ARMS[name].seeds == 0]
    assert sorted(unfunded) == sorted(arms.CUT_ORDER[: len(unfunded)])


def test_describe_lists_every_arm():
    table = arms.describe()
    for name in arms.ARMS:
        assert name in table


def test_the_tranche_fits_one_attempt_of_the_workload_ceiling():
    """
    THE CONSTRAINT THAT SET THE STEP COUNT, ASSERTED SO THAT MOVING EITHER ONE FAILS HERE.

    ``olmo-core-train`` declares maximum_runtime_hours: 24 and ``--hours`` only lowers it.
    The second attempt exists for a lost host, not for a run that outgrows the bound: the
    platform's retry table is ``OnStatusReason "Host EC2*" RETRY`` then two EXITs, and a
    timed-out attempt reaches a retry only through the no-exit-code fall-through, which
    torchrun's non-zero exit on SIGTERM races. So the tranche has to finish inside ONE
    attempt, and the margin below is what pays for step-time drift over eighteen hours.
    """
    longest = max(arms.arm_seconds(arm) for arm in arms.ARMS.values() if arm.seeds) / 3600

    assert longest < 21.0, "a cell would be killed before it finished"
    assert longest < 0.9 * 21.0, "under 21h but with no margin for drift"
    # And the horizon this replaces genuinely does not fit, which is the whole argument.
    full = arms.arm_seconds(arms.ARMS["faithful"], arms.FULL_HORIZON_STEPS) / 3600
    assert full > 24.0


def test_the_cost_model_is_built_from_the_measurement_and_not_from_a_price():
    """
    The rate is an argument. Prices live in reviewed platform configuration that changes
    without anybody being told, so the only honest source is ``edullm check --json``, and a
    number copied into this repository would be right until it silently was not.
    """
    source = pathlib.Path(arms.__file__).read_text()
    assert "10.4926" not in source, "an hourly rate was written into the arm table"

    hours = arms.tranche_hours()
    assert arms.estimated_cost_usd(10.0) == pytest.approx(hours * 10.0)
    # Fifteen runs of about eighteen hours, so the expected spend is well under the ceiling
    # `edullm check` prices and approves against. Both numbers matter and they are different:
    # the ceiling is what the budget has to clear, and this is what arrives on the bill.
    assert 250 < hours < 290
    assert hours == pytest.approx(arms.total_runs() * 17.8, rel=0.05)


def test_the_arm_seconds_model_reproduces_the_probe_it_was_measured_from():
    """
    The probe was 100 steps with an eval every 50 and a checkpoint at 0 and at 100, and it
    took 1,513.5 seconds of wall clock. Rebuilding that shape out of the constants has to land
    on it, or the constants are not the thing they claim to be.
    """
    steps = 100
    evaluations = 3  # on startup, at 50, at 100
    checkpoints = 2  # at 0 and at 100
    monitor = 5 * arms.MONITOR_SECONDS_PER_FIRING  # --monitor-interval 20 on that probe

    modelled = (
        arms.MEASURED_STARTUP_SECONDS
        + steps * arms.MEASURED_SECONDS_PER_STEP
        + evaluations * arms.MEASURED_EVAL_SECONDS
        + checkpoints * arms.MEASURED_CHECKPOINT_SECONDS
        + monitor
    )
    assert modelled == pytest.approx(1513.5, rel=0.05)
