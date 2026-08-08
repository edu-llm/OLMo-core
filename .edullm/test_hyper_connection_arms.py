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
import train_hyper_connections as entry  # noqa: E402

from olmo_core.nn.residual_stream import (  # noqa: E402
    HyperConnectionMode,
    HyperConnectionStream,
    sinkhorn_knopp,
)
from olmo_core.nn.transformer import (  # noqa: E402
    TransformerBlockType,
    TransformerConfig,
)
from olmo_core.train.callbacks import HyperConnectionMonitorCallback  # noqa: E402

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
    Four arms at five seeds and nothing else. The tranche buys H1 (arm 2 against arm 1), H2a
    (arm 3 against arm 2) and H5 (arm 9 against arm 2) at five versus five, and gives up the
    six partial answers an earlier allocation spread the same money over.

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
    assert by_seeds[5] == {"baseline", "faithful", "output-only", "mhc"}
    assert sorted(arms.FUNDED) == ["baseline", "faithful", "mhc", "output-only"]


def test_the_tranche_is_twenty_runs_and_says_so():
    """
    Twenty, because that is what the budget buys and nothing else in this repository states
    it. ``edullm check`` prices a ceiling from the workload profile and never reads the arm
    table, so if this number is wrong nothing downstream disagrees with it.

    It was nine when the design assumed seed sigma fell as 1/sqrt(tokens). DataDecide measures
    it falling as D^-0.172, which makes horizon the worse thing to spend a fixed budget on and
    replicates the better one, so the same money buys five seeds at 6,000 steps instead of
    three at 12,715, which was fifteen. The twentieth through sixteenth are ``mhc``, restored
    from the end of the cut order by a grant above the original $4,000. See the module
    docstring for both pieces of arithmetic.
    """
    assert arms.total_runs() == 20
    assert str(arms.total_runs()) in arms.describe()
    # And the twenty went out as four five-cell submissions, which the table also has to say
    # -- four separate submissions are the thing that can silently disagree with each other.
    assert sum(stage.cells for stage in arms.STAGE_SPECS.values()) == arms.total_runs()
    for stage in arms.STAGE_SPECS.values():
        assert stage.spec in arms.describe()


def test_mhc_was_restored_from_the_end_of_the_cut_order_and_not_from_nowhere():
    """
    H5 is the best-designed hypothesis in the module, and it was placed last in the cut order
    precisely so that a restoration would reach it first. A grant above the original $4,000
    arrived and this is the arm it bought.

    THE ASSERTION THAT MATTERS IS THAT IT IS NO LONGER IN THE CUT ORDER AT ALL. An arm that is
    funded and still listed as cuttable is a budget that can be balanced twice, and the list
    read backwards would then nominate an arm that is already running.
    """
    assert arms.ARMS["mhc"].seeds == arms.STAGE_CELLS
    assert "mhc" not in arms.CUT_ORDER
    assert "mhc" in arms.FUNDED
    assert "FUNDED AS STAGE 3" in arms.ARMS["mhc"].isolates

    # And it did not overtake anything: what remains cuttable is what was ahead of it.
    assert arms.CUT_ORDER[-1] == "no-output-init"


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


def test_eight_sinkhorn_sweeps_pin_the_radius_h5_is_about():
    """
    THE PREMISE OF H5, ASSERTED RATHER THAN CITED. mHC's claim is that the lane-mixing matrix
    has spectral radius exactly 1, and the whole value of this arm's null is that the monitor
    can say whether that held. If it did not hold at the shipped eight sweeps, the arm would be
    measuring an unconstrained matrix under a constrained arm's name.

    IT HOLDS AT ANY SWEEP COUNT, AND THAT IS THE POINT WORTH KNOWING. The last operation of
    every sweep normalizes over ``dim=-2`` before the exponential, so the columns sum to 1
    whether the iteration has converged or not; a nonnegative column-stochastic matrix has 1 as
    an eigenvalue of its transpose and, by Gershgorin on that transpose, nothing above 1 in
    modulus. So the radius does not depend on convergence, which is why the row residual being
    large at eight sweeps -- it is, and ``sinkhorn_knopp``'s docstring says so -- takes nothing
    away from H5.

    The sweep counts below bracket the shipped eight on both sides, because a test that only
    checked eight would pass for the wrong reason if somebody made the default converge.
    """
    torch.manual_seed(0)
    for sigma in (0.5, 1.0, 4.0, 8.0):
        for iters in (1, 8, 16):
            matrices = sinkhorn_knopp(torch.randn(512, 4, 4) * sigma, num_iters=iters)
            columns = matrices.sum(dim=-2)
            assert torch.allclose(columns, torch.ones_like(columns), atol=1e-5)

            radius = torch.linalg.eigvals(matrices).abs().max(dim=-1).values
            assert (radius - 1.0).abs().max() < 1e-4, (sigma, iters)


def test_the_write_applies_the_transpose_so_the_constraint_lands_row_stochastic():
    """
    ``write`` computes ``...ij,...id->...jd``, which is ``A_r^T H``: the operator that acts on
    the lanes is the TRANSPOSE of the matrix Sinkhorn normalized. Worth asserting because the
    two are not the same object and only one of them is what the model applies -- the columns
    of ``A_r`` are the rows of what runs, so the constrained factor arrives row-stochastic,
    which is the textbook case Gershgorin is usually stated for.
    """
    torch.manual_seed(0)
    applied = sinkhorn_knopp(torch.randn(256, 4, 4) * 4.0, num_iters=8).transpose(-1, -2)
    rows = applied.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)


def test_the_mhc_arm_is_the_only_stage_the_lane_gate_would_refuse():
    """
    WHY run.mhc-stage.yaml OMITS ``--fail-closed-by-step`` WHEN THE OTHER TWO TREATMENTS SET
    IT, and the omission would cost five cells if this were wrong in either direction.

    The guard refuses a run in which fewer than half the blocks clear a lane-dispersion floor
    of 5e-03. Sinkhorn drives the mixing matrix towards unit row and column sums, a nonnegative
    matrix with unit sums is close to an averaging operator, and lane dispersion measures how
    far the lanes sit from their own mean -- so the constraint compresses the exact statistic
    the guard reads, and the floor was calibrated on the unconstrained mechanism.

    IT ASSERTS THE GUARD'S OWN CRITERION AND NOT A MAXIMUM, which is not a stylistic choice.
    The guard refuses on the FRACTION of blocks over the floor, and single blocks do cross it
    on this arm -- one of four here, one of eight at step 200 of the longer measurement
    ``MHC_LANE_DISPERSION_AT_GATE`` came from. A test on the maximum would go red on a deeper
    model or a longer run while the guard's answer stayed the same. Short and small on purpose:
    the separation is a factor of several and arrives within a few dozen steps.
    """
    floor = HyperConnectionMonitorCallback.min_lane_norm_spread
    needed = HyperConnectionMonitorCallback.min_differentiated_fraction
    assert arms.MHC_LANE_DISPERSION_AT_GATE < floor, "the measurement no longer motivates this"

    dispersions = {}
    for name in ("mhc", "faithful"):
        torch.manual_seed(17)
        config = arms.ARMS[name].apply(
            TransformerConfig.hc_rehearsal(
                vocab_size=512, d_model=64, n_layers=4, n_heads=4, attn_backend=None
            )
        )
        model = config.build()
        model.init_weights(device=torch.device("cpu"), max_seq_len=32)
        optim = torch.optim.AdamW(model.parameters(), lr=entry.DEFAULT_LEARNING_RATE)
        for _ in range(30):
            batch = torch.randint(0, 512, (2, 32))
            logits = model(batch)[:, :-1].reshape(-1, config.vocab_size)
            torch.nn.functional.cross_entropy(logits, batch[:, 1:].reshape(-1)).backward()
            optim.step()
            optim.zero_grad()
        model.eval()
        dispersions[name] = _lane_dispersions(model, torch.randint(0, 512, (2, 32)))

    alive = {
        name: sum(1 for value in values if value >= floor) / len(values)
        for name, values in dispersions.items()
    }
    assert alive["mhc"] < needed, f"the gate would no longer refuse the mhc arm: {dispersions}"
    assert alive["faithful"] >= needed, f"the gate would now refuse faithful too: {dispersions}"

    # And the arm the guard passes is separated from the arm it refuses by a wide margin rather
    # than by a block or two either side of the floor.
    assert min(dispersions["faithful"]) > 2 * max(dispersions["mhc"]), dispersions


def _lane_dispersions(model, tokens) -> list:
    """
    Lane dispersion per block, computed the way ``HyperConnectionMonitorCallback`` computes it:
    ``mean_k||x_k - x_bar|| / ||x_bar||`` per token, averaged over tokens.

    Duplicated from the callback rather than driven through it, because reaching it needs a
    Trainer and a metric recorder, and what is under test is which side of a threshold an arm's
    activations fall on rather than the callback's plumbing.
    """
    out: list = []

    def hook(module, args, output):
        if not isinstance(output, torch.Tensor) or output.ndim != 4:
            return
        lanes = output.detach().float()
        lane_norms = lanes.norm(dim=-1)
        mean_norm = lanes.mean(dim=-2).norm(dim=-1)
        about_mean = lane_norms.pow(2).mean(dim=-1) - mean_norm.pow(2)
        out.append((about_mean.clamp_min(0).sqrt() / mean_norm.clamp_min(1e-12)).mean().item())

    handles = [
        module.register_forward_hook(hook)
        for module in model.modules()
        if any(isinstance(child, HyperConnectionStream) for child in module.children())
    ]
    with torch.no_grad():
        model(tokens)
    for handle in handles:
        handle.remove()
    return out


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
    # The core is now the four arms the tranche funds. It was {baseline, faithful, n1, mhc}
    # when the budget stretched to seventeen runs; n1 and mhc were cut when it stopped, and mhc
    # came back when a grant above $4,000 arrived. This line moving three times is the record
    # of those three decisions rather than a test being loosened.
    core = {"baseline", "faithful", "output-only", "mhc"}
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
    # Twenty runs of about eighteen hours, so the expected spend is well under the ceiling
    # `edullm check` prices and approves against. Both numbers matter and they are different:
    # the ceiling is what the budget has to clear, and this is what arrives on the bill.
    assert 340 < hours < 380
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
