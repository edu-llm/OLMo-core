"""
The auxiliary-loss weight divisor.

Both aux weights are divided by a per-layer divisor so that a recipe can state the *summed* weight
across the model. Stock divided by the model's **total** depth while only MoE blocks contribute a
term, so on a 24-layer model with 16 MoE blocks the summed balance weight came out
``16 * 0.01/24 = 0.00667`` where the recipe said ``0.01`` -- **1.5x low**, in the coefficient that
governs the routing health the run exists to measure, with no error and no warning.

These tests assert the arithmetic on a built module, because the defect was found by reading the
constructor and confirmed by reading the weights back off the built model. Reading them back is the
check; reasoning about the constructor is what missed it the first time.

CPU only, meta device where possible, nothing heavier than a config build.
"""

import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe import MoEConfig, MoERouterConfig


def _build(*, n_layers, n_moe_layers, lb=0.01, z=0.001, scale=True):
    config = MoEConfig(
        num_experts=8,
        hidden_size=64,
        router=MoERouterConfig(top_k=2, normalize_expert_weights=1.0),
        lb_loss_weight=lb,
        z_loss_weight=z,
        scale_loss_by_num_layers=scale,
        n_moe_layers=n_moe_layers,
    )
    return config.build(d_model=32, n_layers=n_layers, init_device="meta")


def test_unset_divisor_reproduces_the_stock_1_5x_error():
    """
    The measured defect, pinned. ``n_moe_layers=None`` divides by total depth, so the summed weight
    over 16 MoE blocks is 0.00667 rather than 0.01.

    This test asserts the *wrong* number on purpose. It is the regression guard on the default:
    leaving the divisor unset must keep behaving exactly as it did, so that a run comparing against
    an earlier one is comparing like with like, and so that the fix is opt-in rather than a silent
    change to every existing recipe.
    """
    moe = _build(n_layers=24, n_moe_layers=None)
    per_layer = moe.router.lb_loss_weight
    assert per_layer == pytest.approx(0.01 / 24)
    assert 16 * per_layer == pytest.approx(0.006667, abs=1e-6)
    assert moe.aux_loss_divisor == 24


def test_moe_depth_divisor_makes_the_summed_weight_the_recipe_number():
    """With the divisor set to MoE depth, 16 blocks sum to exactly the recipe's 0.01."""
    moe = _build(n_layers=24, n_moe_layers=16)
    per_layer = moe.router.lb_loss_weight
    assert per_layer == pytest.approx(0.01 / 16)
    assert 16 * per_layer == pytest.approx(0.01)
    assert moe.router.z_loss_weight == pytest.approx(0.001 / 16)
    assert 16 * moe.router.z_loss_weight == pytest.approx(0.001)
    assert moe.aux_loss_divisor == 16


def test_ratio_between_the_two_divisors_is_exactly_1_5_at_24_over_16():
    """
    The headline number. Asserted as a ratio so it cannot drift with the weight value.
    """
    stock = _build(n_layers=24, n_moe_layers=None).router.lb_loss_weight
    fixed = _build(n_layers=24, n_moe_layers=16).router.lb_loss_weight
    assert fixed / stock == pytest.approx(24 / 16)
    assert fixed / stock == pytest.approx(1.5)


def test_all_moe_model_is_unaffected_by_the_fix():
    """
    Our own ladder has an MoE block in every layer, so the two divisors coincide and the fix is a
    no-op there.

    That is worth a test rather than a note: it means the fix cannot be blamed for any difference
    between our runs, and it means the correctness of our runs does not *depend* on the fix -- which
    is exactly why the divisor still has to be set explicitly. An invariant that happens to hold is
    not the same as one that is enforced, and it stops holding the moment a dense block appears.
    """
    n = 12
    unset = _build(n_layers=n, n_moe_layers=None).router.lb_loss_weight
    explicit = _build(n_layers=n, n_moe_layers=n).router.lb_loss_weight
    assert unset == explicit == pytest.approx(0.01 / n)


def test_scale_off_leaves_the_weights_untouched_and_reports_divisor_one():
    """
    ``scale_loss_by_num_layers=False`` means the config value IS the per-layer weight. The reported
    divisor must then be 1, not the depth, or the audit metric would claim a division that did not
    happen.
    """
    moe = _build(n_layers=24, n_moe_layers=16, scale=False)
    assert moe.router.lb_loss_weight == pytest.approx(0.01)
    assert moe.router.z_loss_weight == pytest.approx(0.001)
    assert moe.router.aux_loss_divisor == 1


def test_divisor_is_wired_to_the_router_for_logging():
    """
    The correction has to be auditable from the run's own metrics, not from this constructor. If the
    divisor never reaches the router it never reaches the logs.

    Asserts the emitted METRIC, not just the attribute. An attribute set but never emitted is
    exactly the failure this metric exists to prevent -- and `compute_loss_metrics` is what a gate
    reads, so that is what has to be checked.
    """
    moe = _build(n_layers=24, n_moe_layers=16)
    assert moe.router.aux_loss_divisor == 16

    metrics = moe.router.compute_loss_metrics()
    assert metrics["moe/aux_loss_divisor"][0].item() == 16.0
    assert metrics["aux_loss_divisor"][0].item() == 16.0
    # The effective per-layer weight, and the name L5's assertion looks for.
    assert metrics["moe/lb_loss_weight_effective"][0].item() == pytest.approx(0.01 / 16)
    assert metrics["lb_loss_weight_effective"][0].item() == pytest.approx(0.01 / 16)
    assert metrics["moe/z_loss_weight_effective"][0].item() == pytest.approx(0.001 / 16)


def test_moe_depth_exceeding_total_depth_raises():
    """
    A divisor larger than the model is a config error, and it inflates the weight rather than
    deflating it. Raise rather than warn: it is silent otherwise, and any value trains.
    """
    with pytest.raises(OLMoConfigurationError, match="cannot exceed"):
        _build(n_layers=12, n_moe_layers=16)


def test_zero_moe_depth_raises_rather_than_dividing_by_zero():
    with pytest.raises(OLMoConfigurationError, match="at least 1"):
        _build(n_layers=12, n_moe_layers=0)
