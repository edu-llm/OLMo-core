"""
Tests for the GPU smoke script's own guards.

WHY THIS FILE EXISTS
    ``scripts/smoke_gated_conv_gpu.py`` runs on a GPU, so nothing about it is exercised by the
    laptop suite -- and a guard that has never been shown to fire is not a guard. This project has
    shipped three unfireable guards in one file: a ceiling above 100% of peak, a clock check gated
    on an absent library, and a width-only test that passed a 3pp bias. Each read as a pass.

    So each pure guard is extracted from the script as a named function and called here with inputs
    chosen to make it *fail*, not just to make it pass. Every test below asserts both directions.
"""

import importlib.util
import math
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[3] / "scripts" / "smoke_gated_conv_gpu.py"


def _load():
    """
    Import the script without executing its ``__main__`` block or importing torch.

    The script places every argv check above its torch import for exactly this reason: a guard
    below the import cannot be reached without a GPU, which is how two of them became untestable
    last time.
    """
    spec = importlib.util.spec_from_file_location("smoke_gated_conv_gpu", _SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_script_exists_where_the_tests_expect_it():
    """
    A missing artifact must FAIL, not skip.

    A skip counts as a pass in the summary line, so a path typo here would silently remove every
    test below while the suite still read green.
    """
    assert _SCRIPT.is_file(), f"{_SCRIPT} is missing"


def test_loss_band_is_centred_on_the_uniform_entropy():
    mod = _load()
    low, high = mod.loss_band(1024)
    assert low < math.log(1024) < high
    assert high - low == pytest.approx(2 * mod.INITIAL_LOSS_TOLERANCE)
    # And a real bug is outside it: a model reading its own input scores near zero.
    assert not (low <= 0.05 <= high)


def test_gate_liveness_fails_on_a_dead_branch_and_passes_on_a_live_one():
    mod = _load()
    reference = 1.0

    alive, why = mod.gate_is_alive(1e-2, reference)
    assert alive, why

    # Exactly at the floor is NOT alive: 2**-9 down from the reference is already lost to bf16
    # accumulation, so the boundary must be exclusive.
    at_floor = mod.GATE_LIVENESS_RATIO * reference
    alive, why = mod.gate_is_alive(at_floor, reference)
    assert not alive, why

    alive, why = mod.gate_is_alive(0.0, reference)
    assert not alive and "cannot accumulate" in why


def test_gate_liveness_fails_closed_when_the_reference_is_itself_dead():
    """
    ``None`` must never mean fine.

    If the reference parameter has no gradient, the ratio carries no information -- reporting a
    pass would be the guard that cannot fire, dressed as conservatism.
    """
    mod = _load()
    for reference in (None, 0.0):
        alive, why = mod.gate_is_alive(1e-3, reference)
        assert not alive, why
        assert "unmeasurable" in why


def test_gate_liveness_fails_when_no_gate_gradient_was_recorded():
    mod = _load()
    alive, why = mod.gate_is_alive(None, 1.0)
    assert not alive and "no gate gradient" in why


def test_conv_path_check_refuses_an_empty_comparison_set():
    """
    Zero recorded convolutions must be a refusal.

    An empty set trivially satisfies "all fused", which is the exact shape of the four vacuous
    greens this project shipped in one build.
    """
    mod = _load()
    ok, why = mod.conv_path_is_fused([], allow_eager=False)
    assert not ok and "proves nothing" in why
    # And permission to run eager does not rescue an empty set either.
    ok, why = mod.conv_path_is_fused([], allow_eager=True)
    assert not ok


def test_conv_path_check_catches_a_mixed_or_eager_run():
    mod = _load()
    assert mod.conv_path_is_fused(["fused", "fused"], allow_eager=False)[0]

    ok, why = mod.conv_path_is_fused(["fused", "eager"], allow_eager=False)
    assert not ok and "incomparable" in why

    ok, _ = mod.conv_path_is_fused(["eager", "eager"], allow_eager=False)
    assert not ok

    ok, why = mod.conv_path_is_fused(["eager", "eager"], allow_eager=True)
    assert ok and "permitted" in why


def test_a_single_arm_run_is_refused():
    """
    One arm cannot show that the gate changes anything.

    Reachable without torch because the argv checks sit above the import; a version of this check
    placed below it would be unreachable on CPU and would pass on the wrong return.
    """
    mod = _load()
    assert mod.main(["--arms", "kda-gated"]) == 2


def test_an_unknown_arm_is_refused():
    mod = _load()
    assert mod.main(["--arms", "kda-plain,kda-bogus"]) == 2


def test_the_argv_refusals_are_reachable_without_a_gpu():
    """
    Pins the ordering that makes the two tests above meaningful.

    If the torch import moved above the argv checks, ``main`` would exit 3 ("no GPU") before ever
    validating its arguments -- and the two tests above would pass on the wrong return code while
    checking nothing. So this asserts the refusals return 2, not 3.
    """
    mod = _load()
    assert mod.main(["--arms", "kda-gated"]) == 2, "argv check ran after the GPU check"
