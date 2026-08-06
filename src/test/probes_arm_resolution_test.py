"""What ``probes/train_probe.py`` accepts as an arm, and what it refuses.

``probes/`` is vendored at the repository root rather than installed, and ``train_probe`` is a
script rather than a package module, so it is loaded by path here -- the same approach
``edullm_train_on_corpus_test.py`` takes for ``.edullm/``.

Only ``apply_arm`` and the ``ARMS`` registry are exercised. Both are pure argument resolution:
they touch no GPU, build no model and import no kernel, which is what makes them testable in
this repository's CI at all. ``train_probe`` imports ``torch`` at module scope but nothing
under ``olmo_core.nn.attention``, so the load below does not need ``fla``.

WHY THIS FILE EXISTS. On 2026-08-05 a widened conflict guard in ``apply_arm`` broke the
``--match-arm`` path, and the break was invisible until it had killed 24 of 48 cells in each
of two paid array jobs. The guard itself was tested three ways; what was not tested was the
one caller that re-points a copy of the args at a *different* arm, where every field is
supposed to disagree. The end-to-end case at the bottom is that caller.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest


def _load():
    """Load ``probes/train_probe.py`` by path, with ``probes/`` importable for its siblings."""
    root = Path(__file__).parent.parent.parent
    probes = root / "probes"
    if str(probes) not in sys.path:
        sys.path.insert(0, str(probes))
    path = probes / "train_probe.py"
    spec = importlib.util.spec_from_file_location("probes_train_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


probe = _load()


def _args(**overrides) -> argparse.Namespace:
    """A Namespace shaped like the parser's defaults, before any arm is applied."""
    base = dict(arm=None, mixer=None, num_householder=None, beta_regime=None, match_arm=None)
    base.update(overrides)
    return argparse.Namespace(**base)


# The arms an experiment actually launches, with the settings their ids promise. Parametrized
# rather than looped so a single wrong arm names itself in the failure.
@pytest.mark.parametrize(
    "arm,mixer,num_householder,beta_regime,match_arm",
    [
        ("R1", "kda_hh", 1, "strict", None),
        ("DP2-strict", "kda_hh", 2, "strict", None),
        ("DP3-strict", "kda_hh", 3, "strict", None),
        ("DP4-strict", "kda_hh", 4, "strict", None),
        ("R1-refl", "kda_hh", 1, "reflection", None),
        ("Reflection", "kda_hh", 2, "reflection", None),
        ("R1-P", "kda_hh", 1, "strict", "DP2-strict"),
        ("R1-refl-P", "kda_hh", 1, "reflection", "Reflection"),
        ("DP2-P3", "kda_hh", 2, "strict", "DP3-strict"),
        ("DP3-P4", "kda_hh", 3, "strict", "DP4-strict"),
    ],
)
def test_arm_resolves_to_its_declared_settings(arm, mixer, num_householder, beta_regime, match_arm):
    """An arm id sets every field the registry declares for it."""
    args = _args(arm=arm)
    probe.apply_arm(args)
    assert args.mixer == mixer
    assert args.num_householder == num_householder
    assert args.beta_regime == beta_regime
    assert args.match_arm == match_arm


@pytest.mark.parametrize(
    "arm,flag,value",
    [
        ("R1", "num_householder", 3),
        ("DP4-strict", "num_householder", 1),
        ("R1", "beta_regime", "reflection"),
        ("Reflection", "num_householder", 4),
        ("R1-P", "match_arm", "Reflection"),
    ],
)
def test_a_flag_that_contradicts_the_arm_is_refused(arm, flag, value):
    """A surviving command-line flag would make the recorded arm id a lie, so it exits.

    ``--arm R1 --num-householder 3`` used to train at R=1 and record ``arm: "R1"``. The record
    was honest and the caller's intent was gone, with a zero exit either way.
    """
    args = _args(arm=arm, **{flag: value})
    with pytest.raises(SystemExit) as excinfo:
        probe.apply_arm(args)
    message = str(excinfo.value)
    assert flag.replace("_", "-") in message
    assert arm in message


def test_a_flag_that_agrees_with_the_arm_is_accepted():
    """Redundant is not contradictory. Only a *different* value is refused."""
    args = _args(arm="DP3-strict", num_householder=3, beta_regime="strict")
    probe.apply_arm(args)
    assert args.num_householder == 3


def test_unknown_arm_is_refused():
    """Informal short forms are rejected rather than guessed at."""
    with pytest.raises(SystemExit):
        probe.apply_arm(_args(arm="tied-K"))


@pytest.mark.parametrize("arm", ["DP2-budgeted", "R1-2step-tiedK"])
def test_declared_but_unimplemented_arms_raise(arm):
    """Both need a per-factor beta. They must not silently resolve to DP2-strict."""
    with pytest.raises(SystemExit, match="not implemented"):
        probe.apply_arm(_args(arm=arm))


# ---------------------------------------------------------------------------------------
# The end-to-end case: resolving a capacity control's TARGET.
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("arm", ["R1-P", "R1-refl-P", "DP2-P3", "DP3-P4"])
def test_match_arm_target_resolves_from_a_fully_populated_namespace(arm):
    """Call the production resolver, not a copy of it.

    This is the code that broke. It runs on a namespace still carrying the *source* arm's
    mixer, R and regime, so the guard that protects the command line must not fire here --
    every field is meant to be overwritten. The assertion is that the target's R is the
    target's own, not the source's, because reading the source's R would silently match
    against the wrong ledger.

    An earlier version of this test reimplemented the three lines instead of calling them, and
    passed against the broken code. Hence ``resolve_match_target`` exists as a function.
    """
    args = _args(arm=arm, match_non_embedding=None)
    probe.apply_arm(args)
    source_r = args.num_householder

    target_args = probe.resolve_match_target(args)

    expected = probe.ARMS[args.match_arm]
    assert target_args.num_householder == expected["num_householder"]
    assert target_args.num_householder != source_r, "the target must not inherit the source's R"
    # A control matches within its own regime; crossing regimes would fold a capacity control
    # and a regime contrast into one arm.
    assert target_args.beta_regime == args.beta_regime


def test_match_arm_refuses_a_target_that_is_itself_a_control():
    """Matching to a control reads its count *before* its own match, silently."""
    args = _args(arm="DP2-strict", match_non_embedding=None)
    probe.apply_arm(args)
    args.match_arm = "R1-P"
    with pytest.raises(SystemExit, match="itself parameter-matched"):
        probe.resolve_match_target(args)


def test_match_arm_refuses_a_cross_regime_target():
    """A strict control matched to a reflection arm confounds capacity with beta range."""
    args = _args(arm="R1-P", match_non_embedding=None)
    probe.apply_arm(args)
    args.match_arm = "Reflection"
    with pytest.raises(SystemExit, match="regime"):
        probe.resolve_match_target(args)


def test_match_arm_and_match_non_embedding_are_mutually_exclusive():
    """Two targets is not a tie-break, it is an ambiguous run."""
    args = _args(arm="R1-P", match_non_embedding=1_400_524)
    probe.apply_arm(args)
    with pytest.raises(SystemExit, match="at most one"):
        probe.resolve_match_target(args)


@pytest.mark.parametrize("arm", ["R1-P", "R1-refl-P", "DP2-P3", "DP3-P4"])
def test_match_arm_target_is_not_itself_a_control(arm):
    """A control matched to another control would read the target's pre-match count."""
    assert not probe.ARMS[probe.ARMS[arm]["match_arm"]].get("match_arm")


def test_internal_resolution_still_refuses_an_unimplemented_target():
    """``enforce_no_conflict=False`` relaxes the conflict check and nothing else."""
    args = _args(arm="DP2-budgeted")
    with pytest.raises(SystemExit, match="not implemented"):
        probe.apply_arm(args, enforce_no_conflict=False)
