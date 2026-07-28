"""
Tests for the Mamba-3 smoke tests' argv pre-parsing.

These knobs decide *what gets tested*, so every failure mode here is silent by construction: the
run starts, trains, and reports success having exercised a configuration nobody asked for.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

MODULE = Path("src/scripts/train/smoketests/mamba3_cli.py")


@pytest.fixture(scope="module")
def cli() -> ModuleType:
    assert MODULE.exists(), f"{MODULE} not found; run pytest from the repo root"
    spec = importlib.util.spec_from_file_location("mamba3_cli", MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "argv, expected, rest",
    [
        (["--rotation-block-size", "3"], 3, []),
        (["--rotation-block-size=3"], 3, []),
        ([], 2, []),
        (["train", "n", "c", "--rotation-block-size", "3"], 3, ["train", "n", "c"]),
        (["--rotation-block-size", "3", "--other=1"], 3, ["--other=1"]),
    ],
    ids=["spaced", "equals", "absent", "among-positionals", "keeps-other-args"],
)
def test_pop_int_opt_reads_both_forms_without_eating_neighbours(cli, argv, expected, rest):
    assert cli.pop_int_opt(argv, "--rotation-block-size", 2) == (expected, rest)


def test_pop_int_opt_rejects_a_non_integer(cli):
    """A typo must be a usage error, not a silent fall back to the default."""
    with pytest.raises(SystemExit, match="expects an integer"):
        cli.pop_int_opt(["--rotation-block-size", "three"], "--rotation-block-size", 2)


def test_pop_opt_rejects_a_missing_value(cli):
    with pytest.raises(SystemExit, match="requires a value"):
        cli.pop_opt(["train", "--d-state"], "--d-state")


def test_pop_flag_removes_every_occurrence(cli):
    assert cli.pop_flag(["a", "--fast", "b", "--fast"], "--fast") == (True, ["a", "b"])
    assert cli.pop_flag(["a"], "--fast") == (False, ["a"])


def test_popped_tokens_recovers_exactly_what_was_removed(cli):
    """
    The Beaker launch path rebuilds the remote command from what is left in ``argv``. If this
    under-reports, the submitted job silently runs defaults while the launching command line
    says otherwise -- the failure the flags exist to prevent, moved to the remote.
    """
    before = ["launch", "run", "cluster", "--rotation-block-size", "3", "--disable-dp"]
    after = before[:3]
    assert cli.popped_tokens(before, after) == ["--rotation-block-size", "3", "--disable-dp"]


def test_popped_tokens_survives_duplicate_tokens(cli):
    """Naive set/membership differencing drops repeats; order-preserving matching must not."""
    before = ["a", "--flag", "a", "--x", "a"]
    after = ["a", "a", "a"]
    assert cli.popped_tokens(before, after) == ["--flag", "--x"]


def test_popped_tokens_is_empty_when_nothing_was_popped(cli):
    argv = ["train", "run", "cluster"]
    assert cli.popped_tokens(argv, argv) == []


def test_full_pop_sequence_round_trips_through_popped_tokens(cli):
    """End to end: what the scripts pop is exactly what they can hand back to the launcher."""
    before = [
        "launch",
        "run",
        "cluster",
        "--rotation-block-size",
        "3",
        "--launch.priority=low",
        "--require-fast-kernel",
        "--d-state=128",
    ]
    rest = before
    block, rest = cli.pop_int_opt(rest, "--rotation-block-size", 2)
    d_state, rest = cli.pop_int_opt(rest, "--d-state")
    fast, rest = cli.pop_flag(rest, "--require-fast-kernel")

    assert (block, d_state, fast) == (3, 128, True)
    # The launcher's own overrides must survive untouched...
    assert rest == ["launch", "run", "cluster", "--launch.priority=low"]
    # ...and everything taken out must be recoverable for the remote command.
    assert cli.popped_tokens(before, rest) == [
        "--rotation-block-size",
        "3",
        "--require-fast-kernel",
        "--d-state=128",
    ]
