import pytest
import torch

from olmo_core.latentcot import inventory as I

ROOT = "s3://bucket/teams/t/runs/r/checkpoints/"


def test_arm_prefixes_offers_both_slash_spellings():
    double, single = I.arm_prefixes(ROOT, "A0", 1)
    assert double == "s3://bucket/teams/t/runs/r/checkpoints//A0/A0-seed1"
    assert single == "s3://bucket/teams/t/runs/r/checkpoints/A0/A0-seed1"


@pytest.mark.parametrize("root", [ROOT, ROOT.rstrip("/"), ROOT + "///"])
def test_arm_prefixes_normalizes_the_root_it_was_given(root):
    # Whether the caller pasted a trailing slash must not change which keys are probed.
    assert I.arm_prefixes(root, "A2", 1) == I.arm_prefixes(ROOT, "A2", 1)


def test_arm_prefixes_uses_the_seed_in_the_leaf():
    double, _ = I.arm_prefixes(ROOT, "A3", 7)
    assert double.endswith("/A3/A3-seed7")


@pytest.mark.parametrize(
    "names,expected",
    [
        # model.pt is written after the final step, so it outranks every stepN.pt.
        (["step500.pt", "step2000.pt", "model.pt"], "model.pt"),
        # Numeric comparison, not lexicographic: "step500" > "step2000" as strings.
        (["step500.pt", "step1000.pt", "step2000.pt"], "step2000.pt"),
        (["step500.pt", "step1000.pt"], "step1000.pt"),
        (["step500.pt"], "step500.pt"),
        # best.pt is an *earlier* step that scored well; "latest" must not pick it.
        (["best.pt", "step500.pt"], "step500.pt"),
    ],
)
def test_select_latest_picks_the_last_weights_written(names, expected):
    files = {name: f"{ROOT}A0/A0-seed1/{name}" for name in names}
    selected, path = I.select_latest(files)
    assert selected == expected
    assert path == files[expected]


@pytest.mark.parametrize("names", [[], ["best.pt"], ["metrics.json", "train.log", "best.json"]])
def test_select_latest_returns_none_without_a_checkpoint(names):
    assert I.select_latest({name: "x" for name in names}) is None


def _write(dir_path, names):
    dir_path.mkdir(parents=True, exist_ok=True)
    for name in names:
        (dir_path / name).write_text("x")
    return dir_path


def test_discover_arm_and_inventory_over_a_local_root(tmp_path):
    # A local root exercises the same code path as S3 without needing credentials.
    root = tmp_path / "checkpoints"
    _write(root / "A0" / "A0-seed1", ["step500.pt", "step2000.pt", "model.pt", "metrics.json"])
    _write(root / "A2" / "A2-seed1", ["step500.pt", "best.pt"])

    prefix, files = I.discover_arm(str(root), "A0", 1)
    assert prefix is not None and set(files) == {
        "step500.pt",
        "step2000.pt",
        "model.pt",
        "metrics.json",
    }

    inv = I.take_inventory(str(root), ["A0", "A1", "A2"], 1)
    assert inv["A0"]["selected"] == "model.pt"
    assert inv["A0"]["notes"] == ["metrics.json"]
    assert inv["A2"]["selected"] == "step500.pt"
    assert inv["A2"]["notes"] == ["best.pt"]
    # An arm with nothing is still reported -- absence is the finding.
    assert inv["A1"]["selected"] is None
    assert inv["A1"]["files"] == [] and inv["A1"]["prefix"] is None


def test_discover_arm_does_not_raise_on_a_missing_local_prefix(tmp_path):
    # list_directory raises FileNotFoundError locally but yields nothing on S3; both must read
    # as "this arm has no checkpoint" rather than killing the run.
    assert I.discover_arm(str(tmp_path / "nope"), "A4", 1) == (None, {})


@pytest.mark.parametrize(
    "found,gate_a_blocked,gate_b_partial",
    [
        ({"A0", "A1", "A2", "A3", "A4"}, False, False),
        ({"A0", "A2"}, False, True),
        ({"A0", "A1"}, True, True),  # the anchors alone support no gate at all
        ({"A2", "A3", "A4"}, True, False),
        (set(), True, True),
    ],
)
def test_missing_gate_notes_reports_what_cannot_be_answered(found, gate_a_blocked, gate_b_partial):
    notes = " ".join(I.missing_gate_notes(found))
    assert ("gate A CANNOT" in notes) is gate_a_blocked
    assert ("gate B will be PARTIAL" in notes) is gate_b_partial


def test_describe_inventory_names_missing_arms_and_blocked_gates(tmp_path):
    root = tmp_path / "checkpoints"
    _write(root / "A0" / "A0-seed1", ["model.pt"])
    _write(root / "A1" / "A1-seed1", ["model.pt"])
    text = I.describe_inventory(I.take_inventory(str(root), list(I.ARM_NAMES), 1))
    assert "NO CHECKPOINT FOUND" in text
    assert "MISSING: A2, A3, A4" in text
    assert "gate A CANNOT be computed" in text
    assert "No CODI arm is present" in text


class _AlwaysYes(torch.nn.Module):
    """A stand-in whose answer_logits always favour "yes", so predictions are deterministic."""

    device = torch.device("cpu")

    def forward(self, *args, **kwargs):  # pragma: no cover - not reached by these tests
        raise AssertionError("patched out")


def test_single_pass_matches_the_two_pass_figures(monkeypatch):
    """
    The single-pass helper must agree with the two functions it replaces.

    That identity is the whole justification for the change: overall accuracy is the depth
    counts summed before dividing, so one walk yields both numbers. Predictions are stubbed so
    the arithmetic is what is under test, not the model.
    """
    from olmo_core.latentcot import evaluate as E

    examples = [
        {"depth": 2, "reachable": True},
        {"depth": 2, "reachable": False},
        {"depth": 3, "reachable": True},
        {"depth": 3, "reachable": True},
        {"depth": 4, "reachable": False},
    ]
    monkeypatch.setattr(E, "predict_reachable", lambda model, ex, mode, **kw: True)
    model = _AlwaysYes()

    overall, by_depth = E.solve_rates_and_overall(model, examples, "no_cot")
    assert overall == pytest.approx(E.overall_accuracy(model, examples, "no_cot"))
    assert by_depth == E.solve_rate_by_depth(model, examples, "no_cot")
    # Spelled out so a refactor that silently reweights the buckets is caught here.
    assert by_depth == {2: pytest.approx(0.5), 3: pytest.approx(1.0), 4: pytest.approx(0.0)}
    assert overall == pytest.approx(3 / 5)


@pytest.mark.parametrize(
    "names,expected",
    [
        (["step100.pt", "model.pt"], [100]),
        (["step100.pt", "step2000.pt", "step300.pt"], [100, 300, 2000]),  # numeric, not lexical
        (["model.pt", "best.pt", "metrics.json"], []),
    ],
)
def test_steps_available_reads_only_step_checkpoints(names, expected):
    assert I.steps_available(names) == expected


def _inv(**arms):
    """Build an inventory shaped like take_inventory's output from {arm: [basenames]}."""
    return {
        arm: {
            "prefix": f"s3://b/ck//{arm}/{arm}-seed1",
            "files": sorted(names),
            "selected": None,
            "selected_path": None,
            "notes": [],
        }
        for arm, names in arms.items()
    }


def test_select_common_step_is_the_largest_all_arms_reached():
    # A0/A1 are cheap and got further; A2 is the CODI arm that ran out of wall clock.
    inv = _inv(
        A0=["step100.pt", "step200.pt", "step300.pt", "model.pt"],
        A1=["step100.pt", "step200.pt", "step300.pt", "model.pt"],
        A2=["step100.pt", "step200.pt"],
    )
    assert I.select_common_step(inv) == 200


def test_select_common_step_is_none_when_an_arm_never_checkpointed():
    # No matched budget is available, and the caller must say so rather than compare mismatched
    # arms -- this is the case that produced a vacuous result once already.
    inv = _inv(A0=["step100.pt", "model.pt"], A2=["best.pt"])
    assert I.select_common_step(inv) is None
    assert I.select_common_step({}) is None


def test_apply_common_step_repoints_every_arm_and_refuses_to_substitute():
    inv = _inv(
        A0=["step100.pt", "step200.pt", "step300.pt"],
        A2=["step100.pt", "step200.pt"],
        A3=["step100.pt"],  # lacks 200 -> must NOT silently fall back to 100
    )
    out = I.apply_common_step(inv, 200)
    assert out["A0"]["selected"] == "step200.pt"
    assert out["A0"]["selected_path"] == "s3://b/ck//A0/A0-seed1/step200.pt"
    assert out["A2"]["selected"] == "step200.pt"
    assert out["A3"]["selected"] is None and out["A3"]["selected_path"] is None
    # Input is not mutated.
    assert inv["A0"]["selected"] is None


def test_assemble_gates_omits_gate_a_without_both_arms():
    entry = {"mode": "codi", "overall_acc": 0.5, "solve_rate_by_depth": {2: 0.5}}
    from olmo_core.latentcot.evaluate import assemble_gates

    assert "gate_a" not in assemble_gates({"A2": entry})
    assert "gate_a" not in assemble_gates({"A0": entry})
    assert "gate_a" in assemble_gates({"A0": entry, "A2": entry})
    # Only the anchors: no gate A, and an empty gate B rather than a missing key.
    anchors = assemble_gates({"A0": entry, "A1": entry})
    assert "gate_a" not in anchors and anchors["gate_b"] == {}
