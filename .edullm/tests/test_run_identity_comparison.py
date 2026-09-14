"""The freeze contract, executable across runs.

`assert_resume_fingerprint` compares a run to itself. These tests cover the
other half: that two runs which differ in something frozen are reported as
unpoolable, and that two runs differing only along a designed axis are not.

Standard library only, so this runs on the laptop as well as FarmShare.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FARMSHARE = Path(__file__).resolve().parents[1] / "farmshare"
if str(FARMSHARE) not in sys.path:
    sys.path.insert(0, str(FARMSHARE))

import compare_run_identities as cri  # noqa: E402


def _identity(**overrides):
    base = {
        "family": "curriculum-new",
        "arm": "linear_n10-mtld-constant",
        "pacing": "linear_n10",
        "difficulty_metric": "mtld",
        "lr_schedule": "constant",
        "seed": 1,
        "parent": {"metric": "mtld", "manifest_sha256": "a" * 64},
        "model": "TransformerConfig.olmo2_370M",
        "sequence_length": 2048,
        "global_batch_tokens": 4194304,
        "rank_microbatch_tokens": 16384,
        "total_steps": 2500,
        "curriculum_steps": 2300,
        "peak_lr": 4e-4,
        "optimizer": "AdamW",
        "task_loss_nproc": 4,
        "numerics": {"dp_name": "hsdp", "reduce_dtype": "float32"},
        "difficulty_metric_definition": {"formula": "len/zlib"},
    }
    base.update(overrides)
    return base


def _make_run(tmp_path: Path, name: str, identity: dict, *, dirty: bool = False) -> Path:
    run_dir = tmp_path / name
    progress = run_dir / "runs" / identity["arm"] / "progress"
    progress.mkdir(parents=True, exist_ok=True)
    (progress / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    (run_dir / "code_provenance.json").write_text(
        json.dumps({"commit": "b" * 40, "dirty": dirty}), encoding="utf-8"
    )
    return run_dir


def _violations(tmp_path: Path, identities: dict[str, dict], **kwargs) -> list[str]:
    records = [
        cri.RunRecord(_make_run(tmp_path, name, identity))
        for name, identity in identities.items()
    ]
    return cri.compare_identities(records, **kwargs)


def test_identical_runs_are_poolable(tmp_path: Path) -> None:
    assert _violations(
        tmp_path, {"run-a": _identity(seed=1), "run-b": _identity(seed=2)}
    ) == []


@pytest.mark.parametrize(
    "field,value",
    [
        ("rank_microbatch_tokens", 32768),  # changes accumulation depth
        ("total_steps", 2400),
        ("curriculum_steps", 2200),
        ("peak_lr", 3e-4),
        ("optimizer", "SkipStepAdamW"),
        ("task_loss_nproc", 8),  # finding 1e: bpb not comparable across world sizes
        ("model", "TransformerConfig.olmo2_1B"),
        ("sequence_length", 1024),
    ],
)
def test_a_changed_frozen_field_is_a_violation(tmp_path: Path, field, value) -> None:
    violations = _violations(
        tmp_path, {"run-a": _identity(), "run-b": _identity(**{field: value})}
    )
    assert any(field in v for v in violations), violations


def test_a_changed_numerics_knob_is_a_violation(tmp_path: Path) -> None:
    """Finding 1f: reduction dtype changes the arithmetic, so it unpools."""
    violations = _violations(
        tmp_path,
        {
            "run-a": _identity(),
            "run-b": _identity(numerics={"dp_name": "hsdp", "reduce_dtype": "bfloat16"}),
        },
    )
    assert any("numerics" in v for v in violations), violations


def test_designed_axes_are_not_violations(tmp_path: Path) -> None:
    """Varying the manipulation is the experiment, not a contract breach."""
    assert _violations(
        tmp_path,
        {
            "control": _identity(arm="control-constant", pacing="control", difficulty_metric=None),
            "linear": _identity(),
            "cosine": _identity(arm="linear_n10-mtld-cosine", lr_schedule="cosine"),
        },
    ) == []


def test_same_metric_must_mean_the_same_corpus(tmp_path: Path) -> None:
    """Two mtld runs on different manifests means a re-materialization, which
    unpools them even though `parent` is otherwise a free axis."""
    violations = _violations(
        tmp_path,
        {
            "run-a": _identity(),
            "run-b": _identity(parent={"metric": "mtld", "manifest_sha256": "c" * 64}),
        },
    )
    assert any("same metric must mean the same corpus" in v for v in violations), violations


def test_different_metrics_may_have_different_parents(tmp_path: Path) -> None:
    assert _violations(
        tmp_path,
        {
            "mtld": _identity(),
            "flesch": _identity(
                arm="linear_n10-flesch-constant",
                difficulty_metric="flesch",
                parent={"metric": "flesch", "manifest_sha256": "d" * 64},
                difficulty_metric_definition={"formula": "flesch reading ease"},
            ),
        },
    ) == []


def test_an_unclassified_new_field_is_frozen_by_default(tmp_path: Path) -> None:
    """The allowlist is inverted on purpose: a newly added identity field is
    protected without anyone remembering to classify it."""
    violations = _violations(
        tmp_path,
        {"run-a": _identity(), "run-b": _identity(some_future_knob="changed")},
    )
    assert any("some_future_knob" in v for v in violations), violations


def test_a_dirty_run_is_never_poolable(tmp_path: Path) -> None:
    records = [
        cri.RunRecord(_make_run(tmp_path, "clean", _identity(seed=1))),
        cri.RunRecord(_make_run(tmp_path, "dirty", _identity(seed=2), dirty=True)),
    ]
    violations = cri.compare_identities(records)
    assert any("dirty tree" in v for v in violations), violations


def test_replicates_may_differ_only_in_seed(tmp_path: Path) -> None:
    ok = _violations(
        tmp_path,
        {"s1": _identity(seed=1), "s2": _identity(seed=2)},
        replicates=True,
    )
    assert ok == []

    bad = _violations(
        tmp_path,
        {
            "s1": _identity(seed=1),
            "cosine": _identity(arm="linear_n10-mtld-cosine", lr_schedule="cosine", seed=2),
        },
        replicates=True,
    )
    assert any("lr_schedule" in v for v in bad), bad


def test_replicates_must_have_distinct_seeds(tmp_path: Path) -> None:
    violations = _violations(
        tmp_path,
        {"a": _identity(seed=1), "b": _identity(seed=1)},
        replicates=True,
    )
    assert any("not distinct" in v for v in violations), violations
