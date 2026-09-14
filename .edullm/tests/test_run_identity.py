"""Plan finding 1f: the fingerprint must contain every value that changes
what a run computes, not only the values that describe the science.

Before this, `scientific_identity` recorded the arm, metric, schedule, seed
and geometry, but none of the five `EDULLM_BENCH_*` knobs -- data-parallel
strategy, gradient-reduction dtype, expert parallelism, prefetch depth, fp8
linears. Two runs could therefore carry byte-identical fingerprints and
still differ in their arithmetic, which is precisely the poolability leak
the audit named: a pooled result would silently mix incomparable runs.

Also covers plan 1e's eval world size, since `DistributedSampler` pads to
divisibility and bpb from a 4-rank eval is not comparable to bpb from an
8-rank one.

Needs torch and olmo_core, which do not import on Windows (`bettermap`
wants `multiprocessing.context.ForkProcess`), so these skip there and run
on FarmShare and any Linux checkout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

EDULLM_ROOT = Path(__file__).resolve().parents[1]
if str(EDULLM_ROOT) not in sys.path:
    sys.path.insert(0, str(EDULLM_ROOT))

try:
    import curriculum_entrypoint as ce
except ImportError as exc:  # pragma: no cover - environment-dependent
    pytest.skip(
        f"curriculum_entrypoint needs torch + olmo_core, unavailable here: {exc}",
        allow_module_level=True,
    )

NUMERICS_ENV = (
    "EDULLM_BENCH_DP",
    "EDULLM_BENCH_REDUCE_BF16",
    "EDULLM_BENCH_EP_DEGREE",
    "EDULLM_BENCH_PREFETCH",
    "EDULLM_BENCH_FLOAT8",
)


def test_numerics_identity_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in NUMERICS_ENV:
        monkeypatch.delenv(name, raising=False)
    numerics = ce.numerics_identity()
    assert numerics["dp_name"] == "hsdp"
    assert numerics["reduce_dtype"] == "float32"
    assert numerics["ep_degree"] == 0
    assert numerics["prefetch_factor"] == 0
    assert numerics["float8"] is False


@pytest.mark.parametrize(
    "name,value,key,expected",
    [
        ("EDULLM_BENCH_DP", "fsdp", "dp_name", "fsdp"),
        ("EDULLM_BENCH_REDUCE_BF16", "1", "reduce_dtype", "bfloat16"),
        ("EDULLM_BENCH_EP_DEGREE", "2", "ep_degree", 2),
        ("EDULLM_BENCH_PREFETCH", "3", "prefetch_factor", 3),
        ("EDULLM_BENCH_FLOAT8", "1", "float8", True),
    ],
)
def test_every_numerics_knob_moves_the_fingerprint(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str, key: str, expected: object
) -> None:
    """Each knob must be both resolved correctly and visible in the identity.

    A knob that changes the arithmetic but not the fingerprint is the defect;
    asserting the resolved value alone would not catch it, so this checks the
    identity payload itself.
    """
    for other in NUMERICS_ENV:
        monkeypatch.delenv(other, raising=False)
    baseline = ce.numerics_identity()
    monkeypatch.setenv(name, value)
    changed = ce.numerics_identity()
    assert changed[key] == expected
    assert changed != baseline, f"{name} did not change the recorded numerics"


def test_numerics_and_eval_world_size_reach_the_run_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The end-to-end property: flipping a numerics knob or the eval world
    size must change `run_fingerprint.json`, so resume refuses and two such
    runs can never be pooled by accident."""
    from production_contract import checkpoint as checkpoint_contract

    class FakeParent:
        identity = {"metric": "mtld", "sha256": "0" * 64}

    def identity_for(nproc: int) -> dict:
        return ce.scientific_identity(
            arm=ce.Arm(pacing="linear_n10", metric="mtld", lr_schedule="constant"),
            seed=42,
            total_steps=ce.TOTAL_STEPS,
            rank_microbatch_tokens=ce.RANK_MICROBATCH_TOKENS,
            parent=FakeParent(),
            task_loss_nproc=nproc,
        )

    for name in NUMERICS_ENV:
        monkeypatch.delenv(name, raising=False)
    baseline = identity_for(4)
    assert "numerics" in baseline
    assert baseline["task_loss_nproc"] == 4

    # Eval world size is part of the measurement, not a runtime detail.
    assert checkpoint_contract.make_run_fingerprint(
        identity_for(8)
    ) != checkpoint_contract.make_run_fingerprint(baseline)

    # A reduction-dtype change must be equally visible.
    monkeypatch.setenv("EDULLM_BENCH_REDUCE_BF16", "1")
    assert checkpoint_contract.make_run_fingerprint(
        identity_for(4)
    ) != checkpoint_contract.make_run_fingerprint(baseline)


def test_ratified_metric_definition_reaches_the_fingerprint() -> None:
    """Plan section 2: the difficulty metric's definition is frozen.

    compression_ratio is ratified in its RAW form -- length coupling and all
    -- so the definition, its measured rho against document length, and the
    zlib version must travel in the fingerprint. If anyone rescores or edits
    a formula, the fingerprint moves, resume refuses, and runs either side of
    the change cannot be pooled by accident.
    """
    from curriculum_pacing import DIFFICULTY_METRIC_DEFINITIONS
    from production_contract import checkpoint as checkpoint_contract

    class FakeParent:
        identity = {"metric": "compression_ratio", "sha256": "0" * 64}

    def identity_for(metric: str | None, pacing: str) -> dict:
        return ce.scientific_identity(
            arm=ce.Arm(pacing=pacing, metric=metric, lr_schedule="constant"),
            seed=42,
            total_steps=ce.TOTAL_STEPS,
            rank_microbatch_tokens=ce.RANK_MICROBATCH_TOKENS,
            parent=FakeParent(),
            task_loss_nproc=4,
        )

    compression = identity_for("compression_ratio", "linear_n10")
    definition = compression["difficulty_metric_definition"]
    assert "zlib.compress" in definition["formula"]
    assert definition["rho_rank_vs_n_tokens"] == -0.7023
    assert definition["zlib_version_at_verification"] == "1.3"

    # Each metric must carry its own definition, not a shared blob.
    mtld = identity_for("mtld", "linear_n10")
    assert mtld["difficulty_metric_definition"] != definition
    assert checkpoint_contract.make_run_fingerprint(
        mtld
    ) != checkpoint_contract.make_run_fingerprint(compression)

    # Control is ordered by nothing, so it records no definition.
    assert identity_for(None, "control")["difficulty_metric_definition"] is None

    # And every declared metric is covered (curriculum_pacing asserts this at
    # import, but state it where a reader of the contract will look).
    assert set(DIFFICULTY_METRIC_DEFINITIONS) == set(ce.DIFFICULTY_METRICS)
