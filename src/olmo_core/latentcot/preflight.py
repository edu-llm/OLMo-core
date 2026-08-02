"""
Matched-budget dry-run and integrity checks (PRD Phase 7 / the §11 pre-registration gate).

Run before spending any compute. It fails fast if a run would be confounded:

- **Matched config:** arms are identical outside the arm whitelist (:func:`assert_arms_differ_only_in`)
  — no stray LR/seed/schedule/data difference.
- **Same base checkpoint:** all arms fork the *same* base weights.
- **Disjoint seeds:** the train and test problem seeds do not overlap.
- **Per-arm compute report:** the forward-token cost per arm, **with the K continuous-thought
  passes counted, not excluded** — so CODI's higher per-step cost is visible (fairness between
  arms is enforced at *matched inference compute* in the eval, not by equalizing training FLOPs,
  which would be neither possible nor meaningful since the arms use different token views).
"""

import hashlib
from pathlib import Path
from typing import Dict, Sequence, Union

from .arms import ARM_WHITELIST, assert_arms_differ_only_in
from .evaluate import ARM_MODES, inference_token_cost
from .train_module import CodiTransformerTrainModuleConfig

__all__ = [
    "checkpoint_fingerprint",
    "assert_same_base_checkpoint",
    "assert_disjoint_seeds",
    "per_arm_compute",
    "preflight",
]


def checkpoint_fingerprint(path: Union[str, Path]) -> str:
    """A cheap, stable fingerprint of a checkpoint: SHA-1 over sorted (relpath, size)."""
    path = Path(path)
    h = hashlib.sha1()
    if path.is_dir():
        for f in sorted(path.rglob("*")):
            if f.is_file():
                h.update(str(f.relative_to(path)).encode())
                h.update(str(f.stat().st_size).encode())
    elif path.is_file():
        h.update(path.name.encode())
        h.update(str(path.stat().st_size).encode())
    else:
        raise FileNotFoundError(path)
    return h.hexdigest()


def assert_same_base_checkpoint(fingerprints: Dict[str, str]) -> None:
    """Assert every arm forks the same base checkpoint (identical fingerprints)."""
    if len(set(fingerprints.values())) > 1:
        raise AssertionError(f"arms fork different base checkpoints: {fingerprints}")


def assert_disjoint_seeds(train_examples: Sequence[dict], test_examples: Sequence[dict]) -> None:
    """Assert the train and test problem seeds do not overlap."""
    train_seeds = {ex["seed"] for ex in train_examples}
    test_seeds = {ex["seed"] for ex in test_examples}
    overlap = train_seeds & test_seeds
    if overlap:
        raise AssertionError(
            f"{len(overlap)} seed(s) shared between train and test, e.g. {sorted(overlap)[:5]}"
        )


def per_arm_compute(examples: Sequence[dict], arm_names: Sequence[str]) -> Dict[str, dict]:
    """Forward-token cost per arm over the given examples (K passes counted for CODI)."""
    report: Dict[str, dict] = {}
    for arm in arm_names:
        mode = ARM_MODES[arm]
        report[arm] = {
            "mode": mode,
            "problems": len(examples),
            "forward_token_cost": sum(inference_token_cost(ex, mode) for ex in examples),
        }
    return report


def preflight(
    arm_configs: Dict[str, CodiTransformerTrainModuleConfig],
    train_examples: Sequence[dict],
    test_examples: Sequence[dict],
    base_checkpoints: Dict[str, str],
    whitelist: Sequence[str] = ARM_WHITELIST,
) -> dict:
    """
    Run all pre-registration checks; raise on the first failure, else return a report.

    :param arm_configs: ``{arm: CodiTransformerTrainModuleConfig}`` for the arms in the run.
    :param train_examples/test_examples: encoded examples (each carrying its ``"seed"``).
    :param base_checkpoints: ``{arm: fingerprint}`` of the base checkpoint each arm forks.

    :raises AssertionError: if any matched-budget / integrity check fails.
    """
    assert_arms_differ_only_in(list(arm_configs.values()), whitelist)
    assert_same_base_checkpoint(base_checkpoints)
    assert_disjoint_seeds(train_examples, test_examples)
    return {
        "matched_config": True,
        "same_base_checkpoint": True,
        "disjoint_seeds": True,
        "num_train_problems": len(train_examples),
        "num_test_problems": len(test_examples),
        "per_arm_compute": per_arm_compute(train_examples, list(arm_configs)),
    }
