"""
Experiment arms and the confound-control assertion (PRD Phase 5).

All arms share one base training config (model, data, optimizer, WSD schedule, seed,
base checkpoint) and differ *only* in a small whitelist of arm-defining fields, so any
measured difference is attributable to the intervention rather than an accidental
confound. :func:`assert_arms_differ_only_in` enforces this before a run.

Arms:

- **A0** ``explicit_cot`` — CE on the written-out teacher CoT (readable upper anchor).
- **A1** ``no_cot`` — CE on the direct ``question <distill> answer`` view (lower anchor).
- **A2** ``codi`` — continuous thoughts, no vocab regularizer.
- **A3** ``codi`` + ``R1`` — the novel vocabulary-manifold regularizer.
- **A4** ``codi`` + ``L2`` — matched-strength generic regularizer (the confound control
  that isolates the *vocabulary-space direction* of R1 from mere regularization).

The whitelist adds ``arm_mode`` to the PRD's ``(K, vocab_reg, vocab_reg_weight)`` because
A0/A1 are structurally different training objectives, not just field tweaks.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Sequence

from .train_module import CodiTransformerTrainModuleConfig

__all__ = ["Arm", "ARMS", "ARM_WHITELIST", "build_arm_config", "assert_arms_differ_only_in"]

DEFAULT_K = 8
DEFAULT_GAMMA = 0.01

# Fields arms are allowed to differ in. Everything else must be identical.
ARM_WHITELIST = ("arm_mode", "num_continuous_thoughts", "vocab_reg", "vocab_reg_weight")


@dataclass(frozen=True)
class Arm:
    """A single experiment arm (the fields that may vary between arms)."""

    name: str
    arm_mode: str
    num_continuous_thoughts: int
    vocab_reg: str
    vocab_reg_weight: float


# K is held constant across arms (A0/A1 ignore it), so arms differ only in mode + reg.
ARMS: Dict[str, Arm] = {
    "A0": Arm("A0_explicit_cot", "explicit_cot", DEFAULT_K, "none", 0.0),
    "A1": Arm("A1_no_cot", "no_cot", DEFAULT_K, "none", 0.0),
    "A2": Arm("A2_codi", "codi", DEFAULT_K, "none", 0.0),
    "A3": Arm("A3_codi_r1", "codi", DEFAULT_K, "R1", DEFAULT_GAMMA),
    "A4": Arm("A4_codi_l2", "codi", DEFAULT_K, "L2", DEFAULT_GAMMA),
}


def build_arm_config(
    base: CodiTransformerTrainModuleConfig, arm: Arm
) -> CodiTransformerTrainModuleConfig:
    """Return ``base`` with only the arm-defining fields overridden."""
    return replace(
        base,
        arm_mode=arm.arm_mode,
        num_continuous_thoughts=arm.num_continuous_thoughts,
        vocab_reg=arm.vocab_reg,
        vocab_reg_weight=arm.vocab_reg_weight,
    )


def assert_arms_differ_only_in(
    configs: Sequence[CodiTransformerTrainModuleConfig],
    whitelist: Sequence[str] = ARM_WHITELIST,
) -> None:
    """
    Assert that a set of arm configs are identical outside ``whitelist``.

    :raises AssertionError: naming the offending fields if any non-whitelisted field
        differs between arms (an accidental confound, e.g. a different LR or seed).
    """
    allowed = set(whitelist)
    stripped: List[dict] = [
        {k: v for k, v in cfg.as_dict().items() if k not in allowed} for cfg in configs
    ]
    base = stripped[0]
    for i, other in enumerate(stripped[1:], start=1):
        if other != base:
            diffs = sorted(k for k in set(base) | set(other) if base.get(k) != other.get(k))
            raise AssertionError(
                f"arm {i} differs from arm 0 outside the whitelist in fields: {diffs}"
            )
