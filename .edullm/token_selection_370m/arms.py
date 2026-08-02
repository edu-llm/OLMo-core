"""Immutable scientific identities for the approved 370M arms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Method = Literal[
    "rho_excess",
    "rel_ema",
    "middle_ppl",
    "attention_topk",
    "blade",
]

REGMIX = "pretrain/regmix-10b"
REFHQ = "pretrain/refhq-regmix-5p5b"
REFHQ_STEP_1315 = "s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/"
REFHQ_LATE_STEPS = (1000, 1125, 1315)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    method: Method
    dataset_id: str
    run_id: str
    keep_fraction: float = 1.0
    max_tokens: Optional[int] = 9_900_000_000
    reference_contract: Optional[str] = None
    early_reference_contract: Optional[str] = None
    late_reference_contract: Optional[str] = None
    ema_seed: Optional[Literal["zero", "refhq"]] = None
    ema_alpha: Optional[float] = None
    ema_tau: Optional[float] = None
    requires_refhq_stream: bool = False

    @property
    def wandb_project(self) -> str:
        return f"token-selection-{self.name}"

    @property
    def is_online_selection(self) -> bool:
        return True


ARM_SPECS: dict[str, ArmSpec] = {
    "rho-1": ArmSpec(
        "rho-1",
        "rho_excess",
        REGMIX,
        "rho-1-regmix10b-v1",
        keep_fraction=0.6,
        reference_contract=REFHQ_STEP_1315,
    ),
    "rel-ema-exp": ArmSpec(
        "rel-ema-exp",
        "rel_ema",
        REGMIX,
        "rel-ema-exp-10b-scratch-v1",
        keep_fraction=0.6,
        ema_seed="zero",
        ema_tau=300.0,
    ),
    "middle-ppl-token": ArmSpec(
        "middle-ppl-token",
        "middle_ppl",
        REGMIX,
        "middle-ppl-token-10b-v2",
        keep_fraction=0.6,
        late_reference_contract=f"average RefHQ steps {REFHQ_LATE_STEPS}",
    ),
    "attention": ArmSpec(
        "attention",
        "attention_topk",
        REGMIX,
        "attention-topk-10b-scratch-v1",
        keep_fraction=0.6,
    ),
    "blade": ArmSpec(
        "blade",
        "blade",
        REGMIX,
        "blade-regmix10b-v2",
        keep_fraction=0.6,
        requires_refhq_stream=True,
    ),
}


def get_arm(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown token-selection arm {name!r}; expected one of {sorted(ARM_SPECS)}"
        ) from None
