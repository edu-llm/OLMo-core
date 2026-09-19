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
    "random",
    "full",
]

REGMIX = "pretrain/regmix-10b"
# BLADE's HQ/reference-update stream. Pin the immutable version at launch.
REFHQ = "pretrain/refhq-instruct"
RHO_REFERENCE_CHECKPOINT = (
    "s3://edullm-checkpoints/olmo-370m/"
    "edullm-370M-refhq-instruct-v3/checkpoints/step940/"
)
REFHQ_LATE_STEPS = (1000, 1125, 1315)


@dataclass(frozen=True)
class ArmSpec:
    name: str
    method: Method
    dataset_id: str
    run_id: str
    keep_fraction: float = 1.0
    max_tokens: Optional[int] = 9_900_000_000
    wandb_project_override: Optional[str] = None
    reference_contract: Optional[str] = None
    early_reference_contract: Optional[str] = None
    late_reference_contract: Optional[str] = None
    ema_seed: Optional[Literal["zero", "refhq"]] = None
    ema_alpha: Optional[float] = None
    ema_tau: Optional[float] = None
    requires_refhq_stream: bool = False

    @property
    def wandb_project(self) -> str:
        return self.wandb_project_override or f"token-selection-{self.name}"

    @property
    def is_online_selection(self) -> bool:
        return True


ARM_SPECS: dict[str, ArmSpec] = {
    # Full cross-entropy baseline on the identical corpus, step budget, seeds,
    # and 125-step task-loss ladder as every selection arm. method="full" keeps
    # every valid token in the loss, so recipe.py routes this to the stock
    # TransformerTrainModule (no selection callback, no scoring forward pass).
    # Replaces the hpo-ladder clone that the first draft used as its control.
    "full-loss-control": ArmSpec(
        "full-loss-control",
        "full",
        REGMIX,
        "full-loss-control-regmix10b-v3",
        keep_fraction=1.0,
        wandb_project_override="token-selection",
    ),
    "rho-1": ArmSpec(
        "rho-1",
        "rho_excess",
        REGMIX,
        "rho-1-regmix10b-v1",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
        reference_contract=RHO_REFERENCE_CHECKPOINT,
    ),
    "rel-ema-exp": ArmSpec(
        "rel-ema-exp",
        "rel_ema",
        REGMIX,
        "rel-ema-exp-10b-scratch-v1",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
        ema_seed="zero",
        ema_tau=300.0,
    ),
    "middle-ppl-token": ArmSpec(
        "middle-ppl-token",
        "middle_ppl",
        REGMIX,
        "middle-ppl-token-10b-v2",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
        late_reference_contract=f"average RefHQ steps {REFHQ_LATE_STEPS}",
    ),
    "attention": ArmSpec(
        "attention",
        "attention_topk",
        REGMIX,
        "attention-topk-10b-scratch-v1",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
    ),
    "blade": ArmSpec(
        "blade",
        "blade",
        REGMIX,
        "blade-regmix10b-refhq-instruct-v3-v1",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
        requires_refhq_stream=True,
    ),
    "random-control": ArmSpec(
        "random-control",
        "random",
        REGMIX,
        "random-control-regmix10b-v1",
        keep_fraction=0.6,
        wandb_project_override="token-selection",
    ),
}


def get_arm(name: str) -> ArmSpec:
    try:
        return ARM_SPECS[name]
    except KeyError:
        raise ValueError(
            f"unknown token-selection arm {name!r}; expected one of {sorted(ARM_SPECS)}"
        ) from None
