"""
CODI train-module integration (PRD Phase 4.2).

Thin wrapper over :class:`~olmo_core.train.train_module.TransformerTrainModule` that
computes the CODI loss (:func:`olmo_core.latentcot.loss.codi_loss`) on a batch of
encoded examples. The heavy lifting (optimizer, scheduler, grad clipping, checkpointing,
metric plumbing) is inherited unchanged; only the per-batch loss changes.

Batches are ``{"examples": [<encode_example dict>, ...]}`` — the continuous-thought
student is processed per example, so no padded tensor batch is required (see the note in
:mod:`olmo_core.latentcot.loss`).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, cast

import torch
import torch.distributed.checkpoint.state_dict as dist_cp_sd

from olmo_core.config import DType
from olmo_core.nn.transformer import Transformer
from olmo_core.train.common import ReduceType
from olmo_core.train.train_module import (
    TransformerTrainModule,
    TransformerTrainModuleConfig,
)

from .loss import VocabReg, arm_loss
from .moe import (
    collect_router_metrics,
    count_forwards,
    finish_step,
    is_moe_model,
    normalized_aux_losses,
    reset_router_state,
)

__all__ = ["CodiTransformerTrainModule", "CodiTransformerTrainModuleConfig"]


class CodiTransformerTrainModule(TransformerTrainModule):
    """A :class:`TransformerTrainModule` whose per-batch loss is the CODI loss."""

    def __init__(
        self,
        *,
        arm_mode: str = "codi",
        num_continuous_thoughts: int = 8,
        distill_weight: float = 1.0,
        vocab_reg: VocabReg = "none",
        vocab_reg_weight: float = 0.0,
        vocab_reg_entropy_floor: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.arm_mode = arm_mode
        self.num_continuous_thoughts = num_continuous_thoughts
        self.distill_weight = distill_weight
        self.vocab_reg = vocab_reg
        self.vocab_reg_weight = vocab_reg_weight
        self.vocab_reg_entropy_floor = vocab_reg_entropy_floor

    def train_batch(self, batch: Dict[str, Any], dry_run: bool = False) -> None:
        """
        Compute + backprop the arm's loss for a batch of encoded examples.

        This overrides :meth:`TransformerTrainModule.train_batch` wholesale rather than extending
        it, because the CODI loss is per-example rather than over a token-array microbatch. That
        means the parent's MoE bookkeeping is *not* inherited, so it is repeated here — see
        :mod:`olmo_core.latentcot.moe` for what each piece is for and why the missing
        ``normalized_aux_losses`` would be an arm-dependent confound rather than a rough edge.
        """
        examples = batch["examples"]
        moe = is_moe_model(self.model)
        if moe:
            reset_router_state(self.model)
        forwards = count_forwards(examples, mode=self.arm_mode) if moe else 1
        with normalized_aux_losses(self.model, forwards):
            with self._train_microbatch_context(0, 1), self._model_forward_context():
                loss, metrics = arm_loss(
                    self.model,
                    examples,
                    mode=self.arm_mode,
                    distill_weight=self.distill_weight,
                    vocab_reg=self.vocab_reg,
                    vocab_reg_weight=self.vocab_reg_weight,
                    vocab_reg_entropy_floor=self.vocab_reg_entropy_floor,
                    label_ignore_index=self.label_ignore_index,
                )
        loss.backward()

        # The bias_gamma score-bias update, which the parent would have done here.
        finish_step(self.model, dry_run=dry_run)

        if dry_run:
            if moe:
                reset_router_state(self.model)
            return

        # Primary CE (for the SkipStepOptimizer): student for CODI, else the anchor's CE.
        primary = metrics.get("ce_student") or metrics.get("ce_teacher") or metrics.get("ce_answer")
        if primary is not None:
            self.record_ce_loss(torch.tensor(primary, device=self.device), ReduceType.mean)
        for name, value in metrics.items():
            self.record_metric(
                name, torch.tensor(value, device=self.device), ReduceType.mean, namespace="train"
            )
        for name, value in collect_router_metrics(self.model).items():
            self.record_metric(
                name, torch.tensor(value, device=self.device), ReduceType.mean, namespace="train"
            )


@dataclass
class CodiTransformerTrainModuleConfig(TransformerTrainModuleConfig):
    """Config for :class:`CodiTransformerTrainModule` (adds the CODI hyperparameters)."""

    arm_mode: str = "codi"  # one of: explicit_cot, no_cot, codi
    num_continuous_thoughts: int = 8
    distill_weight: float = 1.0
    vocab_reg: VocabReg = "none"  # one of: none, R1, R2, L2
    vocab_reg_weight: float = 0.0
    vocab_reg_entropy_floor: float = 0.0

    def build(
        self, model: Transformer, device: Optional[torch.device] = None
    ) -> CodiTransformerTrainModule:
        """Build the :class:`CodiTransformerTrainModule` (mirrors the base transforms)."""
        if self.pp_config is not None:
            raise NotImplementedError("CODI train module does not support pipeline parallelism")

        kwargs = self.as_dict(exclude_none=True, recurse=False)
        if (autocast_precision := kwargs.pop("autocast_precision", None)) is not None:
            kwargs["autocast_precision"] = cast(DType, autocast_precision).as_pt()
        if (save_opts := kwargs.pop("state_dict_save_opts", None)) is not None:
            kwargs["state_dict_save_opts"] = dist_cp_sd.StateDictOptions(**save_opts)
        if (load_opts := kwargs.pop("state_dict_load_opts", None)) is not None:
            kwargs["state_dict_load_opts"] = dist_cp_sd.StateDictOptions(**load_opts)

        return CodiTransformerTrainModule(model=model, device=device, **kwargs)
