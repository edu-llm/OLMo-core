"""Small OLMo train-module extension for token weights and passive excess loss."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.distributed as dist
from torch import Tensor

from olmo_core.data.utils import get_labels, split_batch
from olmo_core.distributed.utils import get_local_tensor, is_distributed
from olmo_core.nn.lm_head import LMOutputWithLoss
from olmo_core.optim import SkipStepOptimizer
from olmo_core.train import ReduceType
from olmo_core.train.callbacks import Callback
from olmo_core.train.train_module import TransformerTrainModule

from .selection import (
    EMAHistory,
    WeightShadow,
    capture_last_attention,
    ema_alpha,
    scores_from_capture,
    selection_weights,
)


def load_flat_weights(path: str | Path) -> dict[str, Tensor]:
    source = Path(path)
    if not source.is_file() or str(source).startswith("s3://"):
        raise ValueError(f"reference must be a materialized local .pt file: {source}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, Mapping):
                payload = nested
                break
    if not isinstance(payload, Mapping) or not payload:
        raise ValueError(f"reference checkpoint does not contain a state dict: {source}")
    weights = {str(name): value for name, value in payload.items() if isinstance(value, Tensor)}
    if len(weights) != len(payload):
        raise ValueError(f"reference state dict contains non-tensor values: {source}")
    return weights


@dataclass(frozen=True)
class TokenSelectionConfig:
    method: str
    keep_fraction: float
    total_steps: int
    seed: int = 42
    reference_path: Optional[str] = None
    early_reference_path: Optional[str] = None
    late_reference_path: Optional[str] = None
    passive_reference_path: Optional[str] = None
    ema_seed: Optional[str] = None
    ema_alpha: Optional[float] = None
    ema_tau: Optional[float] = None


class TokenSelectionState:
    def __init__(self, config: TokenSelectionConfig, model) -> None:
        self.config = config
        self.completed_steps = 0
        self.reference = (
            WeightShadow.from_state_dict(model, load_flat_weights(config.reference_path))
            if config.reference_path
            else None
        )
        self.early = (
            WeightShadow.from_state_dict(model, load_flat_weights(config.early_reference_path))
            if config.early_reference_path
            else None
        )
        self.late = (
            WeightShadow.from_state_dict(model, load_flat_weights(config.late_reference_path))
            if config.late_reference_path
            else None
        )
        self.passive = (
            WeightShadow.from_state_dict(model, load_flat_weights(config.passive_reference_path))
            if config.passive_reference_path
            else None
        )
        self.ema: Optional[EMAHistory] = None
        if config.method == "rel_ema":
            seed = None
            if config.ema_seed == "refhq":
                if not config.reference_path:
                    raise ValueError("RefHQ-seeded relative EMA requires reference_path")
                seed = load_flat_weights(config.reference_path)
            elif config.ema_seed != "zero":
                raise ValueError("relative EMA must explicitly select zero or refhq initialization")
            self.ema = EMAHistory(model, seed=seed)

    def alpha(self) -> float:
        return ema_alpha(
            self.completed_steps,
            tau=self.config.ema_tau,
            constant=self.config.ema_alpha,
        )

    def after_optimizer_step(self, model) -> None:
        if self.ema is not None:
            self.ema.update(model, self.alpha())
        self.completed_steps += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "completed_steps": self.completed_steps,
            "ema": self.ema.state_dict() if self.ema is not None else None,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported token-selection checkpoint state")
        self.completed_steps = int(state["completed_steps"])
        if self.ema is not None:
            if not isinstance(state.get("ema"), Mapping):
                raise ValueError("relative-EMA resume state is missing EMA weights")
            self.ema.load_state_dict(state["ema"])
        elif state.get("ema") is not None:
            raise ValueError("non-EMA arm received EMA resume state")


class TokenWeightedTrainModule(TransformerTrainModule):
    """Uses OLMo's per-token CE/Z outputs, changing only their final reduction."""

    def __init__(self, *args, selection_config: TokenSelectionConfig, **kwargs):
        super().__init__(*args, **kwargs)
        if self.tp_enabled or self.cp_enabled:
            raise ValueError(
                "token selection requires full per-token losses; TP/CP are unsupported"
            )
        self.selection_config = selection_config
        self.selection_state = TokenSelectionState(selection_config, self.model)

    @staticmethod
    def _valid(labels: Tensor) -> Tensor:
        return labels != -100

    @staticmethod
    def _loss_tensor(output: LMOutputWithLoss, labels: Tensor, name: str) -> Tensor:
        value = getattr(output, name)
        if value is None:
            raise RuntimeError(f"OLMo output did not provide required {name}")
        return get_local_tensor(value).reshape_as(labels)

    @contextlib.contextmanager
    def _score_mode(self):
        was_training = self.model.training
        old_mode = self._model_mode
        self.model.eval()
        try:
            yield
        finally:
            self.model.train(was_training)
            self._model_mode = old_mode

    def _score(self, shadow: WeightShadow | EMAHistory, ids, labels, kwargs) -> Tensor:
        with self._score_mode(), torch.no_grad(), shadow.swap_to(self.model):
            output = self.model_forward(
                ids,
                labels=labels,
                ignore_index=self.label_ignore_index,
                loss_reduction="none",
                return_logits=False,
                **kwargs,
            )
        if not isinstance(output, LMOutputWithLoss):
            raise RuntimeError("OLMo did not return per-token scoring loss")
        self.model.reset_auxiliary_metrics()
        return self._loss_tensor(output, labels, "ce_loss")

    def _planned_weight(self, labels: Tensor, batch: Mapping[str, Any]) -> float:
        valid = self._valid(labels)
        provided = batch.get("token_weight")
        if provided is not None:
            weight = provided.to(device=labels.device, dtype=torch.float32)
            if weight.shape != labels.shape or (weight < 0).any():
                raise ValueError("token_weight must be non-negative and match labels")
            return float((weight * valid).sum().item())
        if self.selection_config.method == "full":
            return float(valid.sum().item())
        count = valid.sum(-1)
        keep = torch.minimum(
            torch.clamp(
                (count.float() * self.selection_config.keep_fraction).round().long(), min=1
            ),
            count,
        )
        return float(keep.sum().item())

    def train_batch(self, batch: dict[str, Any], dry_run: bool = False):
        self._set_model_mode("train")
        if "labels" not in batch:
            batch["labels"] = get_labels(batch, label_ignore_index=self.label_ignore_index)
        labels = batch["labels"]
        divisor = self._planned_weight(labels, batch)
        if divisor <= 0:
            raise RuntimeError("batch contains no weighted target tokens")
        sequence_length = batch["input_ids"].shape[1]
        micro_batches = split_batch(batch, self.rank_microbatch_size // sequence_length)
        ce_batch = torch.zeros((), device=self.device)
        z_batch = (
            torch.zeros((), device=self.device) if self.z_loss_multiplier is not None else None
        )
        observed_weight = 0.0

        for micro_index, micro in enumerate(micro_batches):
            with self._train_microbatch_context(micro_index, len(micro_batches)):
                ids, micro_labels, model_kwargs = self._prepare_batch(dict(micro))
                assert micro_labels is not None
                valid = self._valid(micro_labels)
                state = self.selection_state
                config = self.selection_config
                current = history = reference = early = late = attention = None

                if state.ema is not None:
                    history = self._score(state.ema, ids, micro_labels, model_kwargs)
                elif config.method in {"rho_excess", "middle_ppl"}:
                    if state.reference is None:
                        raise RuntimeError(f"{config.method} is missing its frozen reference")
                    reference = self._score(state.reference, ids, micro_labels, model_kwargs)
                elif config.method == "learnability":
                    if state.early is None or state.late is None:
                        raise RuntimeError("learnability is missing early/late references")
                    early = self._score(state.early, ids, micro_labels, model_kwargs)
                    late = self._score(state.late, ids, micro_labels, model_kwargs)

                capture = (
                    capture_last_attention(self.model)
                    if config.method == "attention_topk"
                    else contextlib.nullcontext()
                )
                with capture as captured:
                    output = self.model_forward(
                        ids,
                        labels=micro_labels,
                        ignore_index=self.label_ignore_index,
                        loss_reduction="none",
                        z_loss_multiplier=self.z_loss_multiplier,
                        return_logits=False,
                        **model_kwargs,
                    )
                if not isinstance(output, LMOutputWithLoss):
                    raise RuntimeError("OLMo did not return per-token train losses")
                token_ce = self._loss_tensor(output, micro_labels, "ce_loss")
                if config.method in {"rho_excess", "rel_ema"}:
                    current = token_ce.detach()
                if config.method == "attention_topk":
                    attention = scores_from_capture(captured)

                supplied = micro.get("token_weight")
                if supplied is not None:
                    weights = supplied.to(self.device, dtype=torch.float32) * valid
                else:
                    weights = selection_weights(
                        config.method,
                        valid=valid,
                        keep_fraction=config.keep_fraction,
                        step=state.completed_steps,
                        seed=config.seed + micro_index,
                        current=current,
                        history=history,
                        reference=reference,
                        early=early,
                        late=late,
                        attention=attention,
                    )
                observed_weight += float(weights.sum().item())
                ce_loss = (token_ce.float() * weights).sum() / divisor
                loss = ce_loss
                if z_batch is not None:
                    token_z = self._loss_tensor(output, micro_labels, "z_loss")
                    z_loss = (token_z.float() * weights).sum() / divisor
                    z_batch += z_loss.detach()
                    loss = loss + z_loss
                ce_batch += ce_loss.detach()
                loss.backward()

                if state.passive is not None:
                    passive_ref = self._score(state.passive, ids, micro_labels, model_kwargs)
                    excess = (token_ce.detach() - passive_ref)[valid].mean()
                    self.record_metric(
                        "passive excess loss",
                        excess,
                        ReduceType.mean,
                        namespace="train",
                    )

        if abs(observed_weight - divisor) > 1e-4:
            raise RuntimeError(
                f"token-weight divisor mismatch: planned {divisor}, observed {observed_weight}"
            )
        self.model.post_batch(dry_run=dry_run)
        if dry_run:
            self.model.reset_auxiliary_metrics()
            return
        if isinstance(self.optim, SkipStepOptimizer):
            if is_distributed():
                ce_batch.div_(self._reduce_divide_factor)
                dist.all_reduce(ce_batch)
                ce_batch.div_(self.world_size)
                ce_batch.mul_(self._reduce_divide_factor)
            self.record_ce_loss(ce_batch)
            self.optim.latest_loss = ce_batch
        else:
            self.record_ce_loss(ce_batch, ReduceType.mean)
        self.record_metric(
            "selected token fraction",
            observed_weight / max(float(labels.numel()), 1.0),
            ReduceType.mean,
            namespace="train",
        )
        if z_batch is not None:
            self.record_metric("Z loss", z_batch, ReduceType.mean, namespace="train")
        for name, (value, reduction) in self.model.compute_auxiliary_metrics(reset=True).items():
            self.record_metric(name, value, reduction, namespace="train")


class TokenSelectionStateCallback(Callback):
    """Advances EMA before the priority-1 checkpointer and persists resume state."""

    priority = 3

    def __init__(self) -> None:
        self._last_step: Optional[int] = None
        self._pending: Optional[Mapping[str, Any]] = None

    def post_attach(self) -> None:
        if self._pending is not None:
            self.trainer.train_module.selection_state.load_state_dict(self._pending)
            self._pending = None

    def post_train_batch(self) -> None:
        step = int(self.trainer.global_step)
        if step == self._last_step:
            return
        self.trainer.train_module.selection_state.after_optimizer_step(
            self.trainer.train_module.model
        )
        self._last_step = step

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "last_step": self._last_step,
            "selection": self.trainer.train_module.selection_state.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported token-selection callback state")
        self._last_step = state.get("last_step")
        if hasattr(self, "trainer"):
            self.trainer.train_module.selection_state.load_state_dict(state["selection"])
        else:
            self._pending = state["selection"]
