"""BLADE dynamic-reference controller integrated through public callback APIs."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

log = logging.getLogger(__name__)

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:  # OLMo's optional runtime dependencies are not available in pure unit-test hosts.
    from olmo_core.data.utils import get_labels, split_batch
    from olmo_core.distributed.utils import get_rank, get_world_size, is_distributed
    from olmo_core.nn.lm_head import LMOutputWithLoss
    from olmo_core.train.callbacks import Callback
except ImportError:  # pragma: no cover - production images take the branch above.
    Callback = object  # type: ignore[assignment,misc]

    class LMOutputWithLoss:  # type: ignore[no-redef]
        pass

    def get_rank() -> int:
        return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    def get_world_size() -> int:
        return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

    def is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def get_labels(batch: dict[str, Any], label_ignore_index: int = -100) -> Tensor:
        labels = batch["input_ids"].clone()
        if batch.get("label_mask") is not None:
            labels.masked_fill_(~batch["label_mask"], label_ignore_index)
        return F.pad(labels[..., 1:], (0, 1), value=label_ignore_index)

    def split_batch(batch: dict[str, Any], num_microbatch_instances: int) -> list[dict[str, Any]]:
        batch_size = batch["input_ids"].shape[0]
        return [
            {
                key: value[start : start + num_microbatch_instances]
                for key, value in batch.items()
            }
            for start in range(0, batch_size, num_microbatch_instances)
        ]


BLADE_START = 500
BLADE_SYNC_STEPS = (500, 875, 1250, 1625, 2000)
BLADE_TAU = 375
BLADE_K = 75
BLADE_GAMMA = 0.6
BLADE_LAMBDA = 1.0
BLADE_REFERENCE_MICROBATCH_TOKENS = 8_192
BLADE_SELECTION_MICROBATCH_TOKENS = 32_768
BLADE_CHECKPOINT_FORMAT = "blade_proxy_dynamic_ref_v2"


@dataclass(frozen=True)
class BladeSchedule:
    start: int = BLADE_START
    sync_steps: tuple[int, ...] = BLADE_SYNC_STEPS
    tau: int = BLADE_TAU
    k_steps: int = BLADE_K
    gamma: float = BLADE_GAMMA
    lambda_penalty: float = BLADE_LAMBDA

    def validate(self, total_steps: int) -> None:
        if self != BladeSchedule():
            raise ValueError("the approved BLADE schedule is locked; use a new run identity")
        if any(step > total_steps for step in self.sync_steps):
            raise ValueError("BLADE sync schedule exceeds the run duration")


class ResumableBatchStream:
    """Infinite stream whose loader position and epoch survive callback checkpoints."""

    def __init__(self, loader, *, epoch: int = 1):
        self.loader = loader
        self.epoch = int(epoch)
        self._iterator = None

    def next(self) -> dict[str, Any]:
        while True:
            if self._iterator is None:
                self.loader.reshuffle(self.epoch)
                self._iterator = iter(self.loader)
            try:
                return next(self._iterator)
            except StopIteration:
                self.epoch += 1
                self._iterator = None

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "epoch": self.epoch, "loader": self.loader.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 1:
            raise ValueError("unsupported BLADE stream state")
        self.epoch = int(state["epoch"])
        self.loader.load_state_dict(state["loader"])
        self._iterator = None


def _output_ce(output: Any, labels: Tensor) -> Tensor:
    if not isinstance(output, LMOutputWithLoss):
        raise RuntimeError("BLADE requires OLMo per-token LM losses")
    return output.ce_loss.reshape_as(labels)


@contextlib.contextmanager
def _autocast(device: torch.device):
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield


def _full_proxy_state(model: nn.Module) -> dict[str, Tensor]:
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

        # This state is loaded into a replicated reference model on every rank.
        # Combining full_state_dict=True with cpu_offload=True only materializes
        # the state on rank 0 and returns an empty dict elsewhere.
        return get_model_state_dict(
            model, options=StateDictOptions(full_state_dict=True, cpu_offload=False)
        )
    except ImportError:
        return {name: value.detach().cpu() for name, value in model.state_dict().items()}


class BladeCallback(Callback):
    """Runs sync/K updates in ``pre_step`` and stores all non-proxy state in checkpoints.

    The proxy model and optimizer remain ordinary OLMo checkpointer state. This callback
    contributes the dynamic reference, its optimizer, both K-update streams, and schedule
    cursor to the trainer state saved in the same checkpoint.
    """

    priority = 4

    def __init__(
        self,
        *,
        total_steps: int,
        reference_factory: Callable[[], nn.Module],
        reference_train_stream: ResumableBatchStream,
        refhq_stream: ResumableBatchStream,
        schedule: BladeSchedule = BladeSchedule(),
        reference_lr: float = 4e-4,
        max_grad_norm: float = 1.0,
        reference_microbatch_tokens: int = BLADE_REFERENCE_MICROBATCH_TOKENS,
        selection_microbatch_tokens: int = BLADE_SELECTION_MICROBATCH_TOKENS,
    ) -> None:
        schedule.validate(total_steps)
        self.total_steps = int(total_steps)
        self.reference_factory = reference_factory
        self.reference_train_stream = reference_train_stream
        self.refhq_stream = refhq_stream
        self.schedule = schedule
        self.reference_lr = float(reference_lr)
        self.max_grad_norm = float(max_grad_norm)
        self.reference_microbatch_tokens = int(reference_microbatch_tokens)
        if self.reference_microbatch_tokens <= 0:
            raise ValueError("BLADE reference microbatch tokens must be positive")
        self.selection_microbatch_tokens = int(selection_microbatch_tokens)
        if self.selection_microbatch_tokens <= 0:
            raise ValueError("BLADE selection microbatch tokens must be positive")
        self.reference: Optional[nn.Module] = None
        self.reference_optim: Optional[torch.optim.Optimizer] = None
        self.completed_step = 0
        self.last_sync: Optional[int] = None
        self._pending: Optional[Mapping[str, Any]] = None

    def _k_progress_path(self) -> Path:
        return self.trainer.work_dir / "blade_k_progress.json"

    def _write_k_progress(self, *, trainer_step: int, k_step: int) -> None:
        if get_rank() != 0:
            return
        payload = {
            "trainer_step": int(trainer_step),
            "k_step": int(k_step),
            "k_total": int(self.schedule.k_steps),
            "updated_at": time.time(),
        }
        path = self._k_progress_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        log.info(
            "BLADE reference K-update %d/%d at trainer step %d",
            k_step,
            self.schedule.k_steps,
            trainer_step,
        )

    def _clear_k_progress(self) -> None:
        if get_rank() != 0:
            return
        with contextlib.suppress(FileNotFoundError):
            self._k_progress_path().unlink()

    def _new_reference(self) -> None:
        self.reference = self.reference_factory()
        self.reference_optim = torch.optim.AdamW(
            self.reference.parameters(),
            lr=self.reference_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            foreach=False,
        )

    def _sync_from_proxy(self) -> None:
        if self.reference is None:
            self._new_reference()
        assert self.reference is not None
        state = _full_proxy_state(self.trainer.train_module.model)
        self.reference.load_state_dict(state, strict=True)

    def _mean_ce(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        loss_div_factor: Optional[Tensor] = None,
    ) -> Tensor:
        ids = batch["input_ids"].to(next(model.parameters()).device)
        labels = get_labels({"input_ids": ids})
        valid = (
            (labels != -100).sum().clamp(min=1)
            if loss_div_factor is None
            else loss_div_factor.to(ids.device)
        )
        kwargs = {
            key: value.to(ids.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
            if key != "input_ids"
        }
        with _autocast(ids.device):
            output = model(
                ids,
                labels=labels,
                ignore_index=-100,
                loss_reduction="sum",
                loss_div_factor=valid,
                return_logits=False,
                **kwargs,
            )
        if not isinstance(output, LMOutputWithLoss):
            raise RuntimeError("BLADE reference update requires LMOutputWithLoss")
        return output.loss

    def _backward_mean_ce(
        self,
        model: nn.Module,
        batch: dict[str, Any],
        *,
        weight: float,
    ) -> None:
        sequence_length = int(batch["input_ids"].shape[1])
        if self.reference_microbatch_tokens < sequence_length:
            raise RuntimeError(
                "BLADE reference microbatch token limit is smaller than sequence length"
            )
        micro_batches = split_batch(
            batch, self.reference_microbatch_tokens // sequence_length
        )
        labels = get_labels(batch)
        divisor = (labels != -100).sum().clamp(min=1)
        for micro_batch in micro_batches:
            loss = self._mean_ce(
                model, micro_batch, loss_div_factor=divisor
            )
            (float(weight) * loss).backward()
            del loss

    def _run_k_updates(self, *, trainer_step: Optional[int] = None) -> None:
        assert self.reference is not None and self.reference_optim is not None
        trainer_step = (
            int(self.trainer.global_step) if trainer_step is None else int(trainer_step)
        )
        self.reference.train()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(True)
        try:
            for k_idx in range(self.schedule.k_steps):
                self._write_k_progress(trainer_step=trainer_step, k_step=k_idx + 1)
                self.reference_optim.zero_grad(set_to_none=True)
                self._backward_mean_ce(
                    self.reference,
                    self.reference_train_stream.next(),
                    weight=self.schedule.lambda_penalty,
                )
                self._backward_mean_ce(
                    self.reference,
                    self.refhq_stream.next(),
                    weight=1.0,
                )
                if is_distributed() and get_world_size() > 1:
                    for parameter in self.reference.parameters():
                        if parameter.grad is not None:
                            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG)
                torch.nn.utils.clip_grad_norm_(self.reference.parameters(), self.max_grad_norm)
                self.reference_optim.step()
        finally:
            self._clear_k_progress()
        self.reference.eval()
        for parameter in self.reference.parameters():
            parameter.requires_grad_(False)

    def _proxy_and_reference_ce_microbatch(
        self, batch: dict[str, Any]
    ) -> tuple[Tensor, Tensor, Tensor]:
        module = self.trainer.train_module
        ids = batch["input_ids"].to(module.device)
        labels = get_labels(batch, label_ignore_index=module.label_ignore_index).to(module.device)
        model_kwargs = {
            key: value.to(module.device) if isinstance(value, Tensor) else value
            for key, value in batch.items()
            if key not in {"input_ids", "labels", "label_mask", "instance_mask", "attention_mask"}
        }
        was_training = module.model.training
        module.model.eval()
        assert self.reference is not None
        self.reference.eval()
        with torch.no_grad():
            proxy = module.model_forward(
                ids,
                labels=labels,
                ignore_index=module.label_ignore_index,
                loss_reduction="none",
                return_logits=False,
                **model_kwargs,
            )
            reference = self.reference(
                ids,
                labels=labels,
                ignore_index=module.label_ignore_index,
                loss_reduction="none",
                return_logits=False,
                **model_kwargs,
            )
        module.model.train(was_training)
        module._model_mode = "train" if was_training else "eval"
        return labels, _output_ce(proxy, labels), _output_ce(reference, labels)

    def _proxy_and_reference_ce(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
        sequence_length = int(batch["input_ids"].shape[1])
        if self.selection_microbatch_tokens < sequence_length:
            raise RuntimeError(
                "BLADE selection microbatch token limit is smaller than sequence length"
            )
        labels, proxy_ce, reference_ce = [], [], []
        for micro_batch in split_batch(
            batch, self.selection_microbatch_tokens // sequence_length
        ):
            micro_labels, micro_proxy_ce, micro_reference_ce = (
                self._proxy_and_reference_ce_microbatch(micro_batch)
            )
            labels.append(micro_labels)
            proxy_ce.append(micro_proxy_ce)
            reference_ce.append(micro_reference_ce)
        return (
            torch.cat(labels, dim=0),
            torch.cat(proxy_ce, dim=0),
            torch.cat(reference_ce, dim=0),
        )

    def _sync_checkpoint_path(self, step: int, phase: str) -> str:
        return str(
            Path(self.trainer.save_folder)
            / "sync_checkpoints"
            / f"step{int(step)}-{phase}"
        )

    def _save_sync_checkpoint(self, *, step: int, phase: str) -> None:
        path = self._sync_checkpoint_path(step, phase)
        if self.trainer.checkpointer.dir_is_checkpoint(path):
            log.info("BLADE %s-sync checkpoint already exists at '%s'", phase, path)
            return
        log.info(
            "Saving BLADE %s-sync checkpoint for step %d to '%s'...",
            phase,
            step,
            path,
        )
        self.trainer._log_metrics()
        self.trainer._join_bookkeeping_ops()
        self.trainer.checkpointer.save(
            path,
            self.trainer.train_module,
            self.trainer.state_dict(),
            ephemeral=False,
        )
        for callback in self.trainer.callbacks.values():
            callback.post_checkpoint_saved(path)
        log.info("BLADE %s-sync checkpoint saved", phase)

    def pre_step(self, batch: dict[str, Any]) -> None:
        step = int(self.trainer.global_step)
        if step in self.schedule.sync_steps and self.last_sync != step:
            # Compatibility fallback for checkpoints created before sync-boundary
            # checkpoints moved the update to the end of the preceding step.
            log.warning("Running BLADE sync %d in pre_step compatibility mode", step)
            self._sync_from_proxy()
            self._run_k_updates(trainer_step=step)
            self.last_sync = step
        if step >= self.schedule.start:
            if self.reference is None:
                raise RuntimeError(
                    "BLADE selection has no dynamic reference; resume state is incomplete"
                )
            labels, proxy_ce, ref_ce = self._proxy_and_reference_ce(batch)
            valid = labels != self.trainer.train_module.label_ignore_index
            # Equation 5 minimizes L_ref - L_proxy over a fixed-size mask, so
            # ranking in descending order must use the equivalent L_proxy - L_ref.
            selection_scores = proxy_ce - ref_ce
            flat_scores = selection_scores[valid]
            keep = max(1, int(torch.ceil(torch.tensor(self.schedule.gamma * flat_scores.numel()))))
            threshold = torch.topk(flat_scores, min(keep, flat_scores.numel())).values[-1]
            batch["labels"] = labels.masked_fill(
                ~(valid & (selection_scores >= threshold)), -100
            )

    def post_train_batch(self) -> None:
        self.completed_step = int(self.trainer.global_step)
        sync_step = self.completed_step + 1
        if sync_step not in self.schedule.sync_steps or self.last_sync == sync_step:
            return

        # The first pre-sync checkpoint still needs a reference state. Initializing
        # it from the just-completed proxy is equivalent to the first sync. The
        # actual sync is repeated below before K updates so phase boundaries remain
        # explicit, while later pre-sync saves preserve the previous reference.
        if self.reference is None:
            self._sync_from_proxy()
        self._save_sync_checkpoint(step=sync_step, phase="pre")

        self._sync_from_proxy()
        self._run_k_updates(trainer_step=sync_step)
        self.last_sync = sync_step
        self._save_sync_checkpoint(step=sync_step, phase="post")

    def post_attach(self) -> None:
        if self._pending is not None:
            pending, self._pending = self._pending, None
            self._restore(pending)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": 2,
            "checkpoint_format": BLADE_CHECKPOINT_FORMAT,
            "completed_step": self.completed_step,
            "last_sync": self.last_sync,
            "schedule": self.schedule.__dict__,
            "dynamic_reference": (
                {name: value.detach().cpu() for name, value in self.reference.state_dict().items()}
                if self.reference is not None
                else None
            ),
            "dynamic_reference_optim": (
                self.reference_optim.state_dict() if self.reference_optim is not None else None
            ),
            "reference_train_stream": self.reference_train_stream.state_dict(),
            "refhq_stream": self.refhq_stream.state_dict(),
        }

    def _restore(self, state: Mapping[str, Any]) -> None:
        if state.get("version") != 2 or state.get("checkpoint_format") != BLADE_CHECKPOINT_FORMAT:
            raise ValueError("unsupported or incomplete BLADE checkpoint state")
        if dict(state["schedule"]) != self.schedule.__dict__:
            raise ValueError("BLADE resume schedule differs from checkpoint")
        self.completed_step = int(state["completed_step"])
        self.last_sync = state.get("last_sync")
        if not 0 <= self.completed_step <= self.total_steps:
            raise ValueError("BLADE completed step is outside the locked run")
        expected_completed_sync = next(
            (step for step in reversed(self.schedule.sync_steps) if step <= self.completed_step),
            None,
        )
        boundary_sync = (
            self.completed_step + 1
            if self.completed_step + 1 in self.schedule.sync_steps
            else None
        )
        if self.last_sync not in {expected_completed_sync, boundary_sync}:
            raise ValueError("BLADE last sync is inconsistent with the completed checkpoint step")
        reference_state = state.get("dynamic_reference")
        if reference_state is not None:
            if state.get("dynamic_reference_optim") is None:
                raise ValueError("BLADE dynamic reference is missing optimizer state")
            self._new_reference()
            assert self.reference is not None and self.reference_optim is not None
            self.reference.load_state_dict(reference_state, strict=True)
            self.reference_optim.load_state_dict(state["dynamic_reference_optim"])
            self.reference.eval()
            for parameter in self.reference.parameters():
                parameter.requires_grad_(False)
        elif self.completed_step >= self.schedule.start or self.last_sync is not None:
            raise ValueError("post-warmup BLADE checkpoint is missing its dynamic reference")
        self.reference_train_stream.load_state_dict(state["reference_train_stream"])
        self.refhq_stream.load_state_dict(state["refhq_stream"])

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if hasattr(self, "trainer"):
            self._restore(state)
        else:
            self._pending = state
