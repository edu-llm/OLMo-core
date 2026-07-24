from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .config import Condition, ExperimentConfig
from .data import load_review_data
from .forever import (
    FOREVER_MEMORY_EPOCHS,
    FOREVER_MEMORY_RATIO,
    FOREVER_REGULARIZATION_COEFFICIENT,
    ForeverClock,
)
from .micro_world import MicroWorldRecord, group_records
from .schedules import ReviewController


log = logging.getLogger(__name__)


def register_model_backend(config: ExperimentConfig) -> None:
    """Register optional Hugging Face model classes required by older OLMo checkpoints."""
    if config.model.backend == "hf_olmo":
        # DataDecide and the original ladder checkpoints use Ai2's legacy
        # ``hf_olmo`` model type. Importing the package registers its config and
        # causal-LM class with Transformers' Auto classes.
        import hf_olmo  # noqa: F401


def seed_everything(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def _device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The OLMo review experiment requires a CUDA GPU. Run data generation and tests on CPU, "
            "then launch training on the GPU machine."
        )
    return torch.device("cuda", torch.cuda.current_device())


def _load_tokenizer(config: ExperimentConfig, checkpoint: Optional[Path] = None):
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    source = (
        str(checkpoint)
        if checkpoint and (checkpoint / "tokenizer_config.json").exists()
        else config.model.name
    )
    # The public DataDecide checkpoints label their tokenizer ``OLMoTokenizer``,
    # a class that is absent from the Transformers version compatible with the
    # legacy hf_olmo model wrapper. The serialized tokenizer.json is standard,
    # so load it through the generic fast-tokenizer class for this backend.
    tokenizer_class = (
        PreTrainedTokenizerFast if config.model.backend == "hf_olmo" else AutoTokenizer
    )
    tokenizer = tokenizer_class.from_pretrained(
        source,
        revision=None if checkpoint else config.model.revision,
        trust_remote_code=config.model.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _enable_gradient_checkpointing(model: nn.Module, adaptation: str) -> None:
    model.gradient_checkpointing_enable()  # type: ignore[attr-defined]
    if adaptation == "lora":
        # Re-entrant checkpointing needs at least one differentiable input. LoRA freezes
        # the embedding weights, so ask Transformers to mark embedding outputs as trainable.
        model.enable_input_require_grads()  # type: ignore[attr-defined]


def _load_model(config: ExperimentConfig, checkpoint: Optional[Path] = None) -> nn.Module:
    from transformers import AutoModelForCausalLM

    register_model_backend(config)

    load_args: Dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "torch_dtype": _dtype(config.model.precision),
    }
    if config.model.attention != "auto":
        load_args["attn_implementation"] = config.model.attention

    if config.model.adaptation == "full" and checkpoint is not None:
        model = AutoModelForCausalLM.from_pretrained(str(checkpoint), **load_args)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            config.model.name,
            revision=config.model.revision,
            **load_args,
        )

    if config.model.adaptation == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model

        if checkpoint is not None and (checkpoint / "adapter_config.json").exists():
            model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=True)
        else:
            lora_config = LoraConfig(
                r=config.model.lora_rank,
                lora_alpha=config.model.lora_alpha,
                lora_dropout=config.model.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            )
            model = get_peft_model(model, lora_config)

    if config.model.gradient_checkpointing:
        _enable_gradient_checkpointing(model, config.model.adaptation)
    model.config.use_cache = False
    model.to(_device())
    if config.training.compile_model:
        model = torch.compile(model)
    return model


def _trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return trainable, total


class EncodedPool:
    def __init__(
        self,
        records: Mapping[str, Sequence[MicroWorldRecord]],
        tokenizer,
        max_length: int,
        *,
        shuffle_seed: int,
    ):
        self.max_length = max_length
        self.pad_token_id = int(tokenizer.pad_token_id)
        self.eos_token_id = int(tokenizer.eos_token_id)
        self.by_skill: Dict[str, List[Dict[str, Any]]] = {}
        for skill, examples in records.items():
            encoded = [self._encode(record, tokenizer) for record in examples]
            # Evaluation takes a bounded prefix, so shuffle it deterministically instead of
            # selecting whichever event happens to sort first. Paired conditions use the same
            # seed and therefore evaluate the exact same examples.
            random.Random(f"{shuffle_seed}:{skill}").shuffle(encoded)
            self.by_skill[skill] = encoded

    def _encode(self, record: MicroWorldRecord, tokenizer) -> Dict[str, Any]:
        prompt_ids = tokenizer.encode(record.prompt, add_special_tokens=False)
        if record.answer:
            answer_ids = tokenizer.encode(" " + record.answer, add_special_tokens=False)
            input_ids = prompt_ids + answer_ids + [self.eos_token_id]
            labels = [-100] * len(prompt_ids) + answer_ids + [self.eos_token_id]
        else:
            input_ids = prompt_ids + [self.eos_token_id]
            labels = list(input_ids)

        input_ids = input_ids[: self.max_length]
        labels = labels[: self.max_length]
        attention_mask = [1] * len(input_ids)
        padding = self.max_length - len(input_ids)
        if padding:
            input_ids.extend([self.pad_token_id] * padding)
            labels.extend([-100] * padding)
            attention_mask.extend([0] * padding)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "example_id": record.example_id,
            "fact_id": record.fact_id,
            "skill": record.skill,
        }

    @property
    def skills(self) -> Tuple[str, ...]:
        return tuple(sorted(self.by_skill))

    def sample(
        self, skill: str, batch_size: int, rng: random.Random, device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        pool = self.by_skill[skill]
        selected = [pool[rng.randrange(len(pool))] for _ in range(batch_size)]
        batch = {
            key: torch.tensor(
                [item[key] for item in selected], dtype=torch.long, device=device
            )
            for key in ("input_ids", "attention_mask", "labels")
        }
        return batch, [str(item["example_id"]) for item in selected]

    def batches(
        self, skill: str, batch_size: int, limit: int, device: torch.device
    ) -> Iterable[Tuple[Dict[str, torch.Tensor], Sequence[Mapping[str, Any]]]]:
        examples = self.by_skill[skill] if limit <= 0 else self.by_skill[skill][:limit]
        for start in range(0, len(examples), batch_size):
            chunk = examples[start : start + batch_size]
            batch = {
                key: torch.tensor([item[key] for item in chunk], dtype=torch.long, device=device)
                for key in ("input_ids", "attention_mask", "labels")
            }
            yield batch, chunk


def _build_pools(config: ExperimentConfig, tokenizer):
    records = load_review_data(config.data)
    old_train = EncodedPool(
        group_records(records, stage="old", split="train"),
        tokenizer,
        config.training.sequence_length,
        shuffle_seed=config.data.seed + 11,
    )
    new_train = EncodedPool(
        group_records(records, stage="new", split="train"),
        tokenizer,
        config.training.sequence_length,
        shuffle_seed=config.data.seed + 23,
    )
    old_eval = EncodedPool(
        group_records(records, stage="old", split="eval"),
        tokenizer,
        config.training.sequence_length,
        shuffle_seed=config.data.seed + 37,
    )
    new_eval = EncodedPool(
        group_records(records, stage="new", split="eval"),
        tokenizer,
        config.training.sequence_length,
        shuffle_seed=config.data.seed + 53,
    )
    return old_train, new_train, old_eval, new_eval


@torch.no_grad()
def evaluate_metrics(
    model: nn.Module,
    pool: EncodedPool,
    *,
    batch_size: int,
    examples_per_skill: int,
    precision: str,
) -> Dict[str, Any]:
    model.eval()
    device = _device()
    losses_by_skill: Dict[str, float] = {}
    items: Dict[str, Dict[str, Any]] = {}
    for skill in pool.skills:
        weighted_loss = 0.0
        active_tokens = 0
        for batch, metadata in pool.batches(skill, batch_size, examples_per_skill, device):
            autocast = (
                torch.autocast(device_type="cuda", dtype=_dtype(precision))
                if precision != "fp32"
                else nullcontext()
            )
            with autocast:
                output = model(**batch)
            shifted_logits = output.logits[:, :-1, :].detach().float()
            shifted_labels = batch["labels"][:, 1:]
            active = shifted_labels != -100
            safe_labels = shifted_labels.masked_fill(~active, 0)
            token_losses = torch.nn.functional.cross_entropy(
                shifted_logits.transpose(1, 2),
                safe_labels,
                reduction="none",
            )
            predictions = shifted_logits.argmax(dim=-1)
            for index, item in enumerate(metadata):
                item_mask = active[index]
                token_count = int(item_mask.sum().item())
                loss_sum = float(token_losses[index][item_mask].sum().item())
                answer_mask = item_mask & (shifted_labels[index] != pool.eos_token_id)
                answer_tokens = int(answer_mask.sum().item())
                exact = bool(
                    answer_tokens
                    and torch.equal(
                        predictions[index][answer_mask],
                        shifted_labels[index][answer_mask],
                    )
                )
                weighted_loss += loss_sum
                active_tokens += token_count
                items[str(item["fact_id"])] = {
                    "example_id": str(item["example_id"]),
                    "skill": skill,
                    "loss": loss_sum / max(token_count, 1),
                    "answer_exact": exact,
                    "answer_tokens": answer_tokens,
                }
        losses_by_skill[skill] = weighted_loss / max(active_tokens, 1)
    model.train()
    exact_values = [bool(item["answer_exact"]) for item in items.values()]
    return {
        "loss_by_skill": losses_by_skill,
        "items": items,
        "exact_match": sum(exact_values) / max(len(exact_values), 1),
        "items_evaluated": len(items),
    }


def _make_optimizer(
    model: nn.Module, config: ExperimentConfig, *, warmup_steps: Optional[int] = None
):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )

    warmup = config.training.warmup_steps if warmup_steps is None else warmup_steps

    def lr_lambda(step: int) -> float:
        if warmup <= 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


def _forever_memory_examples(
    pool: EncodedPool, *, seed: int
) -> List[Dict[str, Any]]:
    """Select the paper's 2% old-task memory with class/skill balancing."""

    total = sum(len(examples) for examples in pool.by_skill.values())
    target = int(total * FOREVER_MEMORY_RATIO)
    if target <= 0:
        return []
    rng = random.Random(f"{seed}:forever-memory")
    skills = list(pool.skills)
    per_skill = max(1, target // len(skills))
    selected: List[Dict[str, Any]] = []
    selected_ids = set()
    for skill in skills:
        candidates = list(pool.by_skill[skill])
        rng.shuffle(candidates)
        for example in candidates[:per_skill]:
            selected.append(example)
            selected_ids.add(str(example["example_id"]))

    if len(selected) < target:
        remainder = [
            example
            for skill in skills
            for example in pool.by_skill[skill]
            if str(example["example_id"]) not in selected_ids
        ]
        rng.shuffle(remainder)
        selected.extend(remainder[: target - len(selected)])
    return selected


@torch.no_grad()
def _parameter_change_norm(
    trainable: Sequence[Tuple[str, nn.Parameter]],
    previous: Mapping[str, torch.Tensor],
) -> float:
    squared = torch.zeros((), dtype=torch.float32, device=_device())
    for name, parameter in trainable:
        squared.add_(
            (parameter.detach().float() - previous[name].float()).pow(2).sum()
        )
    return float(squared.sqrt().item())


@torch.no_grad()
def _refresh_parameter_snapshot(
    trainable: Sequence[Tuple[str, nn.Parameter]],
    snapshot: Dict[str, torch.Tensor],
) -> None:
    for name, parameter in trainable:
        snapshot[name] = parameter.detach().clone()


def _collate_encoded(
    examples: Sequence[Mapping[str, Any]], device: torch.device
) -> Dict[str, torch.Tensor]:
    return {
        key: torch.tensor([item[key] for item in examples], dtype=torch.long, device=device)
        for key in ("input_ids", "attention_mask", "labels")
    }


class MetricsWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def _train_steps(
    model: nn.Module,
    *,
    config: ExperimentConfig,
    steps: int,
    seed: int,
    old_train: EncodedPool,
    new_train: Optional[EncodedPool],
    old_eval: EncodedPool,
    new_eval: Optional[EncodedPool],
    writer: MetricsWriter,
    stage: int,
    controller: Optional[ReviewController] = None,
) -> Dict[str, Any]:
    device = _device()
    optimizer, lr_scheduler = _make_optimizer(model, config)
    old_rngs = {
        skill: random.Random(f"{seed}:{stage}:old:{skill}") for skill in old_train.skills
    }
    new_rngs = (
        {
            skill: random.Random(f"{seed}:{stage}:new:{skill}")
            for skill in new_train.skills
        }
        if new_train is not None
        else {}
    )
    old_stream_hash = hashlib.sha256()
    new_stream_hash = hashlib.sha256()
    old_content_tokens = 0
    new_content_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    model.train()
    autocast_dtype = _dtype(config.model.precision)
    use_scaler = config.model.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    started = time.perf_counter()
    new_skill_cursor = 0
    running_loss = 0.0
    is_forever = (
        stage == 2
        and controller is not None
        and controller.condition == "forever_full"
    )
    forever_clock = ForeverClock() if is_forever else None
    forever_memory = (
        _forever_memory_examples(old_train, seed=seed) if is_forever else []
    )
    forever_review_steps: List[int] = []
    forever_review_reasons: List[str] = []
    forever_replay_optimizer_steps = 0
    forever_replay_examples = 0
    trainable_named = (
        [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if is_forever
        else []
    )
    forever_anchor = (
        {name: parameter.detach().clone() for name, parameter in trainable_named}
        if is_forever
        else {}
    )
    forever_previous = (
        {name: parameter.detach().clone() for name, parameter in trainable_named}
        if is_forever
        else {}
    )
    forever_replay_rng = random.Random(f"{seed}:forever-replay-order")

    def run_forever_replay(step: int, reason: str) -> None:
        nonlocal old_content_tokens
        nonlocal forever_replay_optimizer_steps
        nonlocal forever_replay_examples
        if not forever_memory:
            return
        assert forever_clock is not None
        replay_optimizer, replay_scheduler = _make_optimizer(
            model, config, warmup_steps=5
        )
        replay_optimizer.zero_grad(set_to_none=True)
        replay_losses: List[float] = []
        replay_batch_size = (
            config.training.micro_batch_size
            * config.training.gradient_accumulation_steps
        )
        scale = forever_clock.replay_scale
        for epoch in range(FOREVER_MEMORY_EPOCHS):
            epoch_examples = list(forever_memory)
            forever_replay_rng.shuffle(epoch_examples)
            for start in range(0, len(epoch_examples), replay_batch_size):
                chunk = epoch_examples[start : start + replay_batch_size]
                batch = _collate_encoded(chunk, device)
                content_tokens = int(batch["attention_mask"].sum().item())
                old_content_tokens += content_tokens
                for item in chunk:
                    old_stream_hash.update(str(item["example_id"]).encode("utf-8"))
                    old_stream_hash.update(b"\0")
                autocast = (
                    torch.autocast(device_type="cuda", dtype=autocast_dtype)
                    if config.model.precision != "fp32"
                    else nullcontext()
                )
                with autocast:
                    output = model(**batch)
                regularization = torch.zeros(
                    (), dtype=torch.float32, device=device
                )
                for name, parameter in trainable_named:
                    regularization = regularization + (
                        parameter.float() - forever_anchor[name].float()
                    ).pow(2).sum()
                loss = (
                    output.loss.float()
                    + FOREVER_REGULARIZATION_COEFFICIENT
                    * scale
                    * regularization
                )
                scaler.scale(loss).backward()
                scaler.unscale_(replay_optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )
                scaler.step(replay_optimizer)
                scaler.update()
                replay_optimizer.zero_grad(set_to_none=True)
                replay_scheduler.step()
                replay_losses.append(float(loss.detach().item()))
                forever_replay_optimizer_steps += 1
                forever_replay_examples += len(chunk)

        forever_review_steps.append(step)
        forever_review_reasons.append(reason)
        writer.write(
            {
                "event": "forever_replay",
                "stage": stage,
                "step": step,
                "reason": reason,
                "model_time": forever_clock.tau,
                "model_day": forever_clock.model_day,
                "triggered_days": list(forever_clock.triggered_days),
                "replay_scale": scale,
                "memory_examples": len(forever_memory),
                "memory_epochs": FOREVER_MEMORY_EPOCHS,
                "replay_optimizer_steps_total": forever_replay_optimizer_steps,
                "mean_replay_loss": sum(replay_losses) / len(replay_losses),
            }
        )
        # Replay changes are excluded from model-centric time in FOREVER.
        _refresh_parameter_snapshot(trainable_named, forever_previous)

    for step in range(steps):
        if stage == 1:
            decision_review = True
            selected_skill = old_train.skills[step % len(old_train.skills)]
            reason = "stage1-learning"
            phase = "acquisition"
        else:
            assert new_train is not None and controller is not None
            if step < config.training.stage2_steps:
                decision = controller.decide(step)
                decision_review = decision.review
                reason = decision.reason
                phase = "review-window"
                if decision.review:
                    assert decision.skill is not None
                    selected_skill = decision.skill
                else:
                    selected_skill = new_train.skills[
                        new_skill_cursor % len(new_train.skills)
                    ]
                    new_skill_cursor += 1
            else:
                selected_skill = new_train.skills[new_skill_cursor % len(new_train.skills)]
                new_skill_cursor += 1
                decision_review = False
                reason = "post-review-interference"
                phase = "buffer"

        pool = old_train if decision_review else new_train
        step_loss = 0.0
        for _ in range(config.training.gradient_accumulation_steps):
            assert pool is not None
            rngs = old_rngs if decision_review else new_rngs
            batch, example_ids = pool.sample(
                selected_skill,
                config.training.micro_batch_size,
                rngs[selected_skill],
                device,
            )
            content_tokens = int(batch["attention_mask"].sum().item())
            stream_hash = old_stream_hash if decision_review else new_stream_hash
            for example_id in example_ids:
                stream_hash.update(example_id.encode("utf-8"))
                stream_hash.update(b"\0")
            if decision_review:
                old_content_tokens += content_tokens
            else:
                new_content_tokens += content_tokens
            autocast = (
                torch.autocast(device_type="cuda", dtype=autocast_dtype)
                if config.model.precision != "fp32"
                else nullcontext()
            )
            with autocast:
                output = model(**batch)
                loss = output.loss / config.training.gradient_accumulation_steps
            scaler.scale(loss).backward()
            step_loss += float(loss.detach().float().item())

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.max_grad_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        lr_scheduler.step()
        running_loss += step_loss

        if is_forever and step < config.training.stage2_steps:
            assert forever_clock is not None
            delta = _parameter_change_norm(trainable_named, forever_previous)
            crossed_days = forever_clock.observe_update(delta)
            _refresh_parameter_snapshot(trainable_named, forever_previous)
            writer.write(
                {
                    "event": "forever_clock",
                    "stage": stage,
                    "step": step,
                    "parameter_change_norm": delta,
                    "model_time": forever_clock.tau,
                    "model_day": forever_clock.model_day,
                    "replay_scale": forever_clock.replay_scale,
                    "crossed_days": crossed_days,
                }
            )
            for day in crossed_days:
                run_forever_replay(step, f"model-day-{day}")
                # The official implementation starts a fresh AdamW phase after replay.
                if step + 1 < config.training.stage2_steps:
                    optimizer, lr_scheduler = _make_optimizer(model, config)
            if step == config.training.stage2_steps - 1:
                run_forever_replay(step, "final-replay")

        if step % config.training.log_interval == 0 or step == steps - 1:
            writer.write(
                {
                    "event": "train",
                    "stage": stage,
                    "step": step,
                    "loss": step_loss,
                    "mean_loss": running_loss / (step + 1),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().float().item()),
                    "review": decision_review if stage == 2 else False,
                    "skill": selected_skill,
                    "reason": reason,
                    "phase": phase,
                }
            )

        if stage == 1:
            should_eval = step == steps - 1
        elif config.training.buffer_steps:
            eval_steps = {
                config.training.stage2_steps - 1,
                *(
                    config.training.stage2_steps + delay - 1
                    for delay in config.training.buffer_eval_delays
                ),
                steps - 1,
            }
            if config.training.eval_interval > 0:
                eval_steps.update(
                    range(
                        config.training.eval_interval - 1,
                        config.training.stage2_steps,
                        config.training.eval_interval,
                    )
                )
            should_eval = step in eval_steps
        else:
            should_eval = step == steps - 1 or (
                config.training.eval_interval > 0
                and (step + 1) % config.training.eval_interval == 0
            )
        if should_eval:
            final_evaluation = step == steps - 1
            evaluation_limit = (
                config.training.final_eval_examples_per_skill
                if final_evaluation
                else config.training.eval_examples_per_skill
            )
            old_evaluation = evaluate_metrics(
                model,
                old_eval,
                batch_size=config.training.eval_batch_size,
                examples_per_skill=evaluation_limit,
                precision=config.model.precision,
            )
            new_evaluation = (
                evaluate_metrics(
                    model,
                    new_eval,
                    batch_size=config.training.eval_batch_size,
                    examples_per_skill=evaluation_limit,
                    precision=config.model.precision,
                )
                if new_eval is not None
                else {"loss_by_skill": {}, "items": {}, "exact_match": 0.0}
            )
            if controller is not None:
                controller.observe(old_evaluation["loss_by_skill"])
            writer.write(
                {
                    "event": "eval",
                    "stage": stage,
                    "step": step,
                    "phase": phase,
                    "buffer_delay": (
                        step - config.training.stage2_steps + 1
                        if stage == 2 and step >= config.training.stage2_steps
                        else 0
                    ),
                    "old_loss": old_evaluation["loss_by_skill"],
                    "new_loss": new_evaluation["loss_by_skill"],
                    "old_exact_match": old_evaluation["exact_match"],
                    "new_exact_match": new_evaluation["exact_match"],
                    "old_items": old_evaluation["items"],
                    "new_items": new_evaluation["items"],
                }
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    metadata = {
        "duration_seconds": time.perf_counter() - started,
        "mean_train_loss": running_loss / steps,
        "old_content_tokens": old_content_tokens,
        "new_content_tokens": new_content_tokens,
        "old_stream_digest": old_stream_hash.hexdigest(),
        "new_stream_digest": new_stream_hash.hexdigest(),
        "optimizer_steps": steps + forever_replay_optimizer_steps,
        "current_optimizer_steps": steps,
        "replay_optimizer_steps": forever_replay_optimizer_steps,
    }
    if forever_clock is not None:
        metadata.update(
            {
                "forever_method": {
                    "calibration_steps": forever_clock.calibration_steps,
                    "trigger_days": list(forever_clock.trigger_days),
                    "triggered_days": list(forever_clock.triggered_days),
                    "model_day": forever_clock.model_day,
                    "final_model_time": forever_clock.tau,
                    "memory_ratio": FOREVER_MEMORY_RATIO,
                    "memory_examples": len(forever_memory),
                    "memory_epochs": FOREVER_MEMORY_EPOCHS,
                    "regularization_coefficient": FOREVER_REGULARIZATION_COEFFICIENT,
                    "final_replay": True,
                    "optimizer_reset_after_replay": True,
                },
                "forever_review_steps": forever_review_steps,
                "forever_review_reasons": forever_review_reasons,
                "forever_review_events": len(forever_review_steps),
                "forever_replay_examples": forever_replay_examples,
            }
        )
    return metadata


def _read_last_eval(metrics_path: Path) -> Dict[str, Any]:
    last: Optional[Dict[str, Any]] = None
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "eval":
                last = payload
    if last is None:
        raise ValueError(f"No evaluation record in {metrics_path}")
    return last


def _read_evals(metrics_path: Path) -> List[Dict[str, Any]]:
    evaluations: List[Dict[str, Any]] = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload.get("event") == "eval":
                evaluations.append(payload)
    return evaluations


def _save_model(model: nn.Module, tokenizer, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = getattr(model, "_orig_mod", model)
    unwrapped.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output_dir)


def _clear_model(model: nn.Module) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_root(config: ExperimentConfig) -> Path:
    return Path(config.training.output_dir) / config.experiment_name


def stage1_dir(config: ExperimentConfig, seed: int) -> Path:
    return run_root(config) / "stage1" / f"seed_{seed}"


def condition_dir(config: ExperimentConfig, condition: str, seed: int) -> Path:
    return run_root(config) / condition / f"seed_{seed}"


def prepare_stage1(config: ExperimentConfig, seed: int, *, force: bool = False) -> Path:
    output_dir = stage1_dir(config, seed)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != config.fingerprint():
            raise ValueError(
                f"Stage-one checkpoint at {output_dir} belongs to a different config. "
                "Use a new experiment_name or rerun with --force."
            )
        log.info("Reusing stage-one checkpoint at %s", output_dir)
        return output_dir
    if output_dir.exists() and force:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    tokenizer = _load_tokenizer(config)
    old_train, _, old_eval, _ = _build_pools(config, tokenizer)
    model = _load_model(config)
    trainable, total = _trainable_parameters(model)
    writer = MetricsWriter(output_dir / "metrics.jsonl")
    metadata = _train_steps(
        model,
        config=config,
        steps=config.training.stage1_steps,
        seed=seed,
        old_train=old_train,
        new_train=None,
        old_eval=old_eval,
        new_eval=None,
        writer=writer,
        stage=1,
    )
    metrics_path = output_dir / "metrics.jsonl"
    final_eval = _read_last_eval(metrics_path)
    if config.training.save_stage1:
        _save_model(model, tokenizer, output_dir)
    summary = {
        "event": "stage1-complete",
        "seed": seed,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "config_fingerprint": config.fingerprint(),
        "old_loss": final_eval["old_loss"],
        "mean_old_loss": sum(final_eval["old_loss"].values()) / len(final_eval["old_loss"]),
        "old_exact_match": float(final_eval.get("old_exact_match", 0.0)),
        "old_items": final_eval.get("old_items", {}),
        "mastered_fact_ids": sorted(
            fact_id
            for fact_id, metrics in final_eval.get("old_items", {}).items()
            if metrics.get("answer_exact")
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        **metadata,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _clear_model(model)
    return output_dir


def run_condition(
    config: ExperimentConfig,
    condition: Condition,
    seed: int,
    *,
    force: bool = False,
) -> Path:
    checkpoint = prepare_stage1(config, seed, force=False)
    output_dir = condition_dir(config, condition, seed)
    summary_path = output_dir / "summary.json"
    if summary_path.exists() and not force:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != config.fingerprint():
            raise ValueError(
                f"Completed run at {output_dir} belongs to a different config. "
                "Use a new experiment_name or rerun with --force."
            )
        log.info("Skipping completed run at %s", output_dir)
        return output_dir
    if output_dir.exists() and force:
        import shutil

        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    tokenizer = _load_tokenizer(config, checkpoint)
    old_train, new_train, old_eval, new_eval = _build_pools(config, tokenizer)
    model = _load_model(config, checkpoint)
    trainable, total = _trainable_parameters(model)

    stage1_summary = json.loads((checkpoint / "summary.json").read_text(encoding="utf-8"))
    baseline_losses = {key: float(value) for key, value in stage1_summary["old_loss"].items()}
    baseline_items = {
        str(key): dict(value) for key, value in stage1_summary.get("old_items", {}).items()
    }
    mastered_fact_ids = [
        str(fact_id) for fact_id in stage1_summary.get("mastered_fact_ids", [])
    ]
    if len(mastered_fact_ids) < config.training.minimum_mastered_facts:
        raise RuntimeError(
            f"Only {len(mastered_fact_ids)} old facts met the acquisition criterion; "
            f"need at least {config.training.minimum_mastered_facts}"
        )
    controller = ReviewController(
        condition,
        old_train.skills,
        total_steps=config.training.stage2_steps,
        events=0 if condition in {"no_review", "forever_full"} else config.review.events,
        first_step=config.review.first_step,
        last_step=config.review.last_step,
        expansion_ratio=config.review.expansion_ratio,
        adaptive_opportunity_interval=config.review.adaptive_opportunity_interval,
        loss_delta_threshold=config.review.loss_delta_threshold,
        min_gap=config.review.min_gap,
    )
    controller.set_baseline(baseline_losses)
    writer = MetricsWriter(output_dir / "metrics.jsonl")
    metadata = _train_steps(
        model,
        config=config,
        steps=config.training.stage2_steps + config.training.buffer_steps,
        seed=seed,
        old_train=old_train,
        new_train=new_train,
        old_eval=old_eval,
        new_eval=new_eval,
        writer=writer,
        stage=2,
        controller=controller,
    )
    metrics_path = output_dir / "metrics.jsonl"
    final_eval = _read_last_eval(metrics_path)
    evaluations = _read_evals(metrics_path)
    pre_buffer_eval = next(
        (
            evaluation
            for evaluation in evaluations
            if int(evaluation["step"]) == config.training.stage2_steps - 1
        ),
        None,
    )
    old_deltas = {
        skill: float(final_eval["old_loss"][skill]) - baseline_losses[skill]
        for skill in baseline_losses
    }
    final_items = {
        str(key): dict(value) for key, value in final_eval.get("old_items", {}).items()
    }
    exact_regressed = [
        fact_id
        for fact_id in mastered_fact_ids
        if fact_id in final_items and not bool(final_items[fact_id].get("answer_exact"))
    ]
    loss_regressed = [
        fact_id
        for fact_id in mastered_fact_ids
        if fact_id in final_items
        and fact_id in baseline_items
        and float(final_items[fact_id]["loss"])
        - float(baseline_items[fact_id]["loss"])
        >= config.training.regression_loss_margin
    ]
    mastered_item_deltas = [
        float(final_items[fact_id]["loss"]) - float(baseline_items[fact_id]["loss"])
        for fact_id in mastered_fact_ids
        if fact_id in final_items and fact_id in baseline_items
    ]
    buffer_old_loss_delta = (
        {
            skill: float(final_eval["old_loss"][skill])
            - float(pre_buffer_eval["old_loss"][skill])
            for skill in final_eval["old_loss"]
        }
        if pre_buffer_eval is not None
        else {}
    )
    summary = {
        "event": "condition-complete",
        "condition": condition,
        "seed": seed,
        "model": config.model.name,
        "model_revision": config.model.revision,
        "config_fingerprint": config.fingerprint(),
        "baseline_old_loss": baseline_losses,
        "final_old_loss": final_eval["old_loss"],
        "old_loss_delta": old_deltas,
        "mean_old_loss_delta": sum(old_deltas.values()) / len(old_deltas),
        "final_new_loss": final_eval["new_loss"],
        "mean_new_loss": sum(final_eval["new_loss"].values()) / len(final_eval["new_loss"]),
        "final_old_exact_match": float(final_eval.get("old_exact_match", 0.0)),
        "final_new_exact_match": float(final_eval.get("new_exact_match", 0.0)),
        "mastered_facts": len(mastered_fact_ids),
        "exact_regressed_facts": len(exact_regressed),
        "exact_regression_rate": len(exact_regressed) / max(len(mastered_fact_ids), 1),
        "loss_regressed_facts": len(loss_regressed),
        "loss_regression_rate": len(loss_regressed) / max(len(mastered_fact_ids), 1),
        "mean_mastered_item_loss_delta": (
            sum(mastered_item_deltas) / len(mastered_item_deltas)
            if mastered_item_deltas
            else 0.0
        ),
        "pre_buffer_old_loss": (
            pre_buffer_eval["old_loss"] if pre_buffer_eval is not None else {}
        ),
        "buffer_old_loss_delta": buffer_old_loss_delta,
        "mean_buffer_old_loss_delta": (
            sum(buffer_old_loss_delta.values()) / len(buffer_old_loss_delta)
            if buffer_old_loss_delta
            else 0.0
        ),
        "buffer_steps": config.training.buffer_steps,
        "updates_since_last_review": (
            config.training.stage2_steps
            + config.training.buffer_steps
            - 1
            - (
                metadata["forever_review_steps"][-1]
                if condition == "forever_full"
                else controller.review_steps[-1]
            )
            if (
                metadata.get("forever_review_steps")
                if condition == "forever_full"
                else controller.review_steps
            )
            else None
        ),
        "review_steps": (
            list(metadata["forever_review_steps"])
            if condition == "forever_full"
            else list(controller.review_steps)
        ),
        "review_events": (
            int(metadata["forever_review_events"])
            if condition == "forever_full"
            else controller.used_events
        ),
        "trainable_parameters": trainable,
        "total_parameters": total,
        **metadata,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if config.training.save_final_model:
        _save_model(model, tokenizer, output_dir / "model")
    _clear_model(model)
    return output_dir
