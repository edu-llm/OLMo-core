#!/usr/bin/env python3
"""Train blocked and interleaved arms from the same causal-LM checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_RE = re.compile(r"^[0-9](?: [0-9]){5}$")


def set_seed(seed: int, deterministic: bool) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class ResponseDataset(Dataset):
    def __init__(self, rows: list[dict[str, object]], tokenizer, max_length: int):
        self.rows = rows
        self.examples: list[dict[str, torch.Tensor]] = []
        pad_id = tokenizer.pad_token_id
        eos_id = tokenizer.eos_token_id
        if pad_id is None or eos_id is None:
            raise ValueError("tokenizer must have pad_token_id and eos_token_id")

        for row in rows:
            prompt_ids = tokenizer(str(row["prompt"]), add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(str(row["target"]), add_special_tokens=False)["input_ids"] + [eos_id]
            input_ids = prompt_ids + target_ids
            if len(input_ids) > max_length:
                raise ValueError(f"record {row['id']} tokenizes to {len(input_ids)} > {max_length}")
            attention = [1] * len(input_ids)
            labels = [-100] * len(prompt_ids) + target_ids
            padding = max_length - len(input_ids)
            input_ids += [pad_id] * padding
            attention += [0] * padding
            labels += [-100] * padding
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.examples[index]


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_name: str, device: torch.device, trust_remote_code: bool):
    kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
    if device.type == "cuda":
        kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.to(device)
    return model


def save_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_order_indices(value: str, data_dir: Path) -> list[int]:
    if value == "all":
        indices: list[int] = []
        for path in sorted(data_dir.glob("blocked_order_*.jsonl")):
            match = re.fullmatch(r"blocked_order_(\d+)\.jsonl", path.name)
            if match:
                indices.append(int(match.group(1)))
        if not indices:
            raise ValueError(f"no numbered schedules found in {data_dir}")
        return indices

    indices = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        indices.append(int(item))
    if not indices:
        raise ValueError("--order-indices must name at least one order")
    return indices


def schedule_path(data_dir: Path, arm: str, order_index: int) -> Path:
    numbered = data_dir / f"{arm}_order_{order_index}.jsonl"
    if numbered.exists():
        return numbered
    if order_index == 0:
        legacy = data_dir / f"{arm}_order.jsonl"
        if legacy.exists():
            return legacy
    raise FileNotFoundError(numbered)


def order_labels(data_dir: Path, order_indices: Sequence[int]) -> dict[int, list[str]]:
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result: dict[int, list[str]] = {}
    for row in manifest.get("validation", {}).get("orders", []):
        index = int(row["order_index"])
        if index in order_indices:
            result[index] = [str(label) for label in row["order"]]
    return result


def train_arm(
    arm: str,
    model_name: str,
    tokenizer,
    schedule_path: Path,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, object]]:
    set_seed(args.seed, args.deterministic)
    model = load_model(model_name, device, args.trust_remote_code)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.train()
    rows = read_jsonl(schedule_path)
    dataset = ResponseDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scaler_enabled = False
    logs: list[dict[str, object]] = []
    started = time.monotonic()
    step_limit = args.max_steps if args.max_steps is not None else len(loader)

    for step, batch in enumerate(loader):
        if step >= step_limit:
            break
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            result = model(**batch)
            loss = result.loss
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
        optimizer.step()
        elapsed = time.monotonic() - started
        logs.append(
            {
                "arm": arm,
                "elapsed_seconds": elapsed,
                "grad_norm": float(grad_norm.detach().cpu()),
                "loss": float(loss.detach().cpu()),
                "step": step + 1,
            }
        )
        if step == 0 or (step + 1) % args.log_every == 0:
            print(
                f"{arm} step={step + 1}/{step_limit} "
                f"loss={logs[-1]['loss']:.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    save_jsonl(output_dir / f"{arm}_train_log.jsonl", logs)
    summary = {
        "arm": arm,
        "elapsed_seconds": time.monotonic() - started,
        "final_loss": logs[-1]["loss"] if logs else None,
        "parameters": parameter_count,
        "steps": len(logs),
    }
    if args.save_models:
        model.save_pretrained(output_dir / f"{arm}_model", safe_serialization=True)
    return model, summary


@torch.inference_mode()
def evaluate(
    model,
    tokenizer,
    rows: list[dict[str, object]],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    model.eval()
    previous_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    predictions: list[dict[str, object]] = []
    correct_by_skill: dict[str, list[int]] = defaultdict(list)

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [str(row["prompt"]) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=16,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = generated[:, encoded["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for row, text in zip(batch_rows, texts):
            first_line = text.splitlines()[0].strip() if text.splitlines() else text.strip()
            expected = str(row["target"]).strip()
            parsed = first_line if ANSWER_RE.fullmatch(first_line) else None
            correct = int(parsed == expected)
            skill = str(row["skill"])
            correct_by_skill[skill].append(correct)
            predictions.append(
                {
                    "correct": bool(correct),
                    "expected": expected,
                    "id": row["id"],
                    "prediction": first_line,
                    "raw_generation": text,
                    "skill": skill,
                }
            )

    tokenizer.padding_side = previous_padding_side
    per_skill = {
        skill: sum(values) / len(values)
        for skill, values in sorted(correct_by_skill.items())
    }
    all_correct = [value for values in correct_by_skill.values() for value in values]
    metrics = {
        "accuracy": sum(all_correct) / len(all_correct),
        "examples": len(all_correct),
        "per_skill_accuracy": per_skill,
    }
    return metrics, predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model ID or local converted checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--order-indices",
        default="0",
        help="comma-separated schedule order indices, or 'all'",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size != 32:
        raise ValueError("the frozen schedules require --batch-size 32")
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"device={device}", flush=True)

    set_seed(args.seed, args.deterministic)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_rows = read_jsonl(args.data_dir / "test.jsonl")
    run_summaries: dict[str, object] = {}
    arm_metrics: dict[str, dict[str, object]] = {}
    order_indices = parse_order_indices(args.order_indices, args.data_dir)
    labels_by_order = order_labels(args.data_dir, order_indices)

    set_seed(args.seed, args.deterministic)
    base_model = load_model(args.model, device, args.trust_remote_code)
    base_metrics, base_predictions = evaluate(
        base_model,
        tokenizer,
        test_rows,
        device,
        args.eval_batch_size,
    )
    save_jsonl(args.output_dir / "base_predictions.jsonl", base_predictions)
    arm_metrics["base"] = base_metrics
    print(f"base metrics={json.dumps(base_metrics, sort_keys=True)}", flush=True)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    order_results: dict[str, object] = {}
    effects: list[float] = []
    for order_index in order_indices:
        order_key = f"order_{order_index}"
        order_results[order_key] = {
            "blocked_order": labels_by_order.get(order_index),
            "metrics": {},
            "training": {},
        }
        for arm in ("blocked", "interleaved"):
            arm_name = f"{order_key}_{arm}"
            model, train_summary = train_arm(
                arm_name,
                args.model,
                tokenizer,
                schedule_path(args.data_dir, arm, order_index),
                args.output_dir,
                device,
                args,
            )
            metrics, predictions = evaluate(
                model,
                tokenizer,
                test_rows,
                device,
                args.eval_batch_size,
            )
            save_jsonl(args.output_dir / f"{arm_name}_predictions.jsonl", predictions)
            run_summaries[arm_name] = train_summary
            arm_metrics[arm_name] = metrics
            order_results[order_key]["training"][arm] = train_summary
            order_results[order_key]["metrics"][arm] = metrics
            print(f"{arm_name} metrics={json.dumps(metrics, sort_keys=True)}", flush=True)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        blocked_acc = float(order_results[order_key]["metrics"]["blocked"]["accuracy"])
        interleaved_acc = float(
            order_results[order_key]["metrics"]["interleaved"]["accuracy"]
        )
        effect = interleaved_acc - blocked_acc
        order_results[order_key]["effect_interleaved_minus_blocked"] = effect
        effects.append(effect)

    if order_indices == [0]:
        arm_metrics["blocked"] = arm_metrics["order_0_blocked"]
        arm_metrics["interleaved"] = arm_metrics["order_0_interleaved"]
        run_summaries["blocked"] = run_summaries["order_0_blocked"]
        run_summaries["interleaved"] = run_summaries["order_0_interleaved"]

    effect = sum(effects) / len(effects)
    result = {
        "args": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
        "device": str(device),
        "effect_interleaved_minus_blocked": effect,
        "order_indices": order_indices,
        "order_results": order_results,
        "paired_effects_interleaved_minus_blocked": effects,
        "metrics": arm_metrics,
        "training": run_summaries,
    }
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
