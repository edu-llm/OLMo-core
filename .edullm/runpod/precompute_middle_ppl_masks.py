#!/usr/bin/env python3
"""Precompute exact middle-PPL masks with one replicated reference model per GPU."""

from __future__ import annotations

import argparse
import bisect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

RUNPOD_DIR = Path(__file__).resolve().parent
EDULLM_DIR = RUNPOD_DIR.parent
if str(EDULLM_DIR) not in sys.path:
    sys.path.insert(0, str(EDULLM_DIR))

from token_selection_370m.precomputed import (  # noqa: E402
    MASK_ALGORITHM,
    MASK_SCHEMA_VERSION,
    binding_sha256,
    file_sha256,
    load_mask_manifest,
    weights_to_label_mask,
)
from token_selection_370m.recipe import SEQUENCE_LENGTH  # noqa: E402
from token_selection_370m.selection import selection_weights  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("/workspace/edullm-inputs/token-selection/ready.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/edullm-inputs/token-selection/middle-ppl-masks"),
    )
    parser.add_argument("--batch-sequences", type=int, default=16)
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            nested = payload.get(key)
            if isinstance(nested, dict):
                payload = nested
                break
    if not isinstance(payload, dict) or not payload:
        raise RuntimeError(f"reference checkpoint has no model state: {path}")
    if any(not isinstance(value, torch.Tensor) for value in payload.values()):
        raise RuntimeError("reference checkpoint state contains non-tensor values")
    return payload


def file_layout(
    source_paths: list[Path],
    source_ids: list[str],
    *,
    output_root: Path,
    itemsize: int,
) -> list[dict[str, Any]]:
    records = []
    for index, (source, source_id) in enumerate(zip(source_paths, source_ids)):
        source_size = source.stat().st_size
        mask_size = source_size // itemsize
        records.append(
            {
                "source_id": source_id,
                "source_path": str(source),
                "source_size": source_size,
                "mask_path": str(
                    output_root / f"{index:05d}-{source.name}.middle-ppl-mask.bool.bin"
                ),
                "mask_size": mask_size,
                "instances": mask_size // SEQUENCE_LENGTH,
            }
        )
    return records


def prepare_layout(
    output_root: Path,
    *,
    layout_binding: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    layout_path = output_root / "layout.json"
    expected = {
        "schema_version": MASK_SCHEMA_VERSION,
        "layout_binding": layout_binding,
        "files": records,
    }
    if layout_path.is_file():
        if json.loads(layout_path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError(
                f"partial mask cache at {output_root} belongs to a different corpus/reference"
            )
        return
    for record in records:
        mask = Path(record["mask_path"])
        with mask.open("wb") as handle:
            handle.truncate(int(record["mask_size"]))
    atomic_json(layout_path, expected)


def progress_start(path: Path, *, identity: str, start: int, end: int) -> int:
    if not path.is_file():
        return start
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("identity") != identity
        or int(payload.get("start", -1)) != start
        or int(payload.get("end", -1)) != end
    ):
        raise RuntimeError(f"stale precompute progress file: {path}")
    return int(payload["next"])


def write_progress(path: Path, *, identity: str, start: int, end: int, next_index: int) -> None:
    atomic_json(
        path,
        {
            "identity": identity,
            "start": start,
            "end": end,
            "next": next_index,
            "complete": next_index == end,
        },
    )


def main() -> None:
    args = parse_args()
    if args.batch_sequences != 16:
        raise SystemExit("exact v1 masks require the training-equivalent batch size of 16 sequences")
    if args.progress_every <= 0:
        raise SystemExit("progress interval must be positive")

    from olmo_core.data import NumpyDatasetDType, TokenizerConfig
    from olmo_core.distributed.utils import get_rank, get_world_size
    from olmo_core.nn.lm_head import LMOutputWithLoss
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.train import prepare_training_environment, teardown_training_environment
    from olmo_core.utils import get_default_device

    prepare_training_environment(seed=6198)
    try:
        torch.set_float32_matmul_precision("high")
        rank, world_size = get_rank(), get_world_size()
        device = get_default_device()
        if world_size != 8 or device.type != "cuda":
            raise RuntimeError("middle-PPL precompute requires the production 8-GPU CUDA topology")
        payload = json.loads(args.input_manifest.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != 1
            or payload.get("family") != "token-selection"
            or payload.get("arm") != "middle-ppl-token"
        ):
            raise RuntimeError("input manifest is not the staged middle-PPL arm")
        corpus = payload["corpora"]["pretrain/regmix-10b"]
        source_ids = [str(path) for path in corpus["logical_paths"]]
        source_paths = [Path(record["path"]) for record in corpus["objects"]]
        if len(source_paths) != len(source_ids):
            raise RuntimeError("staged corpus path mapping is not bijective")
        for source, record in zip(source_paths, corpus["objects"]):
            if not source.is_file() or source.stat().st_size != int(record["size"]):
                raise RuntimeError(f"staged source is missing or changed: {source}")
        reference = Path(payload["references"]["late"])
        if not reference.is_file():
            raise RuntimeError(f"staged late reference is missing: {reference}")

        reference_digest = [file_sha256(reference) if rank == 0 else None]
        dist.broadcast_object_list(reference_digest, src=0)
        dtype = NumpyDatasetDType(corpus["dtype"])
        itemsize = dtype.as_np_dtype()(0).itemsize
        layout_binding = {
            "algorithm": MASK_ALGORITHM,
            "sequence_length": SEQUENCE_LENGTH,
            "keep_fraction": 0.6,
            "reference_sha256": reference_digest[0],
            "source_ids": source_ids,
            "source_sizes": [path.stat().st_size for path in source_paths],
            "source_dtype": str(dtype),
            "scoring_batch_sequences": args.batch_sequences,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        layout_identity = binding_sha256(layout_binding)
        records = file_layout(
            source_paths,
            source_ids,
            output_root=args.output_root,
            itemsize=itemsize,
        )
        manifest_path = args.output_root / "manifest.json"
        complete = [manifest_path.is_file() if rank == 0 else None]
        dist.broadcast_object_list(complete, src=0)
        if complete[0]:
            load_mask_manifest(
                manifest_path,
                corpus_paths=[str(path) for path in source_paths],
                source_ids=source_ids,
                source_itemsize=itemsize,
                sequence_length=SEQUENCE_LENGTH,
                keep_fraction=0.6,
                reference_sha256=str(reference_digest[0]),
            )
            if rank == 0:
                print(f"precomputed middle-PPL masks already complete: {manifest_path}", flush=True)
            return

        if rank == 0:
            prepare_layout(
                args.output_root,
                layout_binding=layout_binding,
                records=records,
            )
        dist.barrier()

        model = TransformerConfig.olmo2_370M(
            vocab_size=TokenizerConfig.dolma2().padded_vocab_size(),
            init_seed=6198,
        ).build(init_device=str(device))
        model.load_state_dict(checkpoint_state(reference), strict=True)
        model.to(dtype=torch.bfloat16)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if device.type == "cuda":
            model.apply_compile()

        offsets = [0]
        for record in records:
            offsets.append(offsets[-1] + int(record["instances"]))
        total_instances = offsets[-1]
        range_start = total_instances * rank // world_size
        range_end = total_instances * (rank + 1) // world_size
        progress_path = args.output_root / f"progress-rank{rank:02d}.json"
        cursor = progress_start(
            progress_path,
            identity=layout_identity,
            start=range_start,
            end=range_end,
        )
        batches = 0
        started = time.monotonic()
        open_masks: dict[int, np.memmap] = {}

        while cursor < range_end:
            file_index = bisect.bisect_right(offsets, cursor) - 1
            record = records[file_index]
            local_instance = cursor - offsets[file_index]
            available = int(record["instances"]) - local_instance
            count = min(args.batch_sequences, range_end - cursor, available)
            source = np.memmap(
                record["source_path"],
                mode="r",
                dtype=dtype.as_np_dtype(),
                offset=local_instance * SEQUENCE_LENGTH * itemsize,
                shape=(count, SEQUENCE_LENGTH),
            )
            ids = torch.from_numpy(np.array(source, copy=True)).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            labels = F.pad(ids[:, 1:], (0, 1), value=-100)
            with torch.no_grad():
                output = model(
                    ids,
                    labels=labels,
                    ignore_index=-100,
                    loss_reduction="none",
                    return_logits=False,
                )
            if not isinstance(output, LMOutputWithLoss):
                raise RuntimeError("reference model did not return per-token losses")
            reference_loss = output.ce_loss.reshape_as(labels)
            weights = selection_weights(
                "middle_ppl",
                valid=labels != -100,
                keep_fraction=0.6,
                step=0,
                seed=42,
                reference=reference_loss,
            )
            source_mask = weights_to_label_mask(weights).cpu().numpy()
            mask = open_masks.get(file_index)
            if mask is None:
                mask = np.memmap(record["mask_path"], mode="r+", dtype=np.bool_)
                open_masks[file_index] = mask
            mask_start = local_instance * SEQUENCE_LENGTH
            mask[mask_start : mask_start + count * SEQUENCE_LENGTH] = source_mask.reshape(-1)
            cursor += count
            batches += 1
            model.reset_auxiliary_metrics()

            if batches % args.progress_every == 0 or cursor == range_end:
                for mask_map in open_masks.values():
                    mask_map.flush()
                write_progress(
                    progress_path,
                    identity=layout_identity,
                    start=range_start,
                    end=range_end,
                    next_index=cursor,
                )
                elapsed = max(time.monotonic() - started, 1e-6)
                processed = cursor - range_start
                print(
                    f"rank={rank} instances={cursor}/{range_end} "
                    f"tokens_per_second={processed * SEQUENCE_LENGTH / elapsed:,.0f}",
                    flush=True,
                )

        dist.barrier()
        if rank == 0:
            final_records = []
            for record in records:
                final_records.append(
                    {
                        **record,
                        "mask_sha256": file_sha256(record["mask_path"]),
                    }
                )
            binding = {
                **layout_binding,
                "total_instances": total_instances,
                "selected_tokens": total_instances
                * round((SEQUENCE_LENGTH - 1) * float(layout_binding["keep_fraction"])),
                "mask_files": [
                    {
                        "source_id": record["source_id"],
                        "mask_size": record["mask_size"],
                        "mask_sha256": record["mask_sha256"],
                    }
                    for record in final_records
                ],
            }
            atomic_json(
                manifest_path,
                {
                    "schema_version": MASK_SCHEMA_VERSION,
                    "binding": binding,
                    "binding_sha256": binding_sha256(binding),
                    "files": final_records,
                },
            )
            print(
                f"wrote {total_instances:,} exact middle-PPL masks to {manifest_path}",
                flush=True,
            )
        dist.barrier()
    finally:
        teardown_training_environment()


if __name__ == "__main__":
    main()
