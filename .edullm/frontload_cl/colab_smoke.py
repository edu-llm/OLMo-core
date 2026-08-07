"""
Single-GPU / Colab smoke for frontload-cl 370M.

This is **not** the platform 8×A100 path. It answers: does one A100 hold the
real per-rank microbatch (``24×4096``) with FlashAttention-2 and ``torch.compile``?

Modes
-----
``microbench``
    Build OLMo2-370M, run a few forward+backward steps on random tokens at the
    ladder microbatch. No corpus, no S3, no curriculum.

``write-data``
    Write tiny ``uint32`` little-endian shards under ``tokens/<source>/`` with the
    same source folder names the real corpus uses.

``train``
    Short :class:`~olmo_core.train.Trainer` fit on a **flat** mix of those synthetic
    shards (not the primer/control curriculum). Proves data→train_module on 1 GPU.

Colab::

    # After cloning this branch into /content/OLMo-core (see colab_smoke.ipynb):
    # install with ``pip install -e . --no-deps`` + a few wheels — not ``.[all]``.
    %cd /content/OLMo-core
    !python .edullm/frontload_cl/colab_smoke.py microbench --attn-backend flash_2
    !python .edullm/frontload_cl/colab_smoke.py write-data --out /content/synth
    !python .edullm/frontload_cl/colab_smoke.py train --data /content/synth --steps 5

Local (CPU-safe checks only)::

    pytest -v src/test/edullm/test_frontload_cl_colab_smoke.py
    python .edullm/frontload_cl/colab_smoke.py write-data --out /tmp/synth
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_DIR = Path(__file__).resolve().parent
_PARENT = str(_DIR.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from frontload_cl import constants as C  # noqa: E402

log = logging.getLogger(__name__)

# All sources the real schedule expects under tokens/.
SYNTH_SOURCES = (
    C.SOURCE_FINEWEB_MAIN,
    C.SOURCE_FINEWEB_ANNEAL,
    C.SOURCE_FINEWIKI,
    *C.SFT_LIKE_SOURCES,
)

# Default synthetic size: enough for a few steps at the 1-GPU microbatch, not 10B.
DEFAULT_TOKENS_PER_SOURCE = 256 * C.SEQ_LENGTH  # 1_048_576 tokens each


def _ensure_parent_on_path() -> None:
    if _PARENT not in sys.path:
        sys.path.insert(0, _PARENT)


def gpu_report() -> Dict[str, object]:
    """Facts about the visible accelerator (for the notebook / JSON summary)."""
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "device_name": props.name,
        "total_memory_gib": round(props.total_memory / (1024**3), 2),
        "capability": f"{props.major}.{props.minor}",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def write_synthetic_shards(
    out_dir: Path | str,
    *,
    tokens_per_source: int = DEFAULT_TOKENS_PER_SOURCE,
    vocab_size: int = 100_278,
    seed: int = C.DATA_SEED,
) -> Dict[str, str]:
    """
    Write little-endian ``uint32`` ``.bin`` shards under ``tokens/<source>/``.

    Layout matches ``pretrain/frontload-cl-10b`` source folders so the same path
    grouping logic applies. Token ids stay in ``[0, vocab_size)``.

    :returns: Map of source name → shard path (as a string).
    """
    root = Path(out_dir)
    tokens_root = root / "tokens"
    rng = np.random.default_rng(seed)
    # Align to sequence length so ConcatAndChunk has no awkward remainder.
    n = (tokens_per_source // C.SEQ_LENGTH) * C.SEQ_LENGTH
    if n < C.SEQ_LENGTH:
        raise ValueError(f"tokens_per_source={tokens_per_source} is shorter than seq={C.SEQ_LENGTH}")

    written: Dict[str, str] = {}
    for source in SYNTH_SOURCES:
        folder = tokens_root / source
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / "train-00000.u32le.bin"
        data = rng.integers(0, vocab_size, size=n, dtype=np.uint32)
        data.tofile(path)
        written[source] = str(path.resolve())
        log.info("wrote %s (%d tokens)", path, n)
    return written


def _peak_mem_gib() -> Optional[float]:
    import torch

    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / (1024**3), 3)


def run_microbench(
    *,
    steps: int = 3,
    sequences: int = C.GLOBAL_BATCH_SEQUENCES // 8,  # 24 — one 8×A100 rank
    seq_length: int = C.SEQ_LENGTH,
    attn_backend: str = C.DEFAULT_ATTN_BACKEND,
    compile_model: bool = False,
    with_optim: bool = False,
    device: Optional[str] = None,
) -> Dict[str, object]:
    """
    OLMo2-370M forward+backward at the real per-rank microbatch shape.

    Goal: approximate **activation** memory of one ``gpu-8xa100`` rank (24×4096,
    FA2, bf16). This is intentionally *not* a full replica of HSDP training:

    - Platform HSDP shards params + Adam state across 8 GPUs; this process holds
      the whole model, so enabling ``with_optim`` overestimates rank memory.
    - ``torch.compile`` can spike reserved memory on the first step; leave it off
      until a non-compile step succeeds.

    On a clean 40 GiB A100, a FA2+bf16 fwd/bwd at 24×4096 should fit; leftover
    fragmentation from earlier OOMs often will not — restart the Colab session first.
    """
    import torch

    from frontload_cl.attn import resolve_attn_backend
    from olmo_core.config import DType
    from olmo_core.data import TokenizerConfig
    from olmo_core.nn.transformer import TransformerConfig

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    backend = resolve_attn_backend(attn_backend)
    # Match platform HSDP param_dtype: FA2 only accepts fp16/bf16 activations.
    param_dtype = DType.bfloat16 if device.startswith("cuda") else DType.float32
    tokenizer = TokenizerConfig.dolma2()
    vocab = tokenizer.padded_vocab_size()
    config = TransformerConfig.olmo2_370M(
        vocab_size=vocab, attn_backend=backend, dtype=param_dtype
    )
    log.info(
        "microbench: device=%s attn=%s dtype=%s compile=%s optim=%s shape=(%d,%d) vocab=%d",
        device,
        backend,
        param_dtype,
        compile_model,
        with_optim,
        sequences,
        seq_length,
        vocab,
    )

    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = config.build(init_device=device)
    model.train()
    if compile_model and device.startswith("cuda"):
        # Same intent as the platform trainer (Inductor). First step is slow / spiky.
        model = torch.compile(model)

    # Default: no Adam — platform HSDP shards optim state; a full AdamW here would
    # add several GiB the rank does not carry and confuses the activation question.
    opt = (
        torch.optim.AdamW(model.parameters(), lr=C.PEAK_LR) if with_optim else None
    )
    started = time.monotonic()
    last_loss: Optional[float] = None
    for step in range(steps):
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (sequences, seq_length), device=device
        )
        # Labels are caller-shifted left of input_ids (see Transformer.forward).
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        labels[:, -1] = -100
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        else:
            model.zero_grad(set_to_none=True)
        out = model(input_ids=input_ids, labels=labels)
        loss = out.loss if hasattr(out, "loss") else out
        loss.backward()
        if opt is not None:
            opt.step()
        last_loss = float(loss.detach())
        log.info(
            "step %d/%d loss=%.4f peak_mem_gib=%s",
            step + 1,
            steps,
            last_loss,
            _peak_mem_gib(),
        )

    result = {
        "mode": "microbench",
        "ok": True,
        "steps": steps,
        "sequences": sequences,
        "seq_length": seq_length,
        "attn_backend": str(backend),
        "param_dtype": str(param_dtype),
        "compile": compile_model,
        "with_optim": with_optim,
        "device": device,
        "last_loss": last_loss,
        "seconds": round(time.monotonic() - started, 2),
        "peak_mem_gib": _peak_mem_gib(),
        "gpu": gpu_report(),
        "note": (
            "fwd/bwd activation proxy for one HSDP rank; full Adam/compile are optional "
            "and stricter than the sharded platform path"
        ),
    }
    print(json.dumps(result, indent=2), flush=True)
    return result


def run_synthetic_train(
    data_dir: Path | str,
    *,
    steps: int = 5,
    sequences: int = C.GLOBAL_BATCH_SEQUENCES // 8,
    seq_length: int = C.SEQ_LENGTH,
    attn_backend: str = C.DEFAULT_ATTN_BACKEND,
    compile_model: bool = True,
    work_dir: Optional[str] = None,
    save_folder: Optional[str] = None,
) -> Dict[str, object]:
    """
    Short single-process Trainer fit on a flat mix of synthetic shards.

    Does **not** build the primer/control curriculum (those budgets need the real
    10B corpus). Use ``microbench`` for the pure OOM/flash/compile question.
    """
    from frontload_cl.attn import resolve_attn_backend
    from olmo_core.config import DType
    from olmo_core.data import NumpyDatasetDType, TokenizerConfig
    from olmo_core.data.composable import (
        ConcatAndChunkInstanceSource,
        NumpyDocumentSource,
        ShuffleStrategy,
        ComposableDataLoaderConfig,
    )
    from olmo_core.distributed.parallel import DataParallelType
    from olmo_core.nn.transformer import TransformerConfig
    from olmo_core.optim import CosWithWarmup, OptimGroupOverride, SkipStepAdamWConfig
    from olmo_core.train import (
        Duration,
        TrainerConfig,
        prepare_training_environment,
        teardown_training_environment,
    )
    from olmo_core.train.callbacks import GPUMemoryMonitorCallback
    from olmo_core.train.train_module import (
        TransformerDataParallelConfig,
        TransformerTrainModuleConfig,
    )
    from olmo_core.utils import seed_all

    data_root = Path(data_dir)
    tokens_root = data_root / "tokens" if (data_root / "tokens").is_dir() else data_root
    paths: List[str] = []
    for source in SYNTH_SOURCES:
        shard = tokens_root / source / "train-00000.u32le.bin"
        if not shard.is_file():
            raise FileNotFoundError(
                f"missing {shard}; run: python .edullm/frontload_cl/colab_smoke.py "
                f"write-data --out {data_dir}"
            )
        paths.append(str(shard.resolve()))

    if work_dir is None:
        work_dir = str(data_root / "work")
    if save_folder is None:
        save_folder = str(data_root / "checkpoints")
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    backend = resolve_attn_backend(attn_backend)
    tokenizer = TokenizerConfig.dolma2()
    global_batch = sequences * seq_length

    prepare_training_environment()
    started = time.monotonic()
    try:
        seed_all(C.DATA_SEED)
        model_config = TransformerConfig.olmo2_370M(
            vocab_size=tokenizer.padded_vocab_size(),
            attn_backend=backend,
        )
        # Flat concat of every synthetic source (not the primer/control curriculum).
        doc = NumpyDocumentSource.Config(
            source_paths=paths,
            tokenizer=tokenizer,
            dtype=NumpyDatasetDType.uint32,
            label="colab-synth",
        )
        flat = ConcatAndChunkInstanceSource.Config(
            sources=[doc],
            sequence_length=seq_length,
            label="colab-flat",
        ).build(work_dir)
        needed = steps * global_batch
        if flat.num_tokens < needed:
            raise RuntimeError(
                f"synthetic corpus has {flat.num_tokens:,} tokens but {steps} steps need "
                f"{needed:,}; raise --tokens-per-source on write-data"
            )

        data_loader_config = ComposableDataLoaderConfig(
            tokenizer=tokenizer,
            global_batch_size=global_batch,
            seed=C.DATA_SEED,
            num_workers=0,
            shuffle=True,
            shuffle_strategy=ShuffleStrategy.intra_source,
            work_dir=work_dir,
        )
        train_module_config = TransformerTrainModuleConfig(
            rank_microbatch_size=global_batch,
            max_sequence_length=seq_length,
            optim=SkipStepAdamWConfig(
                lr=C.PEAK_LR,
                weight_decay=C.WEIGHT_DECAY,
                betas=C.ADAM_BETAS,
                group_overrides=[
                    OptimGroupOverride(params=["embeddings.weight"], opts=dict(weight_decay=0.0))
                ],
            ),
            compile_model=compile_model,
            dp_config=TransformerDataParallelConfig(
                name=DataParallelType.fsdp,
                param_dtype=DType.bfloat16,
                reduce_dtype=DType.float32,
            ),
            max_grad_norm=C.GRAD_CLIP,
            scheduler=CosWithWarmup(warmup=min(2, max(steps - 1, 0))),
        )
        trainer_config = TrainerConfig(
            save_folder=save_folder,
            save_overwrite=True,
            metrics_collect_interval=1,
            cancel_check_interval=1,
            max_duration=Duration.steps(steps),
        ).with_callback("gpu_monitor", GPUMemoryMonitorCallback())

        model = model_config.build(init_device="meta")
        train_module = train_module_config.build(model)
        data_loader = data_loader_config.build(
            flat,
            work_dir=work_dir,
            dp_process_group=train_module.dp_process_group,
        )
        trainer = trainer_config.build(train_module, data_loader)
        trainer.fit()
        result = {
            "mode": "train",
            "ok": True,
            "steps": trainer.global_step,
            "sequences": sequences,
            "seq_length": seq_length,
            "attn_backend": str(backend),
            "compile": compile_model,
            "seconds": round(time.monotonic() - started, 2),
            "peak_mem_gib": _peak_mem_gib(),
            "gpu": gpu_report(),
            "note": "flat synthetic mix only; not primer/control curriculum",
        }
        print(json.dumps(result, indent=2), flush=True)
        return result
    finally:
        teardown_training_environment()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="frontload_cl.colab_smoke",
        description="Single-GPU / Colab smoke for frontload-cl 370M.",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("microbench", help="Fwd/bwd at 24×4096 (no corpus).")
    m.add_argument("--steps", type=int, default=3)
    m.add_argument("--sequences", type=int, default=C.GLOBAL_BATCH_SEQUENCES // 8)
    m.add_argument("--seq-length", type=int, default=C.SEQ_LENGTH)
    m.add_argument("--attn-backend", default=C.DEFAULT_ATTN_BACKEND)
    m.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile (off by default; can spike memory on step 1).",
    )
    m.add_argument(
        "--with-optim",
        action="store_true",
        help="Run AdamW on the full model (stricter than HSDP; off by default).",
    )
    m.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")

    w = sub.add_parser("write-data", help="Write synthetic tokens/<source> shards.")
    w.add_argument("--out", required=True, help="Output directory.")
    w.add_argument("--tokens-per-source", type=int, default=DEFAULT_TOKENS_PER_SOURCE)
    w.add_argument("--seed", type=int, default=C.DATA_SEED)

    t = sub.add_parser("train", help="Short Trainer fit on synthetic flat mix.")
    t.add_argument("--data", required=True, help="Directory from write-data.")
    t.add_argument("--steps", type=int, default=5)
    t.add_argument("--sequences", type=int, default=C.GLOBAL_BATCH_SEQUENCES // 8)
    t.add_argument("--seq-length", type=int, default=C.SEQ_LENGTH)
    t.add_argument("--attn-backend", default=C.DEFAULT_ATTN_BACKEND)
    t.add_argument("--no-compile", action="store_true")
    t.add_argument("--work-dir", default=None)
    t.add_argument("--save-folder", default=None)

    sub.add_parser("gpu-info", help="Print visible GPU facts as JSON.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    opts = build_parser().parse_args(argv)
    _ensure_parent_on_path()

    if opts.mode == "gpu-info":
        print(json.dumps(gpu_report(), indent=2), flush=True)
        return 0
    if opts.mode == "write-data":
        written = write_synthetic_shards(
            opts.out,
            tokens_per_source=opts.tokens_per_source,
            seed=opts.seed,
        )
        print(json.dumps({"mode": "write-data", "ok": True, "shards": written}, indent=2))
        return 0
    if opts.mode == "microbench":
        run_microbench(
            steps=opts.steps,
            sequences=opts.sequences,
            seq_length=opts.seq_length,
            attn_backend=opts.attn_backend,
            compile_model=opts.compile,
            with_optim=opts.with_optim,
            device=opts.device,
        )
        return 0
    if opts.mode == "train":
        run_synthetic_train(
            opts.data,
            steps=opts.steps,
            sequences=opts.sequences,
            seq_length=opts.seq_length,
            attn_backend=opts.attn_backend,
            compile_model=not opts.no_compile,
            work_dir=opts.work_dir,
            save_folder=opts.save_folder,
        )
        return 0
    raise AssertionError(opts.mode)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001 — Colab wants a clear last cell error
        log.exception("colab_smoke failed")
        print(
            json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}),
            flush=True,
        )
        raise
