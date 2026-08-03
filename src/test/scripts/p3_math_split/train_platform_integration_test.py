"""Build the complete platform config without AWS or a GPU.

This catches controls that are present in YAML but silently dropped while adapting
`.edullm/train_on_corpus.py`. The resulting config is exactly what `--dry-run`
prints; model construction and actual bytes are covered elsewhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "src" / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location("p3_train_platform", SCRIPTS / "train_platform.py")
assert spec and spec.loader
platform = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = platform
spec.loader.exec_module(platform)

from olmo_core.data import NumpyDatasetDType  # noqa: E402
from olmo_core.nn.attention import AttentionBackendName  # noqa: E402
from olmo_core.nn.transformer.qwen import qwen2_tokenizer_config  # noqa: E402


@pytest.fixture
def built(monkeypatch):
    corpus = platform.Corpus(
        dataset_id="pretrain/formal-proof-premises-500m",
        version="v2",
        paths=["s3://edullm-data/pretrain/formal-proof-premises-500m/v2/tokens/x/train-00000.u32le.bin"],
        dtype=NumpyDatasetDType.uint32,
        tokenizer=qwen2_tokenizer_config(),
        rows=494_862_336,
    )
    monkeypatch.setattr(platform, "resolve_corpus", lambda **_: corpus)
    monkeypatch.setattr(platform, "separator_ids_for", lambda *_: [10952, 15513, 969])
    monkeypatch.delenv("WANDB_PROJECT", raising=False)
    monkeypatch.delenv("EDULLM_WANDB_PROJECT", raising=False)
    args = platform.build_parser().parse_args(
        [
            "test-run",
            "--arm",
            "split",
            "--config",
            str(SCRIPTS / "configs" / "split.yaml"),
            "--dataset-id",
            corpus.dataset_id,
            "--dataset-version",
            corpus.version,
            "--dataset-tokenizer",
            "tokenizer/qwen25-vendored/v1",
            "--save-folder",
            "s3://checkpoints/test-run/",
        ]
    )
    return platform.build_config(args, []), args


def test_dataset_and_loader_controls(built):
    cfg, _ = built
    assert cfg.dataset.paths == [
        "s3://edullm-data/pretrain/formal-proof-premises-500m/v2/tokens/x/train-00000.u32le.bin"
    ]
    assert cfg.dataset.dtype == NumpyDatasetDType.uint32
    assert cfg.dataset.sequence_length == 16_384
    assert cfg.dataset.generate_doc_lengths is True
    assert cfg.data_loader.global_batch_size == 262_144
    assert cfg.data_loader.seed == 42
    assert cfg.data_loader.num_workers == 2


def test_model_and_mask_controls(built):
    cfg, _ = built
    assert cfg.model.vocab_size == 151_936
    assert cfg.model.init_seed == 42
    assert cfg.model.tie_word_embeddings is True
    assert cfg.model.block.sequence_mixer.backend == AttentionBackendName.flash_2
    assert cfg.train_module.arm == "split"
    assert cfg.train_module.separator_ids == [10952, 15513, 969]
    assert cfg.train_module.eos_token_id == 151_643
    assert cfg.train_module.pad_token_id == 151_643
    assert cfg.train_module.fixed_loss_div_factor == 262_144.0


def test_optimizer_schedule_and_checkpoint_controls(built):
    cfg, args = built
    assert cfg.train_module.rank_microbatch_size == 16_384
    assert cfg.train_module.optim.lr == 2e-5
    assert cfg.train_module.optim.betas == (0.9, 0.95)
    assert cfg.train_module.optim.eps == 1e-8
    assert cfg.train_module.optim.weight_decay == 0.0
    assert cfg.train_module.max_grad_norm == 1.0
    assert cfg.train_module.scheduler.warmup == 2_400
    assert cfg.train_module.scheduler.alpha_f == 0.1
    assert args.steps == (494_862_336 * 13) // 262_144
    assert cfg.trainer.max_duration.value == args.steps
    assert cfg.trainer.callbacks["checkpointer"].save_interval == 2_000
    assert cfg.trainer.callbacks["checkpointer"].ephemeral_save_interval is None
    assert cfg.trainer.callbacks["checkpointer"].max_checkpoints is None
    assert cfg.trainer.save_folder == "s3://checkpoints/test-run/"


def test_wandb_comes_from_platform_form_not_local_yaml(built, monkeypatch):
    _, args = built
    assert platform.wandb_project(args) is None
    monkeypatch.setenv("WANDB_PROJECT", "p3-math")
    assert platform.wandb_project(args) == "p3-math"


def test_versioned_published_tokenizer_resolves_to_local_config():
    """The platform pins dependencies as tokenizer/<name>/vN."""
    read = SimpleNamespace(
        paths=["s3://bucket/tokens/x/train-00000.u32le.bin"],
        dtype="uint32",
        byte_order=sys.byteorder,
        header_bytes=0,
        rows=16_384,
    )
    corpus = platform.corpus_from_manifest(
        read,
        dataset_id="pretrain/formal-proof-premises-500m",
        version="v2",
        tokenizer_id="tokenizer/qwen25-vendored/v1",
    )
    assert corpus.tokenizer.vocab_size == 151_936

