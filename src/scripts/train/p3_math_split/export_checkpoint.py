"""Export a P3 OLMo-core checkpoint to a local HuggingFace directory.

Local example:

    python src/scripts/train/p3_math_split/export_checkpoint.py --run runs/dense --out exports/dense

Remote checkpoints are supported, but unsharding and model materialization always
stage locally. Upload is a separate explicit option:

    python export_checkpoint.py --run s3://bucket/run --out /tmp/hf --upload-to s3://bucket/hf
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from cached_path import cached_path
from safetensors import safe_open

from olmo_core.distributed.checkpoint import unshard_checkpoint
from olmo_core.io import (
    copy_dir,
    file_exists,
    is_url,
    join_path,
    list_directory,
    normalize_path,
)
from olmo_core.nn.transformer.qwen import (
    HF_EOS_TOKEN_ID,
    HF_HIDDEN_SIZE,
    HF_INTERMEDIATE_SIZE,
    HF_NUM_ATTENTION_HEADS,
    HF_NUM_KV_HEADS,
    HF_NUM_LAYERS,
    HF_RMS_NORM_EPS,
    HF_ROPE_THETA,
    HF_VOCAB_SIZE,
    QWEN2_0_5B_HF_ID,
    QWEN2_0_5B_HF_REVISION,
    QWEN2_0_5B_HF_WEIGHTS_SHA256,
    QWEN2_0_5B_HF_WEIGHTS_SIZE,
    export_to_hf_state_dict,
    resolve_pinned_hf_snapshot,
)
from olmo_core.train.checkpoint import Checkpointer

sys.path.insert(0, str(Path(__file__).resolve().parent))

from provenance import (
    TOKENIZER_ARTIFACT_ID,
    TOKENIZER_ARTIFACT_VERSION,
    TOKENIZER_COMPOSITE_SHA256,
    TOKENIZER_FILE_SHA256,
    TOKENIZER_PAD_TOKEN_ID,
    TOKENIZERS_VERSION,
    fetch_tokenizer_artifact,
    seal_tokenizer_files,
)

MODEL_PROVENANCE_SCHEMA_VERSION = "p3-model-export-v1"
STEP_DIRECTORY = re.compile(r"^step(\d+)$")
SAFETENSORS_SHARD = re.compile(r"^model-(\d{5})-of-(\d{5})\.safetensors$")
SAFETENSORS_INDEX = "model.safetensors.index.json"
REQUIRED_EVALUATOR_PROVENANCE_FIELDS = (
    "checkpoint_step",
    "arm",
    "base_model_id",
    "base_model_revision",
    "initial_weights_sha256",
    "trained_weight_files",
    "trained_weights_root_sha256",
    "tokenizer_artifact_id",
    "tokenizer_artifact_version",
    "tokenizer_file_sha256",
    "tokenizer_composite_sha256",
    "dataset_id",
    "dataset_version",
    "source_commit",
)


def _step_from_path(path: str) -> Optional[int]:
    name = normalize_path(path).rsplit("/", 1)[-1]
    match = STEP_DIRECTORY.fullmatch(name)
    return int(match.group(1)) if match else None


def resolve_checkpoint(run_dir: str, requested_step: Optional[str]) -> tuple[str, int]:
    """Resolve a complete checkpoint and reject a newer torn step directory."""
    run_dir = normalize_path(run_dir)
    try:
        children = list(list_directory(run_dir, include_files=False))
        latest = Checkpointer.latest_checkpoint(run_dir)
    except FileNotFoundError as error:
        raise RuntimeError(f"no complete checkpoints under {run_dir}") from error

    numbered = {
        step: normalize_path(path)
        for path in children
        if (step := _step_from_path(normalize_path(path))) is not None
    }
    latest_step = _step_from_path(latest)
    if latest_step is None:
        raise RuntimeError(f"latest checkpoint has an unknown directory name: {latest}")
    if latest_step <= 0:
        raise RuntimeError(
            f"checkpoint_step must be a positive integer for reportable export, got {latest_step}"
        )
    if not numbered:
        raise RuntimeError(f"no stepN checkpoint directories under {run_dir}")

    highest_step = max(numbered)
    if highest_step > latest_step:
        highest_path = numbered[highest_step]
        if not Checkpointer.dir_is_checkpoint(highest_path):
            raise RuntimeError(
                f"highest checkpoint step{highest_step} is torn/incomplete; "
                f"latest complete checkpoint is step{latest_step}"
            )
        raise RuntimeError(
            f"checkpoint discovery disagreement: step{highest_step} is complete but "
            f"latest_checkpoint returned step{latest_step}"
        )

    if requested_step is None:
        return normalize_path(latest), latest_step
    requested_match = re.fullmatch(r"(?:step)?(\d+)", str(requested_step))
    if requested_match is None:
        raise RuntimeError(f"unknown checkpoint step {requested_step!r}")
    step = int(requested_match.group(1))
    if step <= 0:
        raise RuntimeError(
            f"checkpoint_step must be a positive integer for reportable export, got {step}"
        )
    checkpoint = numbered.get(step)
    if checkpoint is None:
        raise RuntimeError(f"unknown checkpoint step {step} under {run_dir}")
    if not Checkpointer.dir_is_checkpoint(checkpoint):
        raise RuntimeError(f"checkpoint step{step} is torn/incomplete")
    return checkpoint, step


def load_checkpoint_config(checkpoint: str) -> Dict[str, Any]:
    """Load the checkpoint's config through a local/remote-capable cache."""
    config_url = str(join_path(checkpoint, "config.json"))
    try:
        path = cached_path(config_url, quiet=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"checkpoint has no config.json: {checkpoint}") from error
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"checkpoint config is unreadable: {config_url}") from error


def _require_equal(config: Dict[str, Any], key: str, expected: Any) -> None:
    actual = config.get(key)
    if actual != expected:
        raise RuntimeError(
            f"checkpoint provenance {key!r} is unknown or drifted: "
            f"expected {expected!r}, got {actual!r}"
        )


def _validated_source_identity(config: Dict[str, Any]) -> tuple[str, str, str]:
    run_mode = config.get("run_mode", "train")
    if run_mode == "dry-run":
        raise RuntimeError("dry-run config cannot produce a reportable model export")
    source_commit = config.get("source_commit")
    if not isinstance(source_commit, str) or not source_commit.strip():
        raise RuntimeError("checkpoint source_commit must be nonempty for reportable export")
    manifest_id = config.get("platform_run_manifest_id", "")
    manifest_sha256 = config.get("platform_run_manifest_sha256", "")
    if not isinstance(manifest_id, str) or not isinstance(manifest_sha256, str):
        raise RuntimeError("platform run manifest identity fields must be strings")
    manifest_id = manifest_id.strip()
    manifest_sha256 = manifest_sha256.strip()
    if manifest_sha256:
        if not manifest_id:
            raise RuntimeError("platform run manifest SHA-256 requires a manifest ID")
        if re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) is None:
            raise RuntimeError("platform run manifest SHA-256 must be lowercase 64-hex")
    return source_commit.strip(), manifest_id, manifest_sha256


def validate_saved_config(config: Dict[str, Any]) -> None:
    """Require complete immutable provenance and the pinned Qwen architecture."""
    _validated_source_identity(config)
    if config.get("arm") not in {"dense", "split"}:
        raise RuntimeError(f"checkpoint arm is unknown: {config.get('arm')!r}")
    _require_equal(config, "model_factory", "qwen2_0_5b")
    _require_equal(config, "base_model_id", QWEN2_0_5B_HF_ID)
    _require_equal(config, "base_model_revision", QWEN2_0_5B_HF_REVISION)
    _require_equal(config, "base_model_weight_sha256", QWEN2_0_5B_HF_WEIGHTS_SHA256)
    _require_equal(config, "base_model_weight_size", QWEN2_0_5B_HF_WEIGHTS_SIZE)
    _require_equal(config, "tokenizer_artifact_id", TOKENIZER_ARTIFACT_ID)
    _require_equal(config, "tokenizer_artifact_version", TOKENIZER_ARTIFACT_VERSION)
    _require_equal(config, "tokenizer_file_sha256", TOKENIZER_FILE_SHA256)
    _require_equal(config, "tokenizer_composite_sha256", TOKENIZER_COMPOSITE_SHA256)
    _require_equal(config, "tokenizers_version", TOKENIZERS_VERSION)
    _require_equal(config, "tokenizer_eos_token_id", HF_EOS_TOKEN_ID)
    _require_equal(config, "tokenizer_pad_token_id", TOKENIZER_PAD_TOKEN_ID)
    _require_equal(config, "dataset_id", "pretrain/formal-proof-premises-500m")
    if re.fullmatch(r"v[1-9]\d*", str(config.get("dataset_version", ""))) is None:
        raise RuntimeError(
            f"checkpoint dataset version is unknown: {config.get('dataset_version')!r}"
        )
    if config.get("init_seed") != 42:
        raise RuntimeError(f"checkpoint seed is unknown: {config.get('init_seed')!r}")
    world_size = config.get("world_size")
    if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size < 1:
        raise RuntimeError(f"checkpoint world size is unknown: {world_size!r}")

    model = config.get("model")
    if not isinstance(model, dict):
        raise RuntimeError("checkpoint has no saved model architecture")
    block = model.get("block")
    if not isinstance(block, dict):
        raise RuntimeError("checkpoint model has no transformer block config")
    attention = block.get("sequence_mixer")
    feed_forward = block.get("feed_forward")
    layer_norm = block.get("layer_norm")
    if not all(isinstance(value, dict) for value in (attention, feed_forward, layer_norm)):
        raise RuntimeError("checkpoint model block is not the pinned Qwen architecture")
    architecture = {
        "d_model": model.get("d_model"),
        "vocab_size": model.get("vocab_size"),
        "n_layers": model.get("n_layers"),
        "tie_word_embeddings": model.get("tie_word_embeddings"),
        "n_heads": attention.get("n_heads"),
        "n_kv_heads": attention.get("n_kv_heads"),
        "head_dim": attention.get("head_dim"),
        "attention_bias": attention.get("bias"),
        "rope_theta": (attention.get("rope") or {}).get("theta"),
        "intermediate_size": feed_forward.get("hidden_size"),
        "feed_forward_bias": feed_forward.get("bias"),
        "rms_norm_eps": layer_norm.get("eps"),
    }
    expected_architecture = {
        "d_model": HF_HIDDEN_SIZE,
        "vocab_size": HF_VOCAB_SIZE,
        "n_layers": HF_NUM_LAYERS,
        "tie_word_embeddings": True,
        "n_heads": HF_NUM_ATTENTION_HEADS,
        "n_kv_heads": HF_NUM_KV_HEADS,
        "head_dim": HF_HIDDEN_SIZE // HF_NUM_ATTENTION_HEADS,
        "attention_bias": True,
        "rope_theta": HF_ROPE_THETA,
        "intermediate_size": HF_INTERMEDIATE_SIZE,
        "feed_forward_bias": False,
        "rms_norm_eps": HF_RMS_NORM_EPS,
    }
    if architecture != expected_architecture:
        raise RuntimeError(
            "checkpoint model architecture is not pinned Qwen2.5-0.5B: "
            f"expected {expected_architecture}, got {architecture}"
        )


def load_pinned_hf_config():
    """Load config.json from the exact Qwen revision, locally and without AutoConfig."""
    from transformers import Qwen2Config

    snapshot = resolve_pinned_hf_snapshot(allow_patterns=["config.json"])
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"{QWEN2_0_5B_HF_ID}@{QWEN2_0_5B_HF_REVISION} has no config.json")
    config_dict = json.loads(config_path.read_text(encoding="utf-8"))
    return Qwen2Config.from_dict(config_dict)


def _verify_tokenizer_matches_saved(tokenizer, saved_config: Dict[str, Any]) -> None:
    actual = tokenizer.provenance_dict()
    for key in (
        "tokenizer_artifact_id",
        "tokenizer_artifact_version",
        "tokenizer_file_sha256",
        "tokenizer_composite_sha256",
        "tokenizers_version",
        "tokenizer_eos_token_id",
        "tokenizer_pad_token_id",
    ):
        if actual.get(key) != saved_config.get(key):
            raise RuntimeError(
                f"export tokenizer {key} differs from the training checkpoint: "
                f"{actual.get(key)!r} != {saved_config.get(key)!r}"
            )


def floating_state_dtype(state_dict: Dict[str, torch.Tensor]) -> torch.dtype:
    """Require one coherent BF16 dtype across all floating checkpoint parameters."""
    dtypes = {
        value.dtype
        for value in state_dict.values()
        if isinstance(value, torch.Tensor) and value.is_floating_point()
    }
    if not dtypes:
        raise RuntimeError("checkpoint state has no floating parameters")
    if len(dtypes) != 1:
        names = ", ".join(sorted(str(dtype) for dtype in dtypes))
        raise RuntimeError(f"checkpoint has mixed floating parameter dtypes: {names}")
    dtype = next(iter(dtypes))
    if dtype != torch.bfloat16:
        raise RuntimeError(
            f"reportable P3 checkpoint parameters must be torch.bfloat16, got {dtype}"
        )
    return dtype


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _is_weight_artifact(filename: str) -> bool:
    return (
        filename.endswith(".safetensors")
        or filename.endswith(".safetensors.index.json")
        or filename == "pytorch_model.bin"
        or filename == "pytorch_model.bin.index.json"
        or re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", filename) is not None
    )


def _validated_weight_layout(filenames: set[str]) -> tuple[list[str], Optional[str]]:
    if "model.safetensors" in filenames:
        expected = {"model.safetensors"}
        extra = filenames - expected
        if extra:
            raise RuntimeError(
                "trained weight file set has extra files alongside model.safetensors: "
                + ", ".join(sorted(extra))
            )
        return ["model.safetensors"], None

    if SAFETENSORS_INDEX not in filenames:
        raise RuntimeError(
            "trained weight file set must be model.safetensors or exact shards plus "
            f"{SAFETENSORS_INDEX}"
        )
    shard_names = filenames - {SAFETENSORS_INDEX}
    matches = {name: SAFETENSORS_SHARD.fullmatch(name) for name in shard_names}
    extra = sorted(name for name, match in matches.items() if match is None)
    if extra:
        raise RuntimeError("trained weight file set has extra files: " + ", ".join(extra))
    if not matches:
        raise RuntimeError("trained weight shard index has no safetensors shards")
    totals = {int(match.group(2)) for match in matches.values() if match is not None}
    if len(totals) != 1:
        raise RuntimeError("trained weight shard filenames disagree on shard count")
    total = next(iter(totals))
    expected_shards = {
        f"model-{index:05d}-of-{total:05d}.safetensors" for index in range(1, total + 1)
    }
    if shard_names != expected_shards:
        missing = sorted(expected_shards - shard_names)
        unexpected = sorted(shard_names - expected_shards)
        raise RuntimeError(
            f"trained weight shard set is incomplete: missing={missing}, extra={unexpected}"
        )
    return sorted(expected_shards), SAFETENSORS_INDEX


def _safetensors_keys_and_dtype(path: Path) -> tuple[set[str], str]:
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            keys = set(handle.keys())
            dtypes = {str(handle.get_slice(key).get_dtype()).upper() for key in keys}
    except Exception as error:
        raise RuntimeError(f"trained weight safetensors file is unreadable: {path.name}") from error
    if not keys:
        raise RuntimeError(f"trained weight safetensors file is empty: {path.name}")
    normalized_dtypes = {
        "BF16" if dtype in {"BF16", "BFLOAT16", "TORCH.BFLOAT16"} else dtype for dtype in dtypes
    }
    if normalized_dtypes != {"BF16"}:
        raise RuntimeError(
            f"trained weight file {path.name} must contain only BF16 tensors, "
            f"got {sorted(normalized_dtypes)}"
        )
    return keys, "BF16"


def _validate_trained_weight_file_record(files: Dict[str, Any]) -> None:
    if not isinstance(files, dict) or not files:
        raise RuntimeError("trained weight file inventory must be a nonempty object")
    shard_names, index_name = _validated_weight_layout(set(files))
    for filename in shard_names + ([index_name] if index_name is not None else []):
        entry = files.get(filename)
        if not isinstance(entry, dict) or set(entry) != {"sha256", "bytes", "dtype"}:
            raise RuntimeError(
                f"trained weight inventory entry {filename!r} must contain "
                "exactly sha256, bytes, and dtype"
            )
        digest = entry["sha256"]
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise RuntimeError(f"trained weight inventory {filename!r} sha256 is invalid")
        size = entry["bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise RuntimeError(f"trained weight inventory {filename!r} bytes must be positive")
        expected_dtype = "json" if filename == index_name else "BF16"
        if entry["dtype"] != expected_dtype:
            raise RuntimeError(
                f"trained weight inventory {filename!r} dtype must be {expected_dtype!r}"
            )


def trained_weights_root_sha256(files: Dict[str, Any]) -> str:
    """Hash the canonical trained-weight inventory, including filenames and file metadata."""
    _validate_trained_weight_file_record(files)
    return _stable_json_sha256(files)


def trained_weight_inventory(out_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Validate and content-address the exact HF safetensors payload."""
    out = Path(out_dir)
    filenames = {
        path.name for path in out.iterdir() if path.is_file() and _is_weight_artifact(path.name)
    }
    shard_names, index_name = _validated_weight_layout(filenames)
    expected_keys_by_shard: Dict[str, set[str]] = {}
    if index_name is not None:
        index_path = out / index_name
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError("trained weight safetensors index is unreadable") from error
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("trained weight safetensors index has no nonempty weight_map")
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in weight_map.items()
        ):
            raise RuntimeError("trained weight safetensors index weight_map must map strings")
        if set(weight_map.values()) != set(shard_names):
            raise RuntimeError("trained weight safetensors index does not name the exact shard set")
        expected_keys_by_shard = {
            shard: {key for key, filename in weight_map.items() if filename == shard}
            for shard in shard_names
        }

    inventory: Dict[str, Dict[str, Any]] = {}
    for filename in shard_names:
        path = out / filename
        keys, dtype = _safetensors_keys_and_dtype(path)
        if index_name is not None and keys != expected_keys_by_shard[filename]:
            raise RuntimeError(f"trained weight shard {filename} tensor keys differ from the index")
        inventory[filename] = {
            "sha256": _sha256_file(path),
            "bytes": path.stat().st_size,
            "dtype": dtype,
        }
    if index_name is not None:
        index_path = out / index_name
        inventory[index_name] = {
            "sha256": _sha256_file(index_path),
            "bytes": index_path.stat().st_size,
            "dtype": "json",
        }
    _validate_trained_weight_file_record(inventory)
    return inventory


def write_hf_dir(
    olmo_state_dict: Dict[str, torch.Tensor],
    out_dir: str | Path,
    *,
    saved_config: Dict[str, Any],
    tokenizer,
) -> None:
    """Materialize a tied/untied HF model using saved architecture and sealed tokenizer bytes."""
    from transformers import Qwen2ForCausalLM

    if is_url(out_dir):
        raise ValueError(f"HF output directory must be local, got {out_dir!r}")
    source_dtype = floating_state_dtype(olmo_state_dict)
    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"HF output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    model_config = saved_config.get("model") or {}
    tied = model_config.get("tie_word_embeddings")
    n_layers = model_config.get("n_layers")
    if not isinstance(tied, bool) or not isinstance(n_layers, int):
        raise RuntimeError("saved model config does not declare tying and layer count")

    hf_config = copy.deepcopy(load_pinned_hf_config())
    if hf_config.num_hidden_layers != n_layers:
        raise RuntimeError(
            f"pinned HF config has {hf_config.num_hidden_layers} layers but checkpoint "
            f"declares {n_layers}"
        )
    hf_config.tie_word_embeddings = tied
    hf_config.dtype = source_dtype
    model = Qwen2ForCausalLM(hf_config).to(dtype=source_dtype)
    hf_state = export_to_hf_state_dict(
        olmo_state_dict,
        tied=tied,
        n_layers=n_layers,
    )
    if floating_state_dtype(hf_state) != source_dtype:
        raise RuntimeError("HF state conversion changed the checkpoint floating dtype")
    missing, unexpected = model.load_state_dict(hf_state, strict=False)
    allowed_missing = {"lm_head.weight"} if tied else set()
    real_missing = sorted(set(missing) - allowed_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            f"export mismatch: missing={real_missing[:5]} " f"unexpected={list(unexpected)[:5]}"
        )
    if tied:
        model.tie_weights()
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise RuntimeError("HF export broke the saved embedding tie")
    model.save_pretrained(out, safe_serialization=True)

    for filename in ("tokenizer.json", "tokenizer_config.json"):
        source = Path(tokenizer.root) / filename
        if not source.is_file():
            raise FileNotFoundError(f"sealed training tokenizer is missing {source}")
        shutil.copyfile(source, out / filename)
    exported_tokenizer = seal_tokenizer_files(out)
    _verify_tokenizer_matches_saved(exported_tokenizer, saved_config)


def model_provenance(
    saved_config: Dict[str, Any],
    *,
    checkpoint: str,
    checkpoint_step: int,
    trained_weight_files: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the stable evaluator handoff without local/path-sensitive hashes."""
    if (
        not isinstance(checkpoint_step, int)
        or isinstance(checkpoint_step, bool)
        or checkpoint_step <= 0
    ):
        raise RuntimeError("model provenance checkpoint_step must be a positive integer")
    arm = saved_config.get("arm")
    if arm not in {"dense", "split"}:
        raise RuntimeError(f"model provenance arm must be 'dense' or 'split', got {arm!r}")
    source_commit, manifest_id, manifest_sha256 = _validated_source_identity(saved_config)
    trained_weights_root = trained_weights_root_sha256(trained_weight_files)
    record = {
        "schema_version": MODEL_PROVENANCE_SCHEMA_VERSION,
        "resolved_checkpoint_url": normalize_path(checkpoint),
        "checkpoint_step": checkpoint_step,
        "arm": arm,
        "run_mode": saved_config.get("run_mode", "train"),
        "base_model_id": saved_config["base_model_id"],
        "base_model_revision": saved_config["base_model_revision"],
        "initial_weights_sha256": saved_config["base_model_weight_sha256"],
        "trained_weight_files": copy.deepcopy(trained_weight_files),
        "trained_weights_root_sha256": trained_weights_root,
        "base_model_weight_size": saved_config["base_model_weight_size"],
        "tokenizer_artifact_id": saved_config["tokenizer_artifact_id"],
        "tokenizer_artifact_version": saved_config["tokenizer_artifact_version"],
        "tokenizer_file_sha256": saved_config["tokenizer_file_sha256"],
        "tokenizer_composite_sha256": saved_config["tokenizer_composite_sha256"],
        "tokenizers_version": saved_config["tokenizers_version"],
        "tokenizer_eos_token_id": saved_config["tokenizer_eos_token_id"],
        "tokenizer_pad_token_id": saved_config["tokenizer_pad_token_id"],
        "dataset_id": saved_config["dataset_id"],
        "dataset_version": saved_config["dataset_version"],
        "dataset_release": saved_config.get("dataset_release", ""),
        "world_size": saved_config["world_size"],
        "source_commit": source_commit,
    }
    if manifest_id:
        record["platform_run_manifest_id"] = manifest_id
    if manifest_sha256:
        record["platform_run_manifest_sha256"] = manifest_sha256
    missing = [key for key in REQUIRED_EVALUATOR_PROVENANCE_FIELDS if key not in record]
    if missing:
        raise RuntimeError(f"model provenance is missing required fields: {missing}")
    required_text = (
        "base_model_id",
        "base_model_revision",
        "initial_weights_sha256",
        "trained_weights_root_sha256",
        "tokenizer_artifact_id",
        "tokenizer_artifact_version",
        "tokenizer_composite_sha256",
        "dataset_id",
        "dataset_version",
        "source_commit",
    )
    empty = [
        key for key in required_text if not isinstance(record[key], str) or not record[key].strip()
    ]
    if empty:
        raise RuntimeError(f"model provenance fields must be nonempty: {empty}")
    if record["initial_weights_sha256"] != QWEN2_0_5B_HF_WEIGHTS_SHA256:
        raise RuntimeError("model provenance initial_weights_sha256 is not the pinned Qwen weights")
    file_hashes = record["tokenizer_file_sha256"]
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise RuntimeError("model provenance tokenizer_file_sha256 must be a nonempty mapping")
    return record


def export_run(
    run_dir: str,
    out_dir: str | Path,
    *,
    requested_step: Optional[str] = None,
    work_dir: Optional[str | Path] = None,
    upload_to: Optional[str] = None,
) -> Dict[str, Any]:
    """Export one local or remote run through a mandatory local staging directory."""
    if is_url(out_dir):
        raise ValueError(
            "HF output must be a local directory; use --upload-to for an explicit remote copy"
        )
    if work_dir is not None and is_url(work_dir):
        raise ValueError(f"export work_dir must be local, got {work_dir!r}")
    if upload_to is not None and not is_url(upload_to):
        raise ValueError(f"upload target must be a supported URL, got {upload_to!r}")

    out = Path(out_dir)
    work_root = Path(work_dir) if work_dir is not None else None
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)

    checkpoint, step = resolve_checkpoint(run_dir, requested_step)
    saved_config = load_checkpoint_config(checkpoint)
    validate_saved_config(saved_config)
    model_and_optim = str(join_path(checkpoint, "model_and_optim"))
    if file_exists(join_path(model_and_optim, ".metadata")):
        sharded_model = model_and_optim
    elif file_exists(join_path(checkpoint, ".metadata")):
        sharded_model = checkpoint
    else:
        raise RuntimeError(f"checkpoint step{step} has no distributed model metadata")

    with tempfile.TemporaryDirectory(dir=work_root) as temporary:
        staging = Path(temporary)
        model_path, _ = unshard_checkpoint(
            dir=sharded_model,
            target_dir=staging / "unsharded",
            optim=False,
            save_overwrite=False,
            pre_download=True,
            work_dir=staging / "dcp-cache",
        )
        loaded = torch.load(model_path, map_location="cpu", weights_only=True)
        state = loaded.get("model", loaded)
        if not isinstance(state, dict):
            raise RuntimeError("unsharded checkpoint did not contain a model state dict")

        artifact = (
            f"{saved_config['tokenizer_artifact_id']}/"
            f"{saved_config['tokenizer_artifact_version']}"
        )
        tokenizer = fetch_tokenizer_artifact(artifact, staging / "tokenizer-cache")
        _verify_tokenizer_matches_saved(tokenizer, saved_config)
        write_hf_dir(state, out, saved_config=saved_config, tokenizer=tokenizer)

    provenance = model_provenance(
        saved_config,
        checkpoint=checkpoint,
        checkpoint_step=step,
        trained_weight_files=trained_weight_inventory(out),
    )
    (out / "model_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if upload_to is not None:
        copy_dir(out, upload_to, save_overwrite=False)
    return provenance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Local path or s3:// run checkpoint prefix")
    parser.add_argument("--out", help="Local HF output directory")
    parser.add_argument("--step", help="Specific numeric step; default is latest complete")
    parser.add_argument("--work-dir", help="Optional local parent for temporary unsharding")
    parser.add_argument("--upload-to", help="Optional explicit s3:// output prefix")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.out is None:
        if is_url(args.run):
            raise SystemExit("--out is required and must be local when --run is a URL")
        args.out = str(Path(args.run) / "hf")
    provenance = export_run(
        args.run,
        args.out,
        requested_step=args.step,
        work_dir=args.work_dir,
        upload_to=args.upload_to,
    )
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
