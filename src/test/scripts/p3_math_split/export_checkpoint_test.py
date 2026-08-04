"""Remote-safe, provenance-preserving P3 checkpoint export."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tokenizers import Tokenizer, models, pre_tokenizers

ROOT = Path(__file__).resolve().parents[4]
SCRIPTS = ROOT / "src" / "scripts" / "train" / "p3_math_split"
sys.path.insert(0, str(SCRIPTS))

import compare_arms as compare  # noqa: E402
import export_checkpoint as export  # noqa: E402
import provenance  # noqa: E402
import run_eval  # noqa: E402

from olmo_core.nn.transformer.qwen import (  # noqa: E402
    QWEN2_0_5B_HF_ID,
    QWEN2_0_5B_HF_REVISION,
    QWEN2_0_5B_HF_WEIGHTS_SHA256,
    QWEN2_0_5B_HF_WEIGHTS_SIZE,
    convert_hf_state_dict,
    qwen2_0_5b_config,
)


def _saved_config() -> dict:
    return {
        "model": qwen2_0_5b_config().as_config_dict(),
        "arm": "dense",
        "run_mode": "runtime-smoke",
        "model_factory": "qwen2_0_5b",
        "base_model_id": QWEN2_0_5B_HF_ID,
        "base_model_revision": QWEN2_0_5B_HF_REVISION,
        "base_model_weight_sha256": QWEN2_0_5B_HF_WEIGHTS_SHA256,
        "base_model_weight_size": QWEN2_0_5B_HF_WEIGHTS_SIZE,
        "tokenizer_artifact_id": provenance.TOKENIZER_ARTIFACT_ID,
        "tokenizer_artifact_version": provenance.TOKENIZER_ARTIFACT_VERSION,
        "tokenizer_file_sha256": provenance.TOKENIZER_FILE_SHA256,
        "tokenizer_composite_sha256": provenance.TOKENIZER_COMPOSITE_SHA256,
        "tokenizers_version": provenance.TOKENIZERS_VERSION,
        "tokenizer_eos_token_id": provenance.TOKENIZER_EOS_TOKEN_ID,
        "tokenizer_pad_token_id": provenance.TOKENIZER_PAD_TOKEN_ID,
        "dataset_id": "pretrain/formal-proof-premises-500m",
        "dataset_version": "v3",
        "dataset_release": "formal-proof-premises-500m-v3",
        "world_size": 8,
        "launch_contract": {"final_world_size": 8},
        "source_commit": "b" * 40,
        "platform_run_manifest_id": "manifest-dense-123",
        "platform_run_manifest_sha256": "d" * 64,
        "init_seed": 42,
    }


def _trained_weight_files() -> dict:
    return {
        "model.safetensors": {
            "sha256": "c" * 64,
            "bytes": 7,
            "dtype": "BF16",
        }
    }


def test_latest_checkpoint_rejects_torn_highest_step(monkeypatch):
    run = "s3://checkpoints/run"
    complete = f"{run}/step100"
    torn = f"{run}/step200"

    class FakeCheckpointer:
        @classmethod
        def latest_checkpoint(cls, path):
            assert path == run
            return complete

        @classmethod
        def dir_is_checkpoint(cls, path):
            return path == complete

    monkeypatch.setattr(export, "Checkpointer", FakeCheckpointer)
    monkeypatch.setattr(export, "list_directory", lambda *_args, **_kwargs: iter([complete, torn]))

    with pytest.raises(RuntimeError, match=r"step200.*torn"):
        export.resolve_checkpoint(run, None)


def test_specific_checkpoint_must_be_known_and_complete(monkeypatch):
    run = "s3://checkpoints/run"
    complete = f"{run}/step100"

    class FakeCheckpointer:
        latest_checkpoint = classmethod(lambda cls, _path: complete)
        dir_is_checkpoint = classmethod(lambda cls, path: path == complete)

    monkeypatch.setattr(export, "Checkpointer", FakeCheckpointer)
    monkeypatch.setattr(export, "list_directory", lambda *_args, **_kwargs: iter([complete]))

    assert export.resolve_checkpoint(run, "100") == (complete, 100)
    with pytest.raises(RuntimeError, match="unknown checkpoint step 99"):
        export.resolve_checkpoint(run, "99")


def test_checkpoint_step_zero_is_never_reportable(monkeypatch):
    run = "s3://checkpoints/run"
    step_zero = f"{run}/step0"

    class FakeCheckpointer:
        latest_checkpoint = classmethod(lambda cls, _path: step_zero)
        dir_is_checkpoint = classmethod(lambda cls, _path: True)

    monkeypatch.setattr(export, "Checkpointer", FakeCheckpointer)
    monkeypatch.setattr(export, "list_directory", lambda *_args, **_kwargs: iter([step_zero]))

    with pytest.raises(RuntimeError, match="positive"):
        export.resolve_checkpoint(run, None)
    with pytest.raises(RuntimeError, match="positive"):
        export.model_provenance(
            _saved_config(),
            checkpoint=step_zero,
            checkpoint_step=0,
            trained_weight_files=_trained_weight_files(),
        )


@pytest.mark.parametrize("arm", ["", "both", None])
def test_model_export_metadata_requires_a_known_arm(arm):
    saved = _saved_config()
    saved["arm"] = arm

    with pytest.raises(RuntimeError, match="arm"):
        export.model_provenance(
            saved,
            checkpoint="s3://checkpoints/run/step100",
            checkpoint_step=100,
            trained_weight_files=_trained_weight_files(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_commit", ""),
        ("run_mode", "dry-run"),
    ],
)
def test_model_export_metadata_rejects_unreportable_source_identity(field, value):
    saved = _saved_config()
    saved[field] = value

    with pytest.raises(RuntimeError, match="source_commit|dry-run"):
        export.model_provenance(
            saved,
            checkpoint="s3://checkpoints/run/step100",
            checkpoint_step=100,
            trained_weight_files=_trained_weight_files(),
        )


@pytest.mark.parametrize(
    ("manifest_id", "manifest_sha256"),
    [
        ("", "d" * 64),
        ("manifest-dense-123", "not-a-digest"),
    ],
)
def test_model_export_metadata_validates_available_platform_manifest(manifest_id, manifest_sha256):
    saved = _saved_config()
    saved["platform_run_manifest_id"] = manifest_id
    saved["platform_run_manifest_sha256"] = manifest_sha256

    with pytest.raises(RuntimeError, match="platform run manifest"):
        export.model_provenance(
            saved,
            checkpoint="s3://checkpoints/run/step100",
            checkpoint_step=100,
            trained_weight_files=_trained_weight_files(),
        )


def test_mocked_s3_export_uses_remote_helpers_local_unshard_and_explicit_upload(
    tmp_path, monkeypatch
):
    run = "s3://checkpoints/run"
    checkpoint = f"{run}/step100"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_saved_config()), encoding="utf-8")
    out = tmp_path / "hf"
    calls = {}

    monkeypatch.setattr(export, "resolve_checkpoint", lambda *_args: (checkpoint, 100))

    def cached_path(path, **_kwargs):
        calls["config_url"] = path
        return config_path

    monkeypatch.setattr(export, "cached_path", cached_path)
    monkeypatch.setattr(export, "file_exists", lambda path: path.endswith("/.metadata"))

    def unshard_checkpoint(**kwargs):
        calls["unshard"] = kwargs
        target = Path(kwargs["target_dir"])
        assert "://" not in str(target)
        target.mkdir(parents=True)
        model_path = target / "model.pt"
        torch.save({"model": {"fixture.weight": torch.ones(1)}}, model_path)
        return model_path, None

    monkeypatch.setattr(export, "unshard_checkpoint", unshard_checkpoint)
    sealed = SimpleNamespace(
        root=tmp_path / "tokenizer",
        provenance_dict=lambda: {
            "tokenizer_artifact_id": provenance.TOKENIZER_ARTIFACT_ID,
            "tokenizer_artifact_version": provenance.TOKENIZER_ARTIFACT_VERSION,
            "tokenizer_file_sha256": provenance.TOKENIZER_FILE_SHA256,
            "tokenizer_composite_sha256": provenance.TOKENIZER_COMPOSITE_SHA256,
            "tokenizers_version": provenance.TOKENIZERS_VERSION,
            "tokenizer_eos_token_id": provenance.TOKENIZER_EOS_TOKEN_ID,
            "tokenizer_pad_token_id": provenance.TOKENIZER_PAD_TOKEN_ID,
        },
    )

    def fetch_tokenizer(artifact, work_dir):
        calls["tokenizer"] = (artifact, Path(work_dir))
        return sealed

    monkeypatch.setattr(export, "fetch_tokenizer_artifact", fetch_tokenizer)

    def write_hf_dir(state, out_dir, **kwargs):
        calls["writer"] = (state, Path(out_dir), kwargs)
        Path(out_dir).mkdir(parents=True)
        save_file(
            {"fixture.weight": torch.ones(1, dtype=torch.bfloat16)},
            Path(out_dir) / "model.safetensors",
        )

    monkeypatch.setattr(export, "write_hf_dir", write_hf_dir)
    monkeypatch.setattr(
        export,
        "copy_dir",
        lambda source, target, **kwargs: calls.setdefault("upload", (str(source), target, kwargs)),
    )

    provenance_record = export.export_run(
        run,
        out,
        work_dir=tmp_path / "work",
        upload_to="s3://exports/run/hf",
    )

    assert calls["config_url"] == f"{checkpoint}/config.json"
    assert calls["unshard"]["dir"] == f"{checkpoint}/model_and_optim"
    assert calls["unshard"]["pre_download"] is True
    assert calls["tokenizer"][0] == provenance.TOKENIZER_ARTIFACT
    assert calls["upload"][1] == "s3://exports/run/hf"
    assert provenance_record["checkpoint_step"] == 100
    assert provenance_record["resolved_checkpoint_url"] == checkpoint
    assert provenance_record["schema_version"] == "p3-model-export-v1"
    assert provenance_record["arm"] == "dense"
    assert provenance_record["initial_weights_sha256"] == QWEN2_0_5B_HF_WEIGHTS_SHA256
    assert provenance_record["source_commit"] == "b" * 40
    assert provenance_record["platform_run_manifest_id"] == "manifest-dense-123"
    assert provenance_record["platform_run_manifest_sha256"] == "d" * 64
    assert set(provenance_record["trained_weight_files"]) == {"model.safetensors"}
    assert provenance_record["trained_weight_files"]["model.safetensors"]["dtype"] == "BF16"
    assert len(provenance_record["trained_weights_root_sha256"]) == 64
    assert "base_model_weight_sha256" not in provenance_record
    assert json.loads((out / "model_provenance.json").read_text()) == provenance_record


def test_remote_output_requires_local_staging_and_explicit_upload(tmp_path):
    with pytest.raises(ValueError, match="local"):
        export.export_run(
            "s3://checkpoints/run",
            "s3://exports/run/hf",
            work_dir=tmp_path,
        )


def test_hf_writer_rejects_url_before_constructing_a_local_path(monkeypatch):
    class PathMustNotBeConstructed:
        def __new__(cls, *_args, **_kwargs):
            raise AssertionError("URL must be rejected before Path() can turn s3:// into s3:/")

    monkeypatch.setattr(export, "Path", PathMustNotBeConstructed)
    with pytest.raises(ValueError, match="local"):
        export.write_hf_dir(
            {},
            "s3://exports/run/hf",
            saved_config={},
            tokenizer=None,
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("base_model_revision",), "main"),
        (("tokenizer_composite_sha256",), "0" * 64),
        (("dataset_version",), ""),
        (("source_commit",), ""),
        (("model", "d_model"), 1024),
    ],
)
def test_unknown_checkpoint_provenance_or_architecture_is_refused(path, value):
    config = _saved_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RuntimeError):
        export.validate_saved_config(config)


def _tiny_sealed_tokenizer(tmp_path, monkeypatch):
    root = tmp_path / "tokenizer"
    root.mkdir()
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={"[UNK]": 0, "<|endoftext|>": 1, "hello": 2},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "eos_token": "<|endoftext|>",
                "pad_token": "<|endoftext|>",
            }
        ),
        encoding="utf-8",
    )
    file_hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in provenance.TOKENIZER_REQUIRED_FILES
    }
    monkeypatch.setattr(provenance, "TOKENIZER_FILE_SHA256", file_hashes)
    monkeypatch.setattr(provenance, "TOKENIZER_BACKEND_VOCAB_SIZE", 3)
    monkeypatch.setattr(provenance, "TOKENIZER_EOS_TOKEN_ID", 1)
    monkeypatch.setattr(provenance, "TOKENIZER_PAD_TOKEN_ID", 1)
    monkeypatch.setattr(
        provenance,
        "TOKENIZER_COMPOSITE_SHA256",
        provenance.tokenizer_behavior_sha256(tokenizer),
    )
    return provenance.seal_tokenizer_files(root)


def test_floating_state_dtype_requires_coherent_bfloat16():
    assert export.floating_state_dtype({"a": torch.ones(2, dtype=torch.bfloat16)}) == torch.bfloat16
    with pytest.raises(RuntimeError, match="mixed floating"):
        export.floating_state_dtype(
            {
                "a": torch.ones(2, dtype=torch.bfloat16),
                "b": torch.ones(2, dtype=torch.float32),
            }
        )
    with pytest.raises(RuntimeError, match="bfloat16"):
        export.floating_state_dtype({"a": torch.ones(2, dtype=torch.float32)})


def test_trained_weight_inventory_accepts_exact_single_and_sharded_safetensors(tmp_path):
    single = tmp_path / "single"
    single.mkdir()
    save_file({"weight": torch.ones(2, dtype=torch.bfloat16)}, single / "model.safetensors")

    single_files = export.trained_weight_inventory(single)

    assert set(single_files) == {"model.safetensors"}
    assert single_files["model.safetensors"]["dtype"] == "BF16"
    assert single_files["model.safetensors"]["bytes"] > 0
    assert len(export.trained_weights_root_sha256(single_files)) == 64

    sharded = tmp_path / "sharded"
    sharded.mkdir()
    first = "model-00001-of-00002.safetensors"
    second = "model-00002-of-00002.safetensors"
    save_file({"first": torch.ones(1, dtype=torch.bfloat16)}, sharded / first)
    save_file({"second": torch.ones(1, dtype=torch.bfloat16)}, sharded / second)
    (sharded / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 4},
                "weight_map": {"first": first, "second": second},
            }
        ),
        encoding="utf-8",
    )

    sharded_files = export.trained_weight_inventory(sharded)

    assert set(sharded_files) == {first, second, "model.safetensors.index.json"}
    assert sharded_files[first]["dtype"] == sharded_files[second]["dtype"] == "BF16"
    assert sharded_files["model.safetensors.index.json"]["dtype"] == "json"


def test_trained_weight_inventory_rejects_extra_or_non_bf16_payloads(tmp_path):
    extra = tmp_path / "extra"
    extra.mkdir()
    save_file({"weight": torch.ones(1, dtype=torch.bfloat16)}, extra / "model.safetensors")
    save_file({"other": torch.ones(1, dtype=torch.bfloat16)}, extra / "other.safetensors")
    with pytest.raises(RuntimeError, match="trained weight.*extra"):
        export.trained_weight_inventory(extra)

    wrong_dtype = tmp_path / "wrong-dtype"
    wrong_dtype.mkdir()
    save_file({"weight": torch.ones(1, dtype=torch.float32)}, wrong_dtype / "model.safetensors")
    with pytest.raises(RuntimeError, match="BF16"):
        export.trained_weight_inventory(wrong_dtype)


def test_bf16_tied_export_reloads_and_satisfies_actual_evaluator_contract(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(
        vocab_size=17,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
        rope_theta=10_000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=True,
    )
    torch.manual_seed(7)
    original = transformers.AutoModelForCausalLM.from_config(config).to(dtype=torch.bfloat16)
    original.eval()
    olmo_state = convert_hf_state_dict(
        {key: value.detach().clone() for key, value in original.state_dict().items()},
        tied=True,
        n_layers=1,
    )
    sealed = _tiny_sealed_tokenizer(tmp_path, monkeypatch)
    saved = {
        "model": {
            "n_layers": 1,
            "tie_word_embeddings": True,
        },
        **sealed.provenance_dict(),
    }
    monkeypatch.setattr(export, "load_pinned_hf_config", lambda: copy.deepcopy(config))
    out = tmp_path / "hf"

    export.write_hf_dir(olmo_state, out, saved_config=saved, tokenizer=sealed)

    reloaded = transformers.AutoModelForCausalLM.from_pretrained(
        out,
        local_files_only=True,
    )
    reloaded.eval()
    assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()
    assert {
        parameter.dtype for parameter in reloaded.parameters() if parameter.is_floating_point()
    } == {torch.bfloat16}
    with safe_open(out / "model.safetensors", framework="pt") as handle:
        assert {
            handle.get_tensor(key).dtype
            for key in handle.keys()
            if handle.get_tensor(key).is_floating_point()
        } == {torch.bfloat16}
    for key, expected_parameter in original.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], expected_parameter), key

    input_ids = torch.tensor([[2, 3, 4, 5]])
    with torch.no_grad():
        expected = original(input_ids=input_ids).logits
        actual = reloaded(input_ids=input_ids).logits
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert {
        name: hashlib.sha256((out / name).read_bytes()).hexdigest()
        for name in provenance.TOKENIZER_REQUIRED_FILES
    } == sealed.file_sha256

    metadata = export.model_provenance(
        _saved_config(),
        checkpoint="s3://checkpoints/run/step100",
        checkpoint_step=100,
        trained_weight_files=export.trained_weight_inventory(out),
    )
    (out / "model_provenance.json").write_text(json.dumps(metadata), encoding="utf-8")
    resolved = run_eval.resolve_model_provenance(out)
    compare._validate_model_provenance(
        {"arm": resolved["arm"], "input_provenance": {"model": resolved}},
        "synthetic-export",
    )
    assert metadata["schema_version"] == run_eval.MODEL_EXPORT_SCHEMA_VERSION
    assert resolved["initial_weights_sha256"] == QWEN2_0_5B_HF_WEIGHTS_SHA256
    assert resolved["trained_weights_root_sha256"] == metadata["trained_weights_root_sha256"]
    assert "base_model_weight_sha256" not in metadata
    assert all(
        metadata[key] not in (None, "", {}, [])
        for key in export.REQUIRED_EVALUATOR_PROVENANCE_FIELDS
    )

    valid_metadata = copy.deepcopy(metadata)
    hostile_cases = [
        ("integer schema", {**valid_metadata, "schema_version": 1}, "schema"),
        ("zero step", {**valid_metadata, "checkpoint_step": 0}, "checkpoint_step"),
        (
            "wrong digest field",
            {
                **{
                    key: value
                    for key, value in valid_metadata.items()
                    if key != "initial_weights_sha256"
                },
                "base_model_weight_sha256": valid_metadata["initial_weights_sha256"],
            },
            "initial_weights_sha256",
        ),
    ]
    for _label, hostile, message in hostile_cases:
        (out / "model_provenance.json").write_text(json.dumps(hostile), encoding="utf-8")
        with pytest.raises(RuntimeError, match=message):
            run_eval.resolve_model_provenance(out)

    (out / "model_provenance.json").write_text(json.dumps(valid_metadata), encoding="utf-8")
    resolved = run_eval.resolve_model_provenance(out)
    resolved["export_metadata"] = copy.deepcopy(resolved["export_metadata"])
    resolved["export_metadata"]["initial_weights_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="differs from extracted provenance"):
        compare._validate_model_provenance(
            {"arm": resolved["arm"], "input_provenance": {"model": resolved}},
            "hostile-export",
        )
