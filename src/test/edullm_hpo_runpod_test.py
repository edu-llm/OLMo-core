"""Non-GPU checks for the local, uncommitted HPO RunPod adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNPOD = Path(__file__).resolve().parents[2] / ".edullm" / "runpod"


def load_script(name: str):
    path = RUNPOD / name
    spec = importlib.util.spec_from_file_location(f"hpo_runpod_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def credential_file(path: Path) -> Path:
    path.write_text(
        "\n".join(
            (
                "# generated fixture",
                "unset AWS_PROFILE",
                "export AWS_ACCESS_KEY_ID='test-access'",
                "export AWS_SECRET_ACCESS_KEY='test-secret'",
                "export AWS_SESSION_TOKEN='test-session'",
                "export AWS_REGION='us-east-1'",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def clear_aws(module, monkeypatch: pytest.MonkeyPatch) -> None:
    for key in module.AWS_FILE_KEYS | set(module.AWS_PROVIDER_KEYS):
        monkeypatch.delenv(key, raising=False)


def test_stager_loads_and_destroys_only_explicit_session(tmp_path, monkeypatch):
    module = load_script("stage_inputs.py")
    clear_aws(module, monkeypatch)
    path = credential_file(tmp_path / "aws-session.env")
    module.load_credentials(path)
    assert module.os.environ["AWS_SESSION_TOKEN"] == "test-session"
    module.destroy_credentials(path)
    assert not path.exists()
    assert not any(module.os.environ.get(key) for key in module.AWS_CREDENTIAL_KEYS)


def test_stager_rejects_objects_outside_sealed_bucket(tmp_path):
    module = load_script("stage_inputs.py")
    with pytest.raises(RuntimeError, match="escaped"):
        module.stage_one(object(), object(), tmp_path, "s3://some-other-bucket/data.bin")


def _manifest(tmp_path: Path) -> Path:
    train = tmp_path / "train.bin"
    val = tmp_path / "val.bin"
    train.write_bytes(b"train")
    val.write_bytes(b"val")
    payload = {
        "schema_version": 1,
        "family": "hpo-probe",
        "dataset": {
            "dataset_id": "pretrain/regmix-10b",
            "version": "v1",
            "tokenizer_id": "tokenizer/dolma2-bpe",
            "dtype": "uint32",
            "byte_order": sys.byteorder,
            "header_bytes": 0,
        },
        "train_objects": [{"path": str(train), "size": train.stat().st_size}],
        "val_objects": [{"path": str(val), "size": val.stat().st_size}],
    }
    path = tmp_path / "ready.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_entrypoint_validates_staged_manifest(tmp_path):
    module = load_script("entrypoint.py")
    payload = module.load_manifest(_manifest(tmp_path))
    assert payload["_local_train_paths"] == (str(tmp_path / "train.bin"),)
    assert payload["_local_val_paths"] == (str(tmp_path / "val.bin"),)


def test_runtime_spec_uses_persistent_paths_and_shared_evidence(tmp_path):
    module = load_script("entrypoint.py")
    source = Path(__file__).resolve().parents[2] / ".edullm" / "hpo-full-acronym-soup.json"
    job_root = tmp_path / "runs" / "full"
    shared_root = tmp_path / "runs" / "shared"
    destination = tmp_path / "runtime.json"
    module.materialize_runtime_spec(
        source,
        destination,
        job_root=job_root,
        shared_root=shared_root,
    )
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["controller"]["checkpoint_root"] == str(job_root / "checkpoints")
    assert payload["controller_state_path"] == str(job_root / "controller-state.jsonl")
    assert payload["study_result_path"] == str(job_root / "study-result.json")
    assert payload["proxy_evidence_path"] == str(shared_root / "proxy-evidence.json")
    assert "controller_snapshot_root" not in payload


def test_runtime_spec_args_rewrite_all_cohort_specs(tmp_path):
    module = load_script("entrypoint.py")
    root = Path(__file__).resolve().parents[2]
    result = module.rewrite_spec_args(
        [
            "run-id",
            "--proxy-spec",
            str(root / ".edullm" / "hpo-full-acronym-soup.json"),
            f"--reference-spec={root / '.edullm' / 'hpo-no-proxy.json'}",
        ],
        job_root=tmp_path / "job",
        shared_root=tmp_path / "shared",
    )
    assert result[2].endswith("runtime-specs/hpo-full-acronym-soup.json")
    assert result[3].endswith("runtime-specs/hpo-no-proxy.json")
    assert Path(result[2]).is_file()
    assert Path(result[3].split("=", 1)[1]).is_file()


def test_exact_no_centaur_spec_is_a_clean_centaur_ablation():
    root = Path(__file__).resolve().parents[2] / ".edullm"
    no_proxy = json.loads((root / "hpo-no-proxy.json").read_text(encoding="utf-8"))
    no_centaur = json.loads((root / "hpo-no-centaur-exact.json").read_text(encoding="utf-8"))

    assert no_centaur["arm"] == "no_centaur"
    assert no_centaur["posthoc_variant"] == "proxy_removed_after_failed_admission"
    assert no_centaur["centaur"] is None
    assert no_centaur["fidelity"] == {"kind": "exact"}
    assert no_centaur["experiment_factory"] == no_proxy["experiment_factory"]
    assert no_centaur["model_parameterization"] == no_proxy["model_parameterization"]
    assert no_centaur["search_space"] == no_proxy["search_space"]
    assert no_centaur["controller"] == no_proxy["controller"]
    assert "proxy_evidence_path" not in no_centaur
    assert "proxy_admission" not in no_centaur


def test_winner_checkpoint_comes_from_final_evaluation(tmp_path):
    module = load_script("publish_outputs.py")
    checkpoint = tmp_path / "checkpoints" / "winner"
    checkpoint.mkdir(parents=True)
    state = tmp_path / "controller-state.jsonl"
    state.write_text(
        json.dumps(
            {
                "seq": 0,
                "kind": "final_evaluation",
                "payload": {"checkpoint_ref": str(checkpoint)},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert module.final_checkpoint(state) == checkpoint


def test_launcher_enforces_secrets_proxy_gate_and_four_hour_total_limit():
    launch = (RUNPOD / "launch.sh").read_text(encoding="utf-8")
    assert 'readonly HARD_LIMIT_SECONDS=14400' in launch
    assert "WANDB_ENV_FILE" in launch
    assert "wandb-session.env" in launch
    assert "WANDB_API_KEY is required" in launch
    assert "OPENAI_API_KEY is required for the Centaur arms" in launch
    assert "run MODE=proxy-cohort successfully" in launch
    assert "CONTROLLER_SPEC" in launch
    assert "publish_outputs.py" in launch
    assert "torch.distributed.run" not in launch


def test_bootstrap_pins_commit_and_installs_hpo_extras():
    bootstrap = (RUNPOD / "bootstrap.sh").read_text(encoding="utf-8")
    assert "4f385fe54918b96756042a89d504ac19b928e1b4" in bootstrap
    assert '"${REPO_DIR}[wandb,hpo]"' in bootstrap
    assert "torch.cuda.device_count() == 8" in bootstrap
    assert 'FLASH_ATTN_CUDA_ARCHS="80"' in bootstrap
