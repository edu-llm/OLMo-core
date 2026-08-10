"""Non-GPU checks for the local, uncommitted HPO RunPod adapter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

RUNPOD = Path(__file__).resolve().parents[2] / ".edullm" / "runpod"
if str(RUNPOD) not in sys.path:
    sys.path.insert(0, str(RUNPOD))


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


def test_throughput_smoke_launches_one_world_size_one_worker_per_gpu(tmp_path):
    module = load_script("throughput_smoke.py")

    specs = module.worker_process_specs(
        script=RUNPOD / "throughput_smoke.py",
        profile="olmoe-hpo",
        gpu_count=8,
        warmup_steps=2,
        bench_steps=5,
        work_dir=tmp_path,
    )

    assert len(specs) == 8
    assert [spec.gpu_id for spec in specs] == list(range(8))
    assert [spec.env["CUDA_VISIBLE_DEVICES"] for spec in specs] == [str(i) for i in range(8)]
    assert all(spec.env["WORLD_SIZE"] == "1" for spec in specs)
    assert all(spec.env["LOCAL_WORLD_SIZE"] == "1" for spec in specs)
    assert all(spec.env["RANK"] == "0" for spec in specs)
    assert all(spec.env["LOCAL_RANK"] == "0" for spec in specs)
    assert len({spec.env["MASTER_PORT"] for spec in specs}) == 8
    assert all("torchrun" not in " ".join(spec.argv) for spec in specs)
    assert all("--worker" in spec.argv for spec in specs)
    assert all("--profile" in spec.argv and "olmoe-hpo" in spec.argv for spec in specs)


def test_throughput_smoke_assigns_one_variant_to_each_independent_worker(tmp_path):
    module = load_script("throughput_smoke.py")
    variants = [
        "adam-mb2048",
        "adam-mb4096",
        "adam8-mb2048",
        "adam8-mb4096",
        "adam8-mb8192",
        "adam8-mb16384",
        "adam8-ac-mb8192",
        "adam8-compile-mb4096",
    ]

    specs = module.worker_process_specs(
        script=RUNPOD / "throughput_smoke.py",
        profile="olmoe-hpo",
        gpu_count=8,
        warmup_steps=2,
        bench_steps=5,
        work_dir=tmp_path,
        variants=variants,
    )

    assert [
        spec.argv[spec.argv.index("--variant") + 1]
        for spec in specs
    ] == variants
    assert [spec.env["CUDA_VISIBLE_DEVICES"] for spec in specs] == [str(i) for i in range(8)]


def test_throughput_smoke_variants_are_adam_only():
    module = load_script("throughput_smoke.py")

    assert set(module.VARIANTS) >= {
        "adam-mb2048",
        "adam8-mb2048",
        "adam8-ac-mb8192",
        "adam8-compile-mb4096",
    }
    assert all(variant.optimizer.startswith("adam") for variant in module.VARIANTS.values())


def test_throughput_smoke_exposes_low_memory_fused_loss_adam_variants():
    module = load_script("throughput_smoke.py")

    assert module.VARIANTS["adam8-fused-mb2048"].fused_loss is True
    assert module.VARIANTS["adam8-fused-ac-mb2048"].activation_checkpointing is True
    assert module.VARIANTS["adam8-fused-compile-mb2048"].compile_model is True


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


def _multi_release_manifest(
    tmp_path: Path,
    *,
    include_parent_val: bool = True,
    include_legacy: bool = True,
) -> Path:
    def object_record(name: str) -> dict[str, object]:
        path = tmp_path / name
        path.write_bytes(name.encode("utf-8"))
        return {"path": str(path), "size": path.stat().st_size}

    parent_val = [object_record("parent-val.bin")] if include_parent_val else []
    releases = [
        *(
            [
                {
                    "dataset_id": "pretrain/regmix-10b",
                    "version": "v1",
                    "group": None,
                    "tokenizer_id": "tokenizer/dolma2-bpe",
                    "dtype": "uint32",
                    "byte_order": sys.byteorder,
                    "header_bytes": 0,
                    "train_objects": [object_record("regmix-train.bin")],
                    "val_objects": [object_record("regmix-val.bin")],
                }
            ]
            if include_legacy
            else []
        ),
        {
            "dataset_id": "pretrain/opt-with-synthetic-10b",
            "version": "v1",
            "group": "tokens",
            "tokenizer_id": "tokenizer/dolma2-bpe",
            "profile": "pretrain-tokens/v1",
            "manifest_sha256": "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797",
            "dtype": "uint32",
            "byte_order": "little",
            "header_bytes": 0,
            "train_objects": [object_record("parent-train.bin")],
            "val_objects": parent_val,
        },
        {
            "dataset_id": "curriculum/opt-with-synthetic-10b",
            "version": "v1",
            "group": "mtld",
            "profile": "token-order/v1",
            "manifest_sha256": "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d",
            "dtype": "uint64",
            "byte_order": "little",
            "header_bytes": 0,
            "train_objects": [object_record("mtld-order.bin")],
            "val_objects": [],
        },
    ]
    payload = {
        "schema_version": 2,
        "family": "hpo-probe",
        "releases": releases,
    }
    path = tmp_path / "ready-v2.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_entrypoint_validates_staged_manifest(tmp_path):
    module = load_script("entrypoint.py")
    payload = module.load_manifest(_manifest(tmp_path))
    assert payload["_local_train_paths"] == (str(tmp_path / "train.bin"),)
    assert payload["_local_val_paths"] == (str(tmp_path / "val.bin"),)


def test_entrypoint_serves_legacy_parent_and_mtld_from_one_manifest(tmp_path):
    module = load_script("entrypoint.py")
    payload = module.load_manifest(_multi_release_manifest(tmp_path))

    legacy = module.local_dataset_paths(payload, "pretrain/regmix-10b", "v1")
    parent = module.local_dataset_paths(
        payload, "pretrain/opt-with-synthetic-10b", "v1", group="tokens"
    )
    order = module.local_dataset_paths(
        payload,
        "curriculum/opt-with-synthetic-10b",
        "v1",
        group="mtld",
        split="train",
    )

    assert legacy.paths == [str(tmp_path / "regmix-train.bin")]
    assert legacy.val == [str(tmp_path / "regmix-val.bin")]
    assert parent.train == [str(tmp_path / "parent-train.bin")]
    assert parent.val == [str(tmp_path / "parent-val.bin")]
    assert parent.manifest_sha256 == (
        "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797"
    )
    assert parent.numpy_dtype == "<u4"
    assert order.paths == [str(tmp_path / "mtld-order.bin")]
    assert order.group == "mtld"
    assert order.profile == "token-order/v1"
    assert order.numpy_dtype == "<u8"


def test_entrypoint_accepts_curriculum_only_manifest_without_regmix(tmp_path):
    module = load_script("entrypoint.py")
    payload = module.load_manifest(_multi_release_manifest(tmp_path, include_legacy=False))

    parent = module.local_dataset_paths(
        payload, "pretrain/opt-with-synthetic-10b", "v1", group="tokens"
    )
    order = module.local_dataset_paths(
        payload,
        "curriculum/opt-with-synthetic-10b",
        "v1",
        group="mtld",
        split="train",
    )
    assert parent.val == [str(tmp_path / "parent-val.bin")]
    assert order.paths == [str(tmp_path / "mtld-order.bin")]


def test_entrypoint_rejects_manifest_that_cannot_serve_selected_mode(tmp_path):
    module = load_script("entrypoint.py")
    curriculum = module.load_manifest(_multi_release_manifest(tmp_path, include_legacy=False))
    module.validate_manifest_for_mode(curriculum, "curriculum_quadratic_mtld")
    module.validate_manifest_for_mode(curriculum, "curriculum_quadratic_mtld_no_centaur")
    with pytest.raises(RuntimeError, match="RegMix"):
        module.validate_manifest_for_mode(curriculum, "no_proxy")


@pytest.mark.skipif(sys.platform == "win32", reason="OLMo data loading requires fork support")
def test_multi_release_reader_satisfies_curriculum_core_contract(tmp_path):
    module = load_script("entrypoint.py")
    payload = module.load_manifest(_multi_release_manifest(tmp_path))
    parent = module.local_dataset_paths(
        payload, "pretrain/opt-with-synthetic-10b", "v1", group="tokens"
    )
    order = module.local_dataset_paths(
        payload,
        "curriculum/opt-with-synthetic-10b",
        "v1",
        group="mtld",
        split="train",
    )

    from olmo_core.hpo.curriculum import curriculum_corpus_from_reads

    corpus = curriculum_corpus_from_reads(parent, order)
    assert corpus.train_paths == (str(tmp_path / "parent-train.bin"),)
    assert corpus.val_paths == (str(tmp_path / "parent-val.bin"),)
    assert corpus.order_paths == (str(tmp_path / "mtld-order.bin"),)
    assert corpus.order_dtype.value == "uint64"


def test_entrypoint_rejects_curriculum_parent_without_held_out_split(tmp_path):
    module = load_script("entrypoint.py")
    with pytest.raises(RuntimeError, match="parent.*held-out"):
        module.load_manifest(_multi_release_manifest(tmp_path, include_parent_val=False))


def test_stager_resolves_legacy_parent_and_exact_mtld_release():
    module = load_script("stage_inputs.py")
    calls = []

    class Read:
        paths = ["s3://edullm-data/example/train.bin"]
        train = paths
        val = ["s3://edullm-data/example/val.bin"]
        dtype = "uint32"
        byte_order = sys.byteorder
        header_bytes = 0
        profile = None
        manifest_sha256 = None
        tokenizer_id = "tokenizer/dolma2-bpe"

    def dataset_paths(dataset_id, version, *, s3, group=None):
        calls.append((dataset_id, version, group, s3))
        read = Read()
        if dataset_id == "pretrain/opt-with-synthetic-10b":
            read.profile = "pretrain-tokens/v1"
            read.manifest_sha256 = (
                "e4eb0ce47b27c5d923b97e593a0fdc51edf4a78710caedc4557ae3488777f797"
            )
        if dataset_id.startswith("curriculum/"):
            read.val = None
            read.profile = "token-order/v1"
            read.manifest_sha256 = (
                "8ea6573b84f656c58366dab91d17f2140d6d6f817632d1b9e8ce47633140671d"
            )
            read.tokenizer_id = None
        return read

    releases = module.resolve_release_inputs(dataset_paths, object())

    assert [
        (release["dataset_id"], release["version"], release["group"]) for release in releases
    ] == [
        ("pretrain/regmix-10b", "v1", None),
        ("pretrain/opt-with-synthetic-10b", "v1", "tokens"),
        ("curriculum/opt-with-synthetic-10b", "v1", "mtld"),
    ]
    assert [(dataset_id, version, group) for dataset_id, version, group, _ in calls] == [
        ("pretrain/regmix-10b", "v1", None),
        ("pretrain/opt-with-synthetic-10b", "v1", "tokens"),
        ("curriculum/opt-with-synthetic-10b", "v1", "mtld"),
    ]


def test_stager_curriculum_release_set_omits_regmix():
    module = load_script("stage_inputs.py")
    calls = []

    class Read:
        paths = ["s3://edullm-data/example/train.bin"]
        val = ["s3://edullm-data/example/val.bin"]
        dtype = "uint32"
        byte_order = sys.byteorder
        header_bytes = 0
        profile = "pretrain-tokens/v1"
        manifest_sha256 = module.PARENT_MANIFEST_SHA256
        tokenizer_id = module.TOKENIZER_ID

    def dataset_paths(dataset_id, version, *, s3, group=None):
        del s3
        calls.append((dataset_id, version, group))
        read = Read()
        if dataset_id == module.ORDER_DATASET_ID:
            read.val = None
            read.dtype = "uint64"
            read.profile = "token-order/v1"
            read.manifest_sha256 = module.ORDER_MANIFEST_SHA256
            read.tokenizer_id = None
        return read

    releases = module.resolve_release_inputs(
        dataset_paths,
        object(),
        release_set="curriculum",
    )
    assert [release["dataset_id"] for release in releases] == [
        module.PARENT_DATASET_ID,
        module.ORDER_DATASET_ID,
    ]
    assert all(dataset_id != module.LEGACY_DATASET_ID for dataset_id, _, _ in calls)


def test_stager_rejects_registry_release_with_wrong_immutable_manifest():
    module = load_script("stage_inputs.py")

    class Read:
        paths = ["s3://edullm-data/example/train.bin"]
        val = ["s3://edullm-data/example/val.bin"]
        dtype = "uint32"
        byte_order = sys.byteorder
        header_bytes = 0
        profile = "pretrain-tokens/v1"
        manifest_sha256 = "0" * 64
        tokenizer_id = "tokenizer/dolma2-bpe"

    def dataset_paths(dataset_id, version, *, s3, group=None):
        del dataset_id, version, s3, group
        return Read()

    with pytest.raises(RuntimeError, match="immutable release"):
        module.resolve_release_inputs(dataset_paths, object())


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
    assert Path(result[2]).parts[-2:] == ("runtime-specs", "hpo-full-acronym-soup.json")
    assert Path(result[3].split("=", 1)[1]).parts[-2:] == (
        "runtime-specs",
        "hpo-no-proxy.json",
    )
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
    assert "readonly HARD_LIMIT_SECONDS=14400" in launch
    assert "WANDB_ENV_FILE" in launch
    assert "wandb-session.env" in launch
    assert "WANDB_API_KEY is required" in launch
    assert "OPENAI_API_KEY is required for the Centaur arms" in launch
    assert "run MODE=proxy-cohort successfully" in launch
    assert "CONTROLLER_SPEC" in launch
    assert "publish_outputs.py" in launch
    assert "torch.distributed.run" not in launch
    assert 'MIN_FREE_WORKSPACE_GIB="${MIN_FREE_WORKSPACE_GIB:-300}"' in launch
    assert "insufficient free workspace" in launch


def test_launcher_supports_curriculum_quadratic_mtld_mode_and_dry_run():
    launch = (RUNPOD / "launch.sh").read_text(encoding="utf-8")
    assert "curriculum_quadratic_mtld" in launch
    assert ".edullm/hpo-curriculum-quadratic-mtld.json" in launch
    assert "DRY_RUN" in launch
    assert 'EDULLM_DATASET_ID="pretrain/opt-with-synthetic-10b"' in launch
    assert 'EDULLM_DATASET_VERSION="v1"' in launch


def test_launcher_supports_curriculum_no_centaur_without_openai():
    launch = (RUNPOD / "launch.sh").read_text(encoding="utf-8")
    mode = "curriculum_quadratic_mtld_no_centaur"

    assert mode in launch
    assert ".edullm/hpo-curriculum-quadratic-mtld-no-centaur.json" in launch
    assert f'if [[ "${{MODE}}" == "{mode}"' in launch
    assert "full_acronym_soup|no_proxy|curriculum_quadratic_mtld)" in launch


def test_final_validation_launcher_allows_a_new_wandb_project():
    launch = (RUNPOD / "launch_final_validation.sh").read_text(encoding="utf-8")
    assert 'EDULLM_WANDB_PROJECT="${EDULLM_WANDB_PROJECT:-hpo-final-validation}"' in launch
    assert 'WANDB_PROJECT="${EDULLM_WANDB_PROJECT}"' in launch
    assert 'GLOBAL_BATCH_TOKENS="${GLOBAL_BATCH_TOKENS:-}"' in launch
    assert '"--global-batch-tokens" "${GLOBAL_BATCH_TOKENS}"' in launch


def test_final_validation_routes_torchrun_workers_through_staged_reader():
    module = load_script("final_validation_entrypoint.py")
    validation_script = RUNPOD.parent / "final_validation.py"

    class FinalValidation:
        __file__ = str(validation_script)

        @staticmethod
        def torchrun_command(vector_name, length_tokens, global_batch_tokens):
            return [
                sys.executable,
                "-m",
                "torch.distributed.run",
                str(validation_script),
                "--vector",
                vector_name,
                "--length-tokens",
                str(length_tokens),
                "--global-batch-tokens",
                str(global_batch_tokens),
            ]

    module._patch_validation_worker_entrypoint(FinalValidation)
    command = FinalValidation.torchrun_command("no-proxy-winner", 16_384, 262_144)

    assert command[3] == str((RUNPOD / "final_validation_entrypoint.py").resolve())
    assert str(validation_script) not in command
    assert command[-4:] == ["--length-tokens", "16384", "--global-batch-tokens", "262144"]


def test_bootstrap_pins_commit_and_installs_runtime_dependencies():
    bootstrap = (RUNPOD / "bootstrap.sh").read_text(encoding="utf-8")
    assert "4f385fe54918b96756042a89d504ac19b928e1b4" in bootstrap
    assert '"${REPO_DIR}[wandb,hpo]"' in bootstrap
    assert '-r "${REPO_DIR}/.edullm/requirements-task-loss-eval.txt"' in bootstrap
    assert bootstrap.index("requirements-task-loss-eval.txt") < bootstrap.index(
        'PYTHONPATH="${REPO_DIR}/src:${REPO_DIR}/.edullm"'
    )
    assert "torch.cuda.device_count() == 8" in bootstrap
    assert 'FLASH_ATTN_CUDA_ARCHS="80"' in bootstrap
