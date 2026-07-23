import dataclasses
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from edullm.models import AttemptRecord, JobRequest, JobStatus, Operator
from edullm.policy import load_operators, load_policy


def test_job_request_generates_stable_name():
    request = JobRequest(
        issue_number=42,
        requester="student",
        purpose="Skill-DAG smoke",
        study="skill-dag-v1",
        condition="natural",
        comparison="fixed-uniform",
        commit_sha="a" * 40,
        entrypoint_profile="hypothesis-smoke",
        script_path="src/scripts/train/smoketests/OLMo2-190M-hypothesis-smoke.py",
        launcher="python",
        argv=("train_single", "skilldag-natural", "local", "--seed=0"),
        data_manifest="builtin://generic-smoke-v1",
        data_manifest_sha256="b" * 64,
        data_classification="public",
        seed=0,
        wandb_project="pretraining",
        success_signal="20 steps and finite loss",
        success_metrics=("train/loss",),
        gpu_count=1,
        gpu_preference="l40s",
        max_runtime_minutes=30,
    )
    assert request.request_name == "issue-42-skill-dag-v1-natural"
    assert request.status is JobStatus.REQUESTED


def test_job_request_has_canonical_digest_and_is_immutable(valid_request):
    canonical = valid_request.canonical_json()

    assert canonical == json.dumps(
        dataclasses.asdict(valid_request), sort_keys=True, separators=(",", ":")
    )
    assert valid_request.digest == hashlib.sha256(canonical.encode()).hexdigest()
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid_request.status = JobStatus.READY


def test_job_status_values_are_stable():
    assert [status.value for status in JobStatus] == [
        "requested",
        "validating",
        "ready",
        "assigned",
        "submitted",
        "running",
        "completed",
        "failed",
        "cancelled",
        "preempted",
    ]


def test_supporting_records_are_immutable(valid_resolved_request):
    operator = Operator(
        github="operator",
        slack_user_id="U11111111",
        rotation_order=0,
        apptainer_path="/orcd/pool/edullm.sif",
        apptainer_sha256="c" * 64,
    )
    attempt = AttemptRecord(
        attempt_id="issue-42-attempt-1",
        request_digest=valid_resolved_request.request.digest,
        operator=operator.github,
        slurm_job_id="12345",
        wandb_run_id="issue-42-attempt-1",
        log_path="/scratch/logs/issue-42-attempt-1-12345.log",
    )

    assert valid_resolved_request.slurm_job_id is None
    assert operator.enabled is True
    assert attempt.operator == "operator"
    with pytest.raises(dataclasses.FrozenInstanceError):
        operator.enabled = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        valid_resolved_request.slurm_job_id = "12345"
    with pytest.raises(dataclasses.FrozenInstanceError):
        attempt.slurm_job_id = "54321"


def test_policy_loads_allowed_projects(tmp_path):
    path = tmp_path / "policy.yaml"
    path.write_text(
        "wandb_entity: eduLLM\n" "allowed_wandb_projects: [test]\n" "required_checks: [Lint]\n"
    )
    (tmp_path / "entrypoints.yaml").write_text("entrypoints: {}\n")
    policy = load_policy(path)

    assert policy.wandb_entity == "eduLLM"
    assert policy.allowed_wandb_projects == ("test",)
    assert policy.required_checks == ("Lint",)
    assert policy.max_runtime_minutes == 360
    assert policy.max_gpu_count == 2


def test_production_policy_has_reviewed_limits_and_staged_required_checks():
    policy = load_policy(Path("config/edullm/policy.yaml"))

    assert policy.allowed_wandb_projects == (
        "test",
        "pretraining",
        "posttraining",
        "evaluation",
        "data-pipeline",
    )
    assert policy.max_runtime_minutes == 360
    assert policy.max_gpu_count == 2
    assert policy.allowed_gpu_preferences == ("any", "l40s", "h100", "h200")
    assert policy.reminder_after_minutes == 15
    assert policy.reassign_after_minutes == 30
    assert policy.required_checks == (
        "Lint",
        "Test",
        "Test checkpoint",
        "Test transformer",
        "Test attention",
        "Test examples",
        "Test scripts",
        "Integration tests",
        "Test olmo3 ladder",
        "Type check",
        "Build",
        "Style",
        "Docs",
    )
    assert "Test edullm queue" not in policy.required_checks


def test_generic_profile_owns_the_reviewed_plan_1_smoke_arguments():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["generic-smoke"]

    assert profile["script"] == "src/examples/llm/train.py"
    assert profile["launcher"] == "torchrun"
    assert profile["fixed_launcher_arguments"] == ["--standalone", "--nproc-per-node=1"]
    assert profile["fixed_options"] == {
        "model-factory": "olmo2_190M",
        "sequence-length": 512,
        "save-folder": {
            "type": "derived_path",
            "root_env": "EDULLM_SCRATCH",
            "relative": "runs/{run_name}",
        },
        "work-dir": {
            "type": "derived_path",
            "root_env": "EDULLM_SCRATCH",
            "relative": "runs/{run_name}",
        },
        "data_loader.global_batch_size": 8192,
        "train_module.rank_microbatch_size": 2048,
        "train_module.max_sequence_length": 512,
        "trainer.hard_stop": {"value": 20, "unit": "steps"},
        "trainer.callbacks.lm_evaluator.enabled": False,
        "trainer.callbacks.downstream_evaluator.enabled": False,
        "trainer.callbacks.checkpointer.save_interval": 10,
        "trainer.callbacks.checkpointer.ephemeral_save_interval": None,
        "trainer.callbacks.wandb.enabled": True,
        "trainer.callbacks.wandb.entity": "eduLLM",
        "trainer.callbacks.wandb.project": "test",
        "trainer.callbacks.wandb.group": {"type": "request_field", "field": "study"},
        "trainer.callbacks.wandb.tags": ["orcd", "generic-smoke", "olmo2-190m"],
    }


def test_generic_profile_only_accepts_typed_non_safety_critical_values():
    profile = load_policy(Path("config/edullm/policy.yaml")).entrypoints["generic-smoke"]

    assert profile["positionals"] == 1
    assert profile["allowed_positionals"] == {0: {"type": "slug"}}
    assert profile["allowed_options"] == {}
    serialized = yaml.safe_dump(profile, sort_keys=True)
    assert "$HOME" not in serialized
    assert "EDULLM_SCRATCH" in serialized


def test_operator_files_keep_production_closed_and_examples_inert():
    assert load_operators(Path("config/edullm/operators.yaml")) == ()
    examples = load_operators(Path("config/edullm/operators.example.yaml"))

    assert [operator.github for operator in examples] == ["alice", "bob", "carol"]
    assert [operator.enabled for operator in examples] == [True, False, False]
    assert [operator.rotation_order for operator in examples] == [0, 1, 2]


def test_package_metadata_includes_edullm_without_a_premature_console_script():
    metadata = Path("pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10"' in metadata
    assert 'include = ["olmo_core*", "edullm*"]' in metadata
    assert "[project.scripts]" not in metadata
    assert 'edullm = "edullm.cli:main"' not in metadata
