import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT_ROOT = Path("src/scripts/orcd")
README = SCRIPT_ROOT / "README.md"


def _readme_section(label: str) -> str:
    text = README.read_text(encoding="utf-8")
    marker = f"# {label}\n"
    assert marker in text
    section = text.split(marker, maxsplit=1)[1]
    boundaries = [
        position for boundary in ("\n# ", "\n```") if (position := section.find(boundary)) >= 0
    ]
    return section[: min(boundaries)] if boundaries else section


def _readme_submission(label: str) -> str:
    lines = _readme_section(label).splitlines()
    command_lines = []
    for line in lines:
        if not command_lines:
            if line.startswith("sbatch "):
                command_lines.append(line)
        else:
            command_lines.append(line)
        if command_lines and not line.endswith("\\"):
            break
    assert command_lines
    return " ".join(line.removesuffix("\\").strip() for line in command_lines)


def _export_assignments(command: str) -> dict[str, str]:
    match = re.search(r"--export=([^\s]+)", command)
    assert match is not None
    assignments = (item.split("=", maxsplit=1) for item in match.group(1).split(","))
    return {name: value for name, value in assignments}


def _shell_assignments(label: str) -> dict[str, str]:
    return dict(
        re.findall(r"^export ([A-Z][A-Z0-9_]*)=(.+)$", _readme_section(label), re.MULTILINE)
    )


def _offline_sync_commands() -> str:
    text = README.read_text(encoding="utf-8")
    marker = "mapfile -t OFFLINE_RUN_DIRS"
    assert marker in text
    return marker + text.split(marker, maxsplit=1)[1].split("```", maxsplit=1)[0]


def _run_offline_sync_commands(save_folder: Path) -> subprocess.CompletedProcess[str]:
    script = (
        """
mapfile() {
  local target index=0 line quoted
  for target in "$@"; do :; done
  eval "${target}=()"
  while IFS= read -r line; do
    printf -v quoted '%q' "$line"
    eval "${target}[${index}]=${quoted}"
    index=$((index + 1))
  done
}
wandb() {
  test "$1" = sync
  printf 'synced=%s\\n' "$2"
}
"""
        + _offline_sync_commands()
    )
    return subprocess.run(
        ["bash"],
        input=script,
        env={**os.environ, "SAVE_FOLDER": str(save_folder)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_generic_job_requests_one_l40s():
    text = (SCRIPT_ROOT / "generic_smoke.sbatch").read_text(encoding="utf-8")
    assert "#SBATCH -G l40s:1" in text
    assert "#SBATCH -t 00:45:00" in text
    assert "#SBATCH -c 8" in text
    assert "#SBATCH --mem=64G" in text


def test_generic_smoke_uses_canonical_training_command():
    text = (SCRIPT_ROOT / "run_generic_smoke.sh").read_text(encoding="utf-8")
    expected_arguments = [
        "--model-factory=olmo2_190M",
        "--sequence-length=512",
        "--data_loader.global_batch_size=8192",
        "--train_module.rank_microbatch_size=2048",
        "--train_module.max_sequence_length=512",
        '--trainer.hard_stop="{value: $HARD_STOP_STEPS, unit: steps}"',
        "--trainer.callbacks.lm_evaluator.enabled=false",
        "--trainer.callbacks.downstream_evaluator.enabled=false",
        "--trainer.callbacks.checkpointer.save_interval=10",
        "--trainer.callbacks.checkpointer.ephemeral_save_interval=null",
        "--trainer.callbacks.wandb.enabled=true",
        '--trainer.callbacks.wandb.entity="$WANDB_ENTITY"',
        '--trainer.callbacks.wandb.project="$WANDB_PROJECT"',
        '--trainer.callbacks.wandb.group="$WANDB_GROUP"',
        '--trainer.callbacks.wandb.tags="[orcd,generic-smoke,olmo2-190m]"',
    ]

    assert "torchrun --standalone --nproc-per-node=1" in text
    assert 'HARD_STOP_STEPS="${HARD_STOP_STEPS:-20}"' in text
    assert 'SAVE_FOLDER="${SAVE_FOLDER:-$EDULLM_SCRATCH/runs/$RUN_NAME}"' in text
    assert '--save-folder="$SAVE_FOLDER"' in text
    assert '--work-dir="$SAVE_FOLDER"' in text
    for argument in expected_arguments:
        assert argument in text


def test_resume_uses_stable_save_folder_and_wandb_id():
    text = Path("src/scripts/orcd/run_generic_smoke.sh").read_text()
    assert 'SAVE_FOLDER="${SAVE_FOLDER:-' in text
    assert 'WANDB_RUN_ID="${WANDB_RUN_ID:-$RUN_NAME}"' in text
    assert "WANDB_RESUME=allow" in text


def test_generic_smoke_checkpoint_overrides_build_valid_config(tmp_path):
    text = (SCRIPT_ROOT / "run_generic_smoke.sh").read_text(encoding="utf-8")
    checkpoint_overrides = re.findall(
        r"^\s+--(trainer\.callbacks\.checkpointer\.[^=\s]+=[^\s\\]+)",
        text,
        flags=re.MULTILINE,
    )
    run_dir = tmp_path / "runs"
    command = [
        sys.executable,
        "src/examples/llm/train.py",
        "orcd-config-test",
        "--model-factory=olmo2_190M",
        "--sequence-length=512",
        f"--save-folder={run_dir}",
        f"--work-dir={run_dir}",
        "--dry-run",
        *(f"--{override}" for override in checkpoint_overrides),
    ]
    env = {**os.environ, "OLMO_DATA_ROOT": str(tmp_path / "data")}

    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stderr


def test_generic_smoke_checks_source_and_preserves_offline_wandb_data():
    text = (SCRIPT_ROOT / "run_generic_smoke.sh").read_text(encoding="utf-8")

    assert ': "${EDULLM_REPO_ROOT:?missing EDULLM_REPO_ROOT}"' in text
    assert ': "${EDULLM_SCRATCH:?missing EDULLM_SCRATCH}"' in text
    assert ': "${EDULLM_COMMIT_SHA:?missing EDULLM_COMMIT_SHA}"' in text
    assert ': "${WANDB_API_KEY:?missing WANDB_API_KEY}"' in text
    assert 'test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"' in text
    assert 'test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"' in text
    assert "WANDB_SYNC_DIR" not in text
    assert '--work-dir="$SAVE_FOLDER"' in text
    assert "WANDB_MODE=offline" in text
    assert 'PYTHONPATH="$EDULLM_REPO_ROOT/src"' in text
    assert "WANDB_API_KEY=" not in text
    assert "set -x" not in text


def test_generic_job_routes_logs_to_required_scratch_before_source_checks():
    text = (SCRIPT_ROOT / "generic_smoke.sbatch").read_text(encoding="utf-8")

    scratch_requirement = ': "${EDULLM_SCRATCH:?export EDULLM_SCRATCH before sbatch}"'
    create_log_directory = 'mkdir -p "$EDULLM_SCRATCH/logs"'
    redirect_logs = 'exec >"$EDULLM_SCRATCH/logs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log" 2>&1'
    exact_sha_check = "rev-parse HEAD"
    clean_check = "status --porcelain"

    assert "#SBATCH -o /dev/null" in text
    assert "#SBATCH -e /dev/null" in text
    assert "#SBATCH -o %x-%j.log" not in text
    assert 'EDULLM_SCRATCH="${EDULLM_SCRATCH:-' not in text
    assert (
        text.index(scratch_requirement)
        < text.index(create_log_directory)
        < text.index(redirect_logs)
        < text.index(exact_sha_check)
        < text.index(clean_check)
    )


def test_generic_job_supports_synthetic_and_staged_data():
    text = (SCRIPT_ROOT / "generic_smoke.sbatch").read_text(encoding="utf-8")

    assert 'EDULLM_DATA_MODE="${EDULLM_DATA_MODE:-synthetic}"' in text
    assert 'elif [[ "$EDULLM_DATA_MODE" == "staged" ]]' in text
    assert '--output "$EDULLM_SCRATCH/data/generic-smoke"' in text
    assert ': "${OLMO_DATA_ROOT:?staged mode requires OLMO_DATA_ROOT}"' in text
    assert 'source "$WANDB_ENV"' in text
    assert 'bash "$EDULLM_REPO_ROOT/src/scripts/orcd/run_generic_smoke.sh"' in text


def test_generic_job_forces_scratch_data_root_in_synthetic_mode():
    text = (SCRIPT_ROOT / "generic_smoke.sbatch").read_text(encoding="utf-8")
    synthetic_branch = text.split('if [[ "$EDULLM_DATA_MODE" == "synthetic" ]]; then', maxsplit=1)[
        1
    ].split('elif [[ "$EDULLM_DATA_MODE" == "staged" ]]', maxsplit=1)[0]

    assert 'export OLMO_DATA_ROOT="$EDULLM_SCRATCH/data/generic-smoke"' in synthetic_branch


def test_wandb_environment_example_reads_private_key_and_contains_no_secret():
    text = (SCRIPT_ROOT / "wandb.env.example").read_text(encoding="utf-8")

    assert "# Store the real key in ~/.config/edullm/wandb.key with mode 600." in text
    assert 'export WANDB_API_KEY="$(cat "$HOME/.config/edullm/wandb.key")"' in text
    assert 'export WANDB_ENTITY="eduLLM"' in text
    assert 'export WANDB_PROJECT="test"' in text
    assert 'export WANDB_GROUP="orcd-bootstrap"' in text
    assert "set -x" not in text


def test_setup_job_prints_public_dependency_versions():
    text = (SCRIPT_ROOT / "setup_env.sbatch").read_text(encoding="utf-8")

    for expression in (
        "platform.python_version()",
        "torch.__version__",
        "wandb.__version__",
        "VERSION",
    ):
        assert expression in text
    for label in ("Python:", "Torch:", "W&B:", "OLMo-core:"):
        assert label in text
    assert "WANDB_API_KEY" not in text
    assert "set -x" not in text


def test_readme_requires_real_scratch_and_known_reviewed_sha():
    text = README.read_text(encoding="utf-8")

    assert ': "${EDULLM_COMMIT_SHA:?' in text
    assert ': "${EDULLM_SCRATCH:?' in text
    assert 'export EDULLM_COMMIT_SHA="$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)"' not in text
    assert 'case "$EDULLM_SCRATCH" in' in text
    assert '"$HOME"|"$HOME"/*)' in text
    assert "/*) ;;" in text
    assert 'cd "$EDULLM_REPO_ROOT"' in text
    assert 'test "$(git -C "$EDULLM_REPO_ROOT" rev-parse HEAD)" = "$EDULLM_COMMIT_SHA"' in text
    assert 'test -z "$(git -C "$EDULLM_REPO_ROOT" status --porcelain)"' in text


@pytest.mark.parametrize(
    "label",
    ["Setup environment", "GPU probe", "Initial run", "Resume run", "Forced-offline smoke"],
)
def test_readme_routes_every_submission_to_scratch_logs(label):
    command = _readme_submission(label)

    assert '--output="$EDULLM_SCRATCH/logs/%x-%j.log"' in command
    assert '--error="$EDULLM_SCRATCH/logs/%x-%j.log"' in command


@pytest.mark.parametrize(
    ("label", "expected_assignments", "expected_script"),
    [
        (
            "Setup environment",
            {
                "EDULLM_REPO_ROOT": '"$EDULLM_REPO_ROOT"',
                "EDULLM_COMMIT_SHA": '"$EDULLM_COMMIT_SHA"',
                "EDULLM_SCRATCH": '"$EDULLM_SCRATCH"',
            },
            "src/scripts/orcd/setup_env.sbatch",
        ),
        (
            "GPU probe",
            {
                "EDULLM_REPO_ROOT": '"$EDULLM_REPO_ROOT"',
                "EDULLM_COMMIT_SHA": '"$EDULLM_COMMIT_SHA"',
                "EDULLM_SCRATCH": '"$EDULLM_SCRATCH"',
            },
            "src/scripts/orcd/probe.sbatch",
        ),
        (
            "Initial run",
            {
                "EDULLM_REPO_ROOT": '"$EDULLM_REPO_ROOT"',
                "EDULLM_COMMIT_SHA": '"$EDULLM_COMMIT_SHA"',
                "EDULLM_SCRATCH": '"$EDULLM_SCRATCH"',
                "RUN_NAME": '"$RUN_NAME"',
                "WANDB_RUN_ID": '"$WANDB_RUN_ID"',
                "SAVE_FOLDER": '"$SAVE_FOLDER"',
                "HARD_STOP_STEPS": '"$HARD_STOP_STEPS"',
            },
            "src/scripts/orcd/generic_smoke.sbatch",
        ),
        (
            "Resume run",
            {
                "EDULLM_REPO_ROOT": '"$EDULLM_REPO_ROOT"',
                "EDULLM_COMMIT_SHA": '"$EDULLM_COMMIT_SHA"',
                "EDULLM_SCRATCH": '"$EDULLM_SCRATCH"',
                "RUN_NAME": '"$RUN_NAME"',
                "WANDB_RUN_ID": '"$WANDB_RUN_ID"',
                "SAVE_FOLDER": '"$SAVE_FOLDER"',
                "HARD_STOP_STEPS": '"$HARD_STOP_STEPS"',
            },
            "src/scripts/orcd/generic_smoke.sbatch",
        ),
        (
            "Forced-offline smoke",
            {
                "EDULLM_REPO_ROOT": '"$EDULLM_REPO_ROOT"',
                "EDULLM_COMMIT_SHA": '"$EDULLM_COMMIT_SHA"',
                "EDULLM_SCRATCH": '"$EDULLM_SCRATCH"',
                "RUN_NAME": '"$RUN_NAME"',
                "WANDB_RUN_ID": '"$WANDB_RUN_ID"',
                "SAVE_FOLDER": '"$SAVE_FOLDER"',
                "HARD_STOP_STEPS": '"$HARD_STOP_STEPS"',
                "WANDB_MODE": '"$WANDB_MODE"',
            },
            "src/scripts/orcd/generic_smoke.sbatch",
        ),
    ],
)
def test_readme_submissions_preserve_export_dataflow_and_target(
    label, expected_assignments, expected_script
):
    command = _readme_submission(label)

    assert _export_assignments(command) == expected_assignments
    assert shlex.split(command)[-1] == expected_script
    assert "--export=ALL" not in command
    assert "WANDB_API_KEY" not in command


def test_readme_preserves_smoke_identity_hard_stops_and_offline_mode():
    initial = _shell_assignments("Initial run")
    resume = _shell_assignments("Resume run")
    offline = _shell_assignments("Forced-offline smoke")

    assert initial == {
        "RUN_NAME": "orcd-bootstrap",
        "WANDB_RUN_ID": "orcd-bootstrap",
        "SAVE_FOLDER": '"$EDULLM_SCRATCH/runs/$RUN_NAME"',
        "HARD_STOP_STEPS": "20",
    }
    assert resume == {"HARD_STOP_STEPS": "25"}
    assert offline == {
        "RUN_NAME": "orcd-bootstrap-offline",
        "WANDB_RUN_ID": "orcd-bootstrap-offline",
        "SAVE_FOLDER": '"$EDULLM_SCRATCH/runs/$RUN_NAME"',
        "HARD_STOP_STEPS": "20",
        "WANDB_MODE": "offline",
    }
    assert "WANDB_MODE" not in _export_assignments(_readme_submission("Initial run"))
    assert "WANDB_MODE" not in _export_assignments(_readme_submission("Resume run"))


def test_readme_preserves_private_wandb_key_and_team_visibility_flow():
    text = README.read_text(encoding="utf-8")

    assert 'cp src/scripts/orcd/wandb.env.example "$HOME/.config/edullm/wandb.env"' in text
    assert 'chmod 600 "$HOME/.config/edullm/wandb.env" "$HOME/.config/edullm/wandb.key"' in text
    assert "another `eduLLM` member" in text


def test_readme_syncs_one_exact_offline_run_directory(tmp_path):
    script = (SCRIPT_ROOT / "run_generic_smoke.sh").read_text(encoding="utf-8")
    text = README.read_text(encoding="utf-8")
    offline_run = tmp_path / "wandb" / "wandb" / "offline-run-only"
    offline_run.mkdir(parents=True)

    assert "WANDB_SYNC_DIR" not in script
    assert '--work-dir="$SAVE_FOLDER"' in script
    assert "wandb sync --sync-all" not in text
    result = _run_offline_sync_commands(tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"synced={offline_run}\n"


@pytest.mark.parametrize("candidate_count", [0, 2])
def test_readme_offline_sync_fails_closed_without_exactly_one_candidate(tmp_path, candidate_count):
    wandb_dir = tmp_path / "wandb" / "wandb"
    wandb_dir.mkdir(parents=True)
    for index in range(candidate_count):
        (wandb_dir / f"offline-run-{index}").mkdir()

    result = _run_offline_sync_commands(tmp_path)

    assert result.returncode == 2
    assert "expected exactly one offline W&B run directory" in result.stderr
    assert "synced=" not in result.stdout
