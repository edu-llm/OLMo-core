import os
import re
import subprocess
import sys
from pathlib import Path


SCRIPT_ROOT = Path("src/scripts/orcd")


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
        "--trainer.callbacks.wandb.enabled=true",
        '--trainer.callbacks.wandb.entity="$WANDB_ENTITY"',
        '--trainer.callbacks.wandb.project="$WANDB_PROJECT"',
        '--trainer.callbacks.wandb.group="$WANDB_GROUP"',
        '--trainer.callbacks.wandb.tags="[orcd,generic-smoke,olmo2-190m]"',
    ]

    assert "torchrun --standalone --nproc-per-node=1" in text
    assert 'HARD_STOP_STEPS="${HARD_STOP_STEPS:-20}"' in text
    assert 'SAVE_FOLDER="$EDULLM_SCRATCH/runs/$RUN_NAME"' in text
    assert '--save-folder="$SAVE_FOLDER"' in text
    assert '--work-dir="$EDULLM_SCRATCH/cache/$RUN_NAME"' in text
    for argument in expected_arguments:
        assert argument in text


def test_generic_smoke_checkpoint_overrides_build_valid_config(tmp_path):
    text = (SCRIPT_ROOT / "run_generic_smoke.sh").read_text(encoding="utf-8")
    checkpoint_overrides = re.findall(
        r"^\s+--(trainer\.callbacks\.checkpointer\.[^=\s]+=[^\s\\]+)",
        text,
        flags=re.MULTILINE,
    )
    command = [
        sys.executable,
        "src/examples/llm/train.py",
        "orcd-config-test",
        "--model-factory=olmo2_190M",
        "--sequence-length=512",
        f"--save-folder={tmp_path / 'runs'}",
        f"--work-dir={tmp_path / 'cache'}",
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
    assert 'WANDB_SYNC_DIR="$SAVE_FOLDER/wandb"' in text
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
