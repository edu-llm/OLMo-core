"""
The end-to-end smoke run: a real model, real generated data, real steps, real checkpoints.

Everything else in this suite tests a component. This tests the path -- and the path is where the
bugs were. Four of them survived review, type-checking and a green component suite, and every one
died the first time the code actually ran: a factory that raised ``TypeError`` for every input, a
callback parameter named ``steps`` instead of ``save_steps``, a scheduler refusing both of two
mutually exclusive fields, and a GPU-memory callback calling CUDA APIs on a CPU build.

Run as a subprocess rather than in-process, because the entry point initialises and tears down a
distributed process group and that does not compose with the rest of the suite.

Marked ``slow``: about a minute on CPU. Deselect with ``-m 'not slow'``.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ENTRY_POINT = REPO_ROOT / "src" / "scripts" / "train" / "factcrowd" / "train_cell.py"
SMOKE_CELL = (
    REPO_ROOT
    / "src"
    / "scripts"
    / "train"
    / "factcrowd"
    / "configs"
    / "cells"
    / "smoke"
    / "smoke_13m.yaml"
)


def run_entry_point(*args: str, cwd: Path) -> subprocess.CompletedProcess:
    """Invoke ``train_cell.py`` the way the platform does, with the repo on the path."""
    environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
    return subprocess.run(
        [sys.executable, str(ENTRY_POINT), *args],
        cwd=str(cwd),
        env=environment,
        capture_output=True,
        text=True,
        timeout=1800,
    )


def test_the_dry_run_resolves_a_cell_without_a_gpu(tmp_path):
    """
    The check that precedes a submission, and it has to work on a laptop to be worth running.

    Resolves the cell, generates the entity table, builds the vocabulary, the renderer and the
    offset index, and reports the plan -- so a bad config costs seconds rather than a queue slot.
    """
    result = run_entry_point(
        "dry",
        "--cell",
        str(SMOKE_CELL),
        "--dry-run",
        "--json",
        "--work-dir",
        str(tmp_path / "work"),
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    plan = json.loads(result.stdout)

    assert plan["cell_id"] == "smoke_13m"
    assert plan["non_embedding_params"] == 12_595_456
    assert plan["fact_tokens_measured"] > 0
    assert plan["steps"] >= 3
    # The entropy axis renders one length whatever the entropy, which is its defining property.
    assert plan["tokens_per_bio_min"] == plan["tokens_per_bio_max"]


@pytest.mark.slow
def test_the_smoke_cell_trains_and_the_loss_falls(tmp_path):
    """
    The whole path: generated corpus to a saved checkpoint, with the loss actually moving.

    A run that completes without learning would pass a weaker check while proving nothing -- the
    gradient could be disconnected, the labels shifted, the data all padding. Loss falling from
    roughly 7.6 to under 6 over ten steps is what says the tokens reaching the model are the tokens
    the renderer produced.
    """
    pytest.importorskip("torch")
    save_folder = tmp_path / "ckpt"
    result = run_entry_point(
        "smoke",
        "--cell",
        str(SMOKE_CELL),
        "--save-folder",
        str(save_folder),
        "--work-dir",
        str(tmp_path / "work"),
        "--rank-microbatch-size",
        "2048",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    output = result.stdout + result.stderr

    assert "Training complete" in output
    losses = [float(line.split("=")[1]) for line in output.splitlines() if "train/CE loss=" in line]
    assert len(losses) >= 2, output[-2000:]
    assert losses[0] > 7.0, losses
    assert losses[-1] < 6.0, losses
    assert losses[-1] < losses[0] - 1.0, losses


@pytest.mark.slow
def test_the_smoke_run_writes_resumable_checkpoints(tmp_path):
    """
    Checkpoints have to land where the platform looks, with the state a retry needs.

    A run that trains and saves nothing reachable exits zero and is recorded as a success; one such
    run already exists in this account. The step directories carry model, optimizer and trainer
    state, which is what lets a lost machine resume rather than restart.
    """
    pytest.importorskip("torch")
    save_folder = tmp_path / "ckpt"
    result = run_entry_point(
        "smoke",
        "--cell",
        str(SMOKE_CELL),
        "--save-folder",
        str(save_folder),
        "--work-dir",
        str(tmp_path / "work"),
        "--rank-microbatch-size",
        "2048",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr[-3000:]

    steps = sorted(p.name for p in save_folder.iterdir() if p.name.startswith("step"))
    assert len(steps) >= 3, steps
    assert "step10" in steps, steps
    final = save_folder / "step10"
    assert (final / "model_and_optim").is_dir()
    assert (final / "train").is_dir()


@pytest.mark.slow
def test_the_run_is_reproducible_from_its_seed(tmp_path):
    """
    Two runs of one cell must produce the same loss curve.

    Reproducibility from a seed is what we publish instead of token shards, so it is not enough that
    the corpus be deterministic -- the whole path has to be, including weight initialisation, which
    is why the entry point calls ``seed_all``.
    """
    pytest.importorskip("torch")
    curves = []
    for attempt in ("a", "b"):
        result = run_entry_point(
            "smoke",
            "--cell",
            str(SMOKE_CELL),
            "--save-folder",
            str(tmp_path / f"ckpt-{attempt}"),
            "--work-dir",
            str(tmp_path / f"work-{attempt}"),
            "--rank-microbatch-size",
            "2048",
            cwd=REPO_ROOT,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        output = result.stdout + result.stderr
        curves.append(
            [line.split("=")[1] for line in output.splitlines() if "train/CE loss=" in line]
        )
    assert curves[0] == curves[1], curves
