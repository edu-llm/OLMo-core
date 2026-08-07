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

import csv
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import pytest
from factcrowd import cells as C

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


CONTROL_CELL = SMOKE_CELL.parent / "smoke_13m_ctrl.yaml"
MIXTURE_CELL = SMOKE_CELL.parent / "smoke_13m_reason.yaml"


@pytest.mark.slow
def test_the_reasoning_only_control_trains_with_no_fact_corpus_at_all(tmp_path):
    """
    The control has no entity table, no renderer and no fact stream, and that path runs nowhere else.

    Worth its own minute because the control is what every crowding claim is measured against: a cell
    whose code had only ever been dry-run would be the one cell in the grid never actually executed, and
    the bugs this module exists for were all invisible to a dry run.
    """
    pytest.importorskip("torch")
    result = run_entry_point(
        "smoke-ctrl",
        "--cell",
        str(CONTROL_CELL),
        "--save-folder",
        str(tmp_path / "ckpt"),
        "--work-dir",
        str(tmp_path / "work"),
        "--rank-microbatch-size",
        "2048",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    output = result.stdout + result.stderr

    assert "Training complete" in output
    # One source, and it is the unrelated slice: no facts to mix in, and the related slice needs facts
    # to ask about.
    mixture = output.split("mixture for")[1].split("\n")[0]
    assert "mano" in mixture and "100.0%" in mixture, mixture
    assert "facts" not in mixture, mixture
    losses = [float(line.split("=")[1]) for line in output.splitlines() if "train/CE loss=" in line]
    assert losses and all(loss == loss for loss in losses), losses  # no NaN from an empty slice


@pytest.mark.slow
def test_the_three_way_mixture_trains_at_the_absolute_volumes_the_cell_states(tmp_path):
    """
    Facts plus both reasoning slices, with the two reasoning slices at equal absolute volume.

    The mixer takes ratios, so the absolute counts are derived and then checked -- and a shortfall in
    any source makes it rescale *every* source to preserve the ratios, silently moving the volumes the
    design fixes. Equal instance counts for the two slices is what says the derivation held.
    """
    pytest.importorskip("torch")
    result = run_entry_point(
        "smoke-mix",
        "--cell",
        str(MIXTURE_CELL),
        "--save-folder",
        str(tmp_path / "ckpt"),
        "--work-dir",
        str(tmp_path / "work"),
        "--rank-microbatch-size",
        "2048",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    output = result.stdout + result.stderr

    assert "Training complete" in output

    # Parsed from MixingInstanceSource's own summary of what it actually sampled, NOT from
    # train_cell's log line. That line reports train_cell's own arithmetic, so asserting against it
    # would check the declaration against itself and pass however the mixer behaved.
    sampled = {}
    for line in output.splitlines():
        match = re.search(r"([\d.]+)% (\w+), ([\d,]+) sampled instances", line)
        if match:
            sampled[match.group(2)] = int(match.group(3).replace(",", ""))
    assert set(sampled) == {"facts", "mano", "compare"}, output[-3000:]
    assert sampled["facts"] > sampled["mano"], sampled

    # The two reasoning slices carry deliberately *different* budgets -- the related one is sized on
    # per-entity coverage, not on parity -- so what is checked is each against its own config, in
    # instances, allowing the one-instance truncation calculate_sample_sizes can introduce.
    spec = C.load_cell(MIXTURE_CELL)
    for name in ("mano", "compare"):
        want = spec.slice_budget(name) // spec.sequence_length
        assert abs(sampled[name] - want) <= 1, (name, sampled[name], want)


@pytest.mark.slow
def test_two_ranks_reproduce_the_single_rank_optimisation():
    """
    The check that would have caught the worst defect in this branch, and it needs two processes.

    ``parallelize_model`` gates every wrapper behind ``if dp_config is not None``. Without one, a run
    started with ``torchrun --nproc-per-node=8`` trains **eight independent models**: no wrapper, no
    gradient reduction, and the loader still shards by rank, so each model sees an eighth of the corpus
    at roughly 25 exposures instead of 200. Nothing raises, and every single-process smoke passes --
    world size one is the one size where the omission is invisible.

    Under FSDP with a fixed global batch the optimisation is world-size invariant, so two ranks must
    reproduce the one-rank loss curve step for step. Unsynchronised gradients cannot.
    """
    pytest.importorskip("torch")

    def losses_from(nproc: int, work: Path) -> list:
        command = [sys.executable]
        if nproc > 1:
            command += ["-m", "torch.distributed.run", f"--nproc-per-node={nproc}", "--standalone"]
        command += [
            str(ENTRY_POINT),
            f"ranks{nproc}",
            "--cell",
            str(SMOKE_CELL),
            "--save-folder",
            str(work / "ck"),
            "--work-dir",
            str(work / "wd"),
            "--rank-microbatch-size",
            "1024",
        ]
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src")),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-2500:]
        output = result.stdout + result.stderr
        assert f"(dp={nproc},)" in output, "FSDP was not applied at this world size"
        return [
            float(line.split("=")[1]) for line in output.splitlines() if "train/CE loss=" in line
        ]

    with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
        single = losses_from(1, Path(one))
        doubled = losses_from(2, Path(two))

    assert single and len(single) == len(doubled), (single, doubled)
    for a, b in zip(single, doubled):
        assert abs(a - b) < 0.02, (single, doubled)


@pytest.mark.slow
def test_the_built_trainer_satisfies_every_platform_requirement(tmp_path):
    """
    The settings the platform refuses a run for, or kills it over, read off a real built trainer.

    Run through ``--build-only`` as a subprocess rather than by calling ``build_trainer`` in-process,
    because the trainer now requires a distributed context: ``dp_config`` is set unconditionally, and
    OLMo-core refuses a parallelism config outside distributed training. That refusal is worth keeping
    strict -- quietly dropping FSDP when the process group is missing is precisely the silent failure
    that had eight ranks training eight independent models -- so the test adapts to the program rather
    than the program to the test.

    Each assertion here was a lost run for somebody: ``max_checkpoints`` at its default of three makes a
    prune delete a key the workload role may not delete, and throws away seven of the ten snapshots the
    bits curve needs; ``ephemeral_save_interval`` is refused in the first seconds; the two evaluators
    fail while the trainer is being built; ``max_duration`` defaults to one epoch.
    """
    pytest.importorskip("torch")
    result = run_entry_point(
        "platcheck",
        "--cell",
        str(SMOKE_CELL),
        "--save-folder",
        str(tmp_path / "ck"),
        "--work-dir",
        str(tmp_path / "work"),
        "--rank-microbatch-size",
        "2048",
        "--build-only",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-2500:]
    output = result.stdout + result.stderr

    line = next(text for text in output.splitlines() if text.startswith("BUILD_ONLY "))
    settings = json.loads(line[len("BUILD_ONLY ") :])

    assert settings["max_checkpoints"] is None
    assert settings["ephemeral_save_interval"] is None
    assert settings["max_duration_steps"] > 0
    assert settings["evaluator_callbacks"] == []
    assert settings["save_overwrite"] is False
    # FSDP applied, which is the fix a single-process loss curve cannot verify on its own.
    assert settings["data_parallel"] == "fsdp"
    assert "Applied FSDP" in output
    # W&B off unless the platform named a project, so this never needs an API key locally.
    assert settings["wandb_enabled"] is False
    # And the run records its own cell, without which a finished checkpoint is unscoreable.
    assert settings["records_cell"] is True


@pytest.mark.slow
def test_bookkeeping_is_synchronous_and_repeated_saves_survive_multiple_ranks():
    """
    The defect that killed the first platform run, pinned two ways.

    Async bookkeeping puts metric reductions and the cancel check on a second gloo process group off
    the training thread. It was enabled whenever CUDA was available -- a branch that had never run on a
    real multi-GPU node. On 4xA10G it deadlocked: nineteen steps trained, one complete checkpoint
    written to S3, then `Waiting for bookkeeping ops to finish: 'reduce_metrics'` until gloo's
    1,800-second recv timeout killed the job. Thirty of the run's thirty-one minutes were that timeout.

    So: the flag must read false on the built trainer, and a multi-rank run must get *past* its second
    checkpoint. One save proved nothing -- the first one succeeded in the run that died.
    """
    pytest.importorskip("torch")
    result = run_entry_point(
        "asyncchk",
        "--cell",
        str(CONTROL_CELL),
        "--save-folder",
        str(Path(tempfile.mkdtemp()) / "ck"),
        "--work-dir",
        str(Path(tempfile.mkdtemp()) / "wd"),
        "--rank-microbatch-size",
        "2048",
        "--build-only",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    settings = json.loads(
        next(
            text
            for text in (result.stdout + result.stderr).splitlines()
            if text.startswith("BUILD_ONLY ")
        )[len("BUILD_ONLY ") :]
    )
    assert settings["async_bookkeeping"] is False
    assert len(settings["checkpoint_steps"]) >= 3, settings["checkpoint_steps"]

    # And now actually run it on two ranks, through several saves.
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--standalone",
                str(ENTRY_POINT),
                "asyncrun",
                "--cell",
                str(CONTROL_CELL),
                "--save-folder",
                str(work / "ck"),
                "--work-dir",
                str(work / "wd"),
                "--rank-microbatch-size",
                "1024",
            ],
            cwd=str(REPO_ROOT),
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src")),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == 0, output[-3000:]
        assert "Training complete" in output
        assert "took longer than 30 seconds" not in output, "a bookkeeping op stalled"
        assert "Timed out waiting" not in output
        saved = sorted((work / "ck").glob("step*"))
        assert len(saved) >= 3, [p.name for p in saved]


@pytest.mark.slow
def test_a_trained_checkpoint_scores_end_to_end_into_a_table():
    """
    Train a cell, then score its checkpoints with the real entry point, and read the table.

    The one test that exercises the whole path a finished run takes: sharded checkpoint on disk ->
    rebuilt corpus -> verified fingerprints -> loaded weights -> reasoning endpoints, achieved bits and
    recall -> one CSV row per (checkpoint, endpoint).

    Every stage of that has its own unit tests. This exists because the stages were written against each
    other's docstrings, and the failure that matters is the one at a seam -- which is the same reason
    `train_cell.py` has an end-to-end smoke at all.
    """
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        train = run_entry_point(
            "score-fixture",
            "--cell",
            str(MIXTURE_CELL),
            "--save-folder",
            str(work / "ck"),
            "--work-dir",
            str(work / "train"),
            "--rank-microbatch-size",
            "2048",
            cwd=REPO_ROOT,
        )
        assert train.returncode == 0, train.stdout[-2500:] + train.stderr[-2500:]

        scorer = REPO_ROOT / "src" / "scripts" / "train" / "factcrowd" / "score_run.py"
        scored = subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--prefix",
                str(work / "ck"),
                "--out",
                str(work / "scores.csv"),
                "--work-dir",
                str(work / "score"),
                "--eval-items",
                "32",
                "--bit-entities",
                "32",
                "--batch-size",
                "16",
            ],
            cwd=str(REPO_ROOT),
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src")),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert scored.returncode == 0, scored.stdout[-2500:] + scored.stderr[-2500:]

        import csv

        with (work / "scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))

    # Two endpoints at every checkpoint, and the identity travels with the numbers.
    assert rows, scored.stdout[-2000:]
    assert {row["endpoint"] for row in rows} == {"mano", "compare"}
    assert all(row["cell_id"] == "smoke_13m_reason" for row in rows)
    steps = sorted({int(row["step"]) for row in rows})
    assert len(steps) >= 3, steps
    assert len(rows) == 2 * len(steps)

    for row in rows:
        # Every measurement present and in range, so a downstream read needs no defensive parsing.
        assert 0.0 <= float(row["accuracy"]) <= 1.0
        assert 0.0 <= float(row["floor"]) <= 1.0
        assert float(row["answer_ce_bits"]) > 0.0
        assert float(row["achieved_bits_per_param"]) >= 0.0
        assert row["bits_is_upper_bound"] == "True"
        # Template reconstruction carries its own chance level, so "above chance" is a subtraction.
        # Note the name: this is not closed-book recall, which remains unbuilt (PRD 8.2).
        assert 0.0 < float(row["template_all_chance"]) < 1.0
        assert 0.0 <= float(row["template_all_generation"]) <= 1.0

    # The answer-token CE moves even while accuracy is pinned at zero -- which is the reason a continuous
    # endpoint is reported alongside the count. Observed on a real 20-step run: 11.83 -> 5.67 bits.
    mano = sorted(
        (int(row["step"]), float(row["answer_ce_bits"]))
        for row in rows
        if row["endpoint"] == "mano"
    )
    assert mano[-1][1] < mano[0][1], mano


@pytest.mark.slow
def test_the_entropy_axis_scores_end_to_end_too():
    """
    The count axis is not the whole path, and scoring it first hid a crash.

    Entropy-axis values are four words drawn from a union pool, so this exercises multi-token value spans
    and a wider padded vocabulary. The first attempt at it raised `IndexError` inside the scorer, because
    the output layer is wider than the vocabulary and an untrained model argmaxes into the gap -- 65 ids
    with no word behind them here against 31 on the count axis. The count axis had simply been lucky.

    It also pins the axis-aware endpoint selection from the other side: this cell carries `<mano>` alone,
    because its attributes have no orderable field for `<compare>` to ask about.
    """
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        train = run_entry_point(
            "entropy-fixture",
            "--cell",
            str(SMOKE_CELL.parent / "smoke_13m_entropy.yaml"),
            "--save-folder",
            str(work / "ck"),
            "--work-dir",
            str(work / "train"),
            "--rank-microbatch-size",
            "2048",
            cwd=REPO_ROOT,
        )
        assert train.returncode == 0, train.stdout[-2000:] + train.stderr[-2000:]

        scorer = REPO_ROOT / "src" / "scripts" / "train" / "factcrowd" / "score_run.py"
        scored = subprocess.run(
            [
                sys.executable,
                str(scorer),
                "--prefix",
                str(work / "ck"),
                "--out",
                str(work / "scores.csv"),
                "--work-dir",
                str(work / "score"),
                "--eval-items",
                "16",
                "--bit-entities",
                "16",
                "--batch-size",
                "8",
            ],
            cwd=str(REPO_ROOT),
            env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src")),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        assert scored.returncode == 0, scored.stdout[-2500:] + scored.stderr[-2500:]

        import csv

        with (work / "scores.csv").open() as handle:
            rows = list(csv.DictReader(handle))

    assert rows
    # No gate report was supplied, so nothing may be read as confirmatory (PRD 8.6).
    assert all(row["confirmatory"] == "False" for row in rows)
    assert all("no gate report" in row["admission"] for row in rows)
    # Mano alone: this axis has no orderable attribute, so the related slice is absent by construction.
    assert {row["endpoint"] for row in rows} == {"mano"}
    for row in rows:
        assert 0.0 <= float(row["unparseable_rate"]) <= 1.0
        # Six attributes of eight bits each: the prior is the schema's own, not a guess.
        assert float(row["prior_bits_per_entity"]) == pytest.approx(48.0)
        # Four-word values mean four pools per attribute, reported separately.
        assert float(row["template_attr0_chance"]) > 0.0


@pytest.mark.slow
def test_the_gate_report_is_produced_from_real_runs_and_gates_real_admission():
    """
    The admission seam, in both directions, against a real scored run.

    PRD 8.6 makes admission code-enforced, and the unit tests cover the assembler on synthetic
    `ScoredCheckpoint`s. What they cannot cover is whether `score_run` hands the assembler the shape it
    expects -- the identity fields it reads come out of a *record written by the trainer* and back through
    a checkpoint, which is three seams away from the dataclass a unit test builds.

    Both directions matter. A report that cannot be produced makes the gate unreachable; a report that
    admits everything makes it decorative.
    """
    pytest.importorskip("torch")
    with tempfile.TemporaryDirectory() as raw:
        work = Path(raw)
        train = run_entry_point(
            "gate-fixture",
            "--cell",
            str(MIXTURE_CELL),
            "--save-folder",
            str(work / "ck"),
            "--work-dir",
            str(work / "train"),
            "--rank-microbatch-size",
            "2048",
            cwd=REPO_ROOT,
        )
        assert train.returncode == 0, train.stdout[-2500:] + train.stderr[-2500:]

        def score(
            *extra: str, out: str, extra_prefixes: Sequence[str] = ()
        ) -> subprocess.CompletedProcess:
            # Extra roots join the single --prefix list rather than adding a second --prefix flag:
            # under nargs="+" a second occurrence replaces the first, which would drop the real run.
            return subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "src" / "scripts" / "train" / "factcrowd" / "score_run.py"),
                    "--prefix",
                    str(work / "ck"),
                    *extra_prefixes,
                    "--out",
                    str(work / out),
                    "--work-dir",
                    str(work / "score"),
                    "--eval-items",
                    "32",
                    "--bit-entities",
                    "32",
                    "--batch-size",
                    "16",
                    *extra,
                ],
                cwd=str(REPO_ROOT),
                env=dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src")),
                capture_output=True,
                text=True,
                timeout=1800,
            )

        # --- producing a report ------------------------------------------------------------------
        # A second, nonexistent prefix rides along on purpose. `--prefix` takes several because gate
        # evidence does not share a parent -- the sigma block and the dilution ladder are separate
        # submissions -- and with several roots one bad root must not cost the others. It is warned
        # about and skipped, not fatal, and not silent either.
        report_path = work / "gates-mano.json"
        first = score(
            "--write-gate-report",
            str(report_path),
            "--gate-endpoint",
            "mano",
            out="pass1.csv",
            extra_prefixes=(str(work / "no-such-run"),),
        )
        assert "no checkpoints under" in (first.stdout + first.stderr)
        assert first.returncode == 0, first.stdout[-2500:] + first.stderr[-2500:]
        report = json.loads(report_path.read_text())

        # One verdict per gate, and no gate passes on evidence this run cannot contain: there is no
        # dilution ladder here, no untrained checkpoint, no depth sweep.
        from factcrowd.measure import gates as gates_module

        assert report["version"] == gates_module.GATE_REPORT_VERSION
        assert report["endpoint"] == "mano"
        assert len(report["results"]) == len(gates_module.GATES)
        failures = {r["gate"] for r in report["results"] if not r["passed"]}
        assert "G8" in failures, report["results"]

        # And the run that produced the report is not admitted by it.
        with (work / "pass1.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows and all(row["confirmatory"] == "False" for row in rows)

        # --- consuming a passing report --------------------------------------------------------
        # Hand-built rather than earned: earning it needs five ladder arms, which is a submission and not
        # a smoke test. What is under test here is that a passing report actually flips the column --
        # every other test in this file can only show it staying False.
        passing = [
            {
                "version": gates_module.GATE_REPORT_VERSION,
                "endpoint": name,
                "commit": "0ddba11",
                "results": [
                    {"gate": gate, "passed": True, "detail": "hand-built fixture"}
                    for gate in gates_module.GATES
                ],
            }
            for name in ("mano", "compare")
        ]
        admitted_path = work / "admitted.json"
        admitted_path.write_text(json.dumps(passing))
        second = score("--gate-report", str(admitted_path), out="pass2.csv")
        assert second.returncode == 0, second.stdout[-2500:] + second.stderr[-2500:]
        with (work / "pass2.csv").open() as handle:
            admitted_rows = list(csv.DictReader(handle))

    assert admitted_rows and len(admitted_rows) == len(rows)
    assert all(row["confirmatory"] == "True" for row in admitted_rows)
    assert all("0ddba11" in row["admission"] for row in admitted_rows)

    # A failing report is refused too, and says which gate: the middle state between "no report" and
    # "admitted" is the one a reader is most likely to misread as either.
    broken = dict(passing[0])
    broken["results"] = [{"gate": "G8", "passed": False, "detail": "ladder flat"}]
    assert not gates_module.GateReport.from_dict(broken).passed
    assert gates_module.GateReport.from_dict(broken).failures == ("G8",)


@pytest.mark.slow
def test_checkpoint_saving_is_synchronous_and_opens_no_second_process_group():
    """
    The defect that ate five runs, and the reason a passing smoke test did not catch it.

    `CheckpointerCallback.save_async` defaults to `None`, which `pre_train` resolves to
    `backend_supports_cpu()` -- true on this backend -- and it then calls `dist.new_group()` to obtain a
    second process group for the save. A second process group is precisely what `async_bookkeeping` used,
    and disabling that is already recorded in `train_cell.py` as having cost a run; the same construct
    came back through a different field and kept going.

    The evidence is positional rather than a traceback: across the first count grid, the sigma block and
    the dilution ladder, six runs failed and five of them died 0, 0, 0, 10 and 15 steps after a planned
    checkpoint. The sixth was an unrelated wall-clock kill.

    Asserted on a *built trainer* rather than on the config literal, because the whole failure was a
    default resolving at runtime to something the config never said.
    """
    pytest.importorskip("torch")
    result = run_entry_point(
        "async-off",
        "--cell",
        str(SMOKE_CELL),
        "--save-folder",
        "/tmp/factcrowd-async-off",
        "--work-dir",
        "/tmp/factcrowd-async-off-wd",
        "--build-only",
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout[-2500:] + result.stderr[-2500:]
    match = re.search(r"BUILD_ONLY (\{.*\})", result.stdout)
    assert match is not None, result.stdout[-2500:]
    settings = json.loads(match.group(1))
    assert settings["save_async"] is False, settings
    assert settings["async_bookkeeping"] is False, settings
