"""The multi-cell entrypoint: cell parsing, GPU fit, and per-cell isolation."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("edullm_train", ROOT / ".edullm" / "train.py")
ep = importlib.util.module_from_spec(spec)
sys.modules["edullm_train"] = ep
spec.loader.exec_module(ep)


def test_parse_cells():
    assert ep.parse_cells("dense:0") == [("dense", 0)]
    assert ep.parse_cells("dense:0,split:1") == [("dense", 0), ("split", 1)]
    assert ep.parse_cells(" dense:0 , split:2 ") == [("dense", 0), ("split", 2)]
    assert ep.parse_cells("dense") == [("dense", 0)]
    assert ep.parse_cells("dense:0,,split:0") == [("dense", 0), ("split", 0)]


def test_full_matrix_fits_eight_gpus():
    """4 conditions x 3 seeds = 12 cells, so it must split across two 8-GPU jobs."""
    conds = ["dense", "split", "random_contig", "random_scatter"]
    cells = [f"{c}:{s}" for s in (0, 1, 2) for c in conds]
    assert len(cells) == 12
    first, second = cells[:8], cells[8:]
    assert len(ep.parse_cells(",".join(first))) == 8
    assert len(ep.parse_cells(",".join(second))) == 4
    # No cell appears twice across the split.
    assert len(set(first) | set(second)) == 12


def test_more_cells_than_gpus_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(ep, "visible_gpu_count", lambda: 2)
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "runid", "--save-folder", str(tmp_path),
         "--cells", "dense:0,split:0,random_contig:0"],
    )
    with pytest.raises(SystemExit) as exc:
        ep.main()
    assert "only 2 GPUs visible" in str(exc.value)


def test_single_cell_needs_no_gpu_and_no_subdir(monkeypatch, tmp_path):
    """One cell writes straight to save-folder; many cells get cell-<i>/."""
    seen = {}

    class FakeProc:
        def __init__(self, cmd, cwd=None, env=None):
            seen.setdefault("cmds", []).append(cmd)
            seen.setdefault("envs", []).append(env or {})

        def wait(self):
            return 0

    monkeypatch.setattr(ep.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ep, "visible_gpu_count", lambda: 0)
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "runid", "--save-folder", "s3://b/run", "--condition", "split",
         "--seed", "1"],
    )
    assert ep.main() == 0
    cmd = seen["cmds"][0]
    assert "--checkpoint-dir" in cmd
    assert cmd[cmd.index("--checkpoint-dir") + 1] == "s3://b/run"
    assert cmd[cmd.index("--condition") + 1] == "split"
    assert cmd[cmd.index("--seed") + 1] == "1"


def test_each_cell_gets_its_own_gpu_and_checkpoint_prefix(monkeypatch):
    """Without cell-<i>/ the cells overwrite each other's ckpt.pt."""
    seen = {"cmds": [], "envs": []}

    class FakeProc:
        def __init__(self, cmd, cwd=None, env=None):
            seen["cmds"].append(cmd)
            seen["envs"].append(env or {})

        def wait(self):
            return 0

    monkeypatch.setattr(ep.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ep, "visible_gpu_count", lambda: 4)
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "runid", "--save-folder", "s3://b/run",
         "--cells", "dense:0,split:0,random_contig:0,random_scatter:0"],
    )
    assert ep.main() == 0

    gpus = [e.get("CUDA_VISIBLE_DEVICES") for e in seen["envs"]]
    assert gpus == ["0", "1", "2", "3"], gpus

    ckpts = [c[c.index("--checkpoint-dir") + 1] for c in seen["cmds"]]
    assert ckpts == [f"s3://b/run/cell-{i}" for i in range(4)], ckpts
    assert len(set(ckpts)) == 4, "cells must not share a checkpoint prefix"

    conds = [c[c.index("--condition") + 1] for c in seen["cmds"]]
    assert conds == ["dense", "split", "random_contig", "random_scatter"]


def test_one_failed_cell_fails_the_job(monkeypatch):
    """A matrix with a missing cell is not a matrix; do not exit 0."""
    codes = iter([0, 0, 3, 0])

    class FakeProc:
        def __init__(self, cmd, cwd=None, env=None):
            self.rc = next(codes)

        def wait(self):
            return self.rc

    monkeypatch.setattr(ep.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ep, "visible_gpu_count", lambda: 4)
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "runid", "--save-folder", "s3://b/run",
         "--cells", "dense:0,split:0,random_contig:0,random_scatter:0"],
    )
    assert ep.main() == 1


def test_corpus_is_staged_once_for_the_union_of_sidecars(monkeypatch, tmp_path):
    """8 children each pulling 1.5 GB would waste bandwidth and race."""
    calls = []

    class FakeProc:
        def __init__(self, cmd, cwd=None, env=None):
            calls.append(("proc", cmd))

        def wait(self):
            return 0

    from memsplit import checkpoint_io as cio

    def fake_stage(prefix, dest, names):
        calls.append(("stage", prefix, tuple(names)))
        return tmp_path

    monkeypatch.setattr(cio, "stage_files", fake_stage)
    monkeypatch.setattr(ep.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(ep, "visible_gpu_count", lambda: 2)
    monkeypatch.setattr(
        sys, "argv",
        ["train.py", "runid", "--save-folder", "s3://b/run",
         "--data-root", "s3://b/corpus", "--cells", "dense:0,split:0"],
    )
    assert ep.main() == 0

    stages = [c for c in calls if c[0] == "stage"]
    assert len(stages) == 1, "must stage exactly once, in the parent"
    assert stages[0][2] == ("tokens.bin", "weights.dense.bin", "weights.split.bin")
    # Children receive the LOCAL path, not the s3 prefix.
    for kind, cmd in [c for c in calls if c[0] == "proc"]:
        assert cmd[cmd.index("--data-root") + 1] == str(tmp_path)
