"""Tests for frontload-cl Colab smoke helper (CPU-safe; GPU optional)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_DIR = Path(__file__).resolve().parents[3] / ".edullm"
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from frontload_cl import constants as C  # noqa: E402
from frontload_cl.colab_smoke import (  # noqa: E402
    SYNTH_SOURCES,
    build_parser,
    gpu_report,
    main,
    write_synthetic_shards,
)
from frontload_cl.corpus import group_paths_by_source, source_name_from_path  # noqa: E402


def test_synth_sources_cover_schedule_folders():
    expected = {
        C.SOURCE_FINEWEB_MAIN,
        C.SOURCE_FINEWEB_ANNEAL,
        C.SOURCE_FINEWIKI,
        *C.SFT_LIKE_SOURCES,
    }
    assert set(SYNTH_SOURCES) == expected


def test_write_synthetic_shards_layout(tmp_path: Path):
    written = write_synthetic_shards(tmp_path, tokens_per_source=8 * C.SEQ_LENGTH)
    assert set(written) == set(SYNTH_SOURCES)
    paths = list(written.values())
    grouped = group_paths_by_source(paths)
    assert set(grouped) == set(SYNTH_SOURCES)
    for source, path in written.items():
        p = Path(path)
        assert p.is_file()
        assert source_name_from_path(path) == source
        tokens = np.fromfile(p, dtype=np.uint32)
        assert len(tokens) == 8 * C.SEQ_LENGTH
        assert tokens.max() < 100_278


def test_write_synthetic_shards_refuses_too_short(tmp_path: Path):
    with pytest.raises(ValueError, match="shorter than seq"):
        write_synthetic_shards(tmp_path, tokens_per_source=100)


def test_cli_write_data(tmp_path: Path, capsys):
    out = tmp_path / "synth"
    assert main(["write-data", "--out", str(out), "--tokens-per-source", str(4 * C.SEQ_LENGTH)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["mode"] == "write-data"
    assert (out / "tokens" / C.SOURCE_FINEWEB_MAIN / "train-00000.u32le.bin").is_file()


def test_cli_gpu_info(capsys):
    torch = pytest.importorskip("torch")
    del torch
    assert main(["gpu-info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "cuda" in payload
    assert payload == gpu_report()


def test_source_name_from_windows_local_path():
    path = r"C:\data\tokens\fineweb-edu-main\train-00000.u32le.bin"
    assert source_name_from_path(path) == "fineweb-edu-main"


def test_parser_microbench_defaults():
    opts = build_parser().parse_args(["microbench"])
    assert opts.sequences == C.GLOBAL_BATCH_SEQUENCES // 8
    assert opts.seq_length == C.SEQ_LENGTH
    assert opts.attn_backend == C.DEFAULT_ATTN_BACKEND
    assert opts.compile is False
    assert opts.with_optim is False


@pytest.mark.gpu
def test_microbench_one_step_gpu():
    """Optional: real A100/L4 path. Skipped without GPU / liger-kernel."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("liger_kernel")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    # Prefer torch SDPA so the test does not require flash-attn in CI images.
    assert (
        main(
            [
                "microbench",
                "--steps",
                "1",
                "--sequences",
                "1",
                "--attn-backend",
                "torch",
                "--device",
                "cuda",
            ]
        )
        == 0
    )
