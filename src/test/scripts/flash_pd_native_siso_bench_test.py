import importlib.util
from pathlib import Path
import re

BENCHMARK = Path("src/scripts/benchmarks/flash_pd_native_siso_bench.py")


def _load_benchmark():
    assert BENCHMARK.exists()
    spec = importlib.util.spec_from_file_location("flash_pd_native_siso_bench", BENCHMARK)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_uses_fresh_subprocesses_strict_backend_and_all_chunk_candidates():
    module = _load_benchmark()
    source = BENCHMARK.read_text()
    parser = module.build_parser()
    opts = parser.parse_args([])

    assert module.CHUNK_CANDIDATES == (32, 64, 128)
    assert opts.warmup == 20
    assert opts.iterations == 50
    assert "subprocess.run" in source
    assert "sys.executable" in source
    assert "cuda_mamba3_siso_general_scatter" in source
    assert "torch._dynamo.reset()" in source
    assert "dtype=DType.bfloat16" in source
    assert "mode=NativePDMode.GENERAL_SCATTER" in source


def test_benchmark_never_declares_an_a100_winner_from_other_hardware():
    module = _load_benchmark()
    measurements = [
        {"chunk_size": 32, "median_ms": 3.0},
        {"chunk_size": 64, "median_ms": 2.0},
        {"chunk_size": 128, "median_ms": 4.0},
    ]

    assert module.select_a100_winner("NVIDIA RTX 5050", measurements) is None
    assert module.select_a100_winner("NVIDIA A100-SXM4-80GB", measurements) == 64


def test_benchmark_reports_production_source_hashes_and_working_set():
    module = _load_benchmark()
    hashes = module.production_source_hashes()

    assert set(hashes) == {
        "api.py",
        "cuda.py",
        "mamba3_siso.py",
        "flash_pd_native.cpp",
        "flash_pd_native_cuda.cu",
    }
    assert all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values())
    source = BENCHMARK.read_text()
    assert "working_set_bytes" in source
    assert "model_flops_per_token" in source
    assert "nonlinear_evaluations_per_sequence" in source
    assert "route_comparisons_per_sequence" in source
