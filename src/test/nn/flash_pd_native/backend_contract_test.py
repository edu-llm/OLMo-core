from pathlib import Path
import re

import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    NativePDMode,
    flash_pd_scan,
    get_backend_counters,
    native_cuda_capability,
    reset_backend_counters,
)


def _inputs(*, collision: bool = False, time: int = 257, state: int = 16):
    destination = torch.arange(state, dtype=torch.int16).view(1, 1, state)
    if collision:
        destination[..., 1] = 0
    routes = torch.zeros((2, 1, time), dtype=torch.int16)
    values = [torch.randn(2, 1, time, state) for _ in range(4)]
    return destination, routes, values


def test_auto_dispatch_reports_exact_mode_counter_and_appendix_e_working_set():
    destination, routes, values = _inputs(collision=True)
    reset_backend_counters()

    real, imag, metadata = flash_pd_scan(
        destination,
        routes,
        *values,
        chunk_size=128,
        backend="auto",
        return_metadata=True,
    )

    assert real.shape == imag.shape == (2, 1, 257, 16)
    assert metadata.backend == "reference"
    assert metadata.mode == NativePDMode.GENERAL_SCATTER
    assert metadata.state_shape == (2, 1, 257, 16)
    assert metadata.payload_axes == ()
    assert metadata.shared_memory_bytes == 28 * 16
    assert metadata.forward_launches == 0
    assert metadata.scratch_elements == 2 * 2 * 1 * 3 * 16 * 5
    assert get_backend_counters() == {"reference": 1}


def test_strict_cuda_never_silently_falls_back():
    destination, routes, values = _inputs(time=3)
    capability = native_cuda_capability(destination, routes, *values)

    if capability.available:
        pytest.skip("extension is available; strict execution is covered by CUDA parity tests")
    with pytest.raises(RuntimeError, match=capability.reason):
        flash_pd_scan(destination, routes, *values, backend="cuda")
    assert get_backend_counters()["cuda_rejected"] >= 1


def test_native_cuda_sources_encode_chunkwise_reverse_and_sm80_sm120_build():
    package = Path("src/olmo_core/nn/flash_pd_native")
    source = (package / "csrc/flash_pd_native_cuda.cu").read_text()
    setup = Path("flash_pd_native_setup.py").read_text()
    dockerfile = Path("src/Dockerfile").read_text()
    makefile = Path("Makefile").read_text()

    assert source.count("<<<") == 10
    for kernel in (
        "phase_a_kernel",
        "phase_b_kernel",
        "phase_c_kernel",
        "backward_kernel",
        "paper_backward_phase_a_kernel",
        "paper_backward_phase_b_kernel",
        "paper_backward_phase_c_kernel",
        "paper_dictionary_gradient_kernel",
        "paper_selector_gradient_kernel",
    ):
        assert re.search(rf"{kernel}(?:<scalar_t>)?\s*<<<", source)
    assert "paper_sequence_backward_kernel" not in source
    assert "__match_any_sync" in source
    assert "chunk_carry_real" in source
    assert "chunk_carry_imag" in source
    assert "shared_active_dictionary" in source
    assert "dictionary_size * state <= 12000 - 2 * state - 32" in source
    assert "paper backward chunk_size must be in [1, 128]" in source
    assert (
        "atomicAdd" not in source[source.index("permutation_step") : source.index("scatter_step")]
    )
    assert '"8.0"' in setup
    assert '"12.0"' in setup
    assert "load(" not in package.joinpath("cuda.py").read_text()
    assert "for token in" not in package.joinpath("cuda.py").read_text()
    assert "for dictionary in" not in package.joinpath("cuda.py").read_text()
    assert "active_dictionary_gradient" in source
    assert "selector_score" in source
    assert "torch::zeros({heads, dictionary_size, state}" not in source
    assert "flash_pd_native_setup.py bdist_wheel" in dockerfile
    assert "import _flash_pd_native_cuda" in dockerfile
    assert "import _flash_pd_native_cuda" in makefile
