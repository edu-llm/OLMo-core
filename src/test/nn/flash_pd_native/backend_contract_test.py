from pathlib import Path
import re
import runpy
import sys
import sysconfig
from types import ModuleType

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


def test_native_extension_build_includes_cuda_wheel_component_headers(tmp_path, monkeypatch):
    purelib = tmp_path / "site-packages"
    expected_include_dirs = []
    for component, header in (
        ("cublas", "cublas_v2.h"),
        ("cusparse", "cusparse.h"),
        ("cusolver", "cusolverDn.h"),
    ):
        include_dir = purelib / "nvidia" / component / "include"
        include_dir.mkdir(parents=True)
        include_dir.joinpath(header).touch()
        expected_include_dirs.append(str(include_dir))

    cuda_home = tmp_path / "cuda"
    cuda_home.joinpath("lib").mkdir(parents=True)
    cuda_home.joinpath("lib", "libcudart.so").touch()
    monkeypatch.setenv("CUDA_HOME", str(cuda_home))
    monkeypatch.setattr(sysconfig, "get_path", lambda name: str(purelib))

    captured = {}
    setuptools = ModuleType("setuptools")
    setuptools.setup = lambda **kwargs: captured.update(kwargs)
    cpp_extension = ModuleType("torch.utils.cpp_extension")
    cpp_extension.BuildExtension = type(
        "BuildExtension",
        (),
        {"with_options": staticmethod(lambda **kwargs: kwargs)},
    )
    cpp_extension.CUDAExtension = lambda **kwargs: kwargs
    monkeypatch.setitem(sys.modules, "setuptools", setuptools)
    monkeypatch.setitem(sys.modules, "torch.utils.cpp_extension", cpp_extension)

    runpy.run_path("flash_pd_native_setup.py", run_name="__main__")

    assert captured["ext_modules"][0]["include_dirs"] == expected_include_dirs


def test_scatter_step_aggregates_peers_under_the_peer_mask():
    """
    A warp routing a colliding dictionary holds peer groups of different sizes, so
    ``__match_any_sync`` hands its lanes loop trip counts that differ and the lanes
    in the small groups leave the aggregation loop first. Every warp intrinsic in
    that loop therefore has to name the peer group. Naming the whole warp instead
    makes the surviving lanes wait on lanes that have already left, which wedges
    the block on every architecture with independent thread scheduling.
    """
    source = Path("src/olmo_core/nn/flash_pd_native/csrc/flash_pd_native_cuda.cu").read_text()
    start = source.index("void scatter_step(")
    body = source[start : source.index("\n}\n", start)]

    assert "__match_any_sync" in body
    masks = re.findall(r"__shfl\w*_sync\(\s*([A-Za-z_]\w*)", body)
    assert masks, "scatter_step no longer aggregates its peers with warp shuffles"
    assert set(masks) == {"peers"}, (
        "scatter_step must shuffle under the peer mask so that every lane naming a "
        f"mask is still inside the loop; found {sorted(set(masks))}"
    )


def _kernel_body(source: str, kernel: str) -> str:
    start = source.index(f"__global__ void {kernel}(")
    return source[start : source.index("\n}\n", start)]


@pytest.mark.parametrize(
    "kernel",
    ["phase_a_kernel", "phase_b_kernel", "phase_c_kernel", "mamba3_phase_b_kernel"],
)
def test_permutation_inverse_is_read_into_a_register_before_any_warp_overwrites_it(kernel: str):
    """
    ``permutation_gather`` inverts the routed map in place: a lane reads the
    destination held in its own slot and then writes its lane index into the slot
    that destination names. A block runs one warp per 32 state channels, so above
    a state of 32 the warps are not in lockstep and the write of one warp lands in
    a slot another warp has not read yet. The read therefore has to be hoisted
    into a register and a barrier has to separate it from the write, which is the
    shape ``mamba3_phase_b_kernel`` already carries.
    """
    source = Path("src/olmo_core/nn/flash_pd_native/csrc/flash_pd_native_cuda.cu").read_text()
    body = _kernel_body(source, kernel)
    branch_at = body.index("kPermutationGather")
    branch = body[branch_at : body.index("} else {", branch_at)]

    write = re.search(r"shared_map\[(\w+)\]\s*=\s*lane;", branch)
    assert write, f"{kernel} no longer inverts the routed map in shared memory"
    assert "shared_map[" not in branch[: write.start()], (
        f"{kernel} still reads shared_map inside the permutation branch before writing "
        "the inverse, so a peer warp overwrites the slot before that read retires"
    )

    hoisted = re.search(rf"const int {write.group(1)} = shared_map\[lane\];", body)
    assert hoisted, (
        f"{kernel} must hoist the destination of its own slot into a register while the "
        "map still holds the forward direction"
    )
    assert (
        "__syncthreads();" in body[hoisted.end() : branch_at + write.start()]
    ), f"{kernel} must place a barrier between the hoisted read and the inverse write"
    tail = branch[write.end() :]
    assert tail.index("__syncthreads();") < tail.index("permutation_step("), (
        f"{kernel} must place a barrier between the inverse write and the gather that "
        "reads the inverted map"
    )


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
        assert re.search(rf"{kernel}(?:<[\w:, ]+>)?\s*<<<", source)
    assert "paper_sequence_backward_kernel" not in source
    assert "__match_any_sync" in source
    assert "chunk_carry_real" in source
    assert "chunk_carry_imag" in source
    assert "shared_active_dictionary" in source
    assert "constexpr int kPhaseCReductions = 3;" in source
    assert "constexpr int kMaximumWarps = 32;" in source
    assert "dictionary_size * state <= 12000 - 2 * state - reduction_elements" in source
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
