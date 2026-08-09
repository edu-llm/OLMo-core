import re
from pathlib import Path

import torch

from olmo_core.nn.flash_pd_native import mamba3_siso_surrogate_scan

PACKAGE = Path("src/olmo_core/nn/flash_pd_native")
CUDA_SOURCE = PACKAGE.joinpath("csrc/flash_pd_native_cuda.cu")
API_SOURCE = PACKAGE.joinpath("api.py")


def _between(source: str, start: str, end: str) -> str:
    return source.split(start, 1)[1].split(end, 1)[0]


def test_siso_forward_fuses_trapezoid_preprocessing_into_three_launch_scan():
    source = CUDA_SOURCE.read_text()
    forward = _between(
        source,
        "flash_pd_native_mamba3_forward_cuda(",
        "flash_pd_native_backward_cuda(",
    )

    assert "mamba3_preprocess_kernel" not in forward
    assert "MAMBA3_PHASE_A" in forward
    assert "MAMBA3_PHASE_B" in forward
    assert "MAMBA3_PHASE_C" in forward
    assert not re.search(r"auto bias_(?:real|imag)\s*=", forward)


def test_siso_backward_replays_over_the_row_by_chunk_grid_the_forward_uses():
    source = CUDA_SOURCE.read_text()
    backward = source.split("flash_pd_native_paper_backward_cuda(", 1)[1]
    api_source = API_SOURCE.read_text()

    # The reverse scan carries the same parallelism as the forward: one block per
    # (row, chunk) for the local aggregate and the corrected replay, and a single
    # row-wide block only for the short scan over chunk boundaries.
    assert backward.count("dim3(rows, chunks)") == 2
    assert "paper_backward_phase_a_kernel" in backward
    assert "paper_backward_phase_b_kernel" in backward
    assert "paper_backward_phase_c_kernel" in backward
    assert source.count("mamba3_backward_fused_kernel") == 0
    assert source.count("MAMBA3_FUSED_BACKWARD_BEGIN") == 0
    assert "backward_launches=5 if use_cuda else 0" in api_source


def test_siso_receipt_reports_the_chunk_parallel_backward_working_set():
    batch, heads, time, state, dictionary_size, chunk_size = 1, 2, 70, 8, 3, 32
    torch.manual_seed(17)
    maps = torch.stack([torch.roll(torch.arange(state), shift) for shift in range(dictionary_size)])
    dictionary = torch.randn(heads, dictionary_size, state, state)
    dictionary.scatter_(-2, maps.view(1, dictionary_size, 1, state).expand(heads, -1, -1, -1), 5.0)
    selector = torch.randn(batch, time, heads, dictionary_size)
    values = [torch.randn(batch, heads, time, state) * 0.1 for _ in range(4)]
    values[0].add_(0.9)
    beta = torch.rand(batch, heads, time)
    gamma = torch.rand(batch, heads, time)

    _, _, metadata = mamba3_siso_surrogate_scan(
        dictionary,
        selector,
        *values,
        beta,
        gamma,
        dictionary_temperature=1.0,
        router_temperature=1.0,
        chunk_size=chunk_size,
        mode="general_scatter",
        backend="reference",
        return_metadata=True,
    )
    chunks = (time + chunk_size - 1) // chunk_size

    # The reverse scan stages one int16 map plus six FP32 planes per (row, chunk):
    # the composed local affine map and the exclusive chunk carry.
    assert metadata.scratch_elements == 7 * batch * heads * chunks * state
    # Per-token selector scores feed the router-gradient launch, and the activated
    # dictionary gradient is accumulated in FP32 over (H, K, N).
    assert metadata.training_sequence_elements == (
        batch * heads * time + heads * dictionary_size * state
    )
    assert metadata.dictionary_storage_elements == heads * dictionary_size * state * state
    assert metadata.forward_launches == 0
    assert metadata.backward_launches == 0
