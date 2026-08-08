from pathlib import Path
import re


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


def test_siso_backward_fuses_reverse_replay_and_router_gradient():
    source = CUDA_SOURCE.read_text()
    backward = source.split("flash_pd_native_paper_backward_cuda(", 1)[1]
    fused_path = _between(
        backward,
        "// MAMBA3_FUSED_BACKWARD_BEGIN",
        "// MAMBA3_FUSED_BACKWARD_END",
    )

    assert "mamba3_backward_fused_kernel" in fused_path
    assert "paper_dictionary_gradient_kernel" in fused_path
    assert fused_path.count("FLASH_PD_LAUNCH(") == 2
    assert "selector_score" not in fused_path
    assert "chunk_carry" not in fused_path
    assert "aggregate_destination" not in fused_path


def test_siso_metadata_reports_exact_fused_launches_and_workspace():
    source = API_SOURCE.read_text()
    mamba3_api = source.split("def mamba3_siso_surrogate_scan(", 1)[1]

    assert "forward_launches=3 if use_cuda else 0" in mamba3_api
    assert "backward_launches=2 if use_cuda else 0" in mamba3_api
    assert re.search(
        r"scratch_elements=5\s*\*\s*batch\s*\*\s*heads\s*\*\s*chunks\s*\*\s*state",
        mamba3_api,
    )
    assert re.search(
        r"training_sequence_elements=\(?heads\s*\*\s*dictionary_size\s*\*\s*state\)?",
        mamba3_api,
    )
