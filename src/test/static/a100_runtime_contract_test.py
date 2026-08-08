from pathlib import Path


def test_flash_attention_image_includes_a100_sm80():
    dockerfile = Path("src/Dockerfile").read_text()

    assert 'FLASH_ATTN_CUDA_ARCHS="80;90;100"' in dockerfile


def test_a10g_dense_bf16_peak_halves_the_published_sparse_peak():
    source = Path("src/olmo_core/train/callbacks/speed_monitor.py").read_text()

    assert 'if "A10G" in device_name:\n        return int(125e12 * dense_correction)' in source


def test_image_sets_triton_ieee_and_dynamo_diagnostics_before_python_starts():
    dockerfile = Path("src/Dockerfile").read_text()

    assert dockerfile.count("ENV TRITON_F32_DEFAULT=ieee") >= 2
    assert dockerfile.count("ENV TORCH_LOGS=recompiles,graph_breaks") >= 2
