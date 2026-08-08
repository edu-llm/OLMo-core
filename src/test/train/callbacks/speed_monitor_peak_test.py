import pytest

from olmo_core.train.callbacks.speed_monitor import _dense_bf16_peak_flops


@pytest.mark.parametrize(
    "device_name,expected",
    [
        ("NVIDIA A100-SXM4-40GB", int(312e12)),
        ("NVIDIA L40S", int(362e12 * 0.5)),
        ("NVIDIA A10G", int(125e12 * 0.5)),
        ("NVIDIA L4", int(121e12 * 0.5)),
        ("NVIDIA H100 NVL", int(1671e12 * 0.5)),
        ("NVIDIA H100 PCIe", int(1513e12 * 0.5)),
        ("NVIDIA H100 80GB HBM3", int(1979e12 * 0.5)),
        ("NVIDIA B200", int(4.5e15 * 0.5)),
    ],
)
def test_dense_bf16_peak_table_uses_explicit_device_values(device_name, expected):
    assert _dense_bf16_peak_flops(device_name) == expected


@pytest.mark.parametrize(
    "device_name",
    [
        "NVIDIA GeForce RTX 5050 Laptop GPU",
        "NVIDIA T4",
        "Unknown Accelerator",
    ],
)
def test_dense_bf16_peak_table_never_falls_back_for_unknown_or_unsupported_gpu(
    device_name,
):
    assert _dense_bf16_peak_flops(device_name) is None
