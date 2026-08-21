"""Tests for SpeedMonitorCallback."""

import pytest

from olmo_core.train.callbacks.speed_monitor import get_device_peak_flops_per_second


@pytest.mark.parametrize(
    "device_name, expected_peak_flops_per_second",
    [
        # Data-center dies. Their published figures need only the sparsity correction.
        ("NVIDIA H100 80GB HBM3", 989_500_000_000_000),
        ("NVIDIA H100 NVL", 835_500_000_000_000),
        ("NVIDIA H100 PCIe", 756_500_000_000_000),
        ("NVIDIA B200", 2_250_000_000_000_000),
        ("NVIDIA A100-SXM4-80GB", 312_000_000_000_000),
        ("NVIDIA A100 80GB PCIe", 312_000_000_000_000),
        # Consumer-class dies, which run FP16/BF16 at half rate with an FP32 accumulator.
        # Getting any of these wrong misreports MFU by a factor of two or more, so each one
        # is pinned here: see the derivations in `get_device_peak_flops_per_second`.
        ("NVIDIA L40S", 183_250_000_000_000),
        ("NVIDIA L40", 181_050_000_000_000),
        ("NVIDIA L4", 60_500_000_000_000),
        ("NVIDIA A10G", 70_000_000_000_000),
    ],
)
def test_get_device_peak_flops_per_second(
    device_name: str, expected_peak_flops_per_second: int
) -> None:
    assert get_device_peak_flops_per_second(device_name) == expected_peak_flops_per_second


def test_get_device_peak_flops_per_second_distinguishes_the_ada_cards() -> None:
    # "L4" is a substring of both other Ada names, so reordering those branches would
    # quietly hand an L40S the L4's peak.
    assert get_device_peak_flops_per_second("NVIDIA L4") != get_device_peak_flops_per_second(
        "NVIDIA L40S"
    )


@pytest.mark.parametrize("device_name", ["NVIDIA GeForce RTX 4090", "Tesla T4", ""])
def test_get_device_peak_flops_per_second_returns_none_for_unrecognized_devices(
    device_name: str,
) -> None:
    # Reporting no MFU is better than reporting one computed against another card's peak.
    assert get_device_peak_flops_per_second(device_name) is None
