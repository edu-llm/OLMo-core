import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    NativePDMode,
    flash_pd_scan,
    native_cuda_capability,
)


def _maps(state: int, *, collision: bool, device: torch.device):
    first = torch.arange(state, dtype=torch.int16, device=device)
    second = torch.roll(first, 3)
    if collision:
        second[1] = second[0]
    destination = torch.stack((first, second)).unsqueeze(0)
    return destination


def _parity_case(*, state: int, time: int, dtype: torch.dtype, collision: bool):
    torch.manual_seed(31 + state + time)
    device = torch.device("cuda")
    destination = _maps(state, collision=collision, device=device)
    routes = torch.randint(0, 2, (1, 1, time), device=device, dtype=torch.int16)
    if collision:
        routes[..., 0] = 1
    leaves = [
        (torch.randn(1, 1, time, state, device=device) * 0.1).to(dtype).requires_grad_()
        for _ in range(4)
    ]
    leaves[0].data.add_(0.9)
    mode = NativePDMode.GENERAL_SCATTER if collision else NativePDMode.PERMUTATION_GATHER

    expected_leaves = [leaf.detach().float().requires_grad_() for leaf in leaves]
    expected = flash_pd_scan(
        destination,
        routes,
        *expected_leaves,
        mode=mode,
        backend="reference",
    )
    weight_real = torch.randn_like(expected[0])
    weight_imag = torch.randn_like(expected[1])
    expected_gradients = torch.autograd.grad(
        (expected[0] * weight_real + expected[1] * weight_imag).sum(),
        expected_leaves,
    )

    actual_real, actual_imag, metadata = flash_pd_scan(
        destination,
        routes,
        *leaves,
        mode=mode,
        backend="cuda",
        return_metadata=True,
    )
    actual_gradients = torch.autograd.grad(
        (actual_real.float() * weight_real + actual_imag.float() * weight_imag).sum(),
        leaves,
    )

    tolerance = 5e-2 if dtype == torch.bfloat16 else 4e-4
    torch.testing.assert_close(actual_real.float(), expected[0], atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual_imag.float(), expected[1], atol=tolerance, rtol=tolerance)
    for actual, reference in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual.float(), reference, atol=tolerance, rtol=tolerance)
    assert metadata.backend == "cuda"
    assert metadata.forward_launches == 3
    assert metadata.backward_launches == 1
    assert metadata.payload_axes == ()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(("state", "time"), [(16, 1), (32, 129), (128, 257)])
def test_native_cuda_permutation_tails_outputs_and_gradients(
    dtype: torch.dtype,
    state: int,
    time: int,
):
    capability = native_cuda_capability()
    assert capability.available, capability.reason
    _parity_case(state=state, time=time, dtype=dtype, collision=False)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_cuda_collision_outputs_and_gradients(dtype: torch.dtype):
    capability = native_cuda_capability()
    assert capability.available, capability.reason
    _parity_case(state=16, time=131, dtype=dtype, collision=True)


@pytest.mark.gpu
def test_native_cuda_production_length_probe():
    capability = native_cuda_capability()
    assert capability.available, capability.reason
    _parity_case(state=128, time=1025, dtype=torch.bfloat16, collision=False)


# Parity alone only catches an interleaving that actually lost. Run the test under
# this command on a CUDA host to have racecheck report the hazard itself, whether or
# not the run it watches happens to lose.
RACECHECK_COMMAND = (
    "compute-sanitizer --tool racecheck --racecheck-detect-level analysis "
    "python -m pytest -q -p no:cacheprovider "
    "src/test/nn/flash_pd_native/cuda_test.py -k multiwarp_permutation_inverse"
)


@pytest.mark.gpu
@pytest.mark.parametrize("state", [64, 128])
def test_native_cuda_multiwarp_permutation_inverse_matches_reference_on_every_repeat(state: int):
    """
    A state above 32 spreads the block over several warps, and the in-place
    inversion of the routed map then has one warp writing slots another warp has
    yet to read. The reversing map makes every one of those writes cross a warp
    boundary. A losing interleaving corrupts the gather source for the lanes it
    catches, so the run disagrees with the reference and successive runs disagree
    with each other.
    """
    capability = native_cuda_capability()
    assert capability.available, capability.reason

    torch.manual_seed(97 + state)
    device = torch.device("cuda")
    identity = torch.arange(state, dtype=torch.int16, device=device)
    destination = torch.stack((identity, identity.flip(0))).unsqueeze(0)
    time = 257
    routes = torch.ones((1, 1, time), device=device, dtype=torch.int16)
    routes[..., ::2] = 0
    values = [torch.randn(1, 1, time, state, device=device) * 0.1 for _ in range(4)]
    values[0].add_(0.9)

    expected = flash_pd_scan(
        destination,
        routes,
        *values,
        mode=NativePDMode.PERMUTATION_GATHER,
        backend="reference",
    )

    first = None
    for _ in range(8):
        real, imag = flash_pd_scan(
            destination,
            routes,
            *values,
            mode=NativePDMode.PERMUTATION_GATHER,
            backend="cuda",
        )
        torch.testing.assert_close(real, expected[0], atol=4e-4, rtol=4e-4)
        torch.testing.assert_close(imag, expected[1], atol=4e-4, rtol=4e-4)
        if first is None:
            first = (real.clone(), imag.clone())
        else:
            assert torch.equal(real, first[0]) and torch.equal(imag, first[1]), (
                "repeated permutation_gather runs over identical inputs disagree, which "
                f"is the inversion race; reproduce it with: {RACECHECK_COMMAND}"
            )
