import pytest
import torch

from olmo_core.kernels import fused_twn_available, fused_twn_quantize
from olmo_core.nn.quantization import twn_threshold_and_scale

pytestmark = pytest.mark.gpu


def _reference(w: torch.Tensor, in_dim: int) -> torch.Tensor:
    """The definition from `nn.quantization`, kept here so the test does not depend on
    whichever path `twn_quantize` dispatches to."""
    delta, alpha = twn_threshold_and_scale(w, in_dim=in_dim)
    w32 = w.detach().to(torch.float32)
    return (torch.sign(w32) * (w32.abs() > delta) * alpha).to(w.dtype)


def _requires_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("fused TWN needs CUDA")


# Every orientation from the in_dim table in `twn_quantize`, plus shapes that are not a
# multiple of any block size.
SHAPES = [
    ((64, 128), -1),
    ((64, 128), 1),
    ((512, 2048), -1),
    ((4, 256, 64), 1),
    ((4, 256, 64), 2),
    ((3, 130, 65), 1),
    ((3, 130, 65), 2),
    ((7, 33), -1),
    ((1, 4096), -1),
    ((64, 128), 0),
    ((129, 63), 0),
]


@pytest.mark.parametrize("shape,in_dim", SHAPES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_matches_reference(shape, in_dim, dtype):
    _requires_cuda()
    torch.manual_seed(0)
    w = torch.randn(*shape, device="cuda", dtype=dtype)

    fused = fused_twn_quantize(w, in_dim=in_dim)
    assert fused is not None
    expected = _reference(w, in_dim)

    # The ternary pattern must agree exactly: which weights survive is the quantizer's
    # identity, and a disagreement here is a different quantizer, not a rounding difference.
    torch.testing.assert_close(torch.sign(fused.float()), torch.sign(expected.float()))
    torch.testing.assert_close(fused.float(), expected.float(), rtol=1e-5, atol=1e-6)


def test_disagreement_with_the_reference_is_only_near_threshold_ties():
    """The kernel is not bitwise identical to the reference, and this pins what it does promise.

    `test_fused_matches_reference` asserts exact sign agreement, which holds on the shapes it
    covers but is a stronger claim than the kernel can make in general: it sums each row in a
    different order, so `delta` can differ in its last bits and a weight lying within about one
    bf16 ulp of its row threshold can be classified either way. Measured at up to ~1.7e-5 of
    elements on adversarial draws. What must never happen is a sign inversion or a disagreement
    away from the threshold -- either would mean a different quantizer rather than a tie.
    """
    _requires_cuda()
    for seed in range(4):
        torch.manual_seed(seed)
        w = torch.randn(2048, 1024, device="cuda", dtype=torch.bfloat16)
        fused = fused_twn_quantize(w, in_dim=-1)
        assert fused is not None
        expected = _reference(w, in_dim=-1)

        disagree = fused != expected
        count = int(disagree.sum())
        if count == 0:
            continue

        assert count / w.numel() < 1e-4, f"seed {seed}: {count} disagreements is too many"

        # A tie flips a weight between zero and +-alpha. One of the two sides is therefore zero,
        # so their product is zero; a sign inversion would give a strictly negative product.
        products = torch.sign(fused.float())[disagree] * torch.sign(expected.float())[disagree]
        assert bool((products == 0).all()), f"seed {seed}: sign inversion, not a tie"

        # And every one of them must actually sit next to the threshold it straddles.
        magnitudes = w.detach().float().abs()
        delta, _ = twn_threshold_and_scale(w, in_dim=-1)
        delta = delta.expand_as(magnitudes)
        distance = (magnitudes[disagree] - delta[disagree]).abs() / delta[disagree]
        assert float(distance.max()) < 2**-7, f"seed {seed}: disagreement away from the threshold"


@pytest.mark.parametrize("in_dim", [-1, 0])
def test_each_output_row_holds_at_most_three_values(in_dim):
    """The scale is per output row, so a row spans at most {-alpha, 0, +alpha}.

    ``in_dim=0`` reduces down columns, which makes the *columns* the output rows and also
    exercises the strided path where the reduced axis is not innermost.
    """
    _requires_cuda()
    torch.manual_seed(0)
    w = torch.randn(32, 512, device="cuda", dtype=torch.bfloat16)
    fused = fused_twn_quantize(w, in_dim=in_dim)
    assert fused is not None
    output_rows = fused if in_dim == -1 else fused.T
    for row in output_rows:
        assert len(torch.unique(row)) <= 3


def test_all_zero_row_yields_zero_not_nan():
    _requires_cuda()
    w = torch.zeros(4, 128, device="cuda", dtype=torch.float32)
    w[1] = torch.randn(128, device="cuda")
    fused = fused_twn_quantize(w, in_dim=-1)
    assert fused is not None
    assert torch.all(fused[0] == 0)
    assert not torch.isnan(fused).any()
    torch.testing.assert_close(fused.float(), _reference(w, -1).float(), rtol=1e-5, atol=1e-6)


def test_non_contiguous_input_is_handled():
    _requires_cuda()
    torch.manual_seed(0)
    w = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16).T
    assert not w.is_contiguous()
    fused = fused_twn_quantize(w, in_dim=-1)
    assert fused is not None
    torch.testing.assert_close(fused.float(), _reference(w, -1).float(), rtol=1e-5, atol=1e-6)


def test_declines_on_cpu_so_the_caller_falls_back():
    assert fused_twn_quantize(torch.randn(8, 16), in_dim=-1) is None
    assert not fused_twn_available(torch.randn(8, 16))
