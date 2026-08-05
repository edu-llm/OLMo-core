"""Numerical contract for the P3 Qwen fused-linear loss.

The production shape cannot be exercised on a CPU test runner, but a small CUDA
shape is enough to pin the semantics that matter to the experiment: ignored fact
labels, summed cross-entropy, the arm-shared fixed divisor, and gradients with
respect to both hidden states and the tied-output weight.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("liger_kernel")

from olmo_core.nn.lm_head import LMHead, LMLossImplementation  # noqa: E402


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Liger fused-linear loss requires CUDA")
def test_fused_linear_matches_default_fixed_divisor_loss_and_gradients():
    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size, sequence_length, d_model, vocab_size = 2, 16, 32, 257

    default = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
        dtype=dtype,
        init_device="cuda",
        loss_implementation=LMLossImplementation.default,
    )
    fused = LMHead(
        d_model=d_model,
        vocab_size=vocab_size,
        bias=False,
        dtype=dtype,
        init_device="cuda",
        loss_implementation=LMLossImplementation.fused_linear,
    )
    fused.load_state_dict(default.state_dict(), strict=True)

    hidden = torch.randn(
        batch_size,
        sequence_length,
        d_model,
        device=device,
        dtype=dtype,
    )
    default_hidden = hidden.detach().clone().requires_grad_(True)
    fused_hidden = hidden.detach().clone().requires_grad_(True)
    labels = torch.randint(
        0,
        vocab_size,
        (batch_size, sequence_length),
        device=device,
    )
    labels[:, ::5] = -100
    fixed_divisor = float(labels.numel())

    default_output = default(
        default_hidden,
        labels=labels,
        ignore_index=-100,
        loss_reduction="sum",
        loss_div_factor=fixed_divisor,
        return_logits=False,
    )
    fused_output = fused(
        fused_hidden,
        labels=labels,
        ignore_index=-100,
        loss_reduction="sum",
        loss_div_factor=fixed_divisor,
        return_logits=False,
    )

    torch.testing.assert_close(
        fused_output.loss.float(),
        default_output.loss.float(),
        rtol=2e-2,
        atol=2e-2,
    )

    default_output.loss.backward()
    fused_output.loss.backward()
    assert default_hidden.grad is not None
    assert fused_hidden.grad is not None
    assert default.w_out.weight.grad is not None
    assert fused.w_out.weight.grad is not None
    assert F.cosine_similarity(
        default_hidden.grad.float().flatten(),
        fused_hidden.grad.float().flatten(),
        dim=0,
    ) > 0.99
    assert F.cosine_similarity(
        default.w_out.weight.grad.float().flatten(),
        fused.w_out.weight.grad.float().flatten(),
        dim=0,
    ) > 0.99
