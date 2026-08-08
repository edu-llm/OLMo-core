import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from olmo_core.testing.utils import requires_gpu


def _load_convolution_module():
    flash_api = ModuleType("olmo_core.nn.attention.flash_linear_attn_api")
    flash_api.dispatch_causal_conv1d = None
    flash_api.has_fla = lambda: False
    module_path = Path(__file__).parents[2] / "olmo_core" / "nn" / "convolution.py"
    spec = importlib.util.spec_from_file_location("_convolution_under_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {flash_api.__name__: flash_api, spec.name: module}):
        spec.loader.exec_module(module)
    return module


convolution_module = _load_convolution_module()
CausalConv1d = convolution_module.CausalConv1d


def _depthwise_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    dilation: int = 1,
) -> torch.Tensor:
    output = F.conv1d(
        x.transpose(1, 2),
        weight,
        bias,
        padding=(weight.shape[-1] - 1) * dilation,
        dilation=dilation,
        groups=weight.shape[0],
    )
    output = output[..., : x.shape[1]].transpose(1, 2)
    if activation in ("silu", "swish"):
        output = F.silu(output)
    return output


@pytest.fixture(autouse=True)
def _cpu_tests_must_not_dispatch_to_fla(monkeypatch: pytest.MonkeyPatch):
    def unexpected_dispatch(**_kwargs):
        raise AssertionError("CPU execution unexpectedly dispatched to FLA")

    monkeypatch.setattr(convolution_module, "dispatch_causal_conv1d", unexpected_dispatch)


@pytest.mark.parametrize(
    ("bias", "activation"),
    [
        pytest.param(False, None, id="no-bias-no-activation"),
        pytest.param(True, None, id="bias-no-activation"),
        pytest.param(True, "silu", id="bias-silu"),
        pytest.param(True, "swish", id="bias-swish"),
    ],
)
def test_cpu_fallback_matches_depthwise_conv_and_backward(bias: bool, activation: str | None):
    torch.manual_seed(7)
    conv = CausalConv1d(
        hidden_size=4,
        kernel_size=3,
        bias=bias,
        activation=activation,
        dtype=torch.float64,
    )
    x = torch.randn(2, 7, 4, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = conv.weight.detach().clone().requires_grad_(True)
    bias_ref = conv.bias.detach().clone().requires_grad_(True) if conv.bias is not None else None

    actual = conv(x)
    expected = _depthwise_reference(x_ref, weight_ref, bias_ref, activation)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)

    output_grad = torch.randn_like(actual)
    actual.backward(output_grad)
    expected.backward(output_grad)

    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(conv.weight.grad, weight_ref.grad)
    if conv.bias is not None:
        assert bias_ref is not None
        torch.testing.assert_close(conv.bias.grad, bias_ref.grad)


def test_cpu_fallback_supports_dilation():
    torch.manual_seed(9)
    conv = CausalConv1d(
        hidden_size=4,
        kernel_size=4,
        dilation=3,
        bias=False,
        activation=None,
        dtype=torch.float64,
    )
    x = torch.randn(2, 12, 4, dtype=torch.float64, requires_grad=True)
    x_ref = x.detach().clone().requires_grad_(True)
    weight_ref = conv.weight.detach().clone().requires_grad_(True)

    actual = conv(x)
    expected = _depthwise_reference(
        x_ref,
        weight_ref,
        None,
        None,
        dilation=3,
    )
    torch.testing.assert_close(actual, expected)
    output_grad = torch.randn_like(actual)
    actual.backward(output_grad)
    expected.backward(output_grad)
    torch.testing.assert_close(x.grad, x_ref.grad)
    torch.testing.assert_close(conv.weight.grad, weight_ref.grad)


@requires_gpu
def test_cuda_dilated_fallback_compiles_in_bfloat16(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(convolution_module, "has_fla", lambda: True)
    conv = CausalConv1d(
        hidden_size=4,
        kernel_size=4,
        dilation=3,
        bias=False,
        activation=None,
        dtype=torch.bfloat16,
        init_device="cuda",
    )
    compiled = torch.compile(conv, fullgraph=True)
    x = torch.randn(
        2,
        12,
        4,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    output = compiled(x)
    output.float().square().mean().backward()

    assert output.shape == x.shape
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_cpu_fallback_is_causal():
    torch.manual_seed(11)
    conv = CausalConv1d(hidden_size=3, kernel_size=4, bias=True, activation=None)
    x = torch.randn(2, 8, 3)
    changed_future = x.clone()
    changed_future[:, 5:, :] = torch.randn_like(changed_future[:, 5:, :])

    original_output = conv(x)
    changed_output = conv(changed_future)

    torch.testing.assert_close(original_output[:, :5, :], changed_output[:, :5, :])


def test_cpu_fallback_preserves_context_parallel_channel_slice():
    torch.manual_seed(13)
    conv = CausalConv1d(hidden_size=6, kernel_size=3, bias=True, activation="silu")
    conv._cp_channel_slice = slice(2, 4)
    conv.cp_enabled = True
    x = torch.randn(2, 5, 2)

    actual = conv(x)
    expected = _depthwise_reference(
        x,
        conv.weight[conv._cp_channel_slice],
        conv.bias[conv._cp_channel_slice],
        conv.activation,
    )

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected)


def test_cpu_fallback_rejects_cu_seqlens():
    conv = CausalConv1d(hidden_size=2, kernel_size=3, activation=None)
    x = torch.randn(1, 5, 2)
    cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

    with pytest.raises(NotImplementedError, match="cu_seqlens.*not supported"):
        conv(x, cu_seqlens=cu_seqlens)
