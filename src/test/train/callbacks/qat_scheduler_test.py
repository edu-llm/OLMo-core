import pytest
import torch
import torch.nn as nn

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.quantization import QuantLinear
from olmo_core.train.callbacks import count_quant_modules, set_quant_enabled


def _model(enabled: bool) -> nn.Module:
    return nn.Sequential(
        QuantLinear(8, 8, enabled=enabled),
        nn.ReLU(),
        QuantLinear(8, 8, enabled=enabled),
    )


def test_count_reports_enabled_and_total():
    assert count_quant_modules(_model(enabled=False)) == (0, 2)
    assert count_quant_modules(_model(enabled=True)) == (2, 2)


def test_set_quant_enabled_switches_every_module():
    model = _model(enabled=False)
    assert set_quant_enabled(model, True) == 2
    assert count_quant_modules(model) == (2, 2)


def test_set_quant_enabled_is_idempotent():
    model = _model(enabled=True)
    assert set_quant_enabled(model, True) == 0


def test_set_quant_enabled_refuses_a_model_with_no_quantizer():
    with pytest.raises(OLMoConfigurationError, match="no quantizable modules"):
        set_quant_enabled(nn.Sequential(nn.Linear(8, 8)), True)


def test_switching_on_changes_the_arithmetic():
    torch.manual_seed(0)
    model = _model(enabled=False)
    x = torch.randn(4, 8)

    full_precision = model(x)
    set_quant_enabled(model, True)
    quantized = model(x)

    assert not torch.allclose(full_precision, quantized)


def test_switching_off_restores_full_precision_exactly():
    torch.manual_seed(0)
    model = _model(enabled=True)
    x = torch.randn(4, 8)

    set_quant_enabled(model, False)
    layer = model[0]
    assert isinstance(layer, QuantLinear)
    torch.testing.assert_close(
        model(x)[:, :],
        model[2](torch.relu(nn.functional.linear(x, layer.weight, layer.bias))),
    )
