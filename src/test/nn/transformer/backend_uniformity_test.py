import torch.nn as nn
import pytest

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.transformer import Transformer


class _BackendA(nn.Module):
    pass


class _BackendB(nn.Module):
    pass


class _Attention(nn.Module):
    def __init__(self, backend):
        super().__init__()
        self.backend = backend


class _Block(nn.Module):
    def __init__(self, backend=None):
        super().__init__()
        self.attention = nn.Identity() if backend is None else _Attention(backend)


def _model(*backends):
    model = object.__new__(Transformer)
    nn.Module.__init__(model)
    model.blocks = nn.ModuleDict(
        {str(index): _Block(backend) for index, backend in enumerate(backends)}
    )
    return model


def test_runtime_attention_backend_assertion_accepts_one_backend_across_layers():
    model = _model(_BackendA(), None, _BackendA())

    identity = model.assert_uniform_attention_backend()

    assert identity is not None and identity.endswith("._BackendA")


def test_runtime_attention_backend_assertion_rejects_mixed_realized_backends():
    model = _model(_BackendA(), _BackendB())

    with pytest.raises(OLMoConfigurationError, match="mixed attention backends"):
        model.assert_uniform_attention_backend()
