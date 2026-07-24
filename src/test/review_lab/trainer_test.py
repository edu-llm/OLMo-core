import torch.nn as nn

from olmo_core.review_lab.trainer import _enable_gradient_checkpointing


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True


def test_lora_checkpointing_enables_input_gradients() -> None:
    model = FakeModel()

    _enable_gradient_checkpointing(model, "lora")

    assert model.checkpointing_enabled
    assert model.input_grads_enabled


def test_full_checkpointing_does_not_change_input_gradients() -> None:
    model = FakeModel()

    _enable_gradient_checkpointing(model, "full")

    assert model.checkpointing_enabled
    assert not model.input_grads_enabled
