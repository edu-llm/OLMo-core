import pytest
import torch
from torch import nn

from olmo_core.optim import SkipStepAdamWConfig
from olmo_core.optim.skip_step_optimizer import SkipStepOptimizer
from olmo_core.testing import DEVICES


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding(1024, 16)
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.wte(x)
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)


@pytest.mark.parametrize("device", DEVICES)
def test_skip_step_optimizer(device: torch.device):
    """Test that skip step optimizer skips steps with outlier losses."""
    model = MyModel().to(device)
    optim = SkipStepAdamWConfig(rolling_interval_length=2, sigma_factor=1).build(model)

    # Normal step - should not skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))

    # Outlier step - should skip
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = torch.tensor(1e9, device=device)  # Outlier loss
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(True))

    # Another normal step
    optim.zero_grad(set_to_none=True)
    loss = model(torch.randint(0, 128, (4, 8), device=device)).sum()
    optim.latest_loss = loss.detach()
    loss.backward()
    optim.step()
    assert torch.equal(optim.step_skipped.cpu().detach(), torch.tensor(False))


def _skipped_steps(losses, grad_norms=None, *, rolling_interval_length=128, sigma_factor=6):
    """Replay a loss (and optional grad-norm) series, returning the indices that were skipped."""
    params = [torch.nn.Parameter(torch.zeros(2))]
    optim = SkipStepOptimizer(
        params,
        {},
        rolling_interval_length=rolling_interval_length,
        sigma_factor=sigma_factor,
    )
    skipped = []
    for i, loss in enumerate(losses):
        optim.latest_loss = torch.tensor(float(loss))
        if grad_norms is not None:
            optim.latest_grad_norm = torch.tensor(float(grad_norms[i]))
        if optim.get_step_factor().item() == 0.0:
            skipped.append(i)
    return skipped


@pytest.mark.parametrize("field", ["loss", "grad_norm"])
def test_skip_step_optimizer_recovers_from_a_single_nonfinite_value(field: str):
    """
    One bad step must cost exactly one step, not the whole rolling window.

    ``torch.std_mean`` over a window containing a NaN returns NaN for both statistics, and every
    subsequent ``<=`` comparison against NaN is ``False``. That made a single NaN skip the next
    ``rolling_interval_length + 1`` steps -- 129 at the default -- while the loss looked
    perfectly healthy the whole time, because the weights had simply stopped moving. The
    gradient-norm path matters just as much as the loss path here: the trainer's own finiteness
    check only inspects the CE loss, so a NaN grad norm raises nothing anywhere.
    """
    n, bad = 400, 200
    losses = [float("nan") if (i == bad and field == "loss") else 3.0 for i in range(n)]
    grads = [float("nan") if (i == bad and field == "grad_norm") else 0.5 for i in range(n)]
    assert _skipped_steps(losses, grads) == [bad]


def test_skip_step_optimizer_skips_every_nonfinite_step():
    """
    Sustained non-finite values must be skipped throughout.

    Excluding non-finite entries from the statistics must not become a way for them to slip
    past: once the window is entirely NaN there is no usable threshold, and the fallback has to
    reject rather than admit.
    """
    losses = [3.0] * 20 + [float("nan")] * 40
    skipped = _skipped_steps(losses, rolling_interval_length=8)
    assert skipped == list(range(20, 60))


def test_skip_step_optimizer_resumes_immediately_after_an_inf_burst():
    losses = [3.0] * 20 + [float("inf")] * 5 + [3.0] * 10
    skipped = _skipped_steps(losses, rolling_interval_length=8)
    assert skipped == list(range(20, 25)), "should skip the burst and nothing after it"


def test_skip_step_optimizer_does_not_skip_a_healthy_run():
    """A smoothly decreasing loss must produce no skips at all."""
    assert _skipped_steps([3.0 - i * 0.01 for i in range(30)], rolling_interval_length=8) == []
