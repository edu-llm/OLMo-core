import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    get_backend_counters,
    paper_surrogate_scan,
    reset_backend_counters,
)
from olmo_core.nn.flash_pd_native.reference import _active_hardmax_surrogate_gradient


def _case(
    *,
    collision: bool,
    device: torch.device,
    dtype: torch.dtype,
    state: int = 16,
    time: int = 7,
):
    torch.manual_seed(91 + collision)
    dictionary = 2
    destination = torch.stack((torch.arange(state), torch.roll(torch.arange(state), 3)))
    if collision:
        destination[1, 1] = destination[1, 0]
    dictionary_logits = torch.randn(1, dictionary, state, state, device=device, dtype=torch.float32)
    dictionary_logits.data.scatter_(
        -2,
        destination.view(1, dictionary, 1, state).to(device),
        4.0,
    )
    selector_logits = torch.randn(1, time, 1, dictionary, device=device, dtype=torch.float32)
    selector_logits.data[..., 1] += 1.0
    values = [
        (torch.randn(1, 1, time, state, device=device) * 0.1).to(dtype).requires_grad_()
        for _ in range(4)
    ]
    values[0].data.add_(0.9)
    return dictionary_logits.requires_grad_(), selector_logits.requires_grad_(), values


def test_literal_appendix_c_active_hardmax_counterexamples():
    route_logits = torch.tensor([0.2, -0.1], dtype=torch.float64)
    route = _active_hardmax_surrogate_gradient(
        route_logits, torch.tensor(0), torch.tensor(13.0, dtype=torch.float64), dim=0
    )
    torch.testing.assert_close(
        route,
        torch.tensor([3.177958051979696, -3.1779580519796973], dtype=torch.float64),
    )

    dictionary_logits = torch.tensor([[0.3, 0.1], [-0.2, 0.4]], dtype=torch.float64)
    active = _active_hardmax_surrogate_gradient(
        dictionary_logits,
        torch.tensor([0, 1]),
        torch.tensor([3.0, 10.0], dtype=torch.float64),
        dim=0,
    )
    torch.testing.assert_close(
        active,
        torch.tensor(
            [[0.7050111366047834, -2.444583116907459], [-0.7050111366047835, 2.4445831169074586]],
            dtype=torch.float64,
        ),
    )


@pytest.mark.gpu
@pytest.mark.parametrize("collision", [False, True])
def test_cuda_training_matches_literal_paper_surrogate(collision: bool):
    cpu_dictionary, cpu_selector, cpu_values = _case(
        collision=collision, device=torch.device("cpu"), dtype=torch.float32
    )
    expected = paper_surrogate_scan(
        cpu_dictionary,
        cpu_selector,
        *cpu_values,
        backend="reference",
        mode="general_scatter" if collision else "permutation_gather",
    )
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected, weights)),
        (cpu_dictionary, cpu_selector, *cpu_values),
    )

    dictionary = cpu_dictionary.detach().cuda().requires_grad_()
    selector = cpu_selector.detach().cuda().requires_grad_()
    values = [value.detach().cuda().requires_grad_() for value in cpu_values]
    actual = paper_surrogate_scan(
        dictionary,
        selector,
        *values,
        backend="cuda",
        mode="general_scatter" if collision else "permutation_gather",
    )
    actual_gradients = torch.autograd.grad(
        sum((output.float() * weight.cuda()).sum() for output, weight in zip(actual, weights)),
        (dictionary, selector, *values),
    )

    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output.cpu(), expected_output, atol=4e-4, rtol=4e-4)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float().cpu(), expected_gradient, atol=8e-4, rtol=8e-4
        )


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["auto", "cuda"])
def test_eligible_cuda_training_dispatch_is_native(backend: str):
    reset_backend_counters()
    dictionary, selector, values = _case(
        collision=False, device=torch.device("cuda"), dtype=torch.bfloat16
    )
    real, imag, metadata = paper_surrogate_scan(
        dictionary,
        selector,
        *values,
        backend=backend,
        return_metadata=True,
    )
    (real.float().square().mean() + imag.float().square().mean()).backward()

    assert metadata.backend == "cuda_paper_training"
    assert metadata.forward_launches == 3
    assert metadata.backward_launches == 5
    assert metadata.training_sequence_elements == 1 * 1 * 7 + 1 * 2 * 16
    assert metadata.dictionary_storage_elements == 1 * 2 * 16 * 16
    assert get_backend_counters() == {"cuda_training_permutation_gather": 1}
    assert dictionary.grad is not None and torch.isfinite(dictionary.grad).all()
    assert selector.grad is not None and torch.isfinite(selector.grad).all()


@pytest.mark.gpu
@pytest.mark.parametrize(("state", "time"), [(16, 1), (32, 129), (128, 257)])
def test_bfloat16_paper_training_tails_are_finite_and_match_oracle(state: int, time: int):
    dictionary, selector, cpu_values = _case(
        collision=False,
        device=torch.device("cpu"),
        dtype=torch.float32,
        state=state,
        time=time,
    )
    expected = paper_surrogate_scan(dictionary, selector, *cpu_values, backend="reference")
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected, weights)),
        (dictionary, selector, *cpu_values),
    )
    cuda_dictionary = dictionary.detach().cuda().requires_grad_()
    cuda_selector = selector.detach().cuda().requires_grad_()
    cuda_values = [
        value.detach().cuda().to(torch.bfloat16).requires_grad_() for value in cpu_values
    ]
    actual = paper_surrogate_scan(cuda_dictionary, cuda_selector, *cuda_values, backend="cuda")
    actual_gradients = torch.autograd.grad(
        sum((output.float() * weight.cuda()).sum() for output, weight in zip(actual, weights)),
        (cuda_dictionary, cuda_selector, *cuda_values),
    )

    for tensor in (*actual, *actual_gradients):
        assert torch.isfinite(tensor).all()
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(
            actual_output.float().cpu(), expected_output, atol=8e-2, rtol=8e-2
        )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float().cpu(),
            expected_gradient,
            atol=1e-1,
            rtol=1e-1,
        )
