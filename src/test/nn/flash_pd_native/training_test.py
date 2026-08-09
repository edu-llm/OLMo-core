import pytest
import torch

import olmo_core.nn.flash_pd_native.cuda as cuda_module
from olmo_core.nn.flash_pd_native import (
    NativePDMode,
    compact_hard_selection,
    flash_pd_scan,
    get_backend_counters,
    native_cuda_capability,
    paper_surrogate_scan,
    prove_selected_maps_bijective,
    reset_backend_counters,
)
from olmo_core.nn.flash_pd_native.reference import (
    _active_hardmax_surrogate_gradient,
    paper_surrogate_reference_scan,
    reference_scan,
)


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


def _noncontiguous_case(*, device: torch.device):
    torch.manual_seed(143)
    batch, heads, time, state, dictionary = 2, 2, 5, 16, 2
    base_destination = torch.arange(state)
    destination = torch.stack(
        [
            torch.stack((base_destination, torch.roll(base_destination, shifts=head + 1)))
            for head in range(heads)
        ]
    )
    dictionary_logits = torch.randn(heads, dictionary, state, state) * 0.1
    selector_logits = torch.randn(batch, time, heads, dictionary) * 0.1
    selected_routes = (
        torch.arange(time).view(1, time, 1) + torch.arange(heads).view(1, 1, heads)
    ) % dictionary
    with torch.no_grad():
        dictionary_logits.scatter_(-2, destination.unsqueeze(-2), 4.0)
        selector_logits.scatter_(
            -1,
            selected_routes.expand(batch, -1, -1).unsqueeze(-1),
            4.0,
        )

    values = []
    for index in range(4):
        logical = torch.randn(batch, heads, time, state) * 0.1
        if index == 0:
            logical.add_(0.9)
        backing = logical.permute(0, 2, 1, 3).contiguous().to(device)
        value = backing.permute(0, 2, 1, 3).detach().requires_grad_()
        assert value.shape == (batch, heads, time, state)
        assert not value.is_contiguous()
        values.append(value)
    return (
        dictionary_logits.to(device).detach().requires_grad_(),
        selector_logits.to(device).detach().requires_grad_(),
        values,
    )


class _ReferenceNativeExtension:
    def __init__(self):
        self.forward_operands = {}
        self.mode = NativePDMode.PERMUTATION_GATHER

    def forward(
        self,
        destination,
        inverse_destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        chunk_size,
        mode,
    ):
        del inverse_destination, chunk_size
        self.mode = NativePDMode.PERMUTATION_GATHER if mode == 1 else NativePDMode.GENERAL_SCATTER
        self.forward_operands = {
            "destination": destination,
            "routes": routes,
            "diagonal_real": diagonal_real,
            "diagonal_imag": diagonal_imag,
            "bias_real": bias_real,
            "bias_imag": bias_imag,
        }
        for name, tensor in self.forward_operands.items():
            assert tensor.is_contiguous(), f"{name} passed to forward is non-contiguous"
        return reference_scan(
            destination,
            routes,
            diagonal_real,
            diagonal_imag,
            bias_real,
            bias_imag,
            mode=self.mode,
        )

    def _assert_forward_operands_reused(self, **backward_operands):
        for name, backward_operand in backward_operands.items():
            forward_operand = self.forward_operands[name]
            assert backward_operand.is_contiguous(), f"{name} saved for backward is non-contiguous"
            assert (
                backward_operand.data_ptr() == forward_operand.data_ptr()
            ), f"{name} consumed by backward is not the tensor consumed by forward"
            torch.testing.assert_close(backward_operand, forward_operand)

    def backward(
        self,
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        output_real,
        output_imag,
        grad_output_real,
        grad_output_imag,
    ):
        del output_real, output_imag
        self._assert_forward_operands_reused(
            destination=destination,
            routes=routes,
            diagonal_real=diagonal_real,
            diagonal_imag=diagonal_imag,
        )
        with torch.enable_grad():
            values = [
                tensor.detach().clone().requires_grad_()
                for tensor in (
                    diagonal_real,
                    diagonal_imag,
                    self.forward_operands["bias_real"],
                    self.forward_operands["bias_imag"],
                )
            ]
            output = reference_scan(destination, routes, *values, mode=self.mode)
            return torch.autograd.grad(
                output,
                values,
                (grad_output_real, grad_output_imag),
            )

    def paper_backward(
        self,
        dictionary_logits,
        selector_logits,
        destination,
        routes,
        diagonal_real,
        diagonal_imag,
        bias_real,
        bias_imag,
        output_real,
        output_imag,
        grad_output_real,
        grad_output_imag,
        value_real,
        value_imag,
        beta,
        gamma,
        dictionary_temperature,
        router_temperature,
        chunk_size,
    ):
        del output_real, output_imag, value_real, value_imag, beta, gamma, chunk_size
        assert dictionary_temperature == router_temperature
        self._assert_forward_operands_reused(
            destination=destination,
            routes=routes,
            diagonal_real=diagonal_real,
            diagonal_imag=diagonal_imag,
            bias_real=bias_real,
            bias_imag=bias_imag,
        )
        with torch.enable_grad():
            leaves = [
                tensor.detach().clone().requires_grad_()
                for tensor in (
                    dictionary_logits,
                    selector_logits,
                    diagonal_real,
                    diagonal_imag,
                    bias_real,
                    bias_imag,
                )
            ]
            output = paper_surrogate_reference_scan(
                *leaves,
                temperature=dictionary_temperature,
                mode=self.mode,
            )
            return torch.autograd.grad(
                output,
                leaves,
                (grad_output_real, grad_output_imag),
            )


@pytest.mark.parametrize("operation", ["scan", "paper_training"])
def test_native_backward_reuses_forward_contiguous_operands_and_matches_reference(
    monkeypatch,
    operation: str,
):
    dictionary, selector, values = _noncontiguous_case(device=torch.device("cpu"))
    expected_dictionary = dictionary.detach().clone().requires_grad_()
    expected_selector = selector.detach().clone().requires_grad_()
    expected_values = [value.detach().contiguous().requires_grad_() for value in values]
    selection = compact_hard_selection(dictionary, selector)
    mode = NativePDMode.PERMUTATION_GATHER
    extension = _ReferenceNativeExtension()
    monkeypatch.setattr(cuda_module, "_EXTENSION", extension)

    if operation == "paper_training":
        expected = paper_surrogate_scan(
            expected_dictionary,
            expected_selector,
            *expected_values,
            backend="reference",
            mode=mode,
        )
        actual = cuda_module.native_cuda_paper_surrogate_scan(
            dictionary,
            selector,
            selection.destination,
            selection.routes,
            *values,
            temperature=1.0,
            chunk_size=128,
            mode=mode,
        )
        expected_inputs = (expected_dictionary, expected_selector, *expected_values)
        actual_inputs = (dictionary, selector, *values)
    else:
        expected = flash_pd_scan(
            selection.destination,
            selection.routes,
            *expected_values,
            backend="reference",
            mode=mode,
        )
        proof = prove_selected_maps_bijective(selection.destination, selection.routes)
        assert proof.proven and proof.inverse_destination is not None
        actual = cuda_module._NativeFlashPD.apply(
            selection.destination,
            proof.inverse_destination,
            selection.routes,
            *values,
            128,
            1,
        )
        expected_inputs = tuple(expected_values)
        actual_inputs = tuple(values)

    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected, weights)),
        expected_inputs,
    )
    actual_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(actual, weights)),
        actual_inputs,
    )

    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output, expected_output)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual_gradient, expected_gradient)


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
def test_cuda_training_noncontiguous_bhtn_matches_literal_paper_surrogate():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    capability = native_cuda_capability()
    if not capability.available:
        pytest.skip(capability.reason)

    cpu_dictionary, cpu_selector, cpu_values = _noncontiguous_case(device=torch.device("cpu"))
    expected = paper_surrogate_scan(
        cpu_dictionary,
        cpu_selector,
        *cpu_values,
        backend="reference",
        mode="permutation_gather",
    )
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected, weights)),
        (cpu_dictionary, cpu_selector, *cpu_values),
    )

    dictionary, selector, values = _noncontiguous_case(device=torch.device("cuda"))
    actual = paper_surrogate_scan(
        dictionary,
        selector,
        *values,
        backend="cuda",
        mode="permutation_gather",
    )
    actual_gradients = torch.autograd.grad(
        sum((output.float() * weight.cuda()).sum() for output, weight in zip(actual, weights)),
        (dictionary, selector, *values),
    )

    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(actual_output.cpu(), expected_output, atol=4e-4, rtol=4e-4)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float().cpu(),
            expected_gradient,
            atol=8e-4,
            rtol=8e-4,
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
