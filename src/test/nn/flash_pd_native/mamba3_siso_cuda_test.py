from pathlib import Path
import re

import pytest
import torch

from olmo_core.nn.flash_pd_native import (
    NativePDMode,
    mamba3_siso_surrogate_scan,
)
from olmo_core.nn.flash_pd_native.routes import compact_hard_selection


def test_mamba3_siso_cuda_source_fuses_write_preprocessing_and_exact_backward():
    package = Path("src/olmo_core/nn/flash_pd_native")
    source = package.joinpath("csrc/flash_pd_native_cuda.cu").read_text()
    binding = package.joinpath("csrc/flash_pd_native.cpp").read_text()

    assert not re.search(r"mamba3_preprocess_kernel(?:<scalar_t>)?\s*<<<", source)
    assert "mamba3_phase_a_kernel" in source
    assert "mamba3_phase_b_kernel" in source
    assert "mamba3_phase_c_kernel" in source
    assert "mamba3_backward_fused_kernel" in source
    assert "flash_pd_native_mamba3_forward_cuda" in source
    assert "flash_pd_native_mamba3_forward_cuda" in binding
    assert "transition_input_real" in source
    assert "grad_value_real" in source
    assert "grad_beta" in source
    assert "grad_gamma" in source
    assert "dictionary_temperature" in source
    assert "router_temperature" in source


def _cuda_case(*, collision: bool, dtype: torch.dtype, state: int, time: int):
    torch.manual_seed(211 + state + time + collision)
    dictionary_size = 3
    maps = torch.stack(
        (
            torch.arange(state),
            torch.roll(torch.arange(state), 3),
            torch.roll(torch.arange(state), 7),
        )
    )
    if collision:
        maps[1, 1::4] = maps[1, 0::4]
    dictionary = torch.randn(1, dictionary_size, state, state, dtype=torch.float32)
    dictionary.data.scatter_(-2, maps.view(1, dictionary_size, 1, state), 5.0)
    selector = torch.randn(1, time, 1, dictionary_size, dtype=torch.float32)
    if collision:
        selector.data[..., 1] += 5.0
    values = [(torch.randn(1, 1, time, state) * 0.1).to(dtype).requires_grad_() for _ in range(4)]
    values[0].data.add_(0.9)
    beta = torch.rand(1, 1, time, dtype=dtype, requires_grad=True)
    gamma = torch.rand(1, 1, time, dtype=dtype, requires_grad=True)
    return dictionary.requires_grad_(), selector.requires_grad_(), [*values, beta, gamma]


def _independent_dense_hard_forward(
    dictionary_logits,
    selector_logits,
    diagonal_real,
    diagonal_imag,
    value_real,
    value_imag,
    beta,
    gamma,
):
    def hard_one_hot(logits, dim):
        selected = logits.argmax(dim=dim, keepdim=True)
        return torch.zeros_like(logits).scatter(dim, selected, 1)

    dictionary = hard_one_hot(dictionary_logits, -2)
    selector = hard_one_hot(selector_logits, -1)
    transition = torch.einsum("bthk,hkiq->bthiq", selector, dictionary).permute(0, 2, 1, 3, 4)
    batch, heads, time, state = diagonal_real.shape
    current_real = diagonal_real.new_zeros(batch, heads, state)
    current_imag = diagonal_imag.new_zeros(batch, heads, state)
    previous_real = value_real.new_zeros(batch, heads, state)
    previous_imag = value_imag.new_zeros(batch, heads, state)
    outputs_real = []
    outputs_imag = []
    for token in range(time):
        input_real = current_real + beta[:, :, token, None] * previous_real
        input_imag = current_imag + beta[:, :, token, None] * previous_imag
        product_real = (
            diagonal_real[:, :, token] * input_real - diagonal_imag[:, :, token] * input_imag
        )
        product_imag = (
            diagonal_real[:, :, token] * input_imag + diagonal_imag[:, :, token] * input_real
        )
        current_real = (
            torch.einsum("bhij,bhj->bhi", transition[:, :, token], product_real)
            + gamma[:, :, token, None] * value_real[:, :, token]
        )
        current_imag = (
            torch.einsum("bhij,bhj->bhi", transition[:, :, token], product_imag)
            + gamma[:, :, token, None] * value_imag[:, :, token]
        )
        previous_real = value_real[:, :, token]
        previous_imag = value_imag[:, :, token]
        outputs_real.append(current_real)
        outputs_imag.append(current_imag)
    return torch.stack(outputs_real, 2), torch.stack(outputs_imag, 2)


def _active_selected_row_gradient(logits, active_index, active_gradient, *, dim, temperature):
    probabilities = torch.softmax(logits / temperature, dim=dim)
    index = active_index.unsqueeze(dim)
    active_probability = torch.gather(probabilities, dim, index)
    one_hot = torch.zeros_like(logits).scatter(dim, index, 1)
    return (
        active_gradient.unsqueeze(dim)
        * active_probability
        * (one_hot - probabilities)
        / temperature
    )


def _literal_trapezoidal_proposition2_gradients(
    dictionary_logits,
    selector_logits,
    leaves,
    outputs,
    weights,
    *,
    dictionary_temperature,
    router_temperature,
):
    diagonal_real, diagonal_imag, value_real, value_imag, beta, gamma = leaves
    recurrence_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(outputs, weights)),
        leaves,
        retain_graph=True,
    )
    selection = compact_hard_selection(dictionary_logits, selector_logits)
    destination = selection.destination.long()
    routes = selection.routes.long()
    batch, heads, time, state = diagonal_real.shape
    dictionary_size = destination.shape[1]
    active_dictionary = dictionary_logits.new_zeros(heads, dictionary_size, state)
    selector_score = selector_logits.new_empty(batch, heads, time)
    carry_real = diagonal_real.new_zeros(batch, heads, state)
    carry_imag = diagonal_imag.new_zeros(batch, heads, state)
    expanded_dictionary = destination.unsqueeze(0).expand(batch, -1, -1, -1)

    for token in range(time - 1, -1, -1):
        total_real = weights[0][:, :, token] + carry_real
        total_imag = weights[1][:, :, token] + carry_imag
        route = routes[:, :, token]
        selected = torch.gather(
            expanded_dictionary,
            2,
            route[..., None, None].expand(-1, -1, 1, state),
        ).squeeze(2)
        destination_real = torch.gather(total_real, -1, selected)
        destination_imag = torch.gather(total_imag, -1, selected)
        if token:
            previous_real = outputs[0][:, :, token - 1]
            previous_imag = outputs[1][:, :, token - 1]
            previous_value_real = value_real[:, :, token - 1]
            previous_value_imag = value_imag[:, :, token - 1]
        else:
            previous_real = torch.zeros_like(total_real)
            previous_imag = torch.zeros_like(total_imag)
            previous_value_real = torch.zeros_like(total_real)
            previous_value_imag = torch.zeros_like(total_imag)
        transition_input_real = previous_real + beta[:, :, token, None] * previous_value_real
        transition_input_imag = previous_imag + beta[:, :, token, None] * previous_value_imag
        transformed_input_real = (
            diagonal_real[:, :, token] * transition_input_real
            - diagonal_imag[:, :, token] * transition_input_imag
        )
        transformed_input_imag = (
            diagonal_real[:, :, token] * transition_input_imag
            + diagonal_imag[:, :, token] * transition_input_real
        )
        active = (
            destination_real * transformed_input_real + destination_imag * transformed_input_imag
        )
        route_one_hot = torch.nn.functional.one_hot(route, num_classes=dictionary_size).to(
            active.dtype
        )
        active_dictionary.add_(torch.einsum("bhk,bhn->hkn", route_one_hot, active))
        selector_score[:, :, token] = active.sum(-1)
        token_diagonal_real = diagonal_real[:, :, token]
        token_diagonal_imag = diagonal_imag[:, :, token]
        carry_real = destination_real * token_diagonal_real + destination_imag * token_diagonal_imag
        carry_imag = (
            -destination_real * token_diagonal_imag + destination_imag * token_diagonal_real
        )

    dictionary_gradient = _active_selected_row_gradient(
        dictionary_logits,
        destination,
        active_dictionary,
        dim=-2,
        temperature=dictionary_temperature,
    )
    selector_gradient = _active_selected_row_gradient(
        selector_logits,
        routes.permute(0, 2, 1),
        selector_score.permute(0, 2, 1),
        dim=-1,
        temperature=router_temperature,
    )
    return dictionary_gradient, selector_gradient, *recurrence_gradients


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("collision", [False, True])
def test_reference_surrogate_matches_literal_trapezoidal_proposition2_oracle(
    dtype: torch.dtype,
    collision: bool,
):
    dictionary, selector, leaves = _cuda_case(
        collision=collision,
        dtype=dtype,
        state=16,
        time=7,
    )
    dictionary = dictionary.detach().to(dtype).requires_grad_()
    selector = selector.detach().to(dtype).requires_grad_()
    expected_inputs = [
        dictionary.detach().clone().requires_grad_(),
        selector.detach().clone().requires_grad_(),
        *[leaf.detach().clone().requires_grad_() for leaf in leaves],
    ]
    expected = _independent_dense_hard_forward(*expected_inputs)
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = _literal_trapezoidal_proposition2_gradients(
        expected_inputs[0],
        expected_inputs[1],
        expected_inputs[2:],
        expected,
        weights,
        dictionary_temperature=0.7,
        router_temperature=1.3,
    )

    actual = mamba3_siso_surrogate_scan(
        dictionary,
        selector,
        *leaves,
        dictionary_temperature=0.7,
        router_temperature=1.3,
        chunk_size=32,
        mode="general_scatter" if collision else "permutation_gather",
        backend="reference",
    )
    actual_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(actual, weights)),
        (dictionary, selector, *leaves),
    )

    tolerance = 2e-10 if dtype == torch.float64 else 5e-5
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(
            actual_output,
            expected_output,
            atol=tolerance,
            rtol=tolerance,
        )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient,
            expected_gradient,
            atol=tolerance,
            rtol=tolerance,
        )


def test_trapezoidal_selection_counterexamples_differ_from_dense_two_hardmax_ste():
    dictionary, selector, leaves = _cuda_case(
        collision=True,
        dtype=torch.float64,
        state=16,
        time=7,
    )
    inputs = [
        dictionary.detach().double().requires_grad_(),
        selector.detach().double().requires_grad_(),
        *[leaf.detach().double().requires_grad_() for leaf in leaves],
    ]
    hard_outputs = _independent_dense_hard_forward(*inputs)
    weights = [torch.randn_like(output) for output in hard_outputs]
    literal = _literal_trapezoidal_proposition2_gradients(
        inputs[0],
        inputs[1],
        inputs[2:],
        hard_outputs,
        weights,
        dictionary_temperature=0.7,
        router_temperature=1.3,
    )

    def hard_ste(logits, dim, temperature):
        probabilities = torch.softmax(logits / temperature, dim=dim)
        selected = logits.argmax(dim=dim, keepdim=True)
        one_hot = torch.zeros_like(logits).scatter(dim, selected, 1)
        return one_hot.detach() + probabilities - probabilities.detach()

    dense_dictionary = hard_ste(inputs[0], -2, 0.7)
    dense_selector = hard_ste(inputs[1], -1, 1.3)
    transition = torch.einsum("bthk,hkiq->bthiq", dense_selector, dense_dictionary).permute(
        0, 2, 1, 3, 4
    )
    diagonal_real, diagonal_imag, value_real, value_imag, beta, gamma = inputs[2:]
    current_real = torch.zeros_like(value_real[:, :, 0])
    current_imag = torch.zeros_like(value_imag[:, :, 0])
    previous_real = torch.zeros_like(current_real)
    previous_imag = torch.zeros_like(current_imag)
    dense_outputs = []
    for token in range(value_real.shape[2]):
        input_real = current_real + beta[:, :, token, None] * previous_real
        input_imag = current_imag + beta[:, :, token, None] * previous_imag
        product_real = (
            diagonal_real[:, :, token] * input_real - diagonal_imag[:, :, token] * input_imag
        )
        product_imag = (
            diagonal_real[:, :, token] * input_imag + diagonal_imag[:, :, token] * input_real
        )
        current_real = (
            torch.einsum("bhij,bhj->bhi", transition[:, :, token], product_real)
            + gamma[:, :, token, None] * value_real[:, :, token]
        )
        current_imag = (
            torch.einsum("bhij,bhj->bhi", transition[:, :, token], product_imag)
            + gamma[:, :, token, None] * value_imag[:, :, token]
        )
        previous_real = value_real[:, :, token]
        previous_imag = value_imag[:, :, token]
        dense_outputs.append((current_real, current_imag))
    dense = (
        torch.stack([output[0] for output in dense_outputs], 2),
        torch.stack([output[1] for output in dense_outputs], 2),
    )
    dense_selection = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(dense, weights)),
        inputs[:2],
    )

    assert (literal[0] - dense_selection[0]).abs().max() > 1e-3
    assert (literal[1] - dense_selection[1]).abs().max() > 1e-3


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_mamba3_siso_cuda_matches_hard_recurrence_and_proposition2_selection_oracle(
    dtype: torch.dtype,
    collision: bool,
    chunk_size: int,
):
    dictionary, selector, leaves = _cuda_case(
        collision=collision,
        dtype=dtype,
        state=16,
        time=chunk_size + 3,
    )
    expected_leaves = [leaf.detach().float().requires_grad_() for leaf in leaves]
    expected_dictionary = dictionary.detach().requires_grad_()
    expected_selector = selector.detach().requires_grad_()
    expected = _independent_dense_hard_forward(
        expected_dictionary,
        expected_selector,
        *expected_leaves,
    )
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = _literal_trapezoidal_proposition2_gradients(
        expected_dictionary,
        expected_selector,
        expected_leaves,
        expected,
        weights,
        dictionary_temperature=0.7,
        router_temperature=1.3,
    )

    cuda_dictionary = dictionary.detach().cuda().requires_grad_()
    cuda_selector = selector.detach().cuda().requires_grad_()
    cuda_leaves = [leaf.detach().cuda().requires_grad_() for leaf in leaves]
    actual = mamba3_siso_surrogate_scan(
        cuda_dictionary,
        cuda_selector,
        *cuda_leaves,
        dictionary_temperature=0.7,
        router_temperature=1.3,
        chunk_size=chunk_size,
        mode=(NativePDMode.GENERAL_SCATTER if collision else NativePDMode.PERMUTATION_GATHER),
        backend="cuda",
    )
    actual_gradients = torch.autograd.grad(
        sum((output.float() * weight.cuda()).sum() for output, weight in zip(actual, weights)),
        (cuda_dictionary, cuda_selector, *cuda_leaves),
    )

    tolerance = 1e-1 if dtype == torch.bfloat16 else 1e-3
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(
            actual_output.float().cpu(),
            expected_output,
            atol=tolerance,
            rtol=tolerance,
        )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float().cpu(),
            expected_gradient,
            atol=tolerance,
            rtol=tolerance,
        )


@pytest.mark.gpu
@pytest.mark.parametrize(("state", "time"), [(16, 1), (32, 33), (64, 129), (128, 257)])
def test_mamba3_siso_cuda_tails_have_bounded_linear_workspace(state: int, time: int):
    dictionary, selector, leaves = _cuda_case(
        collision=True,
        dtype=torch.bfloat16,
        state=state,
        time=time,
    )
    selection = compact_hard_selection(dictionary.cuda(), selector.cuda())
    assert selection.destination.dtype == selection.routes.dtype == torch.int16
    real, imag, metadata = mamba3_siso_surrogate_scan(
        dictionary.cuda(),
        selector.cuda(),
        *[leaf.cuda() for leaf in leaves],
        dictionary_temperature=1.0,
        router_temperature=1.0,
        chunk_size=64,
        mode="general_scatter",
        backend="cuda",
        return_metadata=True,
    )
    chunks = (time + 63) // 64

    assert real.shape == imag.shape == (1, 1, time, state)
    assert metadata.forward_launches == 3
    assert metadata.backward_launches == 2
    assert metadata.payload_axes == ()
    assert metadata.scratch_elements == 5 * chunks * state
    assert metadata.training_sequence_elements == 3 * state
