import json
import os
import queue
import re
import subprocess
import sys
import threading
import time as time_module
from pathlib import Path

import pytest
import torch

from olmo_core.config import DType
from olmo_core.nn.flash_pd_native import (
    NativeFlashPDMamba3SISOMixer,
    NativeFlashPDMamba3SISOMixerConfig,
    NativePDBackend,
    NativePDMode,
    mamba3_siso_surrogate_scan,
)
from olmo_core.nn.flash_pd_native.routes import compact_hard_selection
from olmo_core.nn.transformer import InitMethod


def test_mamba3_siso_cuda_source_fuses_write_preprocessing_and_exact_backward():
    package = Path("src/olmo_core/nn/flash_pd_native")
    source = package.joinpath("csrc/flash_pd_native_cuda.cu").read_text()
    binding = package.joinpath("csrc/flash_pd_native.cpp").read_text()

    assert not re.search(r"mamba3_preprocess_kernel(?:<scalar_t>)?\s*<<<", source)
    assert "mamba3_phase_a_kernel" in source
    assert "mamba3_phase_b_kernel" in source
    assert "mamba3_phase_c_kernel" in source
    assert "paper_backward_phase_a_kernel" in source
    assert "paper_backward_phase_b_kernel" in source
    assert "paper_backward_phase_c_kernel" in source
    assert "flash_pd_native_mamba3_forward_cuda" in source
    assert "flash_pd_native_mamba3_forward_cuda" in binding
    assert "transition_input_real" in source
    assert "grad_value_real" in source
    assert "grad_beta" in source
    assert "grad_gamma" in source
    assert "dictionary_temperature" in source
    assert "router_temperature" in source


def _cuda_case(
    *,
    collision: bool,
    dtype: torch.dtype,
    state: int,
    time: int,
    payload_dtype: torch.dtype | None = None,
):
    torch.manual_seed(211 + state + time + collision)
    payload_dtype = dtype if payload_dtype is None else payload_dtype
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
    values = [
        (torch.randn(1, 1, time, state) * 0.1).to(value_dtype).requires_grad_()
        for value_dtype in (dtype, dtype, payload_dtype, payload_dtype)
    ]
    values[0].data.add_(0.9)
    beta = torch.rand(1, 1, time, dtype=payload_dtype, requires_grad=True)
    gamma = torch.rand(1, 1, time, dtype=payload_dtype, requires_grad=True)
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


@pytest.mark.parametrize("collision", [False, True])
def test_reference_mixed_precision_preserves_proposition2_all_gradient_semantics(
    collision: bool,
):
    dictionary, selector, leaves = _cuda_case(
        collision=collision,
        dtype=torch.float32,
        payload_dtype=torch.bfloat16,
        state=16,
        time=7,
    )
    expected_inputs = [
        dictionary.detach().clone().requires_grad_(),
        selector.detach().clone().requires_grad_(),
        *[leaf.detach().float().requires_grad_() for leaf in leaves],
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
        sum((output.float() * weight).sum() for output, weight in zip(actual, weights)),
        (dictionary, selector, *leaves),
    )

    assert all(output.dtype == torch.bfloat16 for output in actual)
    assert tuple(gradient.dtype for gradient in actual_gradients) == tuple(
        value.dtype for value in (dictionary, selector, *leaves)
    )
    for actual_output, expected_output in zip(actual, expected):
        torch.testing.assert_close(
            actual_output.float(),
            expected_output,
            atol=1e-1,
            rtol=1e-1,
        )
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual_gradient.float(),
            expected_gradient,
            atol=1e-1,
            rtol=1e-1,
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
@pytest.mark.parametrize(
    ("diagonal_dtype", "payload_dtype"),
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.bfloat16),
    ],
)
@pytest.mark.parametrize("collision", [False, True])
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
def test_mamba3_siso_cuda_matches_hard_recurrence_and_proposition2_selection_oracle(
    diagonal_dtype: torch.dtype,
    payload_dtype: torch.dtype,
    collision: bool,
    chunk_size: int,
):
    dictionary, selector, leaves = _cuda_case(
        collision=collision,
        dtype=diagonal_dtype,
        state=16,
        time=chunk_size + 3,
        payload_dtype=payload_dtype,
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

    assert all(output.dtype == payload_dtype for output in actual)
    assert tuple(gradient.dtype for gradient in actual_gradients) == tuple(
        value.dtype for value in (cuda_dictionary, cuda_selector, *cuda_leaves)
    )
    tolerance = 1e-1 if payload_dtype == torch.bfloat16 else 1e-3
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
@pytest.mark.parametrize(
    ("diagonal_dtype", "payload_dtype"),
    [(torch.float32, torch.float32), (torch.float32, torch.bfloat16)],
)
@pytest.mark.parametrize("chunk_size", [32, 64, 128])
@pytest.mark.parametrize("time", [63, 64, 65, 129])
def test_mamba3_siso_cuda_chunk_tails_match_proposition2_reference_gradients(
    time: int,
    chunk_size: int,
    diagonal_dtype: torch.dtype,
    payload_dtype: torch.dtype,
):
    state = 16
    dictionary, selector, leaves = _cuda_case(
        collision=True,
        dtype=diagonal_dtype,
        state=state,
        time=time,
        payload_dtype=payload_dtype,
    )
    expected_dictionary = dictionary.detach().requires_grad_()
    expected_selector = selector.detach().requires_grad_()
    expected_leaves = [leaf.detach().float().requires_grad_() for leaf in leaves]
    expected = mamba3_siso_surrogate_scan(
        expected_dictionary,
        expected_selector,
        *expected_leaves,
        dictionary_temperature=0.7,
        router_temperature=1.3,
        chunk_size=chunk_size,
        mode=NativePDMode.GENERAL_SCATTER,
        backend="reference",
    )
    weights = [torch.randn_like(output) for output in expected]
    expected_gradients = torch.autograd.grad(
        sum((output * weight).sum() for output, weight in zip(expected, weights)),
        (expected_dictionary, expected_selector, *expected_leaves),
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
        mode=NativePDMode.GENERAL_SCATTER,
        backend="cuda",
    )
    actual_gradients = torch.autograd.grad(
        sum((output.float() * weight.cuda()).sum() for output, weight in zip(actual, weights)),
        (cuda_dictionary, cuda_selector, *cuda_leaves),
    )

    tolerance = 1e-1 if payload_dtype == torch.bfloat16 else 1e-3
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


def _full_mixer(backend: NativePDBackend) -> NativeFlashPDMamba3SISOMixer:
    """Build the fused fp32 mixer a training step runs, pinned to one backend."""
    return NativeFlashPDMamba3SISOMixerConfig(
        n_heads=2,
        d_state=16,
        dictionary_size=4,
        chunk_size=32,
        dictionary_temperature=0.7,
        router_temperature=1.3,
        mode=NativePDMode.GENERAL_SCATTER,
        backend=backend,
        fuse_input_projections=True,
        dtype=DType.float32,
    ).build(32, layer_idx=0, n_layers=1)


@pytest.mark.gpu
def test_cuda_full_mixer_matches_reference_on_every_parameter_and_input_gradient():
    """
    Drive the CUDA backward through the mixer's own fused projection.

    The scans above hand the kernel freshly allocated leaves, which are dense whatever
    the mixer does. Only a whole forward exposes how the projection is laid out.
    """
    reference_mixer = _full_mixer(NativePDBackend.REFERENCE)
    reference_mixer.init_weights(
        init_method=InitMethod.normal,
        d_model=32,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator().manual_seed(29),
    )
    cuda_mixer = _full_mixer(NativePDBackend.CUDA)
    cuda_mixer.load_state_dict(reference_mixer.state_dict())
    reference_mixer.cuda()
    cuda_mixer.cuda()

    torch.manual_seed(31)
    x = torch.randn(2, 35, 32, device="cuda")
    weight = torch.randn(2, 35, 32, device="cuda")

    def forward_and_gradients(mixer):
        inputs = x.clone().requires_grad_()
        output = mixer(inputs)
        gradients = torch.autograd.grad((output * weight).sum(), (inputs, *mixer.parameters()))
        return output, gradients

    expected_output, expected_gradients = forward_and_gradients(reference_mixer)
    actual_output, actual_gradients = forward_and_gradients(cuda_mixer)

    assert reference_mixer.last_metadata is not None
    assert cuda_mixer.last_metadata is not None
    assert reference_mixer.last_metadata.backend == "reference_mamba3_siso_proposition2"
    assert cuda_mixer.last_metadata.backend == "cuda_mamba3_siso"
    torch.testing.assert_close(actual_output, expected_output, atol=1e-3, rtol=1e-3)
    names = ["input", *(name for name, _ in cuda_mixer.named_parameters())]
    assert len(names) == len(actual_gradients)
    for name, actual, expected in zip(names, actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual,
            expected,
            atol=1e-3,
            rtol=1e-3,
            msg=lambda default, name=name: f"gradient of {name}: {default}",
        )


# A colliding dictionary sends two lanes of a warp to one destination and leaves
# its neighbours alone, so the lanes of that warp aggregate different numbers of
# peers. Every warp intrinsic in the aggregation has to name only the peer group;
# one that names the whole warp waits on lanes that already finished, and the
# block never clears. Nothing about that depends on the sequence length or the
# chunk grid, so a single token at a single chunk is the whole reproduction --
# state 16 is one partial warp, 32 is one full warp, and 48 is a full warp
# followed by a partial one.
_SCATTER_WARP_STATES = (16, 32, 48)
# The child announces `ready` once the extension is loaded and the permutation
# path has cleared, so the per-stage budget below covers only a scatter launch,
# which the CPU reference at this shape finishes in well under a millisecond.
_SCATTER_WARP_SETUP_SECONDS = 180.0
_SCATTER_WARP_STAGE_SECONDS = 10.0

_SCATTER_WARP_PROBE = r"""
import importlib.util
import json
import sys

import torch

spec = importlib.util.spec_from_file_location("_mamba3_scatter_warp_case", sys.argv[1])
case_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = case_module
spec.loader.exec_module(case_module)

from olmo_core.nn.flash_pd_native import NativePDMode, mamba3_siso_surrogate_scan


def emit(**record):
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()


def scan(state, collision, mode, backend):
    device = "cuda" if backend == "cuda" else "cpu"
    dictionary, selector, leaves = case_module._cuda_case(
        collision=collision, dtype=torch.float32, state=state, time=1
    )
    return mamba3_siso_surrogate_scan(
        dictionary.detach().to(device),
        selector.detach().to(device),
        *[leaf.detach().to(device) for leaf in leaves],
        dictionary_temperature=0.7,
        router_temperature=1.3,
        chunk_size=32,
        mode=mode,
        backend=backend,
    )


scan(16, False, NativePDMode.PERMUTATION_GATHER, "cuda")
torch.cuda.synchronize()
emit(stage="ready")

for state in json.loads(sys.argv[2]):
    expected = scan(state, True, NativePDMode.GENERAL_SCATTER, "reference")
    actual = scan(state, True, NativePDMode.GENERAL_SCATTER, "cuda")
    torch.cuda.synchronize()
    emit(
        stage="scatter",
        state=state,
        deviation=max(
            (have.float().cpu() - want.float()).abs().max().item()
            for have, want in zip(actual, expected)
        ),
    )
"""


def _stream_probe(
    source: str,
    argument: str,
    *,
    setup_seconds: float,
    stage_seconds: float,
    records_expected: int,
) -> tuple[list[dict], list[str]]:
    """
    Run a probe child and read its records as they arrive, allowing `setup_seconds`
    for the first one and only `stage_seconds` between later ones. A wedged launch
    is then diagnosed on the budget of the launch itself rather than on the budget
    of loading torch and the extension.
    """
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    process = subprocess.Popen(
        [sys.executable, "-c", source, str(Path(__file__).resolve()), argument],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
    )
    lines: queue.Queue = queue.Queue()

    def pump() -> None:
        for line in process.stdout:  # type: ignore[union-attr]
            lines.put(line.decode(errors="replace").rstrip("\n"))
        lines.put(None)

    threading.Thread(target=pump, daemon=True).start()

    transcript: list[str] = []
    records: list[dict] = []
    deadline = time_module.monotonic() + setup_seconds
    try:
        while len(records) < records_expected:
            remaining = deadline - time_module.monotonic()
            if remaining <= 0:
                break
            try:
                line = lines.get(timeout=remaining)
            except queue.Empty:
                break
            if line is None:
                break
            transcript.append(line)
            if not line.startswith("{"):
                continue
            records.append(json.loads(line))
            deadline = time_module.monotonic() + stage_seconds
    finally:
        process.kill()
        process.wait()
    return records, transcript


@pytest.mark.gpu
def test_mamba3_siso_cuda_scatter_clears_uneven_peer_groups_within_one_warp():
    """
    One general-scatter step over a colliding dictionary has to return and has to
    agree with the reference. This is the narrow sibling of the boundary sweep
    below: it holds the sequence at a single token so that a stall can only be
    blamed on the scatter aggregation itself.
    """
    records, transcript = _stream_probe(
        _SCATTER_WARP_PROBE,
        json.dumps(list(_SCATTER_WARP_STATES)),
        setup_seconds=_SCATTER_WARP_SETUP_SECONDS,
        stage_seconds=_SCATTER_WARP_STAGE_SECONDS,
        records_expected=1 + len(_SCATTER_WARP_STATES),
    )
    tail = "\n".join(transcript[-15:])

    assert any(record.get("stage") == "ready" for record in records), (
        "the probe never loaded the extension and cleared the permutation warm-up "
        f"within {_SCATTER_WARP_SETUP_SECONDS}s, so it says nothing about the "
        f"scatter path. Child output:\n{tail}"
    )
    cleared = {
        record["state"]: record["deviation"]
        for record in records
        if record.get("stage") == "scatter"
    }
    wedged = [state for state in _SCATTER_WARP_STATES if state not in cleared]
    assert not wedged, (
        f"a single general-scatter step never returned for state={wedged[0]} within "
        f"{_SCATTER_WARP_STAGE_SECONDS}s of an already warm device. Cleared before "
        f"the stall: {sorted(cleared)}. Child output:\n{tail}"
    )
    drifted = {state: deviation for state, deviation in cleared.items() if deviation > 1e-3}
    assert not drifted, f"scatter aggregation no longer matches the reference: {drifted}"


_BOUNDARY_WATCHDOG_SECONDS = 120.0
_BOUNDARY_STAGE_BUDGET_SECONDS = 30.0

# Every boundary the chunk-parallel reverse scan can get wrong at chunk 128: a
# chunk that ends one token short of the grid, one that lands on it exactly, and
# ones that spill a one- and two-token tail chunk. Both routing modes and both
# payload precisions are covered because the reverse scan reads the diagonal and
# the payload through separate template parameters.
_BOUNDARY_CASES = (
    ("scatter-chunk128-time127", True, 128, 127, "bfloat16"),
    ("scatter-chunk128-time128", True, 128, 128, "bfloat16"),
    ("scatter-chunk128-time129", True, 128, 129, "bfloat16"),
    ("scatter-chunk128-time130", True, 128, 130, "bfloat16"),
    ("gather-chunk128-time128", False, 128, 128, "bfloat16"),
    ("gather-chunk128-time129", False, 128, 129, "bfloat16"),
    ("scatter-chunk128-time129-fp32", True, 128, 129, "float32"),
    ("scatter-chunk64-time129", True, 64, 129, "bfloat16"),
)

# A wedged kernel cannot be interrupted from the process that launched it, so the
# sweep runs in a child that reports each stage as it clears. Whatever the child
# never printed is the launch that stalled.
_BOUNDARY_PROBE = r"""
import importlib.util
import json
import sys
import time

import torch

spec = importlib.util.spec_from_file_location("_mamba3_boundary_case", sys.argv[1])
case_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = case_module
spec.loader.exec_module(case_module)

from olmo_core.nn.flash_pd_native import NativePDMode, mamba3_siso_surrogate_scan


def emit(label, stage, seconds):
    sys.stdout.write(json.dumps({"label": label, "stage": stage, "seconds": seconds}) + "\n")
    sys.stdout.flush()


for label, collision, chunk_size, steps, payload_name in json.loads(sys.argv[2]):
    mode = NativePDMode.GENERAL_SCATTER if collision else NativePDMode.PERMUTATION_GATHER
    dictionary, selector, leaves = case_module._cuda_case(
        collision=collision,
        dtype=torch.float32,
        state=16,
        time=steps,
        payload_dtype=getattr(torch, payload_name),
    )
    dictionary = dictionary.detach().cuda().requires_grad_()
    selector = selector.detach().cuda().requires_grad_()
    leaves = [leaf.detach().cuda().requires_grad_() for leaf in leaves]

    start = time.perf_counter()
    outputs = mamba3_siso_surrogate_scan(
        dictionary,
        selector,
        *leaves,
        dictionary_temperature=0.7,
        router_temperature=1.3,
        chunk_size=chunk_size,
        mode=mode,
        backend="cuda",
    )
    torch.cuda.synchronize()
    emit(label, "forward", time.perf_counter() - start)

    start = time.perf_counter()
    torch.autograd.grad(
        sum(output.float().sum() for output in outputs),
        (dictionary, selector, *leaves),
    )
    torch.cuda.synchronize()
    emit(label, "backward", time.perf_counter() - start)
"""


def _run_boundary_probe(cases, watchdog_seconds: float) -> dict[tuple[str, str], float]:
    command = [
        sys.executable,
        "-c",
        _BOUNDARY_PROBE,
        str(Path(__file__).resolve()),
        json.dumps([list(case) for case in cases]),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(entry for entry in sys.path if entry)
    try:
        finished = subprocess.run(
            command, capture_output=True, timeout=watchdog_seconds, env=environment
        )
        raw = finished.stdout
    except subprocess.TimeoutExpired as expired:
        raw = expired.stdout or b""

    stages: dict[tuple[str, str], float] = {}
    for line in raw.decode(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        record = json.loads(line)
        stages[(record["label"], record["stage"])] = record["seconds"]
    return stages


@pytest.mark.gpu
def test_mamba3_siso_cuda_clears_every_chunk_boundary_without_stalling():
    """
    The reverse scan must terminate on every chunk boundary, not just finish
    eventually. A stalled launch shows up here as the first stage the child never
    reported, which names the routing mode, chunk width, and sequence length that
    wedged it.
    """
    stages = _run_boundary_probe(_BOUNDARY_CASES, _BOUNDARY_WATCHDOG_SECONDS)

    expected = [(case[0], stage) for case in _BOUNDARY_CASES for stage in ("forward", "backward")]
    stalled = [key for key in expected if key not in stages]
    cleared = ", ".join(
        f"{label}/{stage}={stages[(label, stage)]:.3f}s"
        for label, stage in expected
        if (label, stage) in stages
    )
    assert not stalled, (
        f"{len(stalled)} of {len(expected)} boundary stages never completed within "
        f"{_BOUNDARY_WATCHDOG_SECONDS}s; the scan stalled at "
        f"{stalled[0][1]} of {stalled[0][0]}. Cleared before the stall: [{cleared}]"
    )

    overrun = {
        key: seconds for key, seconds in stages.items() if seconds > _BOUNDARY_STAGE_BUDGET_SECONDS
    }
    assert not overrun, f"boundary stages exceeded the per-stage budget: {overrun}"


def _median_event_milliseconds(run, *, warmup: int, measured: int) -> float:
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    samples = []
    for _ in range(measured):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        stop.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(stop))
    samples.sort()
    return samples[len(samples) // 2]


@pytest.mark.gpu
def test_mamba3_siso_cuda_backward_costs_at_most_four_forwards_at_production_shape():
    if torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3:
        pytest.skip("production-shape timing needs a 16 GiB device")
    batch, heads, time, state, dictionary_size, chunk_size = 2, 16, 4096, 64, 16, 64
    torch.manual_seed(1409)
    maps = torch.stack([torch.roll(torch.arange(state), shift) for shift in range(dictionary_size)])
    dictionary = torch.randn(heads, dictionary_size, state, state, device="cuda")
    dictionary.scatter_(
        -2,
        maps.view(1, dictionary_size, 1, state).expand(heads, -1, -1, -1).cuda(),
        5.0,
    )
    dictionary.requires_grad_()
    selector = torch.randn(batch, time, heads, dictionary_size, device="cuda").requires_grad_()
    diagonal_real = (
        torch.randn(batch, heads, time, state, device="cuda") * 0.1 + 0.9
    ).requires_grad_()
    diagonal_imag = (torch.randn(batch, heads, time, state, device="cuda") * 0.1).requires_grad_()
    payload = [
        (torch.randn(batch, heads, time, state, device="cuda") * 0.1)
        .to(torch.bfloat16)
        .requires_grad_()
        for _ in range(2)
    ]
    beta = torch.rand(batch, heads, time, device="cuda", dtype=torch.bfloat16).requires_grad_()
    gamma = torch.rand(batch, heads, time, device="cuda", dtype=torch.bfloat16).requires_grad_()
    leaves = (dictionary, selector, diagonal_real, diagonal_imag, *payload, beta, gamma)

    def forward():
        return mamba3_siso_surrogate_scan(
            dictionary,
            selector,
            diagonal_real,
            diagonal_imag,
            *payload,
            beta,
            gamma,
            dictionary_temperature=1.0,
            router_temperature=1.0,
            chunk_size=chunk_size,
            mode=NativePDMode.GENERAL_SCATTER,
            backend="cuda",
        )

    outputs = forward()
    seeds = [torch.randn_like(output) for output in outputs]

    def backward():
        return torch.autograd.grad(outputs, leaves, seeds, retain_graph=True)

    forward_milliseconds = _median_event_milliseconds(forward, warmup=20, measured=50)
    backward_milliseconds = _median_event_milliseconds(backward, warmup=20, measured=50)

    assert backward_milliseconds <= 4.0 * forward_milliseconds, (
        f"backward {backward_milliseconds:.3f} ms exceeds four forwards "
        f"({forward_milliseconds:.3f} ms each)"
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
    assert metadata.backward_launches == 5
    assert metadata.payload_axes == ()
    assert metadata.scratch_elements == 7 * chunks * state
    assert metadata.training_sequence_elements == time + 3 * state


@pytest.mark.gpu
def test_cuda_forward_compiles_the_pointwise_regions_apart_from_the_scan():
    """
    Compiled, a CUDA forward must not put the prologue, the scan, and the readout in
    one graph.

    It did, because the scan reaches CUDA through a custom op Dynamo can follow, and
    the arm then got nothing at all out of ``compile_model=True``: 74.6 ms compiled
    against 75.2 ms eager, one bfloat16 forward and backward at the production shape
    ``B=2, T=4096, D=1024``. With the CUDA scan left opaque the two pointwise regions
    are fused and partitioned by themselves and the same step takes 44.4 ms. A host
    forward keeps compiling as one graph, which
    ``test_mixer_forward_compiles_fullgraph_through_the_reference_scan`` holds.
    """
    from torch._dynamo.utils import counters

    mixer = (
        NativeFlashPDMamba3SISOMixerConfig(
            n_heads=1,
            d_state=32,
            dictionary_size=3,
            chunk_size=32,
            mode=NativePDMode.GENERAL_SCATTER,
            backend=NativePDBackend.CUDA,
            dtype=DType.float32,
        )
        .build(32, layer_idx=0, n_layers=1)
        .cuda()
    )
    x = torch.randn(1, 64, 32, device="cuda")

    torch._dynamo.reset()
    counters.clear()
    with torch.no_grad():
        expected = mixer(x)
        actual = torch.compile(mixer, backend="eager")(x)

    torch.testing.assert_close(actual, expected)
    assert counters["stats"]["unique_graphs"] > 1
