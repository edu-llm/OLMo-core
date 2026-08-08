import ast
import inspect
import math
import os
import re
import textwrap

import pytest
import torch

import olmo_core.nn.flash_pd_ssm.mamba3_flash as mamba3_flash_module
from olmo_core.nn.flash_pd_ssm.mamba3_flash import (
    Mamba3FlashPDSSMMixer,
    mamba3_flash_pd_readout,
)
from olmo_core.nn.flash_pd_ssm.mamba3_flash_triton import (
    MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES,
    Mamba3FlashPDTritonCapability,
    get_mamba3_flash_pd_kernel_counts,
    mamba3_flash_pd_triton_capability,
    reset_mamba3_flash_pd_kernel_counts,
)
from olmo_core.nn.transformer.init import InitMethod


def _inputs(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    batch: int = 1,
    heads: int = 1,
    time: int = 5,
    rank: int = 2,
    state: int = 5,
    payload: int = 3,
    dictionary_size: int = 3,
    collisions: bool = True,
):
    generator = torch.Generator(device=device).manual_seed(901)
    dictionary = torch.full(
        (heads, dictionary_size, state, state),
        -4.0,
        dtype=dtype,
        device=device,
    )
    sources = torch.arange(state, device=device)
    for head in range(heads):
        for entry in range(dictionary_size):
            if collisions:
                destination = (sources // 2 + head + entry) % state
            else:
                destination = (sources + head + entry) % state
            dictionary[head, entry, destination, sources] = 4.0
    dictionary.requires_grad_()
    selector = torch.randn(
        batch,
        time,
        heads,
        dictionary_size,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    routes = torch.arange(time, device=device) % dictionary_size
    selector[:, torch.arange(time, device=device), :, routes] += 4.0
    selector.requires_grad_()
    diagonal = torch.complex(
        torch.randn(
            batch,
            heads,
            time,
            state,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
        torch.randn(
            batch,
            heads,
            time,
            state,
            dtype=dtype,
            device=device,
            generator=generator,
        ),
    ).requires_grad_()
    value = torch.randn(
        batch,
        heads,
        time,
        payload,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    b_projection = torch.randn(
        batch,
        heads,
        time,
        rank,
        state,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    c_projection = torch.randn(
        batch,
        heads,
        time,
        rank,
        state,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    mimo_x = torch.randn(
        heads,
        rank,
        payload,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    mimo_o = torch.randn(
        heads,
        rank,
        payload,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    dt = torch.rand(
        batch,
        heads,
        time,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    lam = torch.rand(
        batch,
        heads,
        time,
        dtype=dtype,
        device=device,
        generator=generator,
        requires_grad=True,
    )
    return (
        dictionary,
        selector,
        diagonal,
        value,
        b_projection,
        c_projection,
        mimo_x,
        mimo_o,
        dt,
        lam,
    )


def _clone_inputs(inputs):
    return tuple(tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in inputs)


def _run_readout(
    inputs,
    *,
    use_triton: bool | None,
    chunk_size: int,
    checkpoint_stride: int = 16,
):
    output = mamba3_flash_pd_readout(
        *inputs,
        temperature=0.73,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
        use_triton=use_triton,
    )
    weight = torch.linspace(
        -0.7,
        1.1,
        output.numel(),
        dtype=output.dtype,
        device=output.device,
    ).reshape_as(output)
    gradients = torch.autograd.grad((output * weight).sum(), inputs)
    return output, gradients


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 1e-9), (torch.float32, 1e-5)],
)
def test_cpu_auto_fallback_matches_checkpointed_oracle_and_strict_fails_closed(
    dtype: torch.dtype,
    tolerance: float,
):
    inputs = _inputs(device=torch.device("cpu"), dtype=dtype)
    expected_output, expected_gradients = _run_readout(
        _clone_inputs(inputs),
        use_triton=False,
        chunk_size=3,
    )
    reset_mamba3_flash_pd_kernel_counts()
    actual_output, actual_gradients = _run_readout(
        _clone_inputs(inputs),
        use_triton=None,
        chunk_size=3,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    assert torch.count_nonzero(actual_gradients[0]).item() > 0
    assert torch.count_nonzero(actual_gradients[1]).item() > 0
    assert get_mamba3_flash_pd_kernel_counts() == {"forward": 0, "backward": 0}

    with pytest.raises(RuntimeError, match="required but unavailable"):
        mamba3_flash_pd_readout(
            *_clone_inputs(inputs),
            temperature=0.73,
            chunk_size=3,
            use_triton=True,
        )

    unsupported_stride_output, unsupported_stride_gradients = _run_readout(
        _clone_inputs(inputs),
        use_triton=None,
        chunk_size=3,
        checkpoint_stride=17,
    )
    torch.testing.assert_close(
        unsupported_stride_output,
        expected_output,
        rtol=tolerance,
        atol=tolerance,
    )
    for actual, expected in zip(unsupported_stride_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)
    with pytest.raises(RuntimeError, match="checkpoint_stride"):
        mamba3_flash_pd_readout(
            *_clone_inputs(inputs),
            temperature=0.73,
            chunk_size=3,
            checkpoint_stride=17,
            use_triton=True,
        )


def test_auto_dispatch_selects_only_eligible_kernel_and_strict_never_falls_back(monkeypatch):
    inputs = _inputs(device=torch.device("cpu"))
    calls = []

    monkeypatch.setattr(
        mamba3_flash_module,
        "mamba3_flash_pd_triton_capability",
        lambda *args, **kwargs: Mamba3FlashPDTritonCapability(True, "test kernel"),
    )

    def fake_kernel(*args, **kwargs):
        calls.append((args, kwargs))
        value = args[3]
        return value.permute(0, 2, 1, 3).contiguous()

    monkeypatch.setattr(
        mamba3_flash_module,
        "mamba3_flash_pd_triton_readout",
        fake_kernel,
    )
    output = mamba3_flash_pd_readout(
        *inputs,
        temperature=0.73,
        chunk_size=3,
        use_triton=None,
    )
    assert output.shape == (1, 5, 1, 3)
    assert len(calls) == 1
    assert calls[0][1]["checkpoint_stride"] == 16

    monkeypatch.setattr(
        mamba3_flash_module,
        "mamba3_flash_pd_triton_capability",
        lambda *args, **kwargs: Mamba3FlashPDTritonCapability(False, "unsupported test GPU"),
    )
    with pytest.raises(RuntimeError, match="unsupported test GPU"):
        mamba3_flash_pd_readout(
            *inputs,
            temperature=0.73,
            chunk_size=3,
            use_triton=True,
        )


def test_capability_documents_float32_sm80_sm120_and_bounded_shape_limits():
    assert MAMBA3_FLASH_PD_SUPPORTED_COMPUTE_CAPABILITIES == frozenset({(8, 0), (12, 0)})
    inputs = _inputs(device=torch.device("cpu"))

    assert "CUDA" in mamba3_flash_pd_triton_capability(*inputs, chunk_size=7).reason
    assert "chunk_size" in mamba3_flash_pd_triton_capability(*inputs, chunk_size=129).reason
    assert (
        "checkpoint_stride"
        in mamba3_flash_pd_triton_capability(
            *inputs,
            chunk_size=128,
            checkpoint_stride=17,
        ).reason
    )
    double_inputs = _inputs(device=torch.device("cpu"), dtype=torch.float64)
    assert "float32" in mamba3_flash_pd_triton_capability(*double_inputs, chunk_size=7).reason
    large_state_inputs = _inputs(device=torch.device("cpu"), state=33)
    assert (
        "state"
        in mamba3_flash_pd_triton_capability(
            *large_state_inputs,
            chunk_size=7,
        ).reason
    )


def test_kernel_uses_runtime_loops_and_chunk_bounded_payload_scratch():
    import olmo_core.nn.flash_pd_ssm.mamba3_flash_triton as triton_module

    source = inspect.getsource(triton_module)
    assert source.count("tl.range(0, chunk_size)") >= 2
    gradient_source = source.split("def _mimo_shared_backward_kernel", 1)[1].split(
        "@triton.jit",
        1,
    )[0]
    assert gradient_source.count("tl.range(0, chunk_size)") == 1
    assert "tl.range(0, checkpoint_stride)" in gradient_source
    assert "tl.range(0, n_chunks)" in source
    assert "tl.static_range" not in source
    assert "time * rank * state * payload" not in source
    assert "n_chunks,\n        checkpoints_per_chunk,\n        state,\n        payload" in source
    assert "selected_transition_matrix" not in source


def test_hierarchical_replay_work_and_a100_saved_memory_bound():
    import olmo_core.nn.flash_pd_ssm.mamba3_flash_triton as triton_module

    time = 4096
    chunk_size = 128
    checkpoint_stride = 16
    replay_work = triton_module.estimate_mamba3_flash_pd_replay_work(
        time=time,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    old_replay_work = time * chunk_size
    assert replay_work == checkpoint_stride * time
    assert replay_work <= 16 * time
    old_rank_expanded_replay_work = 4 * replay_work
    assert old_rank_expanded_replay_work == 262_144
    assert old_rank_expanded_replay_work // replay_work == 4
    assert old_replay_work == 524_288
    assert replay_work == 65_536
    assert old_replay_work // replay_work == 8

    checkpoint_bytes = triton_module.estimate_mamba3_flash_pd_checkpoint_bytes(
        batch=2,
        heads=8,
        time=time,
        state=20,
        payload=128,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    compact_input_and_diagonal_bytes = 90_939_392
    saved_bytes_per_layer = compact_input_and_diagonal_bytes + checkpoint_bytes
    full_token_history_bytes_per_layer = 2 * 8 * time * 20 * 128 * 8
    old_rank_expanded_checkpoint_bytes = 335_544_320
    assert checkpoint_bytes == 83_886_080
    assert old_rank_expanded_checkpoint_bytes // checkpoint_bytes == 4
    assert saved_bytes_per_layer == 174_825_472
    assert saved_bytes_per_layer * 12 == 2_097_905_664
    assert saved_bytes_per_layer * 12 < 2 * 1024**3
    assert checkpoint_bytes * checkpoint_stride == full_token_history_bytes_per_layer


def test_triton_has_no_python_sequence_or_dictionary_loops_and_three_launches():
    import olmo_core.nn.flash_pd_ssm.mamba3_flash_triton as triton_module

    forward_source = textwrap.dedent(inspect.getsource(triton_module._launch_mimo_forward))
    forward_tree = ast.parse(forward_source)
    backward_source = textwrap.dedent(
        inspect.getsource(triton_module._Mamba3FlashPDTritonReadout.backward)
    )
    backward_tree = ast.parse(backward_source)
    forbidden_nodes = (
        ast.For,
        ast.While,
        ast.ListComp,
        ast.DictComp,
        ast.SetComp,
        ast.GeneratorExp,
    )
    assert not any(isinstance(node, forbidden_nodes) for node in ast.walk(forward_tree))
    assert not any(isinstance(node, forbidden_nodes) for node in ast.walk(backward_tree))
    assert "_launch_mimo_backward(" in backward_source
    forward_launches = re.findall(r"_mimo_shared_[a-z_]+_kernel\[", forward_source)
    assert len(forward_launches) == 1

    launcher_source = textwrap.dedent(inspect.getsource(triton_module._launch_mimo_backward))
    launcher_tree = ast.parse(launcher_source)
    assert not any(isinstance(node, forbidden_nodes) for node in ast.walk(launcher_tree))
    launches = re.findall(r"_mimo_shared_[a-z_]+_kernel\[", launcher_source)
    assert len(launches) == 2


def _require_triton(inputs, *, chunk_size: int, checkpoint_stride: int = 16):
    capability = mamba3_flash_pd_triton_capability(
        *inputs,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    if not capability.available:
        pytest.skip(capability.reason)


@pytest.mark.gpu
@pytest.mark.parametrize(
    (
        "chunk_size",
        "checkpoint_stride",
        "time",
        "state",
        "collisions",
        "rank",
        "payload",
        "diagonal_scale",
    ),
    [
        (1, 1, 5, 5, True, 2, 3, 1.0),
        (7, 3, 15, 5, False, 2, 3, 1.0),
        (32, 8, 35, 7, True, 2, 3, 1.0),
        (128, 16, 129, 7, False, 2, 3, 0.25),
        (7, 3, 8, 20, True, 4, 128, 1.0),
    ],
)
def test_triton_matches_all_checkpointed_gradients_for_collisions_permutations_and_tails(
    chunk_size: int,
    checkpoint_stride: int,
    time: int,
    state: int,
    collisions: bool,
    rank: int,
    payload: int,
    diagonal_scale: float,
):
    inputs = _inputs(
        device=torch.device("cuda"),
        time=time,
        state=state,
        rank=rank,
        payload=payload,
        collisions=collisions,
    )
    with torch.no_grad():
        inputs[2].mul_(diagonal_scale)
    _require_triton(
        inputs,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    destinations = inputs[0].argmax(dim=-2)
    if collisions:
        assert any(torch.unique(row).numel() < state for row in destinations.flatten(0, 1))
    else:
        assert all(torch.unique(row).numel() == state for row in destinations.flatten(0, 1))

    expected_output, expected_gradients = _run_readout(
        _clone_inputs(inputs),
        use_triton=False,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    actual_output, actual_gradients = _run_readout(
        _clone_inputs(inputs),
        use_triton=True,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=4e-4, atol=4e-4)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert torch.count_nonzero(actual_gradients[0]).item() > 0
    assert torch.count_nonzero(actual_gradients[1]).item() > 0


@pytest.mark.gpu
def test_triton_saved_tensors_are_chunk_bounded_and_backend_identity_is_counted():
    inputs = _inputs(device=torch.device("cuda"), time=15, state=5, rank=2, payload=3)
    chunk_size = 7
    checkpoint_stride = 3
    _require_triton(
        inputs,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )
    saved = []

    def pack(tensor):
        saved.append((tuple(tensor.shape), tensor.dtype))
        return tensor

    reset_mamba3_flash_pd_kernel_counts()
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output = mamba3_flash_pd_readout(
            *inputs,
            temperature=0.73,
            chunk_size=chunk_size,
            checkpoint_stride=checkpoint_stride,
            use_triton=True,
        )
        output.square().mean().backward()

    batch, heads, time, rank, state, payload = 1, 1, 15, 2, 5, 3
    saved_shapes = [shape for shape, _ in saved]
    assert (batch, heads, time, rank, state, payload) not in saved_shapes
    assert ((batch, heads, time, state), torch.int64) not in saved
    assert ((batch, time, heads), torch.int64) not in saved
    assert (
        batch,
        heads,
        math.ceil(time / chunk_size),
        math.ceil(chunk_size / checkpoint_stride),
        state,
        payload,
    ) in saved_shapes
    assert get_mamba3_flash_pd_kernel_counts() == {"forward": 1, "backward": 1}


@pytest.mark.gpu
def test_production_shape_runtime_benchmark():
    if os.environ.get("RUN_FLASH_PD_PRODUCTION_BENCHMARK") != "1":
        pytest.skip("set RUN_FLASH_PD_PRODUCTION_BENCHMARK=1 for the production benchmark")

    inputs = _inputs(
        device=torch.device("cuda"),
        batch=2,
        heads=8,
        time=4096,
        rank=4,
        state=20,
        payload=128,
        dictionary_size=16,
        collisions=True,
    )
    chunk_size = 128
    checkpoint_stride = 16
    with torch.no_grad():
        phase = torch.angle(inputs[2])
        inputs[2].copy_(torch.polar(torch.full_like(phase, 0.99), phase))
    _require_triton(
        inputs,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
    )

    reset_mamba3_flash_pd_kernel_counts()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = mamba3_flash_pd_readout(
        *inputs,
        temperature=0.73,
        chunk_size=chunk_size,
        checkpoint_stride=checkpoint_stride,
        use_triton=True,
    )
    output.square().mean().backward()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end)
    peak_bytes = torch.cuda.max_memory_allocated()
    print(f"production_shape_elapsed_ms={elapsed_ms:.3f} peak_allocated_bytes={peak_bytes}")
    assert torch.isfinite(output).all()
    assert all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in inputs)
    assert get_mamba3_flash_pd_kernel_counts() == {"forward": 1, "backward": 1}


@pytest.mark.gpu
def test_tiny_mixer_auto_dispatch_preserves_selector_dictionary_gradients_and_identity():
    probe = _inputs(device=torch.device("cuda"), time=3, state=5, rank=2, payload=4)
    _require_triton(probe, chunk_size=2)
    torch.manual_seed(902)
    module = Mamba3FlashPDSSMMixer(
        d_model=8,
        n_heads=1,
        head_dim=4,
        d_state=5,
        n_groups=1,
        mimo_rank=2,
        dictionary_size=3,
        scan_chunk_size=2,
        scan_checkpoint_stride=1,
        dtype=torch.float32,
        init_device="cuda",
    )
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=8,
        block_idx=0,
        num_blocks=1,
        generator=torch.Generator(device="cuda").manual_seed(903),
    )
    x = torch.randn(2, 5, 8, device="cuda", requires_grad=True)

    reset_mamba3_flash_pd_kernel_counts()
    module(x).square().mean().backward()

    assert module.last_backend == "triton_mimo_shared_hierarchical_s1"
    assert module.last_fallback_reason is None
    assert module.dictionary_logits.grad is not None
    assert torch.count_nonzero(module.dictionary_logits.grad).item() > 0
    assert module.dynamics_proj.weight.grad is not None
    selector_rows = module.n_heads * module.dictionary_size
    assert torch.count_nonzero(module.dynamics_proj.weight.grad[:selector_rows]).item() > 0
    assert get_mamba3_flash_pd_kernel_counts() == {"forward": 1, "backward": 1}
