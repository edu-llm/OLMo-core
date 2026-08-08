import copy
import hashlib
import inspect
import json
import math

import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.transformer.init import InitMethod


def _config_hash(config: Mamba3Config) -> str:
    payload = json.dumps(config.as_config_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


_MAMBA_CONFIG_BEFORE_IMPORT = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024).as_config_dict()
_MAMBA_HASH_BEFORE_IMPORT = _config_hash(Mamba3Config.mamba3_olmo3_370M(vocab_size=1024))

from olmo_core.nn.flash_pd_ssm.mamba3_flash import (  # noqa: E402
    Mamba3FlashPDSSMMixer,
    Mamba3FlashPDSSMMixerConfig,
    mamba3_flash_pd_readout,
    mamba3_flash_pd_scan,
)


def _hard_ste(values: torch.Tensor, *, dim: int, temperature: float) -> torch.Tensor:
    soft = torch.softmax(values / temperature, dim=dim)
    hard = torch.zeros_like(values).scatter_(dim, values.argmax(dim=dim, keepdim=True), 1.0)
    return (hard - soft).detach() + soft


def _dense_ste_oracle(
    dictionary_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    diagonal: torch.Tensor,
    previous_input: torch.Tensor,
    current_input: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    dictionary = _hard_ste(dictionary_logits, dim=-2, temperature=temperature)
    selector = _hard_ste(selector_logits, dim=-1, temperature=temperature)
    selected = torch.einsum("bthk,hkiq->bthiq", selector, dictionary)

    state = torch.zeros_like(current_input[:, :, 0])
    states = []
    for token_idx in range(diagonal.shape[2]):
        transition = selected[:, token_idx].to(diagonal.dtype)
        transition = transition * diagonal[:, :, token_idx].unsqueeze(-2)
        source = state + previous_input[:, :, token_idx]
        state = torch.einsum("bhiq,bhqp->bhip", transition, source) + current_input[:, :, token_idx]
        states.append(state)
    return torch.stack(states, dim=2)


def _run_dense_or_sparse(*, sparse: bool, dtype: torch.dtype):
    torch.manual_seed(117)
    batch, heads, time, state, payload, dictionary_size = 2, 2, 4, 4, 3, 3
    temperature = 0.63

    dictionary_data = torch.randn(heads, dictionary_size, state, state, dtype=dtype)
    dictionary_data.sub_(2.0)
    for head in range(heads):
        for entry in range(dictionary_size):
            for source in range(state):
                destination = (source // 2 + entry + head) % state
                dictionary_data[head, entry, destination, source] += 6.0
    dictionary = dictionary_data.requires_grad_()

    selector_data = torch.randn(batch, time, heads, dictionary_size, dtype=dtype)
    for token_idx in range(time):
        selector_data[:, token_idx, :, token_idx % dictionary_size] += 3.0
    selector = selector_data.requires_grad_()

    leaves = [
        torch.randn(
            batch,
            heads,
            time,
            state,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(2)
    ]
    diagonal = torch.complex(leaves[0], leaves[1])

    payload_leaves = [
        torch.randn(
            batch,
            heads,
            time,
            state,
            payload,
            dtype=dtype,
            requires_grad=True,
        )
        for _ in range(4)
    ]
    previous_input = torch.complex(payload_leaves[0], payload_leaves[1])
    current_input = torch.complex(payload_leaves[2], payload_leaves[3])

    if sparse:
        output = mamba3_flash_pd_scan(
            dictionary,
            selector,
            diagonal,
            previous_input,
            current_input,
            temperature=temperature,
        )
    else:
        output = _dense_ste_oracle(
            dictionary,
            selector,
            diagonal,
            previous_input,
            current_input,
            temperature=temperature,
        )

    real_weight = torch.randn(output.shape, dtype=dtype)
    imag_weight = torch.randn(output.shape, dtype=dtype)
    loss = (output.real * real_weight + output.imag * imag_weight).sum()
    gradients = torch.autograd.grad(loss, (dictionary, selector, *leaves, *payload_leaves))
    return dictionary.argmax(dim=-2), output, gradients


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 1e-9), (torch.float32, 8e-5)],
)
def test_sparse_payload_scan_matches_dense_collision_oracle_and_all_gradients(
    dtype: torch.dtype,
    tolerance: float,
):
    destination, expected_output, expected_gradients = _run_dense_or_sparse(
        sparse=False, dtype=dtype
    )
    _, actual_output, actual_gradients = _run_dense_or_sparse(sparse=True, dtype=dtype)

    assert any(torch.unique(row).numel() < row.numel() for row in destination.flatten(0, 1))
    torch.testing.assert_close(actual_output, expected_output, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _run_dense_or_checkpointed_readout(
    *,
    checkpointed: bool,
    dtype: torch.dtype,
    rank: int,
):
    torch.manual_seed(217)
    batch, heads, time, state, payload, dictionary_size = 2, 2, 5, 3, 4, 3
    temperature = 0.71

    dictionary_data = torch.randn(heads, dictionary_size, state, state, dtype=dtype)
    dictionary_data.sub_(2.0)
    for head in range(heads):
        for entry in range(dictionary_size):
            for source in range(state):
                destination = (source // 2 + entry + head) % state
                dictionary_data[head, entry, destination, source] += 6.0
    dictionary = dictionary_data.requires_grad_()

    selector_data = torch.randn(batch, time, heads, dictionary_size, dtype=dtype)
    for token_idx in range(time):
        selector_data[:, token_idx, :, token_idx % dictionary_size] += 3.0
    selector = selector_data.requires_grad_()

    diagonal_parts = [
        torch.randn(batch, heads, time, state, dtype=dtype, requires_grad=True) for _ in range(2)
    ]
    diagonal = torch.complex(*diagonal_parts)
    value = torch.randn(batch, heads, time, payload, dtype=dtype, requires_grad=True)
    b_projection = torch.randn(batch, heads, time, rank, state, dtype=dtype, requires_grad=True)
    c_projection = torch.randn(batch, heads, time, rank, state, dtype=dtype, requires_grad=True)
    mimo_x = torch.randn(heads, rank, payload, dtype=dtype, requires_grad=True)
    mimo_o = torch.randn(heads, rank, payload, dtype=dtype, requires_grad=True)
    dt = torch.rand(batch, heads, time, dtype=dtype, requires_grad=True)
    lam = torch.rand(batch, heads, time, dtype=dtype, requires_grad=True)

    if checkpointed:
        output = mamba3_flash_pd_readout(
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
            temperature=temperature,
            chunk_size=2,
        )
    else:
        drive = torch.einsum("bhtrn,hrp,bhtp->bhtnp", b_projection, mimo_x, value)
        prior_drive = torch.cat(
            (torch.zeros_like(drive[:, :, :1]), drive[:, :, :-1]),
            dim=2,
        )
        previous_input = (1.0 - lam)[..., None, None] * dt[..., None, None] * prior_drive
        current_input = lam[..., None, None] * dt[..., None, None] * drive
        states = _dense_ste_oracle(
            dictionary,
            selector,
            diagonal,
            torch.complex(previous_input, torch.zeros_like(previous_input)),
            torch.complex(current_input, torch.zeros_like(current_input)),
            temperature=temperature,
        )
        output = torch.einsum("bhtrn,bhtnp,hrp->bthp", c_projection, states.real, mimo_o)

    weight = torch.randn(output.shape, dtype=dtype)
    loss = (output * weight).sum()
    leaves = (
        dictionary,
        selector,
        *diagonal_parts,
        value,
        b_projection,
        c_projection,
        mimo_x,
        mimo_o,
        dt,
        lam,
    )
    return output, torch.autograd.grad(loss, leaves)


@pytest.mark.parametrize(
    ("dtype", "rank", "tolerance"),
    [
        (torch.float64, 1, 2e-9),
        (torch.float64, 4, 2e-9),
        (torch.float32, 1, 1e-4),
        (torch.float32, 4, 2e-4),
    ],
)
def test_checkpointed_compact_readout_matches_dense_complex_oracle_and_all_gradients(
    dtype: torch.dtype,
    rank: int,
    tolerance: float,
):
    expected_output, expected_gradients = _run_dense_or_checkpointed_readout(
        checkpointed=False,
        dtype=dtype,
        rank=rank,
    )
    actual_output, actual_gradients = _run_dense_or_checkpointed_readout(
        checkpointed=True,
        dtype=dtype,
        rank=rank,
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


def _checkpoint_saved_bytes(
    *,
    batch: int,
    time: int,
    heads: int,
    rank: int,
    state: int,
    payload: int,
    dictionary_size: int,
    chunk_size: int,
) -> int:
    real_bytes = 4
    complex_bytes = 8
    compact_real_elements = (
        heads * dictionary_size * state * state
        + batch * time * heads * dictionary_size
        + batch * heads * time * payload
        + 2 * batch * heads * time * rank * state
        + 2 * batch * heads * time
    )
    diagonal_elements = batch * heads * time * state
    boundary_elements = batch * heads * (math.ceil(time / chunk_size) - 1) * state * payload
    return real_bytes * compact_real_elements + complex_bytes * (
        diagonal_elements + boundary_elements
    )


def test_checkpointed_readout_state_is_shared_and_rank_does_not_multiply_recurrent_bytes():
    torch.manual_seed(218)
    batch, time, heads, state, payload, dictionary_size = 2, 9, 2, 3, 4, 3
    chunk_size = 4
    expected_boundary_shape = (
        batch,
        heads,
        math.ceil(time / chunk_size) - 1,
        state,
        payload,
    )

    def recurrent_saved_bytes(rank: int) -> int:
        dictionary = torch.randn(heads, dictionary_size, state, state, requires_grad=True)
        selector = torch.randn(batch, time, heads, dictionary_size, requires_grad=True)
        diagonal = torch.complex(
            torch.randn(batch, heads, time, state),
            torch.randn(batch, heads, time, state),
        ).requires_grad_()
        value = torch.randn(batch, heads, time, payload, requires_grad=True)
        b_projection = torch.randn(batch, heads, time, rank, state, requires_grad=True)
        c_projection = torch.randn(batch, heads, time, rank, state, requires_grad=True)
        mimo_x = torch.randn(heads, rank, payload, requires_grad=True)
        mimo_o = torch.randn(heads, rank, payload, requires_grad=True)
        dt = torch.rand(batch, heads, time, requires_grad=True)
        lam = torch.rand(batch, heads, time, requires_grad=True)
        saved = []

        def pack(tensor: torch.Tensor):
            saved.append((tuple(tensor.shape), tensor.dtype, tensor.numel()))
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            output = mamba3_flash_pd_readout(
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
                temperature=0.8,
                chunk_size=chunk_size,
            )
            output.square().mean().backward()

        saved_shapes = [shape for shape, _, _ in saved]
        assert expected_boundary_shape in saved_shapes
        assert not any(
            len(shape) >= 6 and shape[-3:] == (rank, state, payload) for shape in saved_shapes
        )
        return sum(
            numel * torch.empty((), dtype=dtype).element_size()
            for shape, dtype, numel in saved
            if shape == expected_boundary_shape
        )

    assert recurrent_saved_bytes(rank=1) == recurrent_saved_bytes(rank=4)

    production_bytes = _checkpoint_saved_bytes(
        batch=32,
        time=4096,
        heads=8,
        rank=4,
        state=20,
        payload=128,
        dictionary_size=16,
        chunk_size=256,
    )
    old_rank_expanded_bytes = 1_766_006_784
    assert production_bytes == 1_530_077_184
    assert old_rank_expanded_bytes - production_bytes == 235_929_600
    assert production_bytes * 12 < 24 * 1024**3
    assert production_bytes < old_rank_expanded_bytes


def _scan_from_mamba3_terms(
    dictionary: torch.Tensor,
    selector: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    drive = torch.einsum("bhtrn,hrp,bhtp->bhtnp", b_projection, mimo_x, value)
    prior_drive = torch.cat((torch.zeros_like(drive[:, :, :1]), drive[:, :, :-1]), dim=2)
    previous_input = (1.0 - lam)[..., None, None] * dt[..., None, None] * prior_drive
    current_input = lam[..., None, None] * dt[..., None, None] * drive
    previous_input = torch.complex(previous_input, torch.zeros_like(previous_input))
    current_input = torch.complex(current_input, torch.zeros_like(current_input))
    return mamba3_flash_pd_scan(
        dictionary,
        selector,
        diagonal,
        previous_input,
        current_input,
        temperature=1.0,
    )


def _direct_trapezoidal_oracle(
    destination: torch.Tensor,
    diagonal: torch.Tensor,
    value: torch.Tensor,
    b_projection: torch.Tensor,
    mimo_x: torch.Tensor,
    dt: torch.Tensor,
    lam: torch.Tensor,
) -> torch.Tensor:
    drive = torch.einsum("bhtrn,hrp,bhtp->bhtnp", b_projection, mimo_x, value)
    state = torch.zeros_like(drive[:, :, 0]).to(diagonal.dtype)
    states = []
    for token_idx in range(value.shape[2]):
        previous_drive = (
            torch.zeros_like(drive[:, :, 0]) if token_idx == 0 else drive[:, :, token_idx - 1]
        )
        source = (
            state
            + ((1.0 - lam[:, :, token_idx]) * dt[:, :, token_idx])[..., None, None] * previous_drive
        )
        transition = F.one_hot(destination, num_classes=destination.shape[-1]).movedim(-1, -2)
        transition = transition.to(diagonal.dtype)
        transition = transition * diagonal[:, :, token_idx].unsqueeze(-2)
        state = torch.einsum("bhiq,bhqp->bhip", transition, source)
        state = (
            state
            + (lam[:, :, token_idx] * dt[:, :, token_idx])[..., None, None] * drive[:, :, token_idx]
        )
        states.append(state)
    return torch.stack(states, dim=2)


def test_exponential_trapezoidal_step_formula_and_lambda_one_euler_limit():
    torch.manual_seed(118)
    batch, heads, time, rank, state, payload = 1, 1, 4, 2, 3, 2
    dictionary = torch.full((heads, 1, state, state), -4.0)
    destination = torch.tensor([[0, 0, 2]])
    dictionary[0, 0, destination[0], torch.arange(state)] = 4.0
    selector = torch.zeros(batch, time, heads, 1)
    dt = torch.tensor([[[0.13, 0.2, 0.07, 0.3]]])
    lam = torch.tensor([[[0.2, 0.6, 0.4, 0.8]]])
    decay = torch.tensor([-0.45])
    theta = torch.randn(batch, heads, time, state)
    magnitude = torch.exp(dt[..., None] * decay.view(1, heads, 1, 1))
    diagonal = torch.polar(magnitude.expand_as(theta), dt[..., None] * theta)
    value = torch.randn(batch, heads, time, payload)
    b_projection = torch.randn(batch, heads, time, rank, state)
    mimo_x = torch.randn(heads, rank, payload)

    actual = _scan_from_mamba3_terms(
        dictionary, selector, diagonal, value, b_projection, mimo_x, dt, lam
    )
    expected = _direct_trapezoidal_oracle(
        destination, diagonal, value, b_projection, mimo_x, dt, lam
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    euler_lambda = torch.ones_like(lam)
    actual_euler = _scan_from_mamba3_terms(
        dictionary,
        selector,
        diagonal,
        value,
        b_projection,
        mimo_x,
        dt,
        euler_lambda,
    )
    expected_euler = _direct_trapezoidal_oracle(
        destination,
        diagonal,
        value,
        b_projection,
        mimo_x,
        dt,
        euler_lambda,
    )
    torch.testing.assert_close(actual_euler, expected_euler, rtol=1e-6, atol=1e-6)


def _mixer_config(*, mimo_rank: int = 2, **kwargs) -> Mamba3FlashPDSSMMixerConfig:
    return Mamba3FlashPDSSMMixerConfig(
        n_heads=2,
        head_dim=4,
        d_state=3,
        n_groups=1,
        mimo_rank=mimo_rank,
        dictionary_size=3,
        ste_temperature=0.7,
        **kwargs,
    )


def _dense_mixer_forward(module: Mamba3FlashPDSSMMixer, x: torch.Tensor) -> torch.Tensor:
    batch, time, _ = x.shape
    H, P, G, R, N, K = (
        module.n_heads,
        module.head_dim,
        module.n_groups,
        module.mimo_rank,
        module.d_state,
        module.dictionary_size,
    )
    xz = module.xz_proj(x).view(batch, time, 2, H, P)
    value, gate = xz[:, :, 0], xz[:, :, 1]
    bc = module.bc_proj(x).view(batch, time, 2, G, R, N)
    b_projection, c_projection = bc[:, :, 0], bc[:, :, 1]
    if module.bc_norm_enabled:
        assert module.bc_norm_b is not None and module.bc_norm_c is not None

        def rms_norm(tensor: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
            original_dtype = tensor.dtype
            tensor_float = tensor.float()
            normalized = tensor_float * torch.rsqrt(
                tensor_float.square().mean(dim=-1, keepdim=True) + module.norm_eps
            )
            return (normalized * weight.float()).to(original_dtype)

        b_projection = rms_norm(b_projection, module.bc_norm_b)
        c_projection = rms_norm(c_projection, module.bc_norm_c)

    dynamics = module.dynamics_proj(x).float()
    selector_flat, dt_logits, lambda_logits, theta_flat = dynamics.split(
        (H * K, H, H, H * N),
        dim=-1,
    )
    selector = selector_flat.view(batch, time, H, K)
    dt = F.softplus(dt_logits + module.dt_bias.view(1, 1, H))
    lam = torch.sigmoid(lambda_logits)
    theta = theta_flat.view(batch, time, H, N)
    decay = -torch.exp(module.A_log.float())
    magnitude = torch.exp(dt * decay.view(1, 1, H))
    phase = dt[..., None] * theta
    diagonal = torch.polar(magnitude[..., None].expand_as(phase), phase).permute(0, 2, 1, 3)
    b_projection = b_projection.repeat_interleave(module.heads_per_group, dim=2).float()
    c_projection = c_projection.repeat_interleave(module.heads_per_group, dim=2).float()
    value = value.float().permute(0, 2, 1, 3)
    b_projection = b_projection.permute(0, 2, 1, 3, 4)
    c_projection = c_projection.permute(0, 2, 1, 3, 4)

    drive = torch.einsum(
        "bhtrn,hrp,bhtp->bhtnp",
        b_projection,
        module.mimo_x.float(),
        value,
    )
    prior_drive = torch.cat(
        (torch.zeros_like(drive[:, :, :1]), drive[:, :, :-1]),
        dim=2,
    )
    previous_input = (
        (1.0 - lam.permute(0, 2, 1))[..., None, None]
        * dt.permute(0, 2, 1)[..., None, None]
        * prior_drive
    )
    current_input = (
        lam.permute(0, 2, 1)[..., None, None] * dt.permute(0, 2, 1)[..., None, None] * drive
    )
    states = _dense_ste_oracle(
        module.dictionary_logits.float(),
        selector,
        diagonal,
        torch.complex(previous_input, torch.zeros_like(previous_input)),
        torch.complex(current_input, torch.zeros_like(current_input)),
        temperature=module.ste_temperature,
    )
    readout = torch.einsum(
        "bhtrn,bhtnp,hrp->bthp",
        c_projection,
        states.real,
        module.mimo_o.float(),
    )
    normalized = rms_norm(
        readout * F.silu(gate.float()),
        module.o_norm_weight,
    )
    return module.out_proj(normalized.reshape(batch, time, H * P).to(x.dtype))


@pytest.mark.parametrize(
    ("dtype", "tolerance"),
    [(torch.float64, 3e-8), (torch.float32, 2e-4)],
)
def test_default_mixer_matches_independent_dense_oracle_through_gate_and_output(
    dtype: torch.dtype,
    tolerance: float,
):
    torch.manual_seed(219)
    actual_module = Mamba3FlashPDSSMMixer(
        d_model=16,
        n_heads=2,
        head_dim=4,
        d_state=3,
        n_groups=1,
        mimo_rank=2,
        dictionary_size=3,
        ste_temperature=0.7,
        dtype=dtype,
        scan_chunk_size=2,
    )
    actual_module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(220),
    )
    expected_module = copy.deepcopy(actual_module)
    actual_x = torch.randn(2, 5, 16, dtype=dtype, requires_grad=True)
    expected_x = actual_x.detach().clone().requires_grad_()

    actual_output = actual_module(actual_x)
    expected_output = _dense_mixer_forward(expected_module, expected_x)
    weight = torch.randn(actual_output.shape, dtype=dtype)
    actual_gradients = torch.autograd.grad(
        (actual_output * weight).sum(),
        (actual_x, *actual_module.parameters()),
    )
    expected_gradients = torch.autograd.grad(
        (expected_output * weight).sum(),
        (expected_x, *expected_module.parameters()),
    )

    torch.testing.assert_close(actual_output, expected_output, rtol=tolerance, atol=tolerance)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("mimo_rank", [1, 3])
def test_mimo_mixer_forward_backward_shapes_and_finite_values(mimo_rank: int):
    torch.manual_seed(119)
    module = _mixer_config(mimo_rank=mimo_rank).build(
        16, layer_idx=0, n_layers=2, init_device="cpu"
    )
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(120),
    )
    x = torch.randn(2, 5, 16, requires_grad=True)

    output = module(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()
    output.square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, f"missing gradient for {name}"
        assert parameter.grad.shape == parameter.shape
        assert torch.isfinite(parameter.grad).all(), f"non-finite gradient for {name}"


@pytest.mark.parametrize("mimo_rank", [1, 4])
def test_official_mimo_parameters_initialize_to_rank_average_and_old_checkpoint_fails(
    mimo_rank: int,
):
    module = _mixer_config(mimo_rank=mimo_rank).build(
        16, layer_idx=0, n_layers=2, init_device="cpu"
    )
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=2,
    )

    assert module.mimo_x.shape == (module.n_heads, mimo_rank, module.head_dim)
    assert module.mimo_o.shape == (module.n_heads, mimo_rank, module.head_dim)
    torch.testing.assert_close(module.mimo_x, torch.full_like(module.mimo_x, 1.0 / mimo_rank))
    torch.testing.assert_close(module.mimo_o, torch.full_like(module.mimo_o, 1.0 / mimo_rank))

    old_rank_expanded_checkpoint = {
        name: value
        for name, value in module.state_dict().items()
        if name not in {"mimo_x", "mimo_o"}
    }
    with pytest.raises(RuntimeError, match="rank-expanded|mimo_x"):
        module.load_state_dict(old_rank_expanded_checkpoint, strict=True)


def test_mixer_has_no_causal_convolution_and_preserves_sparse_collision_path():
    module = _mixer_config().build(16, layer_idx=0, n_layers=1)
    assert not any(isinstance(child, nn.Conv1d) for child in module.modules())

    source = inspect.getsource(
        __import__("olmo_core.nn.flash_pd_ssm.mamba3_flash", fromlist=["mamba3_flash"])
    )
    assert "selected_transition_matrix" not in source
    assert "matrix_exp" not in source
    assert "scatter_add" in source


def test_config_registration_json_roundtrip_meta_num_params_init_and_mamba_hash():
    config = _mixer_config(
        dtype=DType.float32,
        bc_norm=False,
        bc_bias=False,
        norm_eps=1e-6,
        scan_checkpoint_stride=5,
    )
    payload = json.loads(json.dumps(config.as_config_dict()))
    rebuilt = SequenceMixerConfig.from_dict(payload)

    assert isinstance(rebuilt, Mamba3FlashPDSSMMixerConfig)
    assert rebuilt == config
    meta_module = rebuilt.build(16, layer_idx=0, n_layers=2, init_device="meta")
    assert isinstance(meta_module, Mamba3FlashPDSSMMixer)
    assert meta_module.scan_checkpoint_stride == 5
    assert all(parameter.device.type == "meta" for parameter in meta_module.parameters())
    assert rebuilt.num_params(16) == sum(
        parameter.numel() for parameter in meta_module.parameters()
    )
    assert meta_module.num_flops_per_token(seq_len=32) > 0

    module = rebuilt.build(16, layer_idx=0, n_layers=2, init_device="cpu")
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(121),
    )
    assert all(torch.isfinite(parameter).all() for parameter in module.parameters())

    old_payload = dict(payload)
    old_payload.pop("scan_checkpoint_stride")
    old_rebuilt = SequenceMixerConfig.from_dict(old_payload)
    assert isinstance(old_rebuilt, Mamba3FlashPDSSMMixerConfig)
    assert old_rebuilt.scan_checkpoint_stride == 16

    after = Mamba3Config.mamba3_olmo3_370M(vocab_size=1024)
    assert after.as_config_dict() == _MAMBA_CONFIG_BEFORE_IMPORT
    assert _config_hash(after) == _MAMBA_HASH_BEFORE_IMPORT


def test_tiny_transformer_model_forward_backward_through_direct_config_import():
    torch.manual_seed(122)
    config = Mamba3Config.mamba3_hybrid_like(
        d_model=16,
        vocab_size=32,
        n_layers=4,
        n_heads=2,
        intermediate_size=32,
        mamba_n_heads=2,
        mamba_head_dim=8,
        d_state=4,
        n_groups=1,
        mimo_rank=1,
    )
    assert isinstance(config.block, dict)
    config.block["mamba3"].sequence_mixer = _mixer_config()
    model = config.build(init_device="cpu")
    model.init_weights(device=torch.device("cpu"))

    input_ids = torch.randint(0, 32, (1, 6))
    labels = torch.randint(0, 32, (1, 6))
    output = model(input_ids, labels=labels)
    assert output.loss is not None and torch.isfinite(output.loss).all()
    output.loss.sum().backward()


class _Mesh:
    def size(self):
        return 2


def test_tp_cp_packed_reset_initial_state_and_decode_fail_closed():
    module = _mixer_config().build(16, layer_idx=0, n_layers=1)
    x = torch.randn(1, 4, 16)

    with pytest.raises(NotImplementedError, match="Tensor parallelism"):
        module.apply_tp(_Mesh())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="Context parallelism"):
        module.apply_cp(_Mesh())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="packed|cu_doc_lens"):
        module(x, cu_doc_lens=torch.tensor([0, 4], dtype=torch.int32))
    with pytest.raises(NotImplementedError, match="reset"):
        module(x, reset_mask=torch.zeros(1, 4, dtype=torch.bool))
    with pytest.raises(NotImplementedError, match="initial_state"):
        module(x, initial_state=torch.zeros(1))
    with pytest.raises(NotImplementedError, match="decode"):
        module(x, decode=True)


def test_default_training_avoids_dense_per_token_transition(monkeypatch):
    import olmo_core.nn.flash_pd_ssm.transition as transition_module

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("default training materialized a per-token N x N transition")

    monkeypatch.setattr(transition_module, "selected_transition_matrix", forbidden)
    module = _mixer_config().build(16, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=1,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_default_mixer_uses_compact_checkpointed_path(monkeypatch):
    import olmo_core.nn.flash_pd_ssm.mamba3_flash as mamba3_flash_module

    def forbidden(*args, **kwargs):
        del args, kwargs
        raise AssertionError("default mixer called the full-payload oracle scan")

    monkeypatch.setattr(mamba3_flash_module, "mamba3_flash_pd_scan", forbidden)
    module = _mixer_config(scan_chunk_size=2).build(16, layer_idx=0, n_layers=1)
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=16,
        block_idx=0,
        num_blocks=1,
    )
    x = torch.randn(2, 5, 16, requires_grad=True)
    module(x).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_training_flops_include_projection_and_recurrence_backward():
    module = _mixer_config().build(16, layer_idx=0, n_layers=1)
    projection_parameters = sum(
        projection.weight.numel()
        for projection in (
            module.xz_proj,
            module.bc_proj,
            module.dynamics_proj,
            module.out_proj,
        )
    )
    shared_state_elements = module.n_heads * module.d_state * module.head_dim
    rank_boundary_elements = module.mimo_rank * shared_state_elements
    training_lower_bound = (
        6 * projection_parameters + 3 * 10 * shared_state_elements + 3 * 6 * rank_boundary_elements
    )
    assert module.num_flops_per_token(seq_len=128) >= training_lower_bound
