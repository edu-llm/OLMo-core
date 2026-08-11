"""Reference and deferred-GPU tests for native packed TWN training."""

import pytest
import torch
import torch.nn.functional as F

import olmo_core.ops.ternary as ternary_ops
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.moe.mlp import DroplessMoEMLP, MoEMLP
from olmo_core.nn.quantization import (
    QuantBackend,
    QuantConfig,
    QuantLinear,
    audit_quantization,
    twn_quantize,
    twn_quantize_ste,
)
from olmo_core.ops.ternary import (
    TWN_NEGATIVE_CODE,
    TWN_POSITIVE_CODE,
    TWN_ZERO_CODE,
    PackedTWNCache,
    dequantize_packed_twn,
    native_packed_grouped_linear,
    native_packed_linear,
    native_packed_status,
    pack_twn_reference,
    unpack_twn_codes,
)


def test_reference_pack_matches_deepgrove_code_mapping_and_lsb_order():
    # The +/-0.05 entries sit below this sparse row's strict threshold.
    weight = torch.tensor([[-1.0, -0.05, 0.0, 0.05, 1.0] + [0.0] * 11], dtype=torch.bfloat16)
    packed = pack_twn_reference(weight, -1)
    codes = unpack_twn_codes(packed.codes, weight.shape[-1])
    assert codes.tolist() == [
        [
            TWN_NEGATIVE_CODE,
            TWN_ZERO_CODE,
            TWN_ZERO_CODE,
            TWN_ZERO_CODE,
            TWN_POSITIVE_CODE,
            *([TWN_ZERO_CODE] * 11),
        ]
    ]
    expected_word = sum(int(code) << (2 * lane) for lane, code in enumerate(codes[0].tolist()))
    assert int(packed.codes[0, 0].item()) == expected_word


@pytest.mark.parametrize(
    ("shape", "in_dim"),
    [
        ((7, 35), -1),
        ((3, 11, 19), 1),
        ((3, 11, 19), 2),
    ],
)
def test_reference_pack_roundtrip_matches_bf16_twn(shape, in_dim):
    torch.manual_seed(4)
    weight = torch.randn(*shape, dtype=torch.bfloat16)
    packed = pack_twn_reference(weight, in_dim)
    expected = twn_quantize(weight, in_dim=in_dim).movedim(in_dim, -1)
    assert packed.alpha.dtype is torch.bfloat16
    assert packed.codes.dtype is torch.uint32
    assert packed.codes_t.dtype is torch.uint32
    assert torch.equal(dequantize_packed_twn(packed), expected)

    transposed_codes = unpack_twn_codes(packed.codes_t, packed.out_features)
    forward_codes = unpack_twn_codes(packed.codes, packed.in_features)
    assert torch.equal(transposed_codes, forward_codes.transpose(-2, -1))


def test_reference_pack_strict_threshold_and_all_zero_rows():
    weight = torch.zeros(3, 16, dtype=torch.bfloat16)
    # mean(abs)=10 and delta=7 exactly, so boundary values must remain zero.
    weight[1] = torch.tensor([7.0] * 8 + [13.0] * 8, dtype=torch.bfloat16)
    packed = pack_twn_reference(weight, -1)
    codes = unpack_twn_codes(packed.codes, 16)
    assert (codes[0] == TWN_ZERO_CODE).all()
    assert (codes[2] == TWN_ZERO_CODE).all()
    assert packed.alpha[0].item() == 0.0
    assert packed.alpha[2].item() == 0.0
    # Recompute from the BF16 values: this specifically asserts ``>`` rather than ``>=``.
    w32 = weight[1].float()
    delta = 0.7 * w32.abs().mean()
    boundary = w32.abs() == delta
    assert boundary.any()
    assert (codes[1][boundary] == TWN_ZERO_CODE).all()
    assert (codes[1][~boundary] == TWN_POSITIVE_CODE).all()


def test_native_status_rejects_pre_ampere_cuda(monkeypatch):
    monkeypatch.setattr(ternary_ops, "kernels", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (7, 5))
    status = native_packed_status()
    assert status["available"] is False
    assert "SM80" in str(status["reason"])
    assert status["kernel"] == "fake_quant_bf16"


def test_native_status_treats_meta_bf16_weights_as_prospective_cuda(monkeypatch):
    monkeypatch.setattr(ternary_ops, "kernels", object())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (8, 0))
    status = native_packed_status(torch.empty(4, 8, device="meta", dtype=torch.bfloat16))
    assert status["available"] is True
    assert status["kernel"] == "triton_packed_add_sub"
    layer = QuantLinear(
        8,
        4,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
        device="meta",
        dtype=torch.bfloat16,
    )
    audit = audit_quantization(layer)
    assert audit["resolved_backends"] == [QuantBackend.native_packed.value]
    assert audit["resolved_kernels"] == ["triton_packed_add_sub"]


def test_pack_cache_reuses_and_invalidates_version_storage_shape_and_orientation():
    calls = []

    def packer(weight, in_dim):
        calls.append((weight.data_ptr(), int(weight._version), tuple(weight.shape), in_dim))
        return pack_twn_reference(weight, in_dim)

    cache = PackedTWNCache()
    weight = torch.randn(8, 32, dtype=torch.bfloat16)
    first = cache.get_or_pack(weight, in_dim=-1, orientation="dense", packer=packer)
    second = cache.get_or_pack(weight, in_dim=-1, orientation="dense", packer=packer)
    assert first is second
    assert (cache.hits, cache.misses, len(calls)) == (1, 1, 1)

    with torch.no_grad():
        weight.add_(1)
    third = cache.get_or_pack(weight, in_dim=-1, orientation="dense", packer=packer)
    assert third is not second
    assert cache.misses == 2

    replacement = torch.randn_like(weight)
    with torch.no_grad():
        weight.set_(replacement)
    cache.get_or_pack(weight, in_dim=-1, orientation="dense", packer=packer)
    assert cache.misses == 3

    reshaped = torch.randn(4, 32, dtype=torch.bfloat16)
    cache.get_or_pack(reshaped, in_dim=-1, orientation="dense", packer=packer)
    assert cache.misses == 4

    cache.get_or_pack(reshaped, in_dim=-1, orientation="dense_transposed", packer=packer)
    assert cache.misses == 5
    cache.clear()
    cache.get_or_pack(reshaped, in_dim=-1, orientation="dense_transposed", packer=packer)
    assert cache.misses == 6

    # A square transpose has the same pointer, version, shape, dtype, and device. Stride is the
    # only key component that distinguishes the two logical weights.
    square = torch.randn(8, 8, dtype=torch.bfloat16)
    cache.get_or_pack(square, in_dim=-1, orientation="square", packer=packer)
    cache.get_or_pack(square.t(), in_dim=-1, orientation="square", packer=packer)
    assert cache.misses == 8


def test_native_cache_is_not_a_state_dict_entry_and_clears_on_load():
    layer = QuantLinear(32, 8, backend=QuantBackend.native_packed, dtype=torch.bfloat16)
    before = set(layer.state_dict())
    assert before == {"weight"}
    # Populate with the reference pack so this CPU test does not need Triton.
    layer._native_pack_cache.get_or_pack(  # type: ignore[attr-defined]
        layer.weight, in_dim=-1, orientation="test", packer=pack_twn_reference
    )
    assert layer._native_pack_cache._packed is not None  # type: ignore[attr-defined]
    layer.load_state_dict(layer.state_dict())
    assert layer._native_pack_cache._packed is None  # type: ignore[attr-defined]
    assert set(layer.state_dict()) == before


def test_expert_native_caches_are_ephemeral_and_clear_on_load():
    quant = QuantConfig(enabled=True, backend=QuantBackend.native_packed)
    mlp = MoEMLP(d_model=16, hidden_size=8, num_experts=2, quant=quant)
    before = set(mlp.state_dict())
    viewed = mlp.w1.view(2, 16, 8)
    mlp._native_pack_caches["w1"].get_or_pack(  # type: ignore[attr-defined]
        viewed, in_dim=1, orientation="capacity_w1_in1", packer=pack_twn_reference
    )
    assert mlp._native_pack_caches["w1"]._packed is not None  # type: ignore[attr-defined]
    mlp.load_state_dict(mlp.state_dict())
    assert mlp._native_pack_caches["w1"]._packed is None  # type: ignore[attr-defined]
    assert set(mlp.state_dict()) == before == {"w1", "w2", "w3"}


def test_native_backend_cpu_falls_back_exactly_to_fake_quant():
    torch.manual_seed(8)
    fake = QuantLinear(32, 7, backend=QuantBackend.fake_quant, dtype=torch.bfloat16)
    native = QuantLinear(
        32,
        7,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=True,
        dtype=torch.bfloat16,
    )
    native.load_state_dict(fake.state_dict())
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    with pytest.warns(RuntimeWarning, match="falling back"):
        actual = native(x)
    assert torch.equal(actual, fake(x))
    assert native.backend_status()["resolved"] == QuantBackend.fake_quant.value
    assert native.backend_status()["fallback_reason"]


def test_native_backend_cpu_can_be_made_strict():
    layer = QuantLinear(
        8,
        4,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
        dtype=torch.bfloat16,
    )
    with pytest.raises(OLMoConfigurationError, match="native_packed"):
        layer(torch.randn(2, 8, dtype=torch.bfloat16))


def test_capacity_moe_native_fallback_matches_fake_quant():
    torch.manual_seed(11)
    common = dict(d_model=16, hidden_size=8, num_experts=3, dtype=torch.bfloat16)
    fake = MoEMLP(**common, quant=QuantConfig(backend=QuantBackend.fake_quant))
    native = MoEMLP(**common, quant=QuantConfig(backend=QuantBackend.native_packed))
    native.load_state_dict(fake.state_dict())
    x = torch.randn(3, 5, 16, dtype=torch.bfloat16)
    with pytest.warns(RuntimeWarning, match="falling back"):
        actual = native(x)
    assert torch.equal(actual, fake(x))


def test_dropless_moe_native_fallback_matches_fake_quant_with_empty_expert():
    torch.manual_seed(12)
    common = dict(d_model=16, hidden_size=8, num_experts=3, dtype=torch.bfloat16)
    with pytest.warns(UserWarning, match="Grouped GEMM"):
        fake = DroplessMoEMLP(**common, quant=QuantConfig(backend=QuantBackend.fake_quant))
    native = DroplessMoEMLP(**common, quant=QuantConfig(backend=QuantBackend.native_packed))
    native.load_state_dict(fake.state_dict())
    sizes = torch.tensor([0, 4, 3])
    x = torch.randn(7, 16, dtype=torch.bfloat16)
    with pytest.warns(RuntimeWarning, match="falling back"):
        actual = native(x, sizes)
    assert torch.equal(actual, fake(x, sizes))


def test_audit_reports_requested_selected_kernel_and_fallback():
    layer = QuantLinear(
        8,
        4,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=True,
        dtype=torch.bfloat16,
    )
    audit = audit_quantization(layer)
    assert audit["requested_backends"] == ["native_packed"]
    assert audit["resolved_backends"] == ["fake_quant"]
    assert audit["resolved_kernels"] == ["fake_quant_bf16"]
    assert audit["fallback_reasons"]


def test_disabled_native_control_remains_exact_nn_linear():
    torch.manual_seed(9)
    ref = torch.nn.Linear(16, 5, bias=True, dtype=torch.bfloat16)
    control = QuantLinear(
        16,
        5,
        bias=True,
        enabled=False,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
        dtype=torch.bfloat16,
    )
    control.load_state_dict(ref.state_dict())
    x = torch.randn(3, 16, dtype=torch.bfloat16)
    assert torch.equal(control(x), ref(x))
    assert control.backend_status()["resolved"] == "disabled_control"
    audit = audit_quantization(control)
    assert audit["requested_backends"] == ["native_packed"]
    assert audit["resolved_kernels"] == ["disabled_control"]


CUDA_NATIVE = bool(torch.cuda.is_available() and native_packed_status()["available"])
requires_native_cuda = pytest.mark.skipif(
    not CUDA_NATIVE,
    reason=f"native CUDA/Triton backend unavailable: {native_packed_status()['reason']}",
)


@pytest.mark.gpu
@requires_native_cuda
def test_triton_pack_is_bit_exact_to_reference():
    from olmo_core.kernels import ternary as ternary_kernels

    torch.manual_seed(10)
    weight = torch.randn(5, 37, 65, device="cuda", dtype=torch.bfloat16)
    expected = pack_twn_reference(weight, 2)
    actual = ternary_kernels.pack_twn(weight, 2)
    assert torch.equal(actual.codes, expected.codes)
    assert torch.equal(actual.codes_t, expected.codes_t)
    assert torch.equal(actual.alpha, expected.alpha)
    forward_only = ternary_kernels.pack_twn_forward_only(weight, 2)
    assert torch.equal(forward_only.codes, expected.codes)
    assert torch.equal(forward_only.alpha, expected.alpha)
    assert forward_only.codes_t.numel() == 0

    boundary_weight = torch.tensor([[7.0] * 8 + [13.0] * 8], device="cuda", dtype=torch.bfloat16)
    boundary_expected = pack_twn_reference(boundary_weight, -1)
    boundary_actual = ternary_kernels.pack_twn(boundary_weight, -1)
    assert torch.equal(boundary_actual.codes, boundary_expected.codes)
    assert torch.equal(boundary_actual.alpha, boundary_expected.alpha)


@pytest.mark.gpu
@requires_native_cuda
def test_native_dense_forward_and_all_gradients_match_fake_quant():
    torch.manual_seed(1)
    x_ref = torch.randn(37, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    w_ref = torch.randn(23, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b_ref = torch.randn(23, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_native = x_ref.detach().clone().requires_grad_(True)
    w_native = w_ref.detach().clone().requires_grad_(True)
    b_native = b_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(37, 23, device="cuda", dtype=torch.bfloat16)

    expected = F.linear(x_ref, twn_quantize_ste(w_ref, in_dim=-1), b_ref)
    actual = native_packed_linear(
        x_native, w_native, b_native, cache=PackedTWNCache(), orientation="test_dense"
    )
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_native.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(w_native.grad, w_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(b_native.grad, b_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@requires_native_cuda
def test_native_dense_autocast_noncontiguous_leading_dims_and_gradients():
    torch.manual_seed(13)
    x_ref = (
        torch.randn(2, 3, 48, device="cuda", dtype=torch.float32)
        .transpose(0, 1)
        .detach()
        .requires_grad_(True)
    )
    x_native = x_ref.detach().clone(memory_format=torch.preserve_format).requires_grad_(True)
    assert not x_ref.is_contiguous() and not x_native.is_contiguous()
    w_ref = torch.randn(23, 48, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    b_ref = torch.randn(23, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    layer = QuantLinear(
        48,
        23,
        bias=True,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
        device="cuda",
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        layer.weight.copy_(w_ref)
        assert layer.bias is not None
        layer.bias.copy_(b_ref)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected = F.linear(x_ref, twn_quantize_ste(w_ref, in_dim=-1), b_ref)
        actual = layer(x_native)
    assert actual.shape == expected.shape == (3, 2, 23)
    assert actual.dtype is expected.dtype is torch.bfloat16
    assert layer.backend_status()["resolved"] == QuantBackend.native_packed.value
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_native.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(layer.weight.grad, w_ref.grad, atol=2e-2, rtol=2e-2)
    assert layer.bias is not None
    torch.testing.assert_close(layer.bias.grad, b_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@requires_native_cuda
def test_native_dense_empty_batch_forward_and_backward():
    x = torch.empty(0, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bias = torch.randn(17, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = native_packed_linear(
        x, weight, bias, cache=PackedTWNCache(), orientation="empty_dense"
    )
    assert output.shape == (0, 17)
    output.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert torch.count_nonzero(weight.grad) == 0
    assert torch.count_nonzero(bias.grad) == 0


@pytest.mark.gpu
@requires_native_cuda
@pytest.mark.parametrize("in_dim", [1, 2])
def test_native_fixed_grouped_all_orientations(in_dim):
    torch.manual_seed(2)
    e, n, k, o = 4, 9, 32, 20
    logical = torch.randn(e, o, k, device="cuda", dtype=torch.bfloat16)
    weight_ref = logical.movedim(-1, in_dim).detach().requires_grad_(True)
    weight_native = weight_ref.detach().clone().requires_grad_(True)
    x_ref = torch.randn(e, n, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_native = x_ref.detach().clone().requires_grad_(True)
    grad = torch.randn(e, n, o, device="cuda", dtype=torch.bfloat16)
    expected = torch.bmm(
        x_ref, twn_quantize_ste(weight_ref, in_dim=in_dim).movedim(in_dim, -1).transpose(1, 2)
    )
    actual = native_packed_grouped_linear(
        x_native,
        weight_native,
        in_dim=in_dim,
        cache=PackedTWNCache(),
        orientation=f"fixed_in{in_dim}",
    )
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_native.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(weight_native.grad, weight_ref.grad, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
@requires_native_cuda
@pytest.mark.parametrize("in_dim", [1, 2])
def test_native_jagged_grouped_handles_empty_and_imbalanced_experts(in_dim, monkeypatch):
    grouped_mm = getattr(torch, "_grouped_mm", None)
    assert grouped_mm is not None
    grouped_mm_offset_dtypes = []

    def tracked_grouped_mm(left, right, offsets):
        grouped_mm_offset_dtypes.append(offsets.dtype)
        return grouped_mm(left, right, offsets)

    monkeypatch.setattr(torch, "_grouped_mm", tracked_grouped_mm)
    torch.manual_seed(3)
    sizes = torch.tensor([0, 7, 1, 12], device="cuda", dtype=torch.long)
    e, k, o = sizes.numel(), 32, 24
    logical = torch.randn(e, o, k, device="cuda", dtype=torch.bfloat16)
    weight_ref = logical.movedim(-1, in_dim).detach().requires_grad_(True)
    weight_native = weight_ref.detach().clone().requires_grad_(True)
    x_ref = torch.randn(
        int(sizes.sum()), k, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    x_native = x_ref.detach().clone().requires_grad_(True)
    quantized = twn_quantize_ste(weight_ref, in_dim=in_dim).movedim(in_dim, -1)
    expected_parts = []
    start = 0
    for expert, size in enumerate(sizes.tolist()):
        expected_parts.append(F.linear(x_ref[start : start + size], quantized[expert]))
        start += size
    expected = torch.cat(expected_parts)
    actual = native_packed_grouped_linear(
        x_native,
        weight_native,
        in_dim=in_dim,
        cache=PackedTWNCache(),
        orientation=f"jagged_in{in_dim}",
        batch_sizes=sizes,
    )
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x_native.grad, x_ref.grad, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(weight_native.grad, weight_ref.grad, atol=2e-2, rtol=2e-2)
    assert grouped_mm_offset_dtypes == [torch.int32]


@pytest.mark.gpu
@requires_native_cuda
def test_native_jagged_grouped_all_experts_empty():
    e, k, o = 4, 32, 24
    sizes = torch.zeros(e, device="cuda", dtype=torch.long)
    x = torch.empty(0, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(e, o, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    output = native_packed_grouped_linear(
        x,
        weight,
        in_dim=2,
        cache=PackedTWNCache(),
        orientation="jagged_all_empty",
        batch_sizes=sizes,
    )
    assert output.shape == (0, o)
    output.sum().backward()
    assert x.grad is not None and x.grad.shape == x.shape
    assert weight.grad is not None and torch.count_nonzero(weight.grad) == 0


@pytest.mark.gpu
@requires_native_cuda
def test_native_dense_compile_and_gradient_accumulation_reuse_cache():
    layer = QuantLinear(
        32, 16, backend=QuantBackend.native_packed, dtype=torch.bfloat16, device="cuda"
    )
    compiled = torch.compile(layer)
    for _ in range(2):
        compiled(torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)).sum().backward()
    assert layer._native_pack_cache.misses == 1  # type: ignore[attr-defined]
    assert layer._native_pack_cache.hits >= 1  # type: ignore[attr-defined]
    with torch.no_grad():
        layer.weight.add_(1)
    compiled(torch.randn(8, 32, device="cuda", dtype=torch.bfloat16)).sum().backward()
    assert layer._native_pack_cache.misses == 2  # type: ignore[attr-defined]


def test_quant_config_native_fields_roundtrip():
    config = QuantConfig(
        enabled=True,
        backend=QuantBackend.native_packed,
        fallback_to_fake_quant=False,
    )
    serialized = config.as_config_dict()
    assert serialized["backend"] == QuantBackend.native_packed.value
    assert serialized["ste_policy"] == "identity"
    assert serialized["fallback_to_fake_quant"] is False
