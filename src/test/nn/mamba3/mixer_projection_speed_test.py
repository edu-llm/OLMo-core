import json
import inspect

import pytest
import torch
import torch.nn as nn

from olmo_core.nn.attention.base import SequenceMixerConfig
from olmo_core.nn.mamba3.mixer import (
    Mamba3Mixer,
    Mamba3MixerConfig,
    mamba3_modules_to_ignore_for_fp8,
)
from olmo_core.nn.transformer.init import InitMethod


def _mixer(*, fused: bool, block_size: int = 3, rank: int = 3) -> Mamba3Mixer:
    return Mamba3Mixer(
        d_model=12,
        n_heads=3,
        head_dim=4,
        d_state=6,
        n_groups=1,
        mimo_rank=rank,
        rotation_block_size=block_size,
        fuse_input_projections=fused,
        prefer_official_kernel=False,
        dtype=torch.float64,
    )


def _initialize(module: Mamba3Mixer, seed: int = 123) -> None:
    module.init_weights(
        init_method=InitMethod.normal,
        d_model=12,
        block_idx=0,
        num_blocks=2,
        generator=torch.Generator().manual_seed(seed),
    )


def _canonical_tensors(
    module: Mamba3Mixer, tensors: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    canonical = dict(tensors)
    if not module.fuse_input_projections:
        return canonical

    inner = module.n_heads * module.head_dim
    bc_out = module.n_groups * module.mimo_rank * module.d_state
    theta_out = module.n_groups * module.n_rotation_blocks * module.angles_per_block
    canonical["in_x.weight"], canonical["in_z.weight"] = canonical.pop("in_xz.weight").split(
        (inner, inner), dim=0
    )
    canonical["in_B.weight"], canonical["in_C.weight"] = canonical.pop("in_bc.weight").split(
        (bc_out, bc_out), dim=0
    )
    if "in_bc.bias" in canonical:
        canonical["in_B.bias"], canonical["in_C.bias"] = canonical.pop("in_bc.bias").split(
            (bc_out, bc_out), dim=0
        )
    (
        canonical["dt_proj.weight"],
        canonical["lam_proj.weight"],
        canonical["theta_proj.weight"],
    ) = canonical.pop("in_dynamics.weight").split(
        (module.n_heads, module.n_heads, theta_out), dim=0
    )
    return canonical


@pytest.mark.parametrize("block_size", [2, 3])
@pytest.mark.parametrize("rank", [1, 3])
def test_fused_projection_forward_and_all_gradient_parity(block_size: int, rank: int):
    reference = _mixer(fused=False, block_size=block_size, rank=rank)
    fused = _mixer(fused=True, block_size=block_size, rank=rank)
    _initialize(reference)
    fused.load_state_dict(reference.state_dict(), strict=True)

    generator = torch.Generator().manual_seed(456)
    x_reference = torch.randn(
        2, 5, 12, dtype=torch.float64, generator=generator, requires_grad=True
    )
    x_fused = x_reference.detach().clone().requires_grad_(True)
    output_grad = torch.randn(2, 5, 12, dtype=torch.float64, generator=generator)

    y_reference = reference(x_reference)
    y_fused = fused(x_fused)
    torch.testing.assert_close(y_fused, y_reference, rtol=1e-10, atol=1e-12)
    y_reference.backward(output_grad)
    y_fused.backward(output_grad)
    torch.testing.assert_close(x_fused.grad, x_reference.grad, rtol=1e-10, atol=1e-12)

    reference_grads = {name: parameter.grad for name, parameter in reference.named_parameters()}
    fused_grads = _canonical_tensors(
        fused, {name: parameter.grad for name, parameter in fused.named_parameters()}
    )
    assert fused_grads.keys() == reference_grads.keys()
    for name, expected in reference_grads.items():
        assert expected is not None and fused_grads[name] is not None
        torch.testing.assert_close(fused_grads[name], expected, rtol=1e-10, atol=1e-12, msg=name)


def test_fused_projection_initialization_and_checkpoint_conversion_are_exact():
    reference = _mixer(fused=False)
    fused = _mixer(fused=True)
    restored = _mixer(fused=False)
    _initialize(reference, seed=789)
    _initialize(fused, seed=789)

    expected = reference.state_dict()
    actual = _canonical_tensors(fused, fused.state_dict())
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=name)

    fused.load_state_dict(expected, strict=True)
    restored.load_state_dict(fused.state_dict(), strict=True)
    for name, tensor in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)

    nested_reference = nn.Sequential(reference)
    nested_fused = nn.Sequential(_mixer(fused=True))
    nested_fused.load_state_dict(nested_reference.state_dict(), strict=True)


def test_fused_projection_config_default_roundtrip_launch_count_and_fp8_policy():
    default = Mamba3MixerConfig(n_heads=3, d_state=6)
    assert default.fuse_input_projections is None
    assert "fuse_input_projections" not in default.as_config_dict()

    config = Mamba3MixerConfig(
        n_heads=3,
        head_dim=4,
        d_state=6,
        n_groups=1,
        mimo_rank=3,
        rotation_block_size=3,
        fuse_input_projections=True,
        prefer_official_kernel=False,
    )
    rebuilt = SequenceMixerConfig.from_dict(json.loads(json.dumps(config.as_config_dict())))
    assert rebuilt == config
    fused = rebuilt.build(12, layer_idx=0, n_layers=2, init_device="meta")
    reference = Mamba3MixerConfig.from_dict(
        {**config.as_config_dict(), "fuse_input_projections": False}
    ).build(12, layer_idx=0, n_layers=2, init_device="meta")
    assert sum(parameter.numel() for parameter in fused.parameters()) == sum(
        parameter.numel() for parameter in reference.parameters()
    )
    assert rebuilt.num_params(12) == sum(parameter.numel() for parameter in fused.parameters())
    assert mamba3_modules_to_ignore_for_fp8(fused) == {"in_bc", "in_dynamics"}

    runtime = _mixer(fused=True)
    calls = []
    handles = [
        child.register_forward_hook(lambda _module, _inputs, _output, name=name: calls.append(name))
        for name, child in runtime.named_children()
        if isinstance(child, nn.Linear) and name != "out_proj"
    ]
    try:
        runtime(torch.randn(2, 5, 12, dtype=torch.float64))
    finally:
        for handle in handles:
            handle.remove()
    assert calls == ["in_xz", "in_bc", "in_dynamics"]


def test_fusion_escape_hatch_is_appended_after_legacy_config_fields():
    parameters = list(inspect.signature(Mamba3MixerConfig).parameters)
    assert parameters.index("dtype") < parameters.index("fuse_input_projections")


def test_fused_projection_path_fails_closed_for_tensor_parallelism():
    class _Mesh:
        def size(self):
            return 2

    module = _mixer(fused=True)
    with pytest.raises(NotImplementedError, match="Tensor parallelism"):
        module.apply_tp(_Mesh(), float8_enabled=False)  # type: ignore[arg-type]
