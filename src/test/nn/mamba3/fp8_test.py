"""
Tests for the Mamba-3 Float8 recipe helpers.

FP8 on a Mamba-3 hybrid is not "quantize every linear". The mixer has two very different classes
of projection: a few big feed-forward-shaped GEMMs (``in_x``, ``in_z``, ``out_proj``) that carry
almost all the layer's FLOPs and tolerate fp8 fine, and several small projections that *parameterise
the state-space recurrence itself* (``in_B``, ``in_C``, ``dt_proj``, ``lam_proj``, ``theta_proj``).
The small ones contribute negligible FLOPs but decide the SSM's decay, its trapezoidal blend, and --
for ``b >= 3`` -- the non-solvable rotation that the whole NC^1 modification exists to produce.
Rounding those to fp8 is all risk and no speed, so the recipe must be able to name them exactly and
leave them in high precision.

These tests pin *which* projections are treated as sensitive and prove the helper that turns that
decision into the fully-qualified module names ``Float8Config.modules_to_ignore`` needs.
"""

import pytest
import torch
import torch.nn as nn

from olmo_core.float8 import AOFloat8LinearRecipe, Float8Config
from olmo_core.nn.mamba3 import Mamba3Config
from olmo_core.nn.mamba3.mixer import Mamba3Mixer, mamba3_modules_to_ignore_for_fp8
from olmo_core.testing import requires_gpu

# The big, fp8-friendly GEMMs -- the complement of the sensitive set. Named here so the
# completeness assertion below fails loudly if a new linear is added to the mixer without being
# classified either way.
_FP8_SAFE_PROJECTIONS = ("in_x", "in_z", "out_proj")


def _tiny_hybrid_config():
    """A 4-layer 1:3 hybrid config with all-``%16`` linear dims, so fp8 conversion is eligible."""
    return Mamba3Config.mamba3_hybrid_like(
        d_model=64,
        vocab_size=128,
        n_layers=4,
        n_heads=4,
        intermediate_size=128,
        mamba_n_heads=4,
        mamba_head_dim=16,
        d_state=16,
        n_groups=1,
        mimo_rank=2,
    )


def _tiny_hybrid(device: str = "meta"):
    """A 4-layer 1:3 hybrid (3 Mamba-3 mixers + 1 attention). ``meta`` allocates nothing."""
    model = _tiny_hybrid_config().build(init_device=device)
    if device != "meta":
        model.init_weights(device=torch.device(device))
    return model


def test_fp8_sensitive_projections_are_the_ssm_parameterizing_ones():
    """
    Pin the decision itself: the sensitive set is exactly the projections that feed the recurrence.

    ``in_B``/``in_C`` are the state read/write matrices, ``dt_proj`` the timestep, ``lam_proj`` the
    trapezoidal blend, ``theta_proj`` the rotation angles. Everything else is a plain GEMM.
    """
    assert Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS == (
        "in_B",
        "in_C",
        "dt_proj",
        "lam_proj",
        "theta_proj",
    )


def test_every_mixer_linear_is_classified_as_sensitive_or_safe():
    """
    No projection may be silently unclassified.

    If someone adds a new ``nn.Linear`` to the mixer, it must be deliberately placed in the
    sensitive set or the safe set; this catches the "added a linear, forgot fp8 hurts it" bug.
    """
    mixer = next(m for m in _tiny_hybrid().modules() if isinstance(m, Mamba3Mixer))
    linear_leaves = {name for name, m in mixer.named_children() if isinstance(m, nn.Linear)}

    classified = set(Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS) | set(_FP8_SAFE_PROJECTIONS)
    assert linear_leaves == classified


def test_modules_to_ignore_collects_every_mixer_projection_fqn():
    """
    The helper must yield real, fully-qualified ``nn.Linear`` names -- one per sensitive projection
    per Mamba-3 layer -- and never a big GEMM.

    ``Float8Config.apply_float8_linear`` raises if any ignored name fails to match a module, so an
    fqn that is wrong or stale is a hard error at conversion, not a silent miss. Deriving the names
    from the built model (rather than hardcoding a layer count) is what keeps them correct.
    """
    model = _tiny_hybrid()
    mixers = [m for m in model.modules() if isinstance(m, Mamba3Mixer)]
    assert len(mixers) == 3, "the 1:3 tiny hybrid should have three Mamba-3 mixers"

    ignore = mamba3_modules_to_ignore_for_fp8(model)

    # One entry per (mixer, sensitive projection).
    assert len(ignore) == len(mixers) * len(Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS)

    for fqn in ignore:
        leaf = fqn.rsplit(".", 1)[-1]
        assert leaf in Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS, f"{fqn} is not a sensitive projection"
        assert isinstance(model.get_submodule(fqn), nn.Linear), f"{fqn} is not an nn.Linear"

    # The big GEMMs must never be ignored -- that is where the fp8 speedup comes from.
    for fqn in ignore:
        assert fqn.rsplit(".", 1)[-1] not in _FP8_SAFE_PROJECTIONS

    # Sanity that the naming actually resolves to each sensitive projection at least once.
    ignored_leaves = {fqn.rsplit(".", 1)[-1] for fqn in ignore}
    assert ignored_leaves == set(Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS)


def test_modules_to_ignore_is_empty_without_mamba_layers():
    """A model with no Mamba-3 mixer has no SSM projections to protect, so the set must be empty."""
    plain = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
    assert mamba3_modules_to_ignore_for_fp8(plain) == set()


# ------------------------------------------------------------------------------------------
# The recipe composes with the real fp8 conversion (cross-cutting validation, no AWS)
# ------------------------------------------------------------------------------------------


def _is_float8_linear(module: nn.Module) -> bool:
    # Match by name to avoid depending on torchao's internal import path.
    return "Float8Linear" in type(module).__name__


def test_apply_fp8_protects_the_ssm_projections_and_converts_the_big_gemms():
    """
    End-to-end on the real conversion path: the ignore set the helper produces must both (a) be
    accepted by ``apply_fp8`` -- which raises on any ignore name that fails to match a module -- and
    (b) actually leave the SSM projections as plain ``nn.Linear`` while the big GEMMs become fp8.

    This is the composition that matters: a correct ignore list that ``apply_fp8`` rejects, or a
    conversion that silently fp8s ``theta_proj`` anyway, both defeat the point. Runs on CPU; the
    conversion only swaps module classes, the fp8 matmul itself is never executed here.
    """
    torch.manual_seed(0)
    model = _tiny_hybrid(device="cpu")

    ignore = mamba3_modules_to_ignore_for_fp8(model)
    model.apply_fp8(
        Float8Config(ao_recipe=AOFloat8LinearRecipe.rowwise, modules_to_ignore=list(ignore))
    )

    mixers = [m for m in model.modules() if isinstance(m, Mamba3Mixer)]
    assert mixers, "expected Mamba-3 mixers in the hybrid"
    for mixer in mixers:
        for proj in Mamba3Mixer.FP8_SENSITIVE_PROJECTIONS:
            module = getattr(mixer, proj)
            assert isinstance(module, nn.Linear), f"{proj} disappeared"
            assert not _is_float8_linear(module), f"{proj} was converted to fp8 but must not be"
        for proj in _FP8_SAFE_PROJECTIONS:
            assert _is_float8_linear(getattr(mixer, proj)), f"{proj} should be fp8 for the speedup"


@requires_gpu
def test_fp8_forward_backward_stays_finite_and_close_to_bf16():
    """
    The accuracy sanity check the ablation actually rests on, at a size that fits any GPU.

    fp8's failure mode is not a small bias, it is NaN/Inf or divergence. So the load-bearing
    assertion is that the fp8 loss and every gradient are finite; that it also lands within a few
    percent of the bf16 loss on the same batch is corroborating evidence that protecting the SSM
    projections kept the recurrence intact. Uses the ``rowwise`` recipe because it runs on any
    fp8-capable GPU (MXFP8 needs SM100) and lower-bounds MXFP8's accuracy.
    """
    torch.manual_seed(0)
    device = "cuda"
    model = _tiny_hybrid(device=device)

    input_ids = torch.randint(0, 128, (2, 32), device=device)
    labels = torch.randint(0, 128, (2, 32), device=device)

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss_bf16 = model(input_ids, labels=labels).loss.sum().detach().float().item()

    ignore = mamba3_modules_to_ignore_for_fp8(model)
    model.apply_fp8(
        Float8Config(ao_recipe=AOFloat8LinearRecipe.rowwise, modules_to_ignore=list(ignore))
    )

    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        loss_fp8 = model(input_ids, labels=labels).loss.sum()

    assert torch.isfinite(loss_fp8).all(), "fp8 loss is not finite"
    loss_fp8.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "fp8 backward produced no gradients"
    assert all(torch.isfinite(g).all() for g in grads), "fp8 produced a non-finite gradient"

    rel = abs(loss_fp8.detach().float().item() - loss_bf16) / abs(loss_bf16)
    assert rel < 0.15, f"fp8 loss drifted {rel:.3f} from bf16 on the same batch"
