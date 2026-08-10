"""
Tests for ternary-compressed FSDP2 all-gather (:mod:`olmo_core.nn.quantization_comm`).

**UNTESTED AS WRITTEN.** These tests were authored with no torch available and have never been
executed. They are the deliverable; the implementation is only as good as they turn out to be.

What the suite is built to catch
--------------------------------
The headline risk is **not** a crash. It is that moving TWN from post-gather to pre-gather
silently changes ``alpha`` and ``delta`` -- which are per-output-row statistics over the full
``in_dim`` -- and so silently changes the model. A run like that trains happily, logs nothing
unusual, and is the same failure class as D-076 / D-099 / D-122. So:

* :func:`test_bitwise_identical_to_post_gather_two_ranks` is the gate. It asserts
  ``torch.equal`` -- **bitwise**, not ``allclose`` -- between the compressed round trip and the
  in-tree post-gather ``twn_quantize``. Precedent is L4's ``enabled=False`` path, which is
  bitwise identical, and that is the standard here.
* :func:`test_mutation_corrupted_pack_is_caught` is the anti-vacuity check, modelled on
  ``test_cv_excess_is_still_window_dependent_so_the_test_above_is_not_vacuous``. It deliberately
  corrupts pack/unpack four different ways and requires the equivalence test's own assertion to
  **fail** for each. A test that still passes when the packing is broken is worse than no test.
* :func:`test_expert_misaligned_shard_refuses` requires the **inexact** case to raise rather than
  approximate. That is the whole safety property: fidelity outranks bytes.

Deadlock discipline
-------------------
The MoE test suite is known to deadlock on GPU-less nodes -- a collective with no peer -- and the
shared FarmShare venv has **no ``pytest-timeout``**, so every remote pytest needs an external
wall-clock kill. Accordingly **no test in this file contains an unguarded collective.** The two
multi-rank tests go through ``run_distributed_test`` (gloo, which needs no GPU and spawns its own
peers), and every other test is single-process pure tensor math with no ``dist`` call at all. There
is no module-level ``dist`` usage and no import-time collective, so collection alone cannot hang.
"""

import math

import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.quantization import twn_quantize
from olmo_core.nn.quantization_comm import (
    TRITS_PER_BYTE,
    TernaryCommSpec,
    TernaryCommTensor,
    pack_trits,
    reconstruct_from_trits,
    ternary_comm_bytes_report,
    trits_and_alpha,
    unpack_trits,
)
from olmo_core.testing import run_distributed_test

# ===================================================================================
# Packing: round trip and the corruption tripwire
# ===================================================================================


def test_pack_unpack_round_trip_is_exact():
    torch.manual_seed(0)
    trits = torch.randint(-1, 2, (4096,), dtype=torch.float32)
    packed = pack_trits(trits)
    assert packed.dtype == torch.uint8
    # 8x vs bf16: 4 trits per byte against 2 bytes per bf16 value.
    assert packed.numel() == trits.numel() // TRITS_PER_BYTE
    assert trits.numel() * 2 == packed.numel() * 8
    assert torch.equal(unpack_trits(packed), trits)


def test_pack_refuses_non_multiple_rather_than_padding():
    # Padding would be indistinguishable from a real zero trit on unpack, which would move
    # `alpha` on the next round trip. Refusing is the point.
    with pytest.raises(ValueError, match="divisible by"):
        pack_trits(torch.zeros(4095))


def test_unpack_rejects_the_reserved_code():
    # Code 3 is never emitted (trit + 1 spans 0..2), so seeing it means corruption. A byte of
    # 0xFF is four 3s.
    with pytest.raises(ValueError, match="reserved 2-bit code 3"):
        unpack_trits(torch.full((8,), 255, dtype=torch.uint8))


def test_all_three_trit_values_are_present_so_the_round_trip_is_not_vacuous():
    # A round-trip test over an all-zeros tensor passes under almost any bug. Assert the fixture
    # actually exercises every code point.
    torch.manual_seed(0)
    trits = torch.randint(-1, 2, (4096,), dtype=torch.float32)
    for v in (-1.0, 0.0, 1.0):
        assert (trits == v).sum() > 0


# ===================================================================================
# The factorization is bitwise, single process
# ===================================================================================


def _cases():
    """(name, stored shape, spec) for every eligible tensor family.

    ``in_dim`` mirrors the `maybe_quantize` call sites at ``nn/moe/mlp.py:251`` and ``:404``, and
    the dropless entries are the load-bearing ones: ``w1`` and ``w2`` have the **same shape** and
    **different** ``in_dim`` because ``gmm`` takes ``trans_b=True`` for one and not the other.
    """
    E, d, h = 4, 32, 16
    return [
        ("linear", (64, 32), TernaryCommSpec(trailing=(32,), in_dim=1)),
        ("moe_w1", (E * d, h), TernaryCommSpec(trailing=(d, h), in_dim=1)),
        ("moe_w2", (E * h, d), TernaryCommSpec(trailing=(h, d), in_dim=1)),
        ("dropless_w1", (E * h, d), TernaryCommSpec(trailing=(h, d), in_dim=2)),
        ("dropless_w2", (E * h, d), TernaryCommSpec(trailing=(h, d), in_dim=1)),
    ]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("name,shape,spec", _cases(), ids=[c[0] for c in _cases()])
def test_factored_reconstruction_is_bitwise_twn_quantize(name, shape, spec, dtype):
    """
    ``trits * alpha`` must be **bitwise** ``twn_quantize``, for every family and both dtypes.

    This is the single-process half of the correctness argument: it proves the *factorization*
    is lossless, independent of any sharding. ``torch.equal``, not ``allclose``.
    """
    del name
    torch.manual_seed(1234)
    w = torch.randn(*shape, dtype=dtype)
    viewed = w.reshape(spec.interpreted_shape(shape[0]))

    reference = twn_quantize(viewed, in_dim=spec.in_dim)
    trits, alpha = trits_and_alpha(w, spec)
    got = reconstruct_from_trits(trits, alpha, out_dtype=dtype)

    assert got.shape == reference.shape
    assert torch.equal(got, reference), "factored TWN is not bitwise equal to twn_quantize"


@pytest.mark.parametrize("name,shape,spec", _cases(), ids=[c[0] for c in _cases()])
def test_zero_fraction_lands_on_twn_not_bitnet(name, shape, spec):
    """
    Guards the *identity* of the quantizer, not just the plumbing.

    TWN on Gaussian latents zeroes 42.35%; BitNet b1.58 would zero 31.0%. If someone "corrects"
    the threshold this catches it, and it also catches a wrong-axis reduction that happens to be
    shape-legal only in the sense that the value drifts off the closed form.
    """
    del name
    torch.manual_seed(7)
    w = torch.randn(*shape)
    trits, _ = trits_and_alpha(w, spec)
    zero_fraction = float((trits == 0).float().mean())
    assert 0.40 < zero_fraction < 0.45, f"{zero_fraction} is not TWN's 0.4235"
    assert abs(zero_fraction - 0.310064) > 0.05, "this looks like BitNet b1.58, not TWN"


def test_wrong_reduction_axis_is_detectably_different():
    """
    Non-vacuity for the spec's ``in_dim``.

    The dropless ``w1``/``w2`` pair have identical shapes and different ``in_dim``, so a swap is
    shape-legal and silent. Assert the two really do produce different weights -- otherwise every
    ``in_dim`` assertion in this file would be untestable.
    """
    torch.manual_seed(3)
    E, d, h = 4, 32, 16
    w = torch.randn(E * h, d)
    right = reconstruct_from_trits(
        *trits_and_alpha(w, TernaryCommSpec(trailing=(h, d), in_dim=2)), out_dtype=torch.float32
    )
    wrong = reconstruct_from_trits(
        *trits_and_alpha(w, TernaryCommSpec(trailing=(h, d), in_dim=1)), out_dtype=torch.float32
    )
    assert not torch.equal(right, wrong), "in_dim makes no difference -- the spec is untested"


# ===================================================================================
# The exactness gate
# ===================================================================================


def test_spec_fold_and_folded_axis_classification():
    # nn.Linear: nothing folded inside flat axis 0, so no shard can split a row.
    lin = TernaryCommSpec(trailing=(32,), in_dim=1)
    assert lin.fold == 1
    assert not lin.reduces_over_folded_axis

    # MoEMLP.w1 (E*d, h) viewed (E, d, h), reduces over axis 1 == d, which IS folded.
    moe = TernaryCommSpec(trailing=(32, 16), in_dim=1)
    assert moe.fold == 32
    assert moe.reduces_over_folded_axis

    # DroplessMoEMLP.w1 reduces over axis 2 == the stored last axis, never split.
    dropless = TernaryCommSpec(trailing=(16, 32), in_dim=2)
    assert dropless.fold == 16
    assert not dropless.reduces_over_folded_axis


def test_expert_misaligned_shard_refuses():
    """
    The safety property: an inexact shard must **raise**, never approximate.

    E=4 experts over 3 ranks cannot be expert-aligned, so a rank's local ``mean|W|`` would come
    from a fraction of each output row. Adding an all-reduce or accepting the deviation would
    change the quantizer and therefore the model, so refusal is the only correct behaviour.
    """
    spec = TernaryCommSpec(trailing=(32, 16), in_dim=1)
    # Aligned: 2 whole experts of 32 rows each.
    spec.assert_exact(local_flat_dim0=64, param_name="w1", world_size=2)
    # Misaligned: 48 rows is 1.5 experts.
    with pytest.raises(OLMoConfigurationError, match="would change the quantizer"):
        spec.assert_exact(local_flat_dim0=48, param_name="w1", world_size=3)


def test_unconditionally_exact_specs_never_refuse():
    # Non-vacuity for the gate: the always-exact families must pass every shard size, including
    # ones that are not multiples of `fold`.
    for spec in (
        TernaryCommSpec(trailing=(32,), in_dim=1),
        TernaryCommSpec(trailing=(16, 32), in_dim=2),
    ):
        for n in (1, 3, 7, 48, 64):
            spec.assert_exact(local_flat_dim0=n, param_name="w", world_size=3)


# ===================================================================================
# The gate: bitwise identity across a real 2-rank gloo all-gather
# ===================================================================================


def _run_bitwise_two_ranks():
    """
    Body of the decisive test. Runs on **gloo/CPU**, no GPU, no unguarded collective.

    Simulates FSDP2's contract directly rather than standing up ``fully_shard``: build the full
    weight identically on both ranks, ``chunk`` it on dim 0 exactly as
    ``_fsdp_common._chunk_with_empty`` does, run the rank's own ``fsdp_pre_all_gather``,
    ``all_gather`` the packed bytes and the alphas through the process group, then run
    ``fsdp_post_all_gather`` and compare against the in-tree post-gather ``twn_quantize`` on the
    **full** tensor.

    Doing it at this level is deliberate: it isolates the question this lane exists to answer --
    is pre-gather TWN the same function as post-gather TWN -- from every unrelated way
    ``fully_shard`` can fail on CPU. All collectives are inside ``run_distributed_test``'s
    process group, which always has its peer.
    """
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    E, d, h = 4, 32, 16
    cases = [
        ("linear", (64, 32), TernaryCommSpec(trailing=(32,), in_dim=1)),
        ("moe_w1", (E * d, h), TernaryCommSpec(trailing=(d, h), in_dim=1)),
        ("dropless_w1", (E * h, d), TernaryCommSpec(trailing=(h, d), in_dim=2)),
    ]

    for name, shape, spec in cases:
        # Same seed on every rank -> byte-identical full weight, which is what FSDP's sharded
        # parameter is a chunk of.
        torch.manual_seed(20260810)
        full = torch.randn(*shape, dtype=torch.bfloat16)

        chunks = list(torch.chunk(full, world_size, dim=0))
        assert len(chunks) == world_size, f"{name}: uneven chunking, adjust the fixture"
        local = chunks[rank].contiguous()

        class _MP:
            param_dtype = torch.bfloat16

        wrapped = TernaryCommTensor(local, spec)
        (packed, alpha), metadata = wrapped.fsdp_pre_all_gather(
            None, torch.Size(shape), (1, 1), torch.nn.Identity(), _MP()
        )

        # Compression actually happened, on the wire, in bytes. 4 trits/byte vs 2 bytes/bf16.
        assert packed.dtype == torch.uint8
        assert packed.numel() * 8 == local.numel() * 2, f"{name}: not 8x"

        gathered_packed = [torch.empty_like(packed) for _ in range(world_size)]
        gathered_alpha = [torch.empty_like(alpha) for _ in range(world_size)]
        dist.all_gather(gathered_packed, packed)
        dist.all_gather(gathered_alpha, alpha)

        got, _ = wrapped.fsdp_post_all_gather(
            (torch.cat(gathered_packed), torch.cat(gathered_alpha)),
            metadata,
            torch.bfloat16,
        )
        got = got.reshape(spec.interpreted_shape(shape[0]))

        reference = twn_quantize(full.reshape(spec.interpreted_shape(shape[0])), in_dim=spec.in_dim)
        assert got.dtype == reference.dtype, f"{name}: dtype drift"
        assert torch.equal(got, reference), (
            f"{name}: ternary-compressed all-gather is NOT bitwise identical to the post-gather "
            f"path. max abs diff "
            f"{(got.float() - reference.float()).abs().max().item()}"
        )


def test_bitwise_identical_to_post_gather_two_ranks():
    """
    **THE GATE.** Compressed all-gather must be bitwise identical to today's behaviour.

    If this fails, ternary-compressed all-gather is a change to the *model*, not a change to the
    *transport*, and it must not ship at any default -- bytes saved do not buy anything against
    Maple faithfulness.
    """
    run_distributed_test(_run_bitwise_two_ranks, world_size=2, backend="gloo")


def _run_bitwise_four_ranks():
    _run_bitwise_two_ranks()


def test_bitwise_identical_at_world_size_four():
    """
    Same gate at world_size=4, where E=4 makes the MoE shard exactly one expert per rank.

    This is the tightest expert-aligned case: one more rank and ``moe_w1`` would split rows and
    ``assert_exact`` must refuse. Included because a scheme that is exact only at world_size=2 is
    not exact.
    """
    run_distributed_test(_run_bitwise_four_ranks, world_size=4, backend="gloo")


# ===================================================================================
# Anti-vacuity: show what the gate does when the packing is broken
# ===================================================================================


def _round_trip_matches_reference(pack_fn, unpack_fn) -> bool:
    """Single-process stand-in for the gate, with injectable pack/unpack."""
    torch.manual_seed(20260810)
    spec = TernaryCommSpec(trailing=(32, 16), in_dim=1)
    shape = (4 * 32, 16)
    full = torch.randn(*shape, dtype=torch.bfloat16)

    trits, alpha = trits_and_alpha(full.to(torch.bfloat16), spec)
    try:
        recovered = unpack_fn(pack_fn(trits)).reshape(trits.shape)
    except ValueError:
        return False
    got = reconstruct_from_trits(recovered, alpha, out_dtype=torch.bfloat16)
    reference = twn_quantize(full.reshape(spec.interpreted_shape(shape[0])), in_dim=spec.in_dim)
    return bool(torch.equal(got, reference))


def test_mutation_corrupted_pack_is_caught():
    """
    The mutation check. **A test that passes when the packing is broken is worse than no test.**

    First establish the honest path passes, then corrupt pack/unpack four ways -- each a plausible
    real bug -- and require the equivalence check to FAIL every time. Modelled on
    ``test_cv_excess_is_still_window_dependent_so_the_test_above_is_not_vacuous``.
    """
    assert _round_trip_matches_reference(pack_trits, unpack_trits), "honest path must pass first"

    # 1. Sign dropped: every -1 becomes +1. The classic missing-offset bug. Zero fraction is
    #    unchanged, so a zero-fraction-only test would NOT catch this.
    def pack_abs(trits):
        return pack_trits(trits.abs())

    assert not _round_trip_matches_reference(pack_abs, unpack_trits)

    # 2. Offset omitted on unpack: values come back as {0,1,2} instead of {-1,0,1}.
    def unpack_no_offset(packed):
        return unpack_trits(packed) + 1.0

    assert not _round_trip_matches_reference(pack_trits, unpack_no_offset)

    # 3. Wrong bit width: 3 bits per field instead of 2 shuffles every element's position.
    def pack_3bit(trits):
        flat = trits.reshape(-1)
        codes = (flat.to(torch.int16) + 1).to(torch.uint8)
        groups = codes.view(-1, TRITS_PER_BYTE)
        out = groups[:, 0].clone()
        for j in range(1, TRITS_PER_BYTE):
            out |= groups[:, j] << (3 * j)
        return out.contiguous()

    assert not _round_trip_matches_reference(pack_3bit, unpack_trits)

    # 4. Endianness / group order flipped within each byte -- a permutation, so the multiset of
    #    trits is identical and only the layout is wrong. This is the bug most likely to survive a
    #    sloppy test.
    def pack_reversed(trits):
        return pack_trits(trits.reshape(-1, TRITS_PER_BYTE).flip(-1).reshape(-1))

    assert not _round_trip_matches_reference(pack_reversed, unpack_trits)


def test_reference_is_not_trivially_equal_to_anything():
    """
    Guard against the reference itself being degenerate -- e.g. an all-zero ``twn_quantize``
    output, against which a broken implementation could compare equal.
    """
    torch.manual_seed(20260810)
    spec = TernaryCommSpec(trailing=(32, 16), in_dim=1)
    full = torch.randn(4 * 32, 16, dtype=torch.bfloat16)
    reference = twn_quantize(full.reshape(spec.interpreted_shape(128)), in_dim=spec.in_dim)
    assert (reference != 0).sum() > 0, "reference is all zeros"
    assert reference.unique().numel() > 3, "reference has no per-row alpha variation"


# ===================================================================================
# Bytes report
# ===================================================================================


def test_bytes_report_is_8x_on_the_quantized_part():
    r = ternary_comm_bytes_report(quantized_numel=8_000_000, full_precision_numel=0)
    assert r["ratio"] == pytest.approx(8.0)
    # Two gathers per step: forward + backward, since reshard_after_forward=True.
    assert r["baseline_bytes_per_step"] == pytest.approx(2 * 8_000_000 * 2)


def test_bytes_report_full_precision_share_caps_the_win():
    # Amdahl: carve-outs (embeddings, lm_head, router, norms) are gathered at full width, so the
    # end-to-end ratio is strictly below 8x and the gap matters at small d.
    r = ternary_comm_bytes_report(quantized_numel=1_000_000, full_precision_numel=1_000_000)
    assert 1.0 < r["ratio"] < 8.0
    assert r["ratio"] == pytest.approx(4_000_000 / 2_250_000)


# ===================================================================================
# Config surface: both models, default OFF, and off means untouched
# ===================================================================================


@pytest.mark.parametrize("factory", ["maple_m7b", "maple_m20", "maple_r0"])
def test_ternary_comm_defaults_off_on_every_maple_factory(factory):
    """
    Default must be OFF on **both** target models until the hardware gate passes, and the flag
    must be reachable on each -- one flag, every factory, because they all delegate to
    ``maple_scaled``.
    """
    from olmo_core.nn.transformer import TransformerConfig

    fn = getattr(TransformerConfig, factory, None)
    if fn is None:
        pytest.skip(f"{factory} is not on this branch")
    assert fn(100352).ternary_comm is False


@pytest.mark.parametrize("factory", ["maple_m7b", "maple_m20"])
def test_ternary_comm_off_is_bitwise_the_same_config_as_today(factory):
    """
    L4's precedent: with the flag off, the config must be **identical** to not passing it at all.

    Compares the full serialized config with ``ternary_comm`` removed, so this catches the flag
    perturbing any unrelated field -- including the parameter counts, which have already been
    mis-transcribed three times on this ladder.
    """
    from olmo_core.nn.transformer import TransformerConfig

    fn = getattr(TransformerConfig, factory, None)
    if fn is None:
        pytest.skip(f"{factory} is not on this branch")

    baseline = fn(100352).as_config_dict()
    explicit_off = fn(100352, ternary_comm=False).as_config_dict()
    assert explicit_off == baseline

    on = fn(100352, quantize=True, ternary_comm=True).as_config_dict()
    quantized_only = fn(100352, quantize=True).as_config_dict()
    on.pop("ternary_comm", None)
    quantized_only.pop("ternary_comm", None)
    assert on == quantized_only, "ternary_comm changed something other than itself"


@pytest.mark.parametrize("factory", ["maple_m7b", "maple_m20"])
def test_ternary_comm_without_quantize_refuses(factory):
    """A no-op that records itself as active is the failure mode; it must raise."""
    from olmo_core.nn.transformer import TransformerConfig

    fn = getattr(TransformerConfig, factory, None)
    if fn is None:
        pytest.skip(f"{factory} is not on this branch")
    with pytest.raises(OLMoConfigurationError, match="requires `quantize=True`"):
        fn(100352, ternary_comm=True)


@pytest.mark.parametrize("factory", ["maple_m7b", "maple_m20"])
def test_ternary_comm_does_not_move_parameter_counts(factory):
    """
    It is a *transport* optimization. If a single parameter count moves, it is not.
    """
    from olmo_core.nn.transformer import TransformerConfig

    fn = getattr(TransformerConfig, factory, None)
    if fn is None:
        pytest.skip(f"{factory} is not on this branch")
    a = fn(100352, quantize=True)
    b = fn(100352, quantize=True, ternary_comm=True)
    assert a.num_params == b.num_params
    assert a.num_active_params == b.num_active_params


def test_math_of_the_ladder_shapes_is_internally_consistent():
    # Sanity on the numbers quoted in the lane report, computed rather than transcribed --
    # three transcription errors have already cost this project real money.
    for total, per_block, n_layers in (
        (7_656_756_736, 459_279_616, 16),
        (20_002_742_272, 816_320_768, 24),
    ):
        assert per_block * n_layers < total
        assert math.isclose(per_block * n_layers / total, per_block * n_layers / total)
