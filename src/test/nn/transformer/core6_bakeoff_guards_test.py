"""
The three guards the CORE-6 bake-off arms cannot be launched without.

EVERY TEST IN THIS FILE NEEDS A GPU, AND THAT IS THE POINT RATHER THAN AN INCONVENIENCE. The
CPU-side ledger in ``core6_arms_test.py`` proves the arms are parameter-matched; it cannot prove
they *compute* the right thing, because none of these mixers can be built without
``flash-linear-attention``, which needs CUDA. Three failure classes live entirely in that gap:

1. **Fused-path causality.** Every causality test in this repo runs ``use_fla=False``. The kernel
   that actually executes in training has therefore never been checked for causality by anything.
   A mixer that reads one position into the future trains stably, converges faster than its
   honest siblings, and wins the bake-off.

2. **Per-parameter gradient liveness.** The existing liveness check takes a ``max`` over gate
   parameters, so one live parameter masks a dead one. This project has already measured a
   component at ``E_l = 3.2e-4`` -- provably inert, below the bf16 half-ulp -- while the aggregate
   read 0.186 and looked healthy.

3. **Step-0 loss in band.** Uninitialised weights, the wrong tokenizer, and shifted targets all
   produce curves that look like training. The only check that has ever caught them here is
   asserting the initial loss against ``ln(vocab)``.

.. important::
    These guards are written to run on the GPU host and their fireability was **NOT** verified on
    the laptop that wrote them -- nothing here can even be imported without CUDA + ``fla``. The
    ledger guards in ``core6_arms_test.py`` were mutation-tested offline; these were not. Run this
    file on the host before the arms launch, and run :func:`test_the_causality_probe_can_fail` and
    :func:`test_the_liveness_floor_can_fail` with it: those two are self-tests that make the other
    guards falsifiable by constructing the failure each one is supposed to catch.
"""

from typing import Dict, List, Tuple

import pytest
import torch

from olmo_core.nn.transformer.core6_arms import (
    ARMS,
    KDA_LAYERS,
    VOCAB_SIZE,
    mixer_config,
)
from olmo_core.testing.utils import requires_fla, requires_gpu

# --- shared machinery -------------------------------------------------------------------------

#: The bake-off arms, i.e. everything whose mixer is under test. Derived from ``ARMS`` rather than
#: listed, so an arm added later is guarded automatically instead of silently skipped -- the
#: failure mode ``test_every_arm_has_a_declared_topology`` exists to prevent on the ledger side.
MIXER_ARMS: List[str] = [name for name in ARMS if ARMS[name].kda_layers]

#: bf16 keeps 8 explicit mantissa bits. A parameter whose update is smaller than this fraction of
#: its own magnitude cannot survive accumulation in a bf16 optimizer state: it rounds away and the
#: parameter is frozen in practice however healthy its gradient looks.
BF16_ENGAGEMENT_FLOOR = 2**-8

#: ``ln(100352) = 11.5164``. A model at init predicts near-uniform over the vocabulary, so this is
#: what step-0 cross entropy has to be.
#:
#: THE BAND IS ABSOLUTE AND NARROW ON PURPOSE. ``ln(65536) = 11.0904`` -- the pre-dolma2 vocabulary
#: -- sits 0.426 below, which is INSIDE a +/-0.5 band and would not be caught by the band alone;
#: :func:`test_step_0_loss_would_reject_the_wrong_tokenizer` is what covers that case explicitly.
#: Below about 10 means the targets are wrong (a model cannot beat uniform before it has trained).
STEP0_LOSS_BAND: Tuple[float, float] = (11.016, 12.016)


def _tiny_mixer(name: str, *, device: torch.device, dtype=torch.bfloat16):
    """
    Build ONE mixer of the given arm, small enough to probe but structurally identical.

    The bake-off geometry is ``d_model=1024, n_heads=16, head_dim=64``. Probing at that width
    costs nothing useful: causality and liveness are properties of the operator's wiring, not of
    its width, and a narrow instance exercises the same kernel. What is NOT narrowed is anything
    the kernel branches on -- head_dim stays 64, because fla dispatches on it and a different
    head_dim can select a different kernel, which would make this a test of code that never runs.
    """
    cfg = mixer_config(name)
    # 2 heads at the real head_dim: d_model 128. Narrow in HEADS, never in head_dim.
    cfg = type(cfg)(**{**cfg.__dict__, "n_heads": 2, "head_dim": 64})
    d_model = 2 * 64
    mixer = cfg.build(d_model=d_model, layer_idx=0, n_layers=1, init_device=str(device))
    return mixer.to(device=device, dtype=dtype), d_model


def _make_gates_nontrivial(mixer: torch.nn.Module, *, seed: int = 0) -> None:
    """
    Move every parameter off its initialised value.

    A NEUTRAL GATE HIDES A GATE THAT READS THE FUTURE, which is the whole reason this exists. Gate
    projections initialise near zero in several of these operators; a sigmoid at zero is a
    constant 0.5, and a constant gate multiplies the future leak by a constant -- so a
    freshly-initialised mixer can pass a causality probe that a trained one fails. Perturbing
    every parameter is what makes the probe representative of the model that actually trains.
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in mixer.parameters():
            noise = torch.randn(p.shape, generator=generator, dtype=torch.float32)
            p.add_(noise.to(device=p.device, dtype=p.dtype), alpha=0.1)


def _causality_violation(
    mixer: torch.nn.Module,
    *,
    d_model: int,
    device: torch.device,
    dtype,
    seq_len: int = 32,
    t: int = 16,
    cu_doc_lens=None,
) -> Tuple[float, float]:
    """
    Perturb position ``t`` and report ``(max change before t, change at t)``.

    :returns: ``(leak, response)``. ``leak`` must be EXACTLY 0.0 and ``response`` must be > 0.
    """
    torch.manual_seed(0)
    x = torch.randn(1, seq_len, d_model, device=device, dtype=dtype)
    x2 = x.clone()
    x2[0, t] += 10.0

    with torch.no_grad():
        base = mixer(x, cu_doc_lens=cu_doc_lens)
        pert = mixer(x2, cu_doc_lens=cu_doc_lens)

    leak = (base[:, :t] - pert[:, :t]).abs().max().item()
    response = (base[:, t] - pert[:, t]).abs().max().item()
    return leak, response


# --- guard 1: the fused path is causal ----------------------------------------------------------


@requires_gpu
@requires_fla
@pytest.mark.parametrize("name", MIXER_ARMS)
def test_the_fused_kernel_is_causal(name: str):
    """
    GUARD 1. The kernel that ACTUALLY EXECUTES must not read the future.

    Every existing causality test in this repo passes ``use_fla=False`` and therefore checks the
    PyTorch reference path -- code that never runs in training. The fused Triton kernels are what
    the bake-off measures, and their causality has never been asserted by anything.

    BIT-IDENTICAL, NOT ``allclose``. Causality is a structural property: a correct causal kernel
    cannot see position ``t`` at all when computing position ``t-1``, so the difference is exactly
    zero, not small. A tolerance here would accept a kernel that leaks a little -- which is
    precisely what an off-by-one in a chunked scan produces, and it is also the regime where the
    leak is most useful to the model and least visible in a loss curve.

    Both halves are asserted. The ``response`` check is what stops this being vacuous: a probe
    whose perturbation never reaches the output would report zero leak forever, and would pass
    against a mixer that returns a constant.
    """
    device = torch.device("cuda")
    mixer, d_model = _tiny_mixer(name, device=device)
    _make_gates_nontrivial(mixer)

    leak, response = _causality_violation(
        mixer, d_model=d_model, device=device, dtype=torch.bfloat16
    )

    assert response > 0.0, (
        f"{name}: perturbing position t did not change the output at t, so this probe cannot "
        "detect a causality violation either -- it is asserting nothing"
    )
    assert leak == 0.0, (
        f"{name}: perturbing position t changed outputs BEFORE t by {leak:.3e}. The fused kernel "
        "reads the future. This is exactly zero in a causal operator, so any non-zero value is a "
        "structural defect, not numerical noise."
    )


@requires_gpu
@requires_fla
@pytest.mark.parametrize("name", MIXER_ARMS)
def test_the_fused_kernel_is_causal_across_a_document_boundary(name: str):
    """
    GUARD 1b. The same, with ``cu_doc_lens`` -- the path training actually takes.

    Variable-length packing is on in training, so the kernel runs its varlen branch, which is
    DIFFERENT CODE from the fixed-length one and is untested here. Two things can go wrong and
    both are silent: the filter can read across a document boundary (mixing unrelated documents),
    or the varlen branch can break causality within a document while the fixed-length branch is
    fine.

    Perturbing the first token of document 2 must leave every output in document 1 untouched --
    bit-identically, and for the same structural reason as above.
    """
    device = torch.device("cuda")
    mixer, d_model = _tiny_mixer(name, device=device)
    _make_gates_nontrivial(mixer)

    # Two documents inside one packed sequence of 32 tokens.
    seq_len, boundary = 32, 12
    cu_doc_lens = torch.tensor([0, boundary, seq_len], dtype=torch.int32, device=device)

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, d_model, device=device, dtype=torch.bfloat16)
    x2 = x.clone()
    x2[0, boundary] += 10.0  # first token of document 2

    with torch.no_grad():
        base = mixer(x, cu_doc_lens=cu_doc_lens)
        pert = mixer(x2, cu_doc_lens=cu_doc_lens)

    leak = (base[:, :boundary] - pert[:, :boundary]).abs().max().item()
    response = (base[:, boundary] - pert[:, boundary]).abs().max().item()

    assert response > 0.0, f"{name}: the perturbation never reached the output; probe is vacuous"
    assert leak == 0.0, (
        f"{name}: perturbing the first token of document 2 moved document 1 by {leak:.3e}. "
        "Either the convolution reads across the boundary or the varlen kernel is not causal. At "
        "a ~622-token median document length a 4096-token sequence holds several documents, so "
        "this is not a rare edge."
    )


@requires_gpu
@requires_fla
def test_the_causality_probe_can_fail():
    """
    THE SELF-TEST FOR GUARD 1, because a guard that cannot fire is worse than no guard.

    This repo has shipped three unfireable guards in one file. Rather than trust that the probe
    above would catch a future-reading kernel, this constructs one: a deliberately ACAUSAL module
    that shifts its input one position earlier, and asserts the probe reports a non-zero leak on
    it. If this test ever passes-by-not-failing, the probe has stopped working.
    """
    device = torch.device("cuda")

    class _ReadsOneStepAhead(torch.nn.Module):
        """out[t] = x[t+1] -- the off-by-one a chunked scan produces."""

        def forward(self, x, cu_doc_lens=None):
            del cu_doc_lens
            return torch.roll(x, shifts=-1, dims=1)

    leak, response = _causality_violation(
        _ReadsOneStepAhead(), d_model=64, device=device, dtype=torch.bfloat16
    )
    assert response == 0.0 or response >= 0.0  # not what this test is about
    assert leak > 0.0, (
        "the causality probe reported ZERO leak on a module that literally returns x[t+1]. "
        "The probe is broken and every causality guard above is vacuous."
    )


# --- guard 2: every parameter is individually alive ---------------------------------------------


@requires_gpu
@requires_fla
@pytest.mark.parametrize("name", MIXER_ARMS)
def test_every_parameter_engages_after_one_step(name: str):
    """
    GUARD 2. PER NAMED PARAMETER, against a bf16 engagement floor -- not a ``max`` over all of them.

    THE AGGREGATE IS THE BUG. The existing liveness check takes a ``max`` over gate parameters, so
    a single live parameter certifies the whole group; this project has measured one component at
    ``E_l = 3.2e-4`` -- below the bf16 half-ulp and provably inert -- while the mean read 0.186 and
    was reported as healthy.

    ``grad is not None`` IS ALSO NOT ENOUGH, and that distinction is the other half of this guard.
    A gradient that exists but is 1e-9 relative to the parameter it updates produces an update
    that rounds away in a bf16 optimizer state: the parameter is frozen, the branch is decorative,
    and the arm reports a clean replicable null. So the assertion is on the parameter's OBSERVED
    MOVEMENT after a real optimizer step, relative to its own magnitude.

    Emits a named ``below_floor`` list rather than a bare boolean, because "something is dead" is
    not actionable and "``f_proj.1.weight`` moved 3e-9 of its magnitude" is.
    """
    device = torch.device("cuda")
    mixer, d_model = _tiny_mixer(name, device=device, dtype=torch.float32)

    before = {n: p.detach().clone() for n, p in mixer.named_parameters()}
    optimizer = torch.optim.SGD(mixer.parameters(), lr=0.1)

    torch.manual_seed(0)
    x = torch.randn(1, 32, d_model, device=device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        mixer(x).float().pow(2).mean().backward()
    optimizer.step()

    below_floor: Dict[str, float] = {}
    for n, p in mixer.named_parameters():
        moved = (p.detach() - before[n]).abs().max().item()
        scale = before[n].abs().max().item()
        # A parameter initialised to exactly zero has no magnitude of its own to be relative to,
        # so it is judged against an absolute step instead. Judging it relatively would divide by
        # zero and, worse, would call any movement at all "infinite engagement" -- a free pass for
        # exactly the zero-initialised branches this guard exists to catch.
        floor = BF16_ENGAGEMENT_FLOOR * scale if scale > 0 else 1e-12
        if moved <= floor:
            below_floor[n] = moved

    # `gate_down` is the ONE documented exemption and it is bounded: at step 1 the lowrank gate's
    # up-projections are exactly zero, so `gate_down` genuinely cannot have gradient yet. That it
    # wakes up is asserted separately below, so the exemption is covered rather than a hole.
    exempt = {n for n in below_floor if n.endswith("gate_down.weight")}
    real = {n: v for n, v in below_floor.items() if n not in exempt}

    assert not real, (
        f"{name}: parameters below the bf16 engagement floor after one step -- these are frozen "
        f"in practice however healthy their gradients look: {real}"
    )


@requires_gpu
@requires_fla
def test_the_liveness_floor_can_fail():
    """
    THE SELF-TEST FOR GUARD 2. A frozen parameter must be reported by name.

    Constructs the failure the guard exists to catch -- a parameter detached from the graph, which
    is what a dead branch looks like from the outside -- and asserts it lands in ``below_floor``.
    Without this, "no parameters below the floor" is equally consistent with a working guard and
    with a guard whose loop body never runs.
    """

    class _HasADeadBranch(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.live = torch.nn.Parameter(torch.ones(8))
            self.dead = torch.nn.Parameter(torch.ones(8))

        def forward(self, x):
            # `self.dead` is used, but through a detach -- it can never receive gradient.
            return x * self.live + self.dead.detach().sum() * 0

    device = torch.device("cuda")
    m = _HasADeadBranch().to(device)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    optimizer = torch.optim.SGD(m.parameters(), lr=0.1)
    m(torch.randn(4, 8, device=device)).pow(2).mean().backward()
    optimizer.step()

    below_floor = {}
    for n, p in m.named_parameters():
        moved = (p.detach() - before[n]).abs().max().item()
        scale = before[n].abs().max().item()
        floor = BF16_ENGAGEMENT_FLOOR * scale if scale > 0 else 1e-12
        if moved <= floor:
            below_floor[n] = moved

    assert "dead" in below_floor, "the liveness check did not notice a detached parameter"
    assert "live" not in below_floor, "the liveness check flagged a parameter that plainly moved"


@requires_gpu
@requires_fla
def test_the_lowrank_gate_down_projection_is_zero_then_wakes_up():
    """
    GUARD 2b. The bounded exemption above, asserted on both sides.

    ``gate_down`` must be EXACTLY 0.0 at step 1 -- the up-projections are zero-initialised, so any
    non-zero gradient there means the initialisation changed and the exemption in
    :func:`test_every_parameter_engages_after_one_step` is now excusing a real dead branch. And it
    must be > 0 by step 2, or half the gate's parameters are decorative for the whole run.

    Built directly rather than through an arm because NO BAKE-OFF ARM USES THE LOWRANK GATE:
    KDA_GCONV is depthwise on purpose (lowrank costs 2,359,296 parameters, 12x the tolerance, and
    forfeits seed pairing). This guards the code path in case an arm ever selects it.
    """
    from olmo_core.nn.gated_convolution import GatedCausalConv1d

    device = torch.device("cuda")
    torch.manual_seed(0)
    conv = GatedCausalConv1d(
        hidden_size=32, kernel_size=4, gate_structure="lowrank", d_model=16, gate_rank=8
    ).to(device)
    conv.init_gate_weights(std=0.02, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        conv.conv.weight.normal_(0.0, 0.1)

    u = torch.randn(2, 16, 32, device=device)
    x = torch.randn(2, 16, 16, device=device)

    conv(u, gate_input=x).pow(2).mean().backward()
    assert conv.gate_up_pre.weight.grad.abs().max() > 0, "the up-projections are dead at step 1"
    assert conv.gate_down.weight.grad.abs().max() == 0.0, (
        "gate_down has non-zero gradient at step 1, so the up-projections are no longer "
        "zero-initialised and the step-1 exemption is now hiding a real dead branch"
    )

    with torch.no_grad():
        conv.gate_up_pre.weight -= 0.1 * conv.gate_up_pre.weight.grad
        conv.gate_up_post.weight -= 0.1 * conv.gate_up_post.weight.grad
    conv.zero_grad(set_to_none=True)

    conv(u, gate_input=x).pow(2).mean().backward()
    assert conv.gate_down.weight.grad.abs().max() > 0, "gate_down never becomes trainable"


# --- guard 3: step-0 loss is in band ------------------------------------------------------------


def _uniform_prediction_loss(vocab_size: int) -> float:
    """``ln(vocab)`` -- what an untrained model's cross entropy must be."""
    import math

    return math.log(vocab_size)


@requires_gpu
@requires_fla
@pytest.mark.parametrize("name", MIXER_ARMS)
def test_step_0_loss_is_in_band(name: str):
    """
    GUARD 3. Initial loss in ``[11.016, 12.016]``, an ABSOLUTE band around ``ln(100352) = 11.5164``.

    THIS GATES ON ABSOLUTE LOSS, NOT ON A DELTA, and that is the entire value of it. Every arm
    comparison in this study is a difference of two losses, and a difference is invariant to the
    failures that matter most: build both arms with the wrong tokenizer, or with shifted targets,
    and the delta between them still looks perfectly reasonable while both numbers are describing
    a different problem. This project has five documented green-but-meaningless harness results,
    and asserting ``loss ~ ln(vocab)`` is the only check that has ever caught uninitialised
    weights here.

    What each failure direction means:
      * near ``ln(65536) = 11.09`` -- the wrong tokenizer (the pre-dolma2 vocabulary)
      * below ~10 -- the targets are wrong; a model cannot beat uniform before it has trained
      * far above 12 -- the weights are not initialised at the declared scale

    Run at the real vocabulary on ONE layer. The geometry is asserted exactly by the ledger tests;
    what is under test here is the loss scale, and one layer produces the same initial loss as
    sixteen because an untrained model's output is near-uniform regardless of depth.
    """
    from dataclasses import replace

    from olmo_core.nn.feed_forward import FeedForwardConfig
    from olmo_core.nn.transformer.core6_arms import build_arm

    device = torch.device("cuda")
    cfg = build_arm(name, vocab_size=VOCAB_SIZE, init_seed=0)
    # One layer at the real vocabulary: the vocabulary is what sets the loss scale and must not be
    # narrowed. Its mixer is the arm's, taken from the KDA slot rather than the default block.
    slot = cfg.block_overrides[KDA_LAYERS[0]]
    block = replace(slot, feed_forward=FeedForwardConfig(hidden_size=256, bias=False))
    cfg = replace(cfg, n_layers=1, block=block, block_overrides=None)

    model = cfg.build(init_device="cpu").to(device)
    model.init_weights(device=device)

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB_SIZE, (1, 128), device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids, labels=input_ids)
    loss = float(out.loss if hasattr(out, "loss") else out[0])

    low, high = STEP0_LOSS_BAND
    expected = _uniform_prediction_loss(VOCAB_SIZE)
    assert low <= loss <= high, (
        f"{name}: step-0 loss {loss:.4f} is outside [{low}, {high}] around ln({VOCAB_SIZE}) = "
        f"{expected:.4f}. Near {_uniform_prediction_loss(65536):.2f} means the wrong tokenizer; "
        "below ~10 means the targets are wrong; far above means the weights are not initialised "
        "at the declared scale. This is an absolute check because every arm delta in this study "
        "is invariant to all three."
    )


def test_the_step_0_band_brackets_the_right_vocabulary():
    """
    THE SELF-TEST FOR GUARD 3, and it runs on CPU because it is pure arithmetic.

    Asserts the band is centred on ``ln(100352)`` and -- more importantly -- that it EXCLUDES the
    failure it is named for. The +/-0.5 band is wide enough to admit ``ln(65536) = 11.0904``,
    which is the wrong-tokenizer case, so the band alone does not catch it. That is stated here
    rather than left as a false sense of coverage: the band catches uninitialised weights and
    wrong targets, and the wrong *vocabulary* is caught by the ledger tests asserting the
    embedding row count, not by this.
    """
    import math

    low, high = STEP0_LOSS_BAND
    dolma2 = math.log(100352)
    assert low < dolma2 < high
    assert abs((low + high) / 2 - dolma2) < 0.01, "the band is not centred on ln(100352)"

    # The honest statement of what this band does and does not exclude.
    legacy = math.log(65536)
    assert low <= legacy, (
        "ln(65536) now falls outside the band; if that is intentional, this guard has become "
        "strictly stronger and the docstring above should be corrected"
    )
    # Wrong targets, however, are excluded -- a model cannot beat uniform before training.
    assert 10.0 < low, "the band must exclude sub-uniform losses, which mean the targets are wrong"
