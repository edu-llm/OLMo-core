"""Arm builder for Exp-2: S1/S2/S3/S4 x {hybrid, allliv} x W in {2,3,4,8}.

Owner: sub-agent A. Binding spec: ``docs/dynconv-review/build/exp2/SPEC.md``.

THE FOUR ARMS (SPEC §1)
----------------------
======  ==========  ===================================================================
 arm     name        mechanism
======  ==========  ===================================================================
 S1      static      static-LIV baseline; ``ShortConv`` exactly as released.
 S2      permuted    **the permuted-conditioning control.** Identical parameters,
                     identical FLOPs and the identical kernel to S4, but ``z`` is
                     shuffled along the sequence axis so it carries zero positional
                     content. The only arm that can separate "input-dependent local
                     composition" from "one more multiplicative degree of freedom" --
                     which is the live confound, not capacity.
 S3      dynqkv      static LIV everywhere + dynamic conv on Q/K/V inside the attention
                     blocks. The *ungated* slot, which is where the published effect was
                     actually measured. **N/A in ``allliv``** -- there are no attention
                     blocks. Raises; never silently substitutes S1.
 S4      dynamic     Dynamic-LIV in every LIV block.
======  ==========  ===================================================================

**Pre-registered decision rule: if S4 beats S1 but does NOT beat S2, the hypothesis is
unsupported.**

THREE TOPOLOGIES, AND WHY ``allliv`` IS NOT OPTIONAL
----------------------------------------------------
``hybrid`` (4 LIV + 2 attention) is LFM2-shaped and is the proposal's configuration. But R5 F5(i)
measured the in-tree probe -- a structurally near-identical hybrid with attention at 2 of 4
layers -- at **100% success, every seed 1.00** on ``N128_D8`` and ``N256_D16``, and the probe's
own README states the cliff is *not* a receptive-field limit "because the attention layers are
global". A metric pinned at 1.00 cannot show a mechanism difference, so ``hybrid`` alone risks a
null for the wrong reason. ``allliv`` (6 LIV, 0 attention) is the ceiling guard: the only
configuration in which the conv mechanism is load-bearing on MQAR.

Attention layer indices in ``hybrid`` are **(2, 5)**, pre-registered. LFM2-16L places attention at
``[2, 5, 8, 10, 12, 14]``; the first six layers of that published pattern are exactly ``{2, 5}``,
so this is LFM2's own late-heavy spacing rather than a tuned choice.

``attn1`` (5 LIV + 1 attention at layer 2) EXISTS BECAUSE BOTH ENDS SATURATED
----------------------------------------------------------------------------
Added 2026-08-05, after measurement rather than in anticipation. The topology axis as originally
pre-registered is saturated at **both** ends, in opposite directions, so no sigma is measurable on
either and the S4-vs-S2 contrast is unreadable on both:

* ``hybrid`` is at **CEILING** -- attention solves MQAR by itself (measured 1.000, every seed).
* ``allliv`` is at **FLOOR** -- measured ``acc 0.0092`` against a ``0.25`` floor at the FULL
  512,000-example budget on the EASIEST rung (``N64_D4``), parked at the ``ln(128) = 4.852``
  wrong-half plateau, one rung *below* the guess-among-D floor. The arithmetic explains it and
  makes it structural rather than empirical: a stacked causal conv reaches ``1 + L(W-1)`` = **13
  tokens at L=6, W=3**, against ~60 needed for ``N64_D4``. **No W in the swept grid reaches even
  the easiest rung**; at the primary operating point the gap is 34x.

So ``attn1`` is the hypothesis that **one attention layer of six is the topology in between** --
enough global reach to make the task solvable, few enough that the conv still carries most of the
recall work.

The index is **2**, derived the same way ``hybrid``'s is rather than tuned: LFM2-16L's published
attention pattern is ``[2, 5, 8, 10, 12, 14]`` and its FIRST element is 2, exactly as ``hybrid``
takes the first two. It is also the right place on independent grounds -- layers 0-1 conv form local
features, layer 2 gathers globally, layers 3-5 conv consume the result -- so the two rationales
agree and neither was fitted to an outcome.

**This topology's operating point is NOT assumed.** It is calibrated on the **baseline arm only**
via :mod:`calibration`, which structurally cannot see a treatment arm (there is no arm flag and
``BASELINE_ARM`` is the only model it builds). Choosing a difficulty by peeking at S4 would tune the
experiment toward the hypothesis.

WHAT THIS FILE FIXES THAT THE IN-TREE PROBE GOT WRONG
-----------------------------------------------------
``Brainlifts/liv_experiment_research/probes/mqar/mqar_model.py`` constructs ``ShortConv``
**directly and never calls ``init_weights``** (repo memory ``fan-in-correct-one-branch-only``).
The consequence on that probe was that the grouped-gate arm ran at ~1/128 of dense activation
scale -- *on the probe used to justify that arm*. Exp-2 adds a new arm with a new zero-init leg to
the same shape, so the identical failure mode is available, with one arm corrected and another
not, biasing the contrast toward the hypothesis. Here **every arm, including S1, is initialized
through an explicit ``init_weights``**, and preflight check 8 asserts step-0 loss parity across
arms to prove it.

PAIRED SEEDING -- READ THIS BEFORE CHANGING ``init_weights``
------------------------------------------------------------
The power analysis is paired across arms, so shared parameters must be **bit-identical** at the
same seed. A single sequential RNG stream **cannot** do this: S4 owns ``V``, ``U`` and ``alpha``
that S1 does not, so the stream diverges at the first new tensor and every subsequent draw in the
arm is misaligned. "Paired init seeds" would then be simply false (R7 FP1) and the power analysis
void.

This file therefore derives one generator **per init unit, keyed by that unit's qualified name**,
via BLAKE2b over ``f"{seed}/{name}"``. Consequences:

* adding, removing or reordering a module changes no other module's draws;
* the generator's own tensors draw from a separate ``:dyn`` stream, so ``V`` cannot perturb the
  shared block;
* the keying is ``hashlib``-based, not ``hash()``. Python salts ``hash(str)`` per process, so a
  ``hash()``-keyed scheme reproduces within a process and silently fails across processes -- which
  is exactly how the arms of a multi-process sweep would stop being paired.

``preflight`` check 6 asserts this with ``torch.equal``, not ``allclose``. ``ArmSpec.seeding =
"sequential"`` reproduces the broken scheme, and ``test_arms.py`` proves check 6 fails under it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from olmo_core.nn.attention.short_conv import ShortConv
from olmo_core.nn.transformer.init import InitMethod, _apply_init, init_linear

from dynamic_conv import (
    DynamicFilterGen,
    DynamicQKVConv,
    DynamicShortConv,
    PermuteMode,
    depthwise_causal_conv_static,
    gen_param_count,
    iter_generators,
)

__all__ = [
    "ArmSpec",
    "MQARModel",
    "ARMS",
    "TOPOLOGIES",
    "WIDTHS",
    "HYBRID_ATTENTION_LAYERS",
    "ATTN1_ATTENTION_LAYERS",
    "TOPOLOGY_ATTENTION_LAYERS",
    "UNDEFINED_ARM_TOPOLOGY",
    "D_MODEL",
    "N_LAYERS",
    "VOCAB_SIZE",
    "RANK",
    "FFN_MULT",
    "N_HEADS",
    "build_arm",
    "expected_param_count",
    "expected_dynamic_layers",
    "arm_grid",
    "derive_generator",
    "ArmNotDefined",
]

# ---- Geometry, SPEC §7. Pre-registered; do not tune. ----------------------------------------
D_MODEL = 128
N_LAYERS = 6
VOCAB_SIZE = 256
"""Calibrated, and deliberately NOT Zoology's 8192. An 8192-way softmax over 4 answers spends
capacity on the output distribution rather than the binding at this budget; the first in-tree
sweep returned 0.000 everywhere for exactly this reason (job 1670922)."""
RANK = 16
"""``R/d = 1/8`` at d=128, a **deliberate deviation** from the 350M spec's ``R/d = 1/64`` (which
would give R=2 -- degenerately small). The cited paper's own rank curve is still descending at
R=128, so R is the steep axis and starving it here would test the wrong thing. Pre-registered per
SPEC §7; do not tune."""
FFN_MULT = 2
N_HEADS = 1
HYBRID_ATTENTION_LAYERS: Tuple[int, ...] = (2, 5)

ATTN1_ATTENTION_LAYERS: Tuple[int, ...] = (2,)
"""ONE attention layer of six, at index 2 -- the first element of LFM2-16L's published pattern
``[2, 5, 8, 10, 12, 14]``, exactly as ``hybrid`` takes its first two. Added 2026-08-05 because both
ends of the pre-registered topology axis measured saturated (see the module docstring). Its
operating point is calibrated on the BASELINE ARM ONLY."""

ARMS: Tuple[str, ...] = ("S1", "S2", "S3", "S4")
TOPOLOGIES: Tuple[str, ...] = ("hybrid", "attn1", "allliv")

TOPOLOGY_ATTENTION_LAYERS: Dict[str, Tuple[int, ...]] = {
    "hybrid": HYBRID_ATTENTION_LAYERS,
    "attn1": ATTN1_ATTENTION_LAYERS,
    "allliv": (),
}
"""The single source of truth for which layers are attention. ``ArmSpec.attn_idx`` reads THIS rather
than counting layers, so a topology cannot disagree with itself: the previous code special-cased
``allliv`` against a default field value, which meant adding a topology silently inherited
``hybrid``'s indices."""
WIDTHS: Tuple[int, ...] = (2, 3, 4, 8)
"""``{2,3,4}`` is the floor; W=3 must always be present (LFM2 fidelity anchor).

**W=2 is a falsification control, not a data point.** ``orch_verify_W_minus_2.py`` verifies that
the dynamic block at W=2 is an *exact* reparameterization of the static block (max log-residual
8.3e-16, constructive), so the dynamic arm has **zero** new degrees of freedom there. A W=2
dynamic-vs-static difference exceeding seed noise is a bug, not a result."""


UNDEFINED_ARM_TOPOLOGY: Tuple[Tuple[str, str], ...] = (("S3", "allliv"),)
"""The (arm, topology) pairs with no definition. **Exactly one**, and it is a property of the ARM,
not of attention-free-ness in general: S3 puts a dynamic conv on Q/K/V, so it needs at least one
attention block. ``attn1`` has one, so S3 **is** defined there.

Named because the grid-size arithmetic needs it. The count of undefined pairs is 1 regardless of how
many topologies exist, and a test that instead wrote ``len(TOPOLOGIES) // 2`` passed by coincidence
at both 2 and 3 topologies while being wrong as a formula."""


class ArmNotDefined(ValueError):
    """Raised for (S3, allliv). Explicitly NOT a silent fallback to S1."""


# ---------------------------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------------------------


def derive_generator(seed: int, name: str, device: str = "cpu") -> torch.Generator:
    """A generator keyed by ``(seed, name)``, stable across processes.

    ``hashlib.blake2b`` rather than ``hash()``: Python salts ``hash(str)`` per process
    (``PYTHONHASHSEED``), so a ``hash()``-keyed scheme is reproducible *within* a process and
    silently unpaired across the processes of a sweep -- the worst possible failure shape,
    because every local test passes.
    """
    digest = hashlib.blake2b(f"{seed}/{name}".encode(), digest_size=8).digest()
    g = torch.Generator(device=device)
    g.manual_seed(int.from_bytes(digest, "big") % (2**63 - 1))
    return g


# ---------------------------------------------------------------------------------------------
# Arm specification
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmSpec:
    """One cell of the grid. Frozen so a spec cannot be mutated after a count is computed."""

    arm: Literal["S1", "S2", "S3", "S4"]
    topology: Literal["hybrid", "attn1", "allliv"]
    width: int
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    vocab_size: int = VOCAB_SIZE
    rank: int = RANK
    ffn_mult: int = FFN_MULT
    n_heads: int = N_HEADS
    attention_layers: Optional[Tuple[int, ...]] = None
    """``None`` means "use the topology's canonical indices" -- see :attr:`attn_idx`. Defaulting this
    to ``HYBRID_ATTENTION_LAYERS`` was a live trap: any topology other than ``allliv`` silently
    inherited ``hybrid``'s two indices, so ``attn1`` would have been built with TWO attention layers
    while reporting itself as one. An explicit tuple still overrides, for tests."""
    init_method: InitMethod = InitMethod.normal
    init_std: float = 0.02
    permute_mode: PermuteMode = "full"
    permute_seed: int = 0
    nonlinear: bool = False
    """SPEC §7's optional ``sigma(Vh)`` SiLU ablation. Off by default: linear is primary."""

    # ---- Mutation switches. These exist ONLY so the negative controls can flip them. -------
    # Per the repo scar `test-must-call-not-recompute`, a test that re-derives the code's own
    # formula passes when the code changes, and per HANDOFF.md "a guard that has never failed is
    # not known to work." Each of these makes a specific preflight check fail, and
    # `test_preflight.py` proves it does.
    alpha_init: float = 1.0
    """Set 0.0 to reproduce the ``U = 0 AND alpha = 0`` exact saddle -- the $6,100 bug. Check 5b
    must fail."""
    alpha_learnable: bool = True
    """Set False to reproduce ``alpha = 0/1 fixed``. Check 2 and check 5b must fail."""
    conv_activation: Optional[str] = None
    """Set ``"silu"`` to reproduce ``CausalConv1d``'s default. Check 12 must fail."""
    seeding: Literal["per_module", "sequential"] = "per_module"
    """Set ``"sequential"`` to reproduce R7 FP1's single-stream seeding. Check 6 must fail."""
    wire_slot: str = "sequence_mixer"
    """Set ``"attention"`` to reproduce the silent-no-op trap. Check 7 must report 0 dynamic
    modules **while the forward pass still succeeds**."""

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm '{self.arm}'")
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unknown topology '{self.topology}'")
        if self.width < 1:
            raise ValueError(f"width must be >= 1, got {self.width}")
        bad = [i for i in self.attn_idx if not 0 <= i < self.n_layers]
        if bad:
            raise ValueError(
                f"attention layer indices {bad} are outside [0, {self.n_layers}). An out-of-range "
                f"index does not raise anywhere else -- `i in attn` simply never matches, so the "
                f"model is built attention-free while declaring itself hybrid."
            )
        if (self.arm, self.topology) in UNDEFINED_ARM_TOPOLOGY:
            raise ArmNotDefined(
                f"{self.arm} (dynamic conv on Q/K/V) is undefined in the {self.topology} topology: "
                "there are no attention blocks to put a dynamic conv in. Report it as N/A. "
                "Substituting S1 here would create a duplicate baseline masquerading as a "
                "treatment arm."
            )

    # -- derived ---------------------------------------------------------------------------

    @property
    def cell(self) -> str:
        return f"{self.arm}-{self.topology}-W{self.width}"

    @property
    def attn_idx(self) -> Tuple[int, ...]:
        """The attention layer indices, from :data:`TOPOLOGY_ATTENTION_LAYERS` unless overridden.

        Reads the topology TABLE rather than special-casing one topology against a default field
        value. The previous form -- ``() if topology == "allliv" else self.attention_layers`` --
        made every non-``allliv`` topology inherit ``hybrid``'s ``(2, 5)``, so a new 1-attention
        topology would have been built with two attention layers and no check would have noticed:
        ``expected_param_count`` and ``dynamic_layers`` both derive from ``attn_idx``, so the
        declaration and the build would have agreed with each other and disagreed with the design.
        That is the empty-comparison-set defect (EXP2-DESIGN.md Sec 12.4) in yet another costume.
        """
        idx = (
            TOPOLOGY_ATTENTION_LAYERS[self.topology]
            if self.attention_layers is None
            else tuple(self.attention_layers)
        )
        return tuple(sorted(idx))

    @property
    def liv_idx(self) -> Tuple[int, ...]:
        return tuple(i for i in range(self.n_layers) if i not in set(self.attn_idx))

    @property
    def dynamic_liv(self) -> bool:
        return self.arm in ("S2", "S4")

    @property
    def dynamic_qkv(self) -> bool:
        return self.arm == "S3"

    @property
    def permute_z(self) -> bool:
        return self.arm == "S2"

    @property
    def dynamic_layers(self) -> Tuple[int, ...]:
        """The layer INDICES that the arm DECLARES should carry a generator.

        Check 7 asserts these against what the built module actually has, not just the count -- an
        exact total can hide two offsetting errors.

        **This is deliberately independent of ``wire_slot``.** An earlier version returned ``()``
        when ``wire_slot != "sequence_mixer"``, on the reasoning that no generator lands. That made
        the wrong-attribute negative control *pass*: declared 0, built 0, agreed, silent. The
        declaration is the arm's INTENT, so the mutation now shows up as a real disagreement --
        which is the entire point of the check.
        """
        if self.dynamic_liv:
            return self.liv_idx
        if self.dynamic_qkv:
            return self.attn_idx
        return ()


# ---------------------------------------------------------------------------------------------
# Model components. Each owns an explicit `init_weights(generator=...)`.
# ---------------------------------------------------------------------------------------------


class Attention(nn.Module):
    """Minimal causal MHA for the attention slots, with an optional dynamic Q/K/V conv (S3).

    Deliberately plain -- no RoPE, no GQA, no KV cache. These layers exist to supply the global
    routing a short conv cannot, not to reproduce the production attention stack.
    """

    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        qkv_conv: Optional[DynamicQKVConv] = None,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.qkv_conv = qkv_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        if self.qkv_conv is not None:
            # `x` is the normalized block input, i.e. `h` -- the conditioning signal, per SPEC §7.
            q, k, v = self.qkv_conv(x, q, k, v)
        q, k, v = (
            z.view(b, t, self.n_heads, self.head_dim).transpose(1, 2) for z in (q, k, v)
        )
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(o.transpose(1, 2).reshape(b, t, -1))

    @torch.no_grad()
    def init_weights(
        self,
        *,
        std: float = 0.02,
        out_std: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        dyn_generator: Optional[torch.Generator] = None,
    ) -> None:
        # Draw order is qkv then out, and it is IDENTICAL whether or not `qkv_conv` is present.
        # That is what keeps S1's and S3's attention blocks bit-identical (check 6).
        init_linear(self.qkv, std=std, generator=generator)
        init_linear(self.out, std=out_std if out_std is not None else std, generator=generator)
        if self.qkv_conv is not None:
            self.qkv_conv.init_weights(dyn_generator=dyn_generator)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    @torch.no_grad()
    def init_weights(
        self,
        *,
        std: float = 0.02,
        out_std: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
    ) -> None:
        init_linear(self.w1, std=std, generator=generator)
        init_linear(self.w3, std=std, generator=generator)
        init_linear(self.w2, std=out_std if out_std is not None else std, generator=generator)


class Block(nn.Module):
    """Pre-norm block. The mixer attribute is ``sequence_mixer`` -- and only that.

    **The trap this naming defends against:** an override applied to ``block.attention`` instead
    of ``block.sequence_mixer`` **silently no-ops and trains happily**. :meth:`replace_mixer`
    reproduces that silence on purpose so the negative control can exercise it; the production
    path is :func:`build_arm`, which asserts the resulting module count and layer indices.
    """

    SLOTS = ("sequence_mixer",)

    def __init__(self, *, d_model: int, mixer: nn.Module, ffn_mult: int):
        super().__init__()
        self.mixer_norm = nn.RMSNorm(d_model)
        self.sequence_mixer = mixer
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_mult * d_model)

    def replace_mixer(self, slot: str, mixer: nn.Module) -> bool:
        """Apply a mixer override to ``slot``. Returns whether it landed.

        Mirrors config-driven override resolution, which does **not** validate the key: a slot
        that is not a real slot is ignored without error, the block keeps its original mixer, and
        the model trains -- just as the baseline. Returns False so a caller that checks can catch
        it; ``build_arm`` does check.
        """
        if slot not in self.SLOTS:
            return False
        setattr(self, slot, mixer)
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sequence_mixer(self.mixer_norm(x))
        return x + self.ffn(self.ffn_norm(x))


class MQARModel(nn.Module):
    """The Exp-2 model shell: 6 pre-norm blocks, MQAR-sized, one arm's worth of mechanism.

    Topologically the same as the in-tree ``MQARHybrid`` so the calibrated operating point
    transfers, but with two differences that matter: the mixer attribute is ``sequence_mixer``
    (never ``attention``), and **every** parameter is initialized through :meth:`init_weights`.
    """

    def __init__(self, spec: ArmSpec):
        super().__init__()
        self.spec = spec
        d, W = spec.d_model, spec.width
        self.embed = nn.Embedding(spec.vocab_size, d)

        attn = set(spec.attn_idx)
        blocks: List[Block] = []
        for i in range(spec.n_layers):
            if i in attn:
                qkv_conv = None
                if spec.dynamic_qkv and spec.wire_slot == "sequence_mixer":
                    qkv_conv = DynamicQKVConv(
                        d_model=d,
                        kernel_size=W,
                        rank=spec.rank,
                        alpha_init=spec.alpha_init,
                        permute_z=spec.permute_z,
                        permute_mode=spec.permute_mode,
                        permute_seed=spec.permute_seed + 1000 * i,
                        nonlinear=spec.nonlinear,
                    )
                mixer: nn.Module = Attention(
                    d_model=d, n_heads=spec.n_heads, qkv_conv=qkv_conv
                )
            else:
                mixer = ShortConv(
                    d_model=d, kernel_size=W, use_fla=False
                )  # plain nn.Conv1d: the correct, activation-free operator, and it runs on CPU
            blk = Block(d_model=d, mixer=mixer, ffn_mult=spec.ffn_mult)
            if spec.dynamic_liv and i not in attn:
                dyn = DynamicShortConv(
                    d_model=d,
                    kernel_size=W,
                    rank=spec.rank,
                    alpha_init=spec.alpha_init,
                    permute_z=spec.permute_z,
                    permute_mode=spec.permute_mode,
                    # A distinct permutation stream per layer. A single shared stream would make
                    # every layer shuffle identically, which is a weaker control than intended.
                    permute_seed=spec.permute_seed + 1000 * i,
                    nonlinear=spec.nonlinear,
                    conv_activation=spec.conv_activation,
                )
                # THE TRAP, on purpose: `spec.wire_slot` is "sequence_mixer" in every production
                # arm. "attention" is a negative control and lands nowhere.
                blk.replace_mixer(spec.wire_slot, dyn)
            blocks.append(blk)
        self.blocks = nn.ModuleList(blocks)

        self.out_norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, spec.vocab_size, bias=False)

        if not spec.alpha_learnable:
            # `alpha = 0/1 fixed` -- SPEC §3 rows 3 and 4. Delta_w becomes structurally
            # unreachable for the whole run. Reproduced only so check 5b can be shown to fail.
            for _, gen in iter_generators(self):
                gen.alpha.requires_grad_(False)

        self.init_weights()

    # -- init ----------------------------------------------------------------------------------

    @torch.no_grad()
    def init_weights(self, seed: Optional[int] = None) -> None:
        """Initialize EVERY parameter, with one generator per init unit keyed by its name.

        ``seed=None`` uses a fixed default so a freshly-constructed model is never uninitialized
        (a module operating on uninitialized memory usually reads as *inert* rather than broken,
        which is the harder bug to see).
        """
        spec = self.spec
        s = 0 if seed is None else int(seed)
        std = spec.init_std

        def gen_for(name: str) -> Optional[torch.Generator]:
            if spec.seeding == "sequential":
                # R7 FP1 reproduced: ONE stream for the whole model. It diverges at the first
                # tensor an arm does not share, so every later draw is misaligned and the arms
                # are not paired -- while every individual model still looks perfectly fine.
                return self._sequential_generator(s)
            return derive_generator(s, name)

        # Embedding and head. `init_linear` handles nn.Linear/Conv1d; the embedding is a bare
        # parameter, so route it through _apply_init directly (never `w[...] = x`).
        _apply_init(
            nn.init.trunc_normal_,
            self.embed.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std,
            generator=gen_for("embed"),
        )
        init_linear(self.head, std=std, generator=gen_for("head"))
        _apply_init(nn.init.ones_, self.out_norm.weight)

        for i, blk in enumerate(self.blocks):
            _apply_init(nn.init.ones_, blk.mixer_norm.weight)
            _apply_init(nn.init.ones_, blk.ffn_norm.weight)
            blk.ffn.init_weights(
                std=std,
                out_std=self._out_std(std, i),
                generator=gen_for(f"blocks.{i}.ffn"),
            )
            mixer = blk.sequence_mixer
            mname = f"blocks.{i}.sequence_mixer"
            if isinstance(mixer, DynamicShortConv):
                # ShortConv.init_weights draws value_proj, gate_proj, conv, out_proj in that
                # exact order from `generator`; the generator's V draws from a SEPARATE stream.
                # That is what makes S1's and S4's shared tensors bit-identical.
                mixer.init_weights(
                    init_method=spec.init_method,
                    d_model=spec.d_model,
                    block_idx=i,
                    num_blocks=spec.n_layers,
                    std=std,
                    generator=gen_for(mname),
                    dyn_generator=gen_for(f"{mname}:dyn"),
                )
            elif isinstance(mixer, ShortConv):
                mixer.init_weights(
                    init_method=spec.init_method,
                    d_model=spec.d_model,
                    block_idx=i,
                    num_blocks=spec.n_layers,
                    std=std,
                    generator=gen_for(mname),
                )
            elif isinstance(mixer, Attention):
                mixer.init_weights(
                    std=std,
                    out_std=self._out_std(std, i),
                    generator=gen_for(mname),
                    dyn_generator=gen_for(f"{mname}:dyn"),
                )
            else:  # pragma: no cover - defensive
                raise TypeError(f"no init path for mixer {type(mixer).__name__}")

        if not spec.alpha_learnable:
            for _, gen in iter_generators(self):
                gen.alpha.requires_grad_(False)

    def _sequential_generator(self, seed: int) -> torch.Generator:
        g = getattr(self, "_seq_gen", None)
        if g is None or self._seq_gen_seed != seed:  # type: ignore[has-type]
            g = torch.Generator()
            g.manual_seed(seed)
            object.__setattr__(self, "_seq_gen", g)
            object.__setattr__(self, "_seq_gen_seed", seed)
        return g

    def _out_std(self, std: float, block_idx: int) -> float:
        m = self.spec.init_method
        n = self.spec.n_layers
        if m == InitMethod.llama:
            return std / (2 * n) ** 0.5
        if m == InitMethod.llama_depth:
            return std / (2 * (block_idx + 1)) ** 0.5
        if m == InitMethod.normalized:
            return std / (2 * n) ** 0.5
        return std

    # -- forward -------------------------------------------------------------------------------

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """:param tokens: ``(B, T)`` int64. :returns: logits ``(B, T, vocab)``."""
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.out_norm(x))

    # -- introspection -------------------------------------------------------------------------

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def dynamic_module_layers(self) -> Tuple[int, ...]:
        """The layer indices that actually carry a generator, read off the built module tree.

        Independent of :attr:`ArmSpec.dynamic_layers`, which is the *declared* answer. Check 7
        compares the two; agreement is the point.
        """
        out = []
        for i, blk in enumerate(self.blocks):
            if any(isinstance(m, DynamicFilterGen) for m in blk.modules()):
                out.append(i)
        return tuple(out)

    def n_dynamic_modules(self) -> int:
        return len(iter_generators(self))

    def static_filters(self) -> List[Tuple[str, torch.Tensor]]:
        out = []
        for i, blk in enumerate(self.blocks):
            m = blk.sequence_mixer
            if isinstance(m, ShortConv):
                out.append(
                    (
                        f"blocks.{i}.sequence_mixer",
                        m.conv.weight.view(self.spec.d_model, self.spec.width),
                    )
                )
            elif isinstance(m, Attention) and m.qkv_conv is not None:
                out.append((f"blocks.{i}.sequence_mixer.qkv_conv", m.qkv_conv.filters))
        return out


# ---------------------------------------------------------------------------------------------
# Analytic parameter counts
# ---------------------------------------------------------------------------------------------


def _short_conv_params(d: int, W: int) -> int:
    """``value_proj + gate_proj + out_proj + depthwise conv``, no biases (LFM2 uses none)."""
    return d * d + 2 * d * d + d * d + W * d


def _attention_params(d: int) -> int:
    return 3 * d * d + d * d


def _swiglu_params(d: int, hidden: int) -> int:
    return 2 * d * hidden + hidden * d


def expected_param_count(spec: ArmSpec) -> Dict[str, int]:
    """Exact integer parameter count, derived analytically, component by component.

    Reported per component and not only as a total, because **an exact total can hide two
    offsetting errors** -- the in-tree 350M reconciliation caught two geometry omissions
    (+67,108,864 from untied embeddings defaulting on, and a +768 residual from missing per-head
    QK-norm) precisely because it reconciled components rather than a single number.
    """
    d, W, L = spec.d_model, spec.width, spec.n_layers
    hidden = spec.ffn_mult * d
    attn = set(spec.attn_idx)
    liv = [i for i in range(L) if i not in attn]

    parts: Dict[str, int] = {}
    parts["embed"] = spec.vocab_size * d
    parts["head"] = d * spec.vocab_size
    parts["norms"] = (2 * L + 1) * d  # mixer_norm + ffn_norm per block, plus out_norm
    parts["ffn"] = L * _swiglu_params(d, hidden)
    parts["liv_mixers"] = len(liv) * _short_conv_params(d, W)
    parts["attn_mixers"] = len(attn) * _attention_params(d)

    # DECLARED, i.e. intent -- deliberately independent of `wire_slot`, exactly as
    # `ArmSpec.dynamic_layers` is. An earlier version zeroed these when `wire_slot` was mutated,
    # which made checks 7a/7b PASS on the wrong-attribute control: declared 0, built 0, agreed,
    # silent. The declaration must describe what the arm is FOR, so that a mechanism which failed
    # to land shows up as a disagreement in the count as well as in the indices.
    n_dyn_liv = len(liv) if spec.dynamic_liv else 0
    n_dyn_qkv = len(attn) if spec.dynamic_qkv else 0
    parts["dyn_liv_gen"] = n_dyn_liv * gen_param_count(d, spec.rank, W, n_streams=1)
    parts["dyn_qkv_gen"] = n_dyn_qkv * (
        gen_param_count(d, spec.rank, W, n_streams=3) + 3 * d * W  # + the 3 static filters
    )
    parts["total"] = sum(v for k, v in parts.items() if k != "total")
    return parts


def expected_dynamic_layers(spec: ArmSpec) -> Tuple[int, ...]:
    return spec.dynamic_layers


# ---------------------------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------------------------


def build_arm(spec: ArmSpec, seed: int = 0, *, strict: bool = True) -> MQARModel:
    """Build and initialize one arm at one seed.

    :param strict: refuse the ``wire_slot`` negative-control knob, and assert the declared module
        count, the declared layer indices and the analytic parameter total against the built
        module. Production callers leave this True. The negative controls set it False, because
        their whole purpose is to build a model whose structure is wrong and then watch preflight
        catch it.
    """
    model = MQARModel(spec)
    model.init_weights(seed=seed)
    if strict:
        # Refuse the mutation switches outright, BEFORE the structural checks. `wire_slot` is a
        # negative-control knob, not a configuration option: any value other than
        # "sequence_mixer" reproduces the trap in which the mechanism is attached to an attribute
        # nothing reads, so it silently no-ops and TRAINS HAPPILY to a clean-looking null. The
        # count checks below would also catch it, but a caller who asked for `strict=True` is
        # entitled to have the builder refuse rather than hand back a quietly-baseline model.
        if spec.wire_slot != "sequence_mixer":
            raise AssertionError(
                f"{spec.cell}: wire_slot={spec.wire_slot!r} is a NEGATIVE CONTROL, refused under "
                "strict=True. Wiring the mechanism to any attribute other than "
                "'block.sequence_mixer' silently no-ops -- the module is constructed, the forward "
                "pass succeeds, the loss looks normal, and the arm trains as the static baseline "
                "while reporting itself as the treatment. Pass strict=False if you are "
                "deliberately exercising the trap."
            )
        exp_layers = expected_dynamic_layers(spec)
        got_layers = model.dynamic_module_layers()
        if got_layers != exp_layers:
            raise AssertionError(
                f"{spec.cell}: dynamic layer indices {got_layers} != declared {exp_layers}. "
                "This is the silent-no-op trap: a mechanism wired to the wrong attribute "
                "trains happily and produces a clean-looking null."
            )
        n_exp = len(exp_layers)
        if model.n_dynamic_modules() != n_exp:
            raise AssertionError(
                f"{spec.cell}: {model.n_dynamic_modules()} DynamicFilterGen modules != {n_exp}"
            )
        exp_total = expected_param_count(spec)["total"]
        if model.n_params != exp_total:
            raise AssertionError(
                f"{spec.cell}: {model.n_params} params != analytic {exp_total} "
                f"(delta {model.n_params - exp_total})"
            )
    return model


def arm_grid(
    widths: Sequence[int] = WIDTHS,
    arms: Sequence[str] = ARMS,
    topologies: Sequence[str] = TOPOLOGIES,
    **kwargs,
) -> List[ArmSpec]:
    """Every buildable cell. (S3, allliv) is skipped and must be reported as **N/A**."""
    out: List[ArmSpec] = []
    for topo in topologies:
        for arm in arms:
            for W in widths:
                try:
                    out.append(ArmSpec(arm=arm, topology=topo, width=W, **kwargs))  # type: ignore[arg-type]
                except ArmNotDefined:
                    continue
    return out


def na_cells(
    widths: Sequence[int] = WIDTHS,
    arms: Sequence[str] = ARMS,
    topologies: Sequence[str] = TOPOLOGIES,
) -> List[str]:
    """The cells that are undefined by construction, named so a results table can print N/A
    rather than a silently substituted baseline."""
    return [
        f"{a}-{t}-W{w}"
        for t in topologies
        for a in arms
        for w in widths
        if (a, t) in UNDEFINED_ARM_TOPOLOGY
    ]


def param_count_table(widths: Sequence[int] = WIDTHS) -> str:
    """A markdown table of exact integer counts per (arm, topology, W), verified against a
    built module. What goes in the design doc."""
    rows = [
        "| arm | topology | W | total params | dyn params | dyn modules | layer indices |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for spec in arm_grid(widths):
        m = build_arm(spec, seed=0)
        dyn = sum(p.numel() for _, g in iter_generators(m) for p in g.parameters())
        if spec.arm == "S3":
            dyn += sum(
                blk.sequence_mixer.qkv_conv.filters.numel()
                for blk in m.blocks
                if isinstance(blk.sequence_mixer, Attention)
                and blk.sequence_mixer.qkv_conv is not None
            )
        rows.append(
            f"| {spec.arm} | {spec.topology} | {spec.width} | {m.n_params:,} | {dyn:,} | "
            f"{m.n_dynamic_modules()} | {list(m.dynamic_module_layers())} |"
        )
    for cell in na_cells(widths):
        a, t, w = cell.split("-")
        rows.append(f"| {a} | {t} | {w[1:]} | **N/A** | — | — | — |")
    return "\n".join(rows)


if __name__ == "__main__":  # pragma: no cover
    print(param_count_table())
