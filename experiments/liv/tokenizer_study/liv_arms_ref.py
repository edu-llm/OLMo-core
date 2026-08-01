"""
Declarative arm builder for the LFM2/LIV hybrid study.

Every arm in the study is one entry in :data:`ARMS`, and every arm is built by the same
function. The point is that an arm is a *declaration*, not a script: two arms that differ in
one field differ in exactly one field, and the difference is visible on one line.

The frozen 350M geometry (see ``docs/liv-kda-gqa-sub500m-experiment.md``):

===========================  ==============
Decoder layers               16
``d_model``                  1,024
Query / KV heads             16 / 8
Head dimension               64
SwiGLU branch width          4,608
Vocabulary (tied)            65,536
Attention layer indices      2, 5, 8, 10, 12, 14
LIV convolution              causal depthwise, kernel 3
===========================  ==============

``L0``, the released-shape control, must come to **354,483,968** parameters. That number is
asserted in the tests: it is the check that the whole ledger is right.

.. important::
    The SwiGLU width is **4,608**, which is *not* what ``llama_like`` computes by default. The
    released config field ``block_ff_dim=6656`` is transformed by LFM2's implementation into
    an effective per-branch width of 4,608 via ``256 * ceil(int(2/3 * block_ff_dim) / 256)``.
    Miss this and the MLP -- 69% of the model -- is roughly 50% wrong. This module passes the
    width explicitly rather than reproducing the transform.

.. important::
    Arms must be matched on ``num_flops_per_token`` at the target context, **not** on
    parameter count. Attention has a term that grows with sequence length and short
    convolutions do not, so two arms with identical parameters can differ by tens of percent
    in compute at 32K. :func:`arm_report` prints both.
"""

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Set, Tuple

from olmo_core.config import DType
from olmo_core.nn.attention.short_conv import GateStructure, ShortConvConfig
from olmo_core.nn.feed_forward import FeedForwardConfig
from olmo_core.nn.transformer.config import TransformerBlockConfig, TransformerConfig

__all__ = [
    "LivArm",
    "ARMS",
    "build_arm",
    "arm_report",
    "solve_swiglu_width",
    "solve_d_model",
    "L0_PARAM_TARGET",
]


# --- frozen geometry ----------------------------------------------------------------------

N_LAYERS = 16
D_MODEL = 1024
N_HEADS = 16
N_KV_HEADS = 8
HEAD_DIM = 64
SWIGLU_WIDTH = 4608
VOCAB_SIZE = 65536
KERNEL_SIZE = 3
ATTENTION_LAYERS: Tuple[int, ...] = (2, 5, 8, 10, 12, 14)

L0_PARAM_TARGET = 354_483_968
"""Exact released-shape parameter count for ``L0``, from the frozen protocol."""


@dataclass(frozen=True)
class LivArm:
    """
    One experimental arm, declared rather than scripted.

    :param name: Short arm identifier, e.g. ``"L0"``.
    :param role: What this arm is for -- control, treatment, or competitor. Written out so a
        reader can tell at a glance which arms are load-bearing.
    :param attention_layers: Indices that use attention. Every other layer uses ``ShortConv``.
    :param kernel_size: LIV convolution taps. The P3 width arms vary only this.
    :param gate_structure: ``"dense"``, ``"lowrank"``, or ``"grouped"``.
    :param gate_rank: Bottleneck rank when ``gate_structure="lowrank"``.
    :param gate_groups: Block count when ``gate_structure="grouped"``.
    :param d_model: Model width. Only ``N-narrow`` changes this.
    :param swiglu_width: SwiGLU branch width. Capacity controls change this.
    :param n_kv_heads: KV heads. The MQA secondary arm sets this to 1.
    """

    name: str
    role: str
    attention_layers: Tuple[int, ...] = ATTENTION_LAYERS
    kernel_size: int = KERNEL_SIZE
    gate_structure: GateStructure = "dense"
    gate_rank: Optional[int] = None
    gate_groups: Optional[int] = None
    d_model: int = D_MODEL
    swiglu_width: int = SWIGLU_WIDTH
    n_kv_heads: int = N_KV_HEADS
    notes: str = ""

    @property
    def n_liv_layers(self) -> int:
        return N_LAYERS - len(self.attention_layers)


# --- the arms -------------------------------------------------------------------------------
#
# Grouped by the question each one answers. Controls are listed with their treatment so that
# dropping a treatment without its control is visibly wrong.

ARMS: Dict[str, LivArm] = {
    a.name: a
    for a in [
        # -- topology ------------------------------------------------------------------------
        LivArm(
            "L0",
            "control: released LFM2 shape",
            notes="The baseline everything is measured against. Must hit L0_PARAM_TARGET.",
        ),
        LivArm(
            "A16-P",
            "control: all-attention topology",
            attention_layers=tuple(range(N_LAYERS)),
            swiglu_width=4820,  # solved by solve_swiglu_width(); asserted in tests
            notes="All-GQA competitor, parameter-matched to L0 (-94,976, 0.027%). NOTE it is "
            "NOT compute-matched: 1.27x L0's FLOPs/token at 4K and 1.94x at 32K, because "
            "attention's score term grows with context and a convolution's does not. Match on "
            "FLOPs for any compute-controlled comparison -- see arm_report.",
        ),
        # -- P1: gate structure ----------------------------------------------------------------
        # The latency claim is dead (measured: fused r=128 + CUDA graphs is 8.2% SLOWER than
        # dense on L40S). These arms now test QUALITY at reduced parameter cost.
        LivArm(
            "F-r128",
            "treatment: P1 low-rank gates",
            gate_structure="lowrank",
            gate_rank=128,
            notes="Retains 92.6% of activation-weighted energy at 0.25x gate params.",
        ),
        LivArm("F-r256", "treatment: P1 low-rank gates", gate_structure="lowrank", gate_rank=256),
        LivArm(
            "G-grouped",
            "competitor: block-diagonal gates",
            gate_structure="grouped",
            gate_groups=4,
            notes="Cost-identical to F-r128. Wins latency (+15.3%) but retains only 0.130 of "
            "activation-weighted energy vs low-rank's 0.929. In STAR's search space and NOT "
            "selected, so it is the incumbent structure to beat.",
        ),
        LivArm(
            "N-narrow",
            "control: just build a narrower model",
            d_model=976,
            swiglu_width=4668,
            notes="MANDATORY -- the obvious competing way to spend the parameters F-r128 saves. "
            "Both dims are SOLVED against F-r128, never guessed: d_model on the 16-multiple "
            "grid (head count), then SwiGLU width to close the remainder. Lands +49,200 "
            "(0.0145%). d_model alone only reaches 0.815%, too coarse for a capacity control.",
        ),
        # -- P3: kernel width ------------------------------------------------------------------
        # Widths-first, per the frozen decision: if width is flat inside the gate, a router has
        # nothing to route between, so this decides whether the router is even coherent.
        LivArm("W-k5", "treatment: P3 wider kernel", kernel_size=5),
        LivArm("W-k9", "treatment: P3 wider kernel", kernel_size=9),
        LivArm(
            "W-k15",
            "treatment: P3 wider kernel",
            kernel_size=15,
            notes="Also the dense-equivalent control for a 4-branch dilated block, which fuses "
            "losslessly into one 15-tap kernel when branch weights are fixed.",
        ),
        # -- P2: attention budget ----------------------------------------------------------------
        LivArm(
            "A-fewer3",
            "competitor: 3 attention layers instead of 6",
            attention_layers=(5, 10, 14),
            notes="MANDATORY and the strongest competitor to P2. Matches CLA2's resident KV "
            "capacity AND halves read bandwidth, which cross-layer sharing structurally cannot.",
        ),
        LivArm(
            "Q-mqa",
            "secondary: MQA",
            n_kv_heads=1,
            notes="Monitored secondary only. GQA's Appendix A reports MQA-from-scratch had "
            "'frequent loss spikes' and diverged; CLA's MQA results were uptrained. Watch for "
            "spikes and do not promote to primary.",
        ),
    ]
}


def build_arm(
    arm: LivArm | str,
    *,
    vocab_size: int = VOCAB_SIZE,
    init_device: str = "meta",
    dtype: DType = DType.float32,
    **kwargs,
) -> TransformerConfig:
    """
    Build the :class:`TransformerConfig` for one arm.

    Defaults to ``init_device="meta"`` so parameter counts can be checked without allocating.

    :param arm: A :class:`LivArm` or the name of one in :data:`ARMS`.
    :param vocab_size: Vocabulary size. The frozen ledger uses 65,536, tied.
    :param init_device: Where to place parameters, e.g. ``"meta"``, ``"cpu"``.
    :param dtype: Parameter dtype.

    :returns: A config whose per-layer mixers follow ``arm.attention_layers``.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]

    cfg = TransformerConfig.llama_like(
        d_model=arm.d_model,
        vocab_size=vocab_size,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_kv_heads=arm.n_kv_heads,
        head_dim=HEAD_DIM,
        # Pass the width explicitly: llama_like would otherwise derive 8/3*d rounded to 256,
        # which is not the 4,608 the released config implies.
        feed_forward=FeedForwardConfig(hidden_size=arm.swiglu_width, bias=False, dtype=dtype),
        # LFM2's attention applies RMSNorm to Q and K per head -- `Lfm2Attention` builds
        # `q_layernorm` and `k_layernorm` of size `head_dim`. Both default to False in
        # `llama_like`. Omitting them costs exactly 6 layers x 2 norms x 64 = 768 parameters,
        # which is how the ledger discrepancy was found.
        qk_norm=True,
        use_head_qk_norm=True,
        dtype=dtype,
        **kwargs,
    )
    # The frozen ledger specifies TIED embeddings; `llama_like` defaults to untied, which adds
    # a second vocab x d_model tensor -- 67,108,864 parameters at the frozen geometry, or ~19%
    # of the model. Large enough to invalidate every arm-matching decision silently.
    cfg.tie_word_embeddings = True

    # Per-layer overrides. NOTE the field is `sequence_mixer`, NOT `attention`: setting
    # `.attention` on a block *config* silently creates a new attribute, the override is
    # ignored, and every layer stays attention -- a model that builds, trains, and answers a
    # different question. The tests assert layer types for exactly this reason.
    assert isinstance(
        cfg.block, TransformerBlockConfig
    ), "llama_like should return a single block config, not a named-block dict"
    liv_block = replace(
        cfg.block,
        sequence_mixer=ShortConvConfig(
            kernel_size=arm.kernel_size,
            gate_structure=arm.gate_structure,
            gate_rank=arm.gate_rank,
            gate_groups=arm.gate_groups,
            dtype=dtype,
        ),
    )
    overrides = {i: liv_block for i in range(N_LAYERS) if i not in arm.attention_layers}
    cfg.block_overrides = overrides or None  # all-attention arm: no overrides at all

    return cfg


def arm_report(
    names: Optional[List[str]] = None,
    *,
    contexts: Tuple[int, ...] = (4096, 32768),
    vocab_size: int = VOCAB_SIZE,
) -> str:
    """
    Tabulate parameters and FLOPs/token for each arm, relative to ``L0``.

    Reports FLOPs at both a short and a long context because that is where param-matching and
    compute-matching diverge: attention's score term grows with context, a convolution's does
    not, so a parameter-matched pair can be badly compute-mismatched at 32K.

    :param names: Arms to include. Defaults to all of :data:`ARMS`.
    :param contexts: Sequence lengths at which to report FLOPs per token.
    :param vocab_size: Vocabulary size to build with.

    :returns: A printable table.
    """
    names = names or list(ARMS)
    base = build_arm("L0", vocab_size=vocab_size)
    base_model = base.build(init_device="meta")
    base_params = sum(p.numel() for p in base_model.parameters())
    base_flops = {t: base_model.num_flops_per_token(t) for t in contexts}

    head = f"{'arm':<12}{'params':>14}{'vs L0':>9}"
    for t in contexts:
        head += f"{'flops@' + (str(t // 1024) + 'K'):>14}{'vs L0':>9}"
    lines = [head, "-" * len(head)]

    for name in names:
        cfg = build_arm(name, vocab_size=vocab_size)
        model = cfg.build(init_device="meta")
        n = sum(p.numel() for p in model.parameters())
        row = f"{name:<12}{n:>14,}{n / base_params:>8.3f}x"
        for t in contexts:
            f = model.num_flops_per_token(t)
            row += f"{f:>14,}{f / base_flops[t]:>8.3f}x"
        lines.append(row)

    return "\n".join(lines)


def _count_params(cfg: TransformerConfig) -> int:
    """Deduplicated parameter count -- tied embeddings appear under two names."""
    model = cfg.build(init_device="meta")
    seen: Set[int] = set()
    total = 0
    for p in model.parameters():
        if id(p) not in seen:
            seen.add(id(p))
            total += p.numel()
    return total


def solve_swiglu_width(
    arm: LivArm | str,
    *,
    target_params: int = L0_PARAM_TARGET,
    multiple_of: int = 4,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[int, int]:
    """
    Find the SwiGLU width bringing an arm closest to ``target_params``.

    Use for capacity controls (``A16-P``, and the ``L0-P``-style padded variants) where the
    protocol says "FFN width solved to match", so the width is *derived* rather than chosen.

    :param arm: Arm or arm name to solve for.
    :param target_params: Parameter count to match. Defaults to ``L0``'s exact ledger.
    :param multiple_of: Constrain the width to a multiple of this, for kernel friendliness.
    :param vocab_size: Vocabulary size to build with.

    :returns: ``(width, achieved_params)``.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]

    # Parameters are exactly linear in the SwiGLU width (3 * d * ff per layer), so two probes
    # determine the line and the solve is direct rather than a search.
    w0 = 1024
    w1 = 8192
    p0 = _count_params(build_arm(replace(arm, swiglu_width=w0), vocab_size=vocab_size))
    p1 = _count_params(build_arm(replace(arm, swiglu_width=w1), vocab_size=vocab_size))
    slope = (p1 - p0) / (w1 - w0)
    ideal = w0 + (target_params - p0) / slope
    width = max(multiple_of, int(round(ideal / multiple_of)) * multiple_of)
    return width, _count_params(build_arm(replace(arm, swiglu_width=width), vocab_size=vocab_size))


def solve_d_model(
    arm: LivArm | str,
    *,
    target_params: int,
    multiple_of: int = 16,
    vocab_size: int = VOCAB_SIZE,
) -> Tuple[int, int]:
    """
    Find the model width bringing an arm closest to ``target_params``.

    Use for ``N-narrow``, whose whole point is "just build a narrower model with the parameters
    the low-rank arm saved" -- so its width must be *solved against* the low-rank arm, never
    guessed, or the control is not a control.

    ``multiple_of`` defaults to 16 -- the head count, which is the binding constraint. A
    coarser grid (64) leaves ``N-narrow`` ~8.9M parameters short of ``F-r128``, which is 2.6%
    of the model and far too large a gap for a capacity control.

    :param arm: Arm or arm name to solve for.
    :param target_params: Parameter count to match, e.g. ``F-r128``'s.
    :param multiple_of: Constrain the width to a multiple of this.
    :param vocab_size: Vocabulary size to build with.

    :returns: ``(d_model, achieved_params)``.
    """
    if isinstance(arm, str):
        arm = ARMS[arm]

    # Parameters are quadratic in d_model, so scan the plausible band rather than interpolate.
    best: Optional[Tuple[int, int]] = None
    for d in range(multiple_of * 8, D_MODEL + multiple_of, multiple_of):
        if d % N_HEADS != 0:
            continue
        got = _count_params(build_arm(replace(arm, d_model=d), vocab_size=vocab_size))
        if best is None or abs(got - target_params) < abs(best[1] - target_params):
            best = (d, got)
    assert best is not None
    return best


if __name__ == "__main__":  # pragma: no cover
    print(arm_report())
