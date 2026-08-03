"""The run matrix: James's three best loss-weighting configs, on Impl 5's targets.

Everything governing comparability is imported from ``impl5.config5`` (which imports it from
``impl4.config``) rather than restated — 923 steps, the 22-point checkpoint grid, the 24/8
block layout, seed 13, LoRA r=16/α=32, lr 2e-4, ``per_device_batch 8 × grad_accum 4``. Only
the loss multiplier is defined here.

**The batch shape is load-bearing and is not a tuning knob.** ``per_device_batch 8 ×
grad_accum 4`` is A1's and D4's. ``impl4_ssd/probe_loss_norm.py`` measured which loss
normalisation the Trainer actually applies across an accumulation group, and Impl 5 inherited
that result — check 4 in its acceptance checks — on the grounds of "same recipe, same pins,
same PEFT wrapping". Regrouping the micro-batches changes what each example contributes
whenever the group's token counts are uneven, so it would void that inheritance and put a
second variable into a contrast that exists to isolate one. GPU utilisation is taken from
running the arms concurrently instead; see ``run_klw.py``.

## The arms

Chosen on the +SI blind pedagogy judge (2026-08-03, ``llm_judge/auto/test_results_all10/``),
where James's arms took the top three places and Impl 5's D4 came fourth:

===========  =========  =====  ===========================  ==========================
arm          variant    T      judge (+SI, gold corpus)     James's gold-corpus notes
===========  =========  =====  ===========================  ==========================
``bT1``      b          1      0.913 — 1st of 10            KL(noSI) 0.096, hinted 0.612
``bT2``      b          2      0.910 — 2nd                  KL(noSI) 0.118, hinted 0.548
``aT8``      a          8      0.896 — 3rd                  KL(noSI) 0.067, hinted 0.652
``bT451``    b          451    — (control, not judged)      James's T→∞ limit check
===========  =========  =====  ===========================  ==========================

``bT451`` is not a fourth condition, it is the **implementation check James recommends**:
"T → ∞ recovers vanilla SFT exactly. We verified this: ``b-T451`` reproduces the SFT baseline
to within 0.002 on every metric. It is a cheap and strong implementation check — recommend
you run the equivalent." Here the thing it must reproduce is **D4**, so it prices the entire
apparatus in one run: if ``bT451`` does not land on D4, no other arm's number means anything.
It occupies the fourth GPU, which would otherwise idle.

## Two things these numbers are not

**The judge scores above are on James's gold-trained adapters, on a different corpus and a
different pipeline.** They are recorded here only to document *why* these three arms were
picked. They are not baselines for the new runs. The baseline for every arm here is **D4**.

**The temperatures are not transferable.** §4.1's softmax is global over the pedagogy stream,
and Impl 5's stream is ~37% base-model text where both signals are systematically smaller.
The same numeric T therefore applies different pressure. Quote ``multiplier.ess`` from the
precompute (see ``weighting.describe``) when comparing an arm here to its gold-corpus twin —
it is the cross-corpus-comparable quantity; T is not.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._impl5 import config5

# --- Inherited verbatim. Do not restate; import. ------------------------------
BASE_MODEL = config5.BASE_MODEL
SEED = config5.SEED
MAX_LEN = config5.MAX_LEN
BLOCK_SIZE = config5.BLOCK_SIZE
PED_PER_BLOCK = config5.PED_PER_BLOCK
GEN_PER_BLOCK = config5.GEN_PER_BLOCK
N_BLOCKS = config5.N_BLOCKS            # 923
N_PED = config5.N_PED                  # 22,152
N_GEN = config5.N_GEN                  # 7,384
N_TRAIN = config5.N_TRAIN              # 29,536
CKPT_GRID = config5.CKPT_GRID          # the 22-point union grid
POC_N_BLOCKS = config5.POC_N_BLOCKS
POC_CKPT_GRID = config5.POC_CKPT_GRID
PER_DEVICE_BATCH = config5.config4.PER_DEVICE_BATCH   # 8  — see the module docstring
GRAD_ACCUM = config5.config4.GRAD_ACCUM               # 4
LEARNING_RATE = config5.config4.LEARNING_RATE
WARMUP_RATIO = config5.config4.WARMUP_RATIO
NUM_EPOCHS = config5.config4.NUM_EPOCHS
LORA_R = config5.config4.LORA_R
LORA_ALPHA = config5.config4.LORA_ALPHA
LORA_DROPOUT = config5.config4.LORA_DROPOUT

#: The arm supplying the training data. Every arm here trains on **D4's** mix, byte-for-byte:
#: the distilled pedagogy pool at realised δ = 0.368 label tokens, in A1's replay slot and
#: A1's block positions. The data is not rebuilt per arm, it is built once and shared.
DATA_ARM = "D4"

#: Variant b's reference π_SFT. §4.1: "Variant b requires an already-trained vanilla SFT
#: model as its reference. We used checkpoint-923 (the POC's Impl-2 adapter) for every b run
#: — keep this fixed, since changing it changes both the signal and the precompute cache key."
#:
#: His definition (§1) is "a vanilla SFT run on identical data". On Impl 5's mix that is D4,
#: not the gold Impl-2 adapter: D4 is vanilla SFT — no reweighting — on exactly the training
#: file these arms use. Using the gold adapter would measure how far gold-SFT moved from base
#: on contexts that gold-SFT never saw. Fixed for every b arm here, as §4.1 requires.
REFERENCE_ADAPTER_ARM = "D4"
REFERENCE_ADAPTER_STEP = 923


@dataclass(frozen=True)
class ArmKLW:
    """One (variant, temperature) cell. Nothing else varies between arms."""

    name: str
    variant: str
    temperature: float
    role: str = "condition"          # "condition" | "control"
    judge_gold: float | None = None  # +SI judge on James's GOLD adapter — provenance only
    note: str = ""

    def __post_init__(self):
        if self.variant not in ("a", "b"):
            raise ValueError(f"{self.name}: variant must be 'a' or 'b', got {self.variant!r}")
        if not self.temperature > 0:
            raise ValueError(f"{self.name}: temperature must be > 0")

    @property
    def needs_reference(self) -> bool:
        """Variant b needs π_SFT; variant a needs only the frozen base."""
        return self.variant == "b"

    @property
    def impl3_name(self) -> str:
        """James's own name for the same cell, for cross-referencing his tables."""
        t = self.temperature
        ts = f"{t:g}"
        return f"{self.variant}-T{ts}"


ARMS: dict[str, ArmKLW] = {
    "bT1": ArmKLW("bT1", "b", 1.0, judge_gold=0.913,
                  note="1st of 10 on the +SI judge; James's KL(noSI) 0.096, hinted 0.612."),
    "bT2": ArmKLW("bT2", "b", 2.0, judge_gold=0.910,
                  note="2nd; KL(noSI) 0.118, hinted 0.548."),
    "aT8": ArmKLW("aT8", "a", 8.0, judge_gold=0.896,
                  note="3rd, and the arm to beat: on gold it beats D4 on forgetting "
                       "(math_hint 0.652 vs 0.572) at the same KL. §6.3 calls it 'the "
                       "interesting anomaly' — it gets there by gating on the SI rather "
                       "than by staying close to base."),
    "bT451": ArmKLW("bT451", "b", 451.0, role="control",
                    note="T -> inf limit. MUST reproduce D4. James's equivalent reproduced "
                         "his SFT baseline to within 0.002 on every metric; if this one "
                         "misses D4, nothing else here is interpretable."),
}

#: The three the user asked for, in judge order, plus the control last.
CONDITION_ARMS = ("bT1", "bT2", "aT8")
CONTROL_ARMS = ("bT451",)
ALL_ARMS = CONDITION_ARMS + CONTROL_ARMS
ARM_CHOICES = tuple(ARMS)

#: Run-name prefix in the results files. Impl 3's runs are ``impl3-*``, Impl 4's ``impl4-*``,
#: Impl 5's ``impl5-*``; these are the product of two of them.
RUN_PREFIX = "impl3x5-"


def resolve_arm(name: str) -> ArmKLW:
    if name not in ARMS:
        raise KeyError(f"unknown arm {name!r}; known: {', '.join(ARMS)}")
    return ARMS[name]


def variants_needed(arms=ALL_ARMS) -> tuple[str, ...]:
    """Which signals the precompute has to produce for this set of arms.

    Both variants come out of one pass (see ``precompute_signal.py``): variant a's
    ``−log π₀(y_t)`` is a by-product of the same base forward whose full distribution variant
    b needs. So this is used to decide whether the *reference* model has to be loaded, not to
    split the pass in two.
    """
    return tuple(sorted({resolve_arm(a).variant for a in arms}))


def n_blocks(poc: bool = False) -> int:
    return POC_N_BLOCKS if poc else N_BLOCKS


def checkpoint_grid(poc: bool = False) -> tuple[int, ...]:
    return POC_CKPT_GRID if poc else CKPT_GRID
