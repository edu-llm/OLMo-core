"""MQAR (multi-query associative recall) data, faithful to Zoology's generator.

Layout, causal, single pass, answers strictly after all pairs::

    k1 v1 k2 v2 ... kD vD  <filler ...>  q1 q2 ... qD
    ^^^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^   ^^^^^^^^^^^^
    2D tokens              distance       D queries

Labels are ``-100`` everywhere except query positions, where the label is the value token bound
to that query's key. ``-100`` is ``F.cross_entropy``'s default ``ignore_index``, so the loss
needs no special handling -- but **accuracy does**, or the ignored positions count as wrong.

THE TWO DIFFICULTY AXES ARE DECOUPLED, deliberately
---------------------------------------------------
MQAR difficulty has two independent knobs and varying both at once confounds them:

* **num_pairs (D)** -- how much must be held. A *capacity* limit on a fixed-size state.
* **distance** -- how long it must be held. What a forget gate, or a conv's receptive field,
  governs.

Holding D fixed while stretching the filler turns a length sweep into a pure retention-distance
test. This mirrors ``probes/mqar_patch.py``'s ``MQAR_MAX_PAIRS`` design.

TWO REPRODUCTION GOTCHAS, both pinned here
-------------------------------------------
Zoology's published configs differ from its own class defaults, and both differences silently
change results:

1. ``random_non_queries=False`` in the configs, but the class default is ``True``. With ``True``,
   filler is random tokens; with ``False`` it is a dedicated padding token. Random filler is
   *harder* (it can collide with keys) -- so an unpinned run is not the published task.
2. ``power_a=0.01`` gives strongly power-law gaps between pairs; ``1.0`` is uniform. The default
   clusters pairs early, which shortens effective retention distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

IGNORE_INDEX = -100


@dataclass(frozen=True)
class MQARConfig:
    """
    One point in the MQAR difficulty grid.

    :param seq_len: Total sequence length in tokens.
    :param num_pairs: Number of key-value pairs (``D``) -- the capacity axis.
    :param vocab_size: Total vocabulary. Split into disjoint key and value halves.
    :param power_a: Power-law exponent for gaps between pairs. ``0.01`` matches Zoology's
        published configs; ``1.0`` is uniform.
    :param random_non_queries: Fill non-query positions with random tokens. Zoology's class
        default is ``True`` but its published configs set ``False``. Pinned ``False`` here.
    """

    seq_len: int
    num_pairs: int
    vocab_size: int = 8192
    power_a: float = 0.01
    random_non_queries: bool = False

    def __post_init__(self):
        # 2D pair tokens + D query tokens must fit, with room for at least some distance.
        if 3 * self.num_pairs > self.seq_len:
            raise ValueError(
                f"num_pairs={self.num_pairs} needs >= {3 * self.num_pairs} positions "
                f"but seq_len={self.seq_len}"
            )
        if self.vocab_size % 2:
            raise ValueError("vocab_size must be even to split into key/value halves")
        if self.num_pairs > self.vocab_size // 2:
            raise ValueError("num_pairs exceeds the key vocabulary")

    @property
    def label(self) -> str:
        return f"N{self.seq_len}_D{self.num_pairs}"


def make_mqar_batch(
    cfg: MQARConfig,
    batch_size: int,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one batch of MQAR sequences.

    :param cfg: The difficulty point to sample.
    :param batch_size: Number of sequences.
    :param generator: Seeded RNG, for paired-seed designs.

    :returns: ``(tokens, labels)``, both ``[batch_size, seq_len]`` int64. ``labels`` is
        :data:`IGNORE_INDEX` except at query positions.
    """
    g = generator or torch.Generator().manual_seed(0)
    d, n, v = cfg.num_pairs, cfg.seq_len, cfg.vocab_size
    half = v // 2  # keys are [0, half), values are [half, v)

    # Filler token: the last value id, reserved and never used as a real value. Keys are drawn
    # without replacement so the key->value mapping is unambiguous.
    filler = v - 1

    tokens = torch.full((batch_size, n), filler, dtype=torch.long)
    labels = torch.full((batch_size, n), IGNORE_INDEX, dtype=torch.long)

    for b in range(batch_size):
        keys = torch.randperm(half, generator=g)[:d]
        values = torch.randint(half, v - 1, (d,), generator=g)  # exclude the filler id

        # Pair placement. Power-law gaps concentrate pairs early when power_a is small, which
        # is what Zoology's power_a=0.01 does; sort so the layout stays causal and ordered.
        slots_for_pairs = n - d  # reserve the tail for queries
        max_start = slots_for_pairs - 2 * d
        if max_start > 0:
            u = torch.rand(d, generator=g).numpy()
            gaps = np.power(u, 1.0 / cfg.power_a) if cfg.power_a > 0 else np.zeros(d)
            offsets = np.sort((gaps * max_start).astype(np.int64))
            starts = torch.from_numpy(offsets) + torch.arange(d) * 2
        else:
            starts = torch.arange(d) * 2

        for i in range(d):
            s = int(starts[i])
            tokens[b, s] = keys[i]
            tokens[b, s + 1] = values[i]

        if cfg.random_non_queries:
            mask = tokens[b] == filler
            tokens[b][mask] = torch.randint(0, v, (int(mask.sum()),), generator=g)

        # Queries occupy the final D positions, in shuffled order so position cannot be used
        # as a shortcut for which key is being asked about.
        order = torch.randperm(d, generator=g)
        q0 = n - d
        for j, i in enumerate(order.tolist()):
            tokens[b, q0 + j] = keys[i]
            labels[b, q0 + j] = values[i]

    return tokens, labels


def mqar_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Fraction of query positions answered correctly.

    Only positions where ``labels != IGNORE_INDEX`` count -- unmasked, the ignored positions
    would be scored as wrong and every model would look near-zero.

    :param logits: ``[batch, seq_len, vocab]``.
    :param labels: ``[batch, seq_len]`` with :data:`IGNORE_INDEX` at non-query positions.
    """
    mask = labels != IGNORE_INDEX
    if not mask.any():
        return float("nan")
    pred = logits.argmax(dim=-1)
    return float((pred[mask] == labels[mask]).float().mean())


# --- calibrated settings, established by the positive control (FarmShare 1670928) ------------
#
# VOCAB_SIZE = 256, NOT Zoology's 8192. Measured: at 8192 the best of 6 configurations reached
# 0.214 and four sat exactly at the "it's a value token" plateau (loss 8.32 = ln(4096), i.e. the
# size of the value half). At 256 two configurations reached 0.995 and 1.000. An 8192-way softmax
# over 4 possible answers spends the model's capacity on the output distribution rather than on
# the binding, at this budget. Raise it only alongside a proportionally larger training budget.
CALIBRATED_VOCAB = 256

# Measured solving configurations, both at vocab 256, 8000 steps x batch 64:
#   lr 1e-3, attention at (2,)     -> 0.995
#   lr 3e-3, attention at (1, 3)   -> 1.000
CALIBRATED_LR = 3e-3
CALIBRATED_ATTENTION_LAYERS = (1, 3)

# TRAINING BUDGET IS PART OF THE CALIBRATION, not a free knob.
#
# Job 1670963 failed exactly here: the script had the corrected vocab/lr/attention but the sbatch
# file still carried the old 3000 x 32 = 96k budget, 5.3x below the control's 512k. The same
# N64_D4 config the control solved at 1.000 then scored 0.24/0.25/0.25/0.26/0.93 -- four runs
# parked on the 1/D floor. A budget-starved run is indistinguishable from a too-hard task unless
# you already know the config is solvable, so callers must not silently under-train.
CALIBRATED_STEPS = 8000
CALIBRATED_BATCH_SIZE = 64
CALIBRATED_EXAMPLES = CALIBRATED_STEPS * CALIBRATED_BATCH_SIZE  # 512,000

# For reference: Zoology trains ~6.4M example-presentations (100k examples x 64 epochs). The
# control's 512k is 8% of that and sufficed for the easiest rung -- so harder rungs may well need
# more, and a zero at high D should be read as "budget or difficulty", not difficulty alone.
ZOOLOGY_EXAMPLES = 6_400_000

# THE 1/D FLOOR -- read every result against this, or a degenerate model looks like a good one.
#
# A model that learns "the answer is one of the D values present in this sequence" but cannot
# bind key->value scores exactly 1/D by guessing. At D=4 that is 0.250, and six of the twelve
# control trials landed in 0.208-0.274 with losses of 1.40-1.76 against ln(4) = 1.386. That is
# not partial recall; it is a distinct, fully-learned, wrong algorithm.
#
# Consequence: **the chance baseline is 1/D, not 1/vocab**, and it MOVES with the config. At
# D=64 the floor is 0.016, so an arm scoring 0.10 there is doing real work while an arm scoring
# 0.10 at D=4 is below the degenerate strategy. Always report accuracy against 1/D.
def degenerate_floor(cfg: "MQARConfig") -> float:
    """Accuracy of the "guess among the D values present" strategy: exactly ``1 / num_pairs``."""
    return 1.0 / cfg.num_pairs


# The calibration grid. Zoology's published (N, D) points, plus a harder rung: at the frozen
# geometry a k=3 conv reaches only 2 tokens per layer, so the interesting regime for a
# short-conv hybrid is where distance far exceeds the stacked receptive field.
CALIBRATION_GRID = (
    MQARConfig(seq_len=64, num_pairs=4, vocab_size=CALIBRATED_VOCAB),
    MQARConfig(seq_len=128, num_pairs=8, vocab_size=CALIBRATED_VOCAB),
    MQARConfig(seq_len=256, num_pairs=16, vocab_size=CALIBRATED_VOCAB),
    MQARConfig(seq_len=512, num_pairs=64, vocab_size=CALIBRATED_VOCAB),
)

# Pure retention-distance sweep: D fixed, distance stretched 16x. Any degradation here is about
# holding information, not about how much is held -- and because D is fixed, the 1/D floor is
# constant across the sweep, so the rungs are directly comparable.
DISTANCE_SWEEP = tuple(
    MQARConfig(seq_len=n, num_pairs=8, vocab_size=CALIBRATED_VOCAB)
    for n in (64, 128, 256, 512, 1024)
)
