"""
The alternating group A_5 and its word problem, computed exactly.

A_5 is the smallest non-solvable group. By Barrington's theorem the word problem of any
non-solvable group is NC^1-complete, which makes "track the running product of a sequence of
A_5 generators" the standard probe for whether a sequence model has escaped TC^0. A model whose
transition monoid is abelian -- which is what the Mamba-3 mixer is at ``rotation_block_size=2``,
since ``R(a) R(b) = R(a+b)`` -- provably cannot solve it at length, however well it fits the
training distribution.

The group is represented as permutations of five symbols and composed exactly. Labels are
therefore ground truth with no floating-point tolerance, which matters because this harness is
the negative control: if it leaks, every downstream capability claim is worthless.

Note the deliberate contrast with the ``SO(3)`` construction in ``rotation_test.py``. That one
asks whether the *model's parameterization* can express A_5 and necessarily works in floating
point. This one generates the *data*, and must be exact.
"""

import itertools
from typing import Dict, Tuple

import torch

__all__ = ["ELEMENTS", "IDENTITY", "GENERATORS", "compose", "make_word_problem"]

Permutation = Tuple[int, ...]


def _is_even(perm: Permutation) -> bool:
    """Parity by inversion count: even permutations are exactly the members of A_5."""
    return (
        sum(1 for i in range(len(perm)) for j in range(i + 1, len(perm)) if perm[i] > perm[j]) % 2
        == 0
    )


#: The 60 even permutations of five symbols, in lexicographic order so indices are stable
#: across runs (they end up in checkpoints and eval logs).
ELEMENTS: Tuple[Permutation, ...] = tuple(
    p for p in itertools.permutations(range(5)) if _is_even(p)
)

_INDEX: Dict[Permutation, int] = {perm: i for i, perm in enumerate(ELEMENTS)}

#: Index of the identity permutation.
IDENTITY: int = _INDEX[(0, 1, 2, 3, 4)]


def _apply(left: Permutation, right: Permutation) -> Permutation:
    """Function composition ``left . right``: apply ``right`` first."""
    return tuple(left[right[i]] for i in range(5))


# Full 60x60 Cayley table. Building it once keeps `compose` a constant-time lookup, which
# matters because label generation calls it once per token.
_CAYLEY: Tuple[Tuple[int, ...], ...] = tuple(
    tuple(_INDEX[_apply(a, b)] for b in ELEMENTS) for a in ELEMENTS
)

#: A generating pair for A_5: the 5-cycle (0 1 2 3 4) and the 3-cycle (0 1 2). Both are even,
#: and together they reach all 60 elements -- asserted in the tests, because a generating set
#: that only reached a proper subgroup would make the task solvable and silently trivial.
GENERATORS: Tuple[int, ...] = (
    _INDEX[(1, 2, 3, 4, 0)],
    _INDEX[(1, 2, 0, 3, 4)],
)


def compose(a: int, b: int) -> int:
    """
    Compose two group elements by index, returning the index of ``a . b``.

    :param a: Index of the left (outer) element.
    :param b: Index of the right (inner) element.

    :returns: Index of the product in :data:`ELEMENTS`.
    """
    return _CAYLEY[a][b]


def make_word_problem(
    batch_size: int, seq_len: int, *, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample uniform generator words and label each position with the running prefix product.

    The label at position ``t`` is the index of ``g_t . g_{t-1} ... g_1`` -- newest on the left,
    matching the left-multiplying recurrence the mixer implements. Predicting it at position
    ``t`` requires having tracked the exact group element through all ``t`` steps; there is no
    shortcut, which is precisely why abelian models fail as the sequence grows.

    :param batch_size: Number of sequences.
    :param seq_len: Length of each sequence.
    :param seed: Seed for reproducibility. A failed evaluation has to be re-examinable.

    :returns: ``(inputs, labels)``, both ``(batch_size, seq_len)`` int64. ``inputs`` holds
        indices into :data:`GENERATORS`; ``labels`` holds indices into :data:`ELEMENTS`.
    """
    rng = torch.Generator().manual_seed(seed)
    inputs = torch.randint(len(GENERATORS), (batch_size, seq_len), generator=rng, dtype=torch.long)

    labels = torch.empty_like(inputs)
    running = [IDENTITY] * batch_size
    for t in range(seq_len):
        for b in range(batch_size):
            running[b] = compose(GENERATORS[int(inputs[b, t])], running[b])
            labels[b, t] = running[b]
    return inputs, labels
