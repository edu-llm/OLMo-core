"""
Tests for the exact A_5 group and word-problem dataset used by the state-tracking harness.

This is the negative control's foundation, so it has to be exact. ``rotation_test.py`` asks a
different question -- whether the mixer's ``SO(3)`` parameterization can *express* A_5 -- and
works in floating point. Here the group is represented as permutations and composed exactly,
so labels are ground truth with no tolerance.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

MODULE_PATH = Path("src/scripts/train/smoketests/a5_group.py")


@pytest.fixture(scope="module")
def a5() -> ModuleType:
    assert MODULE_PATH.exists(), f"{MODULE_PATH} not found; run pytest from the repo root"
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("a5_group", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_group_has_exactly_sixty_distinct_elements(a5):
    """|A_5| = 60. Anything else means the element set is wrong."""
    assert len(a5.ELEMENTS) == 60
    assert len(set(a5.ELEMENTS)) == 60


def test_every_element_is_an_even_permutation_of_five_symbols(a5):
    """
    A_5 is the *alternating* group: even permutations only.

    Including odd permutations would give S_5, which is also non-solvable, but it would no
    longer be the group the labels claim to be.
    """
    for perm in a5.ELEMENTS:
        assert sorted(perm) == [0, 1, 2, 3, 4]
        inversions = sum(1 for i in range(5) for j in range(i + 1, 5) if perm[i] > perm[j])
        assert inversions % 2 == 0, f"{perm} is an odd permutation"


def test_group_is_closed_and_has_inverses(a5):
    """Closure and inverses make it a group rather than an arbitrary set of permutations."""
    for a in range(60):
        assert a5.compose(a, a5.IDENTITY) == a
        assert a5.compose(a5.IDENTITY, a) == a
        assert any(a5.compose(a, b) == a5.IDENTITY for b in range(60)), "no inverse"
    for a in range(0, 60, 7):
        for b in range(0, 60, 5):
            assert 0 <= a5.compose(a, b) < 60, "composition left the group"


def test_composition_is_associative_and_non_abelian(a5):
    """
    Associativity is what the prefix-product scan relies on; non-commutativity is what makes
    the word problem NC^1-hard in the first place.
    """
    for a, b, c in ((1, 2, 3), (5, 11, 23), (59, 7, 31)):
        assert a5.compose(a5.compose(a, b), c) == a5.compose(a, a5.compose(b, c))
    assert any(
        a5.compose(a, b) != a5.compose(b, a) for a in range(60) for b in range(60)
    ), "group is abelian, so the word problem would be in TC^0"


def test_generators_actually_generate_the_whole_group(a5):
    """
    A generating set that only reaches a subgroup would silently make the task easier -- and a
    proper subgroup of A_5 is solvable, so the task would no longer be NC^1-hard at all.
    """
    assert len(a5.GENERATORS) >= 2
    reached = {a5.IDENTITY}
    frontier = [a5.IDENTITY]
    while frontier:
        nxt = []
        for elem in frontier:
            for gen in a5.GENERATORS:
                candidate = a5.compose(gen, elem)
                if candidate not in reached:
                    reached.add(candidate)
                    nxt.append(candidate)
        frontier = nxt
    assert len(reached) == 60


def test_word_problem_labels_are_the_running_prefix_product(a5):
    """
    The label at position t must be the index of ``g_t . g_{t-1} ... g_1``.

    This is recomputed here independently of however the generator produced it, so an
    off-by-one or a reversed composition order shows up rather than being baked into both
    sides.
    """
    inputs, labels = a5.make_word_problem(batch_size=4, seq_len=16, seed=0)
    assert inputs.shape == (4, 16)
    assert labels.shape == (4, 16)
    assert inputs.dtype == torch.long and labels.dtype == torch.long

    for b in range(inputs.shape[0]):
        running = a5.IDENTITY
        for t in range(inputs.shape[1]):
            running = a5.compose(a5.GENERATORS[int(inputs[b, t])], running)
            assert int(labels[b, t]) == running


def test_word_problem_outputs_are_in_range(a5):
    inputs, labels = a5.make_word_problem(batch_size=8, seq_len=32, seed=1)
    assert int(inputs.min()) >= 0 and int(inputs.max()) < len(a5.GENERATORS)
    assert int(labels.min()) >= 0 and int(labels.max()) < 60


def test_word_problem_is_reproducible_and_seed_dependent(a5):
    """Fixed seeds must be reproducible, or a failed eval cannot be re-examined."""
    first, _ = a5.make_word_problem(batch_size=4, seq_len=16, seed=7)
    same, _ = a5.make_word_problem(batch_size=4, seq_len=16, seed=7)
    different, _ = a5.make_word_problem(batch_size=4, seq_len=16, seed=8)
    assert torch.equal(first, same)
    assert not torch.equal(first, different)


def test_labels_are_not_trivially_predictable(a5):
    """
    A long random word must land near-uniformly across the 60 classes.

    If the label distribution collapsed, a model could score well by predicting a constant and
    the negative control would be meaningless.
    """
    _, labels = a5.make_word_problem(batch_size=64, seq_len=128, seed=3)
    tail = labels[:, -1]
    most_common = int(torch.bincount(tail, minlength=60).max())
    assert most_common < tail.numel() * 0.25
