"""
Tests for the A_5 state-tracking harness (model construction + train/eval loop).

The most important test here is the negative control's own control: an *untrained* model must
score at chance. If it does not, the harness is leaking the answer and every capability number
downstream -- including the headline "b=2 fails, b>=3 succeeds" claim -- is worthless.

Everything runs on CPU at toy sizes. The real capability sweep lives in
``src/test/nn/mamba3/state_tracking_test.py`` and needs a GPU.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

MODULE_PATH = Path("src/scripts/train/smoketests/a5_harness.py")


@pytest.fixture(scope="module")
def harness() -> ModuleType:
    assert MODULE_PATH.exists(), f"{MODULE_PATH} not found; run pytest from the repo root"
    sys.path.insert(0, str(MODULE_PATH.parent))
    try:
        spec = importlib.util.spec_from_file_location("a5_harness", MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _tiny(harness, **kwargs):
    defaults = dict(rotation_block_size=2, n_layers=1, d_model=32, n_heads=2, d_state=12, seed=0)
    defaults.update(kwargs)
    return harness.build_a5_model(**defaults)


def test_model_is_a_pure_mamba3_stack(harness):
    """
    No attention layers.

    The default hybrid inserts attention every fourth block, and an attention layer can
    memorize short sequences outright -- it would solve the training lengths without any state
    tracking and confound the entire experiment.
    """
    model = _tiny(harness)
    type_names = {type(m).__name__ for m in model.modules()}
    assert any("Mamba3Mixer" == n for n in type_names), "no Mamba-3 mixer in the model"
    assert not any("Attention" in n for n in type_names), f"attention present: {type_names}"


@pytest.mark.parametrize("block_size", [2, 3, 4])
def test_model_respects_rotation_block_size(harness, block_size: int):
    """The whole experiment is a comparison across this axis, so it must actually be applied."""
    model = _tiny(harness, rotation_block_size=block_size)
    sizes = {m.rotation_block_size for m in model.modules() if type(m).__name__ == "Mamba3Mixer"}
    assert sizes == {block_size}


def test_harness_default_d_state_covers_at_least_the_library_sweep(harness):
    """
    Pin the relationship between the harness's ``d_state`` and the library's.

    Both answer the same question -- which ``b`` this state size can sweep -- but they were
    chosen independently and landed on different numbers, with nothing tying them together. The
    harness must sweep at least whatever the presets can, and must reach ``b=3``, the smallest
    non-solvable block and the point of the experiment. Read the real default off the signature
    rather than restating it, so this fails if either moves.

    Subset, not *strict* subset: the two agreed once ``DEFAULT_D_STATE`` gained ``b=3``, and that
    agreement is the desired end state rather than a regression.
    """
    import inspect

    from olmo_core.nn.mamba3 import DEFAULT_D_STATE, admissible_block_sizes

    harness_default = inspect.signature(harness.build_a5_model).parameters["d_state"].default

    assert 3 in admissible_block_sizes(harness_default), (
        f"harness d_state={harness_default} cannot express b=3, so the A_5 sweep is impossible"
    )
    assert set(admissible_block_sizes(DEFAULT_D_STATE)) <= set(
        admissible_block_sizes(harness_default)
    ), (
        f"harness d_state={harness_default} must sweep at least what "
        f"DEFAULT_D_STATE={DEFAULT_D_STATE} can"
    )


def test_harness_rejects_a_block_size_its_d_state_cannot_express(harness):
    """The error must name the admissible set, not just say 'not divisible'."""
    with pytest.raises(ValueError, match=r"cannot express rotation_block_size \(5\).*admits"):
        _tiny(harness, rotation_block_size=5)


def test_untrained_model_scores_at_chance(harness):
    """
    An untrained model must be near 1/60 (~1.7%).

    This is the harness's own negative control. A leak -- labels visible in the input, an
    off-by-one making the target the current generator rather than the running product -- would
    show up here as a high score before any training at all.
    """
    model = _tiny(harness)
    accuracy = harness.evaluate_a5(model, seq_len=32, batch_size=16, seed=0)
    assert 0.0 <= accuracy <= 0.15, f"untrained accuracy {accuracy:.1%} is far above chance"


def test_evaluation_is_deterministic_for_a_seed(harness):
    """A failed eval has to be reproducible to be diagnosable."""
    model = _tiny(harness)
    first = harness.evaluate_a5(model, seq_len=16, batch_size=8, seed=3)
    again = harness.evaluate_a5(model, seq_len=16, batch_size=8, seed=3)
    assert first == again


def test_training_reduces_loss(harness):
    """
    The loop must actually optimize.

    This asserts only that the optimizer is wired up and loss descends on a short-sequence
    batch -- not that the model solves A_5, which is a GPU-scale question.
    """
    model = _tiny(harness, d_model=64, n_heads=4)
    losses = harness.train_a5(model, train_len=12, steps=30, batch_size=16, lr=3e-3, seed=0)
    assert len(losses) == 30
    assert all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_accuracy_is_measured_at_the_final_position(harness):
    """
    Scoring must use the final position, not an average over all of them.

    Early positions are easy -- position 0 has only as many possible labels as there are
    generators -- so an all-position average is inflated by prefixes that need no real state
    tracking. Length extrapolation claims have to rest on the hardest position.
    """
    model = _tiny(harness)
    accuracy = harness.evaluate_a5(model, seq_len=8, batch_size=4, seed=0)
    # 16 sequences would give resolution 1/16; 4 sequences gives 1/4. Any returned value must
    # be expressible as k/batch_size if only the last position is scored.
    assert (accuracy * 4) == pytest.approx(round(accuracy * 4)), (
        f"accuracy {accuracy} is not a multiple of 1/batch_size, so it is averaging over "
        f"positions rather than scoring the final one"
    )
