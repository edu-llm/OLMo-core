"""Tests for the runner wrapper.

``train_recurrent.py`` rebinds three module globals on ``train_on_corpus``, so what has to
be pinned is that the rebinds take effect through ``train_on_corpus.main``, and that the
residual scale is re-derived after the config merge rather than before it.

``train_on_corpus.py`` is a sibling in the deployed layout. Point ``EDULLM_DIR`` at a
checkout's ``.edullm/`` to run these from this repository::

    EDULLM_DIR=/path/to/OLMo-core/.edullm pytest integrations/olmo-core/
"""

import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo_core")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
if os.environ.get("EDULLM_DIR"):
    sys.path.insert(0, os.environ["EDULLM_DIR"])

train_on_corpus = pytest.importorskip(
    "train_on_corpus",
    reason="needs the fork's .edullm/train_on_corpus.py; set EDULLM_DIR to point at it",
)

import olmo_recurrent as R  # noqa: E402
import train_recurrent  # noqa: E402
from olmo_core.data import NumpyDatasetDType, TokenizerConfig  # noqa: E402


@pytest.fixture
def corpus(monkeypatch, tmp_path):
    """Stand in for the sealed manifest, which needs AWS and a private reader package."""
    shard = tmp_path / "part-00-00000.npy"
    shard.write_bytes(b"\0" * 4096)
    stub = train_on_corpus.Corpus(
        dataset_id="pretrain/regmix-10b",
        version="v1",
        paths=[str(shard)],
        dtype=NumpyDatasetDType.uint32,
        tokenizer=TokenizerConfig.dolma2(),
        rows=None,
    )
    monkeypatch.setattr(train_on_corpus, "resolve_corpus", lambda **_: stub)
    return stub


def parse(*argv):
    return train_on_corpus.build_parser().parse_known_args(list(argv))


def test_the_runner_defaults_to_the_recurrent_factory():
    """The platform default is olmo2_190M, and silently training that would be hard to spot."""
    opts, _ = parse("run")
    assert opts.model_factory == "recurrent_olmo3_370M"


def test_build_config_produces_a_recurrent_model(corpus):
    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt", "--sequence-length", "256")
    config = train_on_corpus.build_config(opts, overrides)
    assert isinstance(config.model, R.RecurrentTransformerConfig)
    assert (config.model.n_prelude, config.model.n_recurrent_layers, config.model.n_coda) == (
        2,
        12,
        2,
    )
    # The wrapper is reached through train_on_corpus's own global, not called directly.
    assert train_on_corpus.build_config is train_recurrent.build_config


def test_n_loops_moves_the_depth_and_the_residual_scale_together(corpus):
    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt", "--n-loops", "2")
    config = train_on_corpus.build_config(opts, overrides)
    model = config.model
    assert model.default_n_loops == 2
    assert model.max_loops == 2
    expected = 1.0 / (2 * math.sqrt(12))
    assert model.residual_alpha == pytest.approx(expected)
    assert model.block_overrides[2].attention_residual_alpha == pytest.approx(expected)


def test_a_dotted_override_of_max_loops_also_re_derives_the_scale(corpus):
    """`config.merge` runs inside build_config, after the factory already wrote the alphas."""
    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt", "model.max_loops=8")
    config = train_on_corpus.build_config(opts, overrides)
    assert config.model.max_loops == 8
    expected = 1.0 / (8 * math.sqrt(12))
    assert config.model.residual_alpha == pytest.approx(expected)
    assert config.model.block_overrides[7].attention_residual_alpha == pytest.approx(expected)


def test_the_depth_schedule_is_off_unless_asked_for(corpus):
    opts, _ = parse("run", "--save-folder", "/tmp/ckpt")
    assert opts.depth_schedule is False
    opts, _ = parse("run", "--save-folder", "/tmp/ckpt", "--depth-schedule")
    assert opts.depth_schedule is True


def test_activation_checkpointing_is_reachable_only_through_the_flag(corpus):
    """train_module.ac_config is None by default, and a dotted merge cannot set a field on None.

    That is the whole reason the flag exists, so both halves are pinned: the override route
    fails, and the flag route works.
    """
    from olmo_core.config import Config

    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt")
    assert train_on_corpus.build_config(opts, overrides).train_module.ac_config is None

    opts, overrides = parse(
        "run", "--save-folder", "/tmp/ckpt", "--activation-checkpointing", "full"
    )
    ac = train_on_corpus.build_config(opts, overrides).train_module.ac_config
    assert ac is not None and ac.mode == "full"

    opts, overrides = parse(
        "run",
        "--save-folder",
        "/tmp/ckpt",
        "--activation-checkpointing",
        "selected_blocks",
        "--ac-block-interval",
        "2",
    )
    ac = train_on_corpus.build_config(opts, overrides).train_module.ac_config
    assert ac.mode == "selected_blocks" and ac.block_interval == 2

    opts, overrides = parse(
        "run", "--save-folder", "/tmp/ckpt", "train_module.ac_config.mode=full"
    )
    with pytest.raises((ValueError, TypeError, AttributeError)):
        train_on_corpus.build_config(opts, overrides)
    del Config


def test_the_wrapper_leaves_a_non_recurrent_factory_alone(corpus):
    """Asking for a stock model through this runner has to give the stock model."""
    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt", "--model-factory", "olmo3_370M")
    config = train_on_corpus.build_config(opts, overrides)
    assert not isinstance(config.model, R.RecurrentTransformerConfig)
    assert config.model.block.attention_residual_alpha is None


def test_the_saved_config_names_the_recurrent_class_and_the_corpus(corpus):
    opts, overrides = parse("run", "--save-folder", "/tmp/ckpt")
    config = train_on_corpus.build_config(opts, overrides)
    as_dict = config.as_config_dict()
    assert as_dict["model"]["_CLASS_"] == "olmo_recurrent.RecurrentTransformerConfig"
    assert as_dict["dataset_id"] == "pretrain/regmix-10b"
    assert as_dict["model"]["default_n_loops"] == 4
