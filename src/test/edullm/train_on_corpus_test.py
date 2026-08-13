"""What ``--scheduler`` resolves to, held to the four answers it has.

The module under test is a script beside the Dockerfile rather than a package, so it is
loaded by path. It reaches ``edullm_data`` only inside ``resolve_corpus``, which nothing
here calls, so the load needs no platform install.
"""

import importlib.util
from pathlib import Path

import pytest

from olmo_core.optim import WSD, CosWithWarmup

TRAIN_ON_CORPUS = Path(__file__).resolve().parents[3] / ".edullm" / "train_on_corpus.py"


@pytest.fixture(scope="module")
def train_on_corpus():
    spec = importlib.util.spec_from_file_location("train_on_corpus", TRAIN_ON_CORPUS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scheduler(module, argv):
    opts, _ = module.build_parser().parse_known_args(argv)
    return module.build_scheduler(opts)


def test_the_default_is_still_cosine(train_on_corpus):
    """Every run this platform has recorded was cosine, so the default may not move."""
    scheduler = _scheduler(train_on_corpus, [])
    assert isinstance(scheduler, CosWithWarmup)
    assert scheduler.warmup == train_on_corpus.build_parser().get_default("warmup_steps")


@pytest.mark.parametrize("argv", [["--scheduler", "cos"], []])
def test_cosine_carries_the_warmup_it_was_given(train_on_corpus, argv):
    scheduler = _scheduler(train_on_corpus, [*argv, "--warmup-steps", "2000"])
    assert isinstance(scheduler, CosWithWarmup)
    assert scheduler.warmup == 2000


def test_wsd_left_alone_decays_by_fraction(train_on_corpus):
    """An unset --decay-steps leaves WSD's own 0.1, which is right when nothing branches."""
    scheduler = _scheduler(train_on_corpus, ["--scheduler", "wsd", "--warmup-steps", "2000"])
    assert isinstance(scheduler, WSD)
    assert scheduler.warmup == 2000
    assert scheduler.decay is None
    assert scheduler.decay_fraction == pytest.approx(0.1)


def test_naming_a_decay_clears_the_fraction(train_on_corpus):
    """WSD refuses a config carrying both, so this is the assertion that keeps it buildable."""
    scheduler = _scheduler(
        train_on_corpus,
        ["--scheduler", "wsd", "--warmup-steps", "2000", "--decay-steps", "8020"],
    )
    assert isinstance(scheduler, WSD)
    assert scheduler.decay == 8020
    assert scheduler.decay_fraction is None


def test_decay_steps_does_not_reach_cosine(train_on_corpus):
    """The flag is documented as ignored by cos; a cosine that read it would be a surprise."""
    scheduler = _scheduler(train_on_corpus, ["--scheduler", "cos", "--decay-steps", "8020"])
    assert isinstance(scheduler, CosWithWarmup)


def test_an_unknown_schedule_is_refused(train_on_corpus):
    with pytest.raises(SystemExit):
        train_on_corpus.build_parser().parse_known_args(["--scheduler", "linear"])
