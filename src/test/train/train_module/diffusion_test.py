import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.train.train_module.transformer import (
    DiffusionLossWeighting,
    DiffusionSchedule,
    DiffusionTransformerTrainModuleConfig,
    MaskedDiffusionConfig,
)

MASK_ID = 100278
IGNORE = -100


def _config(**kwargs) -> MaskedDiffusionConfig:
    kwargs.setdefault("mask_token_id", MASK_ID)
    return MaskedDiffusionConfig(**kwargs)


def test_corrupt_labels_are_not_shifted():
    """The property that separates diffusion from autoregression.

    An off-by-one here trains a model that denoises one position left of the mask it can see. It
    converges, the loss looks fine, and it is not a diffusion model -- so this is pinned rather
    than left to the shape check that would not catch it.
    """
    cfg = _config()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 1000, (4, 64), generator=g)

    corrupted, labels, _ = cfg.corrupt(ids, generator=g)
    masked = corrupted == MASK_ID

    # At a masked position the label is that same position's original token.
    torch.testing.assert_close(labels[masked], ids[masked])
    # Everywhere else there is no label at all.
    assert (labels[~masked] == IGNORE).all()
    # And the clean tokens are untouched.
    torch.testing.assert_close(corrupted[~masked], ids[~masked])


def test_corrupt_hits_the_drawn_rate():
    cfg = _config(antithetic_sampling=False)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 1000, (16, 4096), generator=g)

    corrupted, _, p = cfg.corrupt(ids, generator=g)
    empirical = (corrupted == MASK_ID).float().mean(dim=1)

    # 4096 Bernoulli draws per row, so the standard error is under 0.008.
    torch.testing.assert_close(empirical, p, atol=0.03, rtol=0)


def test_antithetic_draws_are_paired():
    cfg = _config(antithetic_sampling=True, min_mask_probability=0.0)
    p = cfg.sample_mask_probability(8, device=torch.device("cpu"))
    torch.testing.assert_close(p[:4] + p[4:], torch.ones(4), atol=1e-6, rtol=0)


def test_corrupt_respects_unscoreable_positions():
    """Padding and filtered instances must be neither corrupted nor scored."""
    cfg = _config()
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, 1000, (4, 64), generator=g)
    scoreable = torch.zeros_like(ids, dtype=torch.bool)
    scoreable[:, :20] = True

    corrupted, labels, _ = cfg.corrupt(ids, scoreable=scoreable, generator=g)

    torch.testing.assert_close(corrupted[:, 20:], ids[:, 20:])
    assert (labels[:, 20:] == IGNORE).all()


@pytest.mark.parametrize("schedule", [s for s in DiffusionSchedule])
def test_schedules_stay_in_range_and_are_monotone(schedule: DiffusionSchedule):
    t = torch.linspace(0.0, 1.0, 101)
    p = schedule.mask_probability(t)
    assert p.min() >= 0.0 and p.max() <= 1.0
    assert (p.diff() >= -1e-6).all(), "a noise schedule must not decrease in t"


def test_min_mask_probability_is_clamped():
    """A sequence drawn at zero spends a whole forward and backward pass on no loss at all."""
    cfg = _config(min_mask_probability=0.01)
    p = cfg.sample_mask_probability(256, device=torch.device("cpu"))
    assert p.min() >= 0.01


def test_elbo_weighting_is_refused_rather_than_ignored():
    with pytest.raises(OLMoConfigurationError, match="not implemented"):
        _config(loss_weighting=DiffusionLossWeighting.elbo)


def test_bad_probability_bounds_are_refused():
    with pytest.raises(OLMoConfigurationError):
        _config(min_mask_probability=0.5, max_mask_probability=0.5)


def test_train_module_config_requires_diffusion():
    from olmo_core.optim import AdamWConfig

    cfg = DiffusionTransformerTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=AdamWConfig(lr=1e-4),
    )
    with pytest.raises(OLMoConfigurationError, match="'diffusion' is required"):
        cfg.build(model=None)  # type: ignore[arg-type]


def test_train_module_config_refuses_pipeline_parallelism():
    """`TransformerPipelineTrainModule` has its own train_batch, so it would train AR in silence."""
    from olmo_core.optim import AdamWConfig
    from olmo_core.train.train_module.transformer import (
        TransformerPipelineParallelConfig,
    )

    cfg = DiffusionTransformerTrainModuleConfig(
        rank_microbatch_size=1024,
        max_sequence_length=512,
        optim=AdamWConfig(lr=1e-4),
        diffusion=_config(),
        pp_config=TransformerPipelineParallelConfig(degree=2),
    )
    with pytest.raises(OLMoConfigurationError, match="pipeline parallelism"):
        cfg.build(model=None)  # type: ignore[arg-type]
