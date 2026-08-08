import pytest
import torch

from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.generate.diffusion import (
    DiffusionSamplingConfig,
    EarlySkippingConfig,
    EarlySkippingPolicy,
    RemaskingStrategy,
    block_schedule,
    commit_counts,
    select_commits,
)


def test_block_schedule_covers_the_canvas_exactly_once():
    config = DiffusionSamplingConfig(mask_token_id=100278, max_new_tokens=96, block_length=32)
    spans = block_schedule(prompt_len=10, config=config)

    assert spans == [(10, 42), (42, 74), (74, 106)]
    covered = [p for start, end in spans for p in range(start, end)]
    assert covered == list(range(10, 106)), "blocks must tile the canvas with no gap or overlap"


@pytest.mark.parametrize(
    "block_length,steps", [(32, 8), (32, 32), (32, 5), (7, 3), (1, 1), (64, 7)]
)
def test_commit_counts_sum_to_the_block_and_are_front_loaded(block_length: int, steps: int):
    counts = commit_counts(block_length, steps)
    assert len(counts) == steps
    assert sum(counts) == block_length, "every position in the block must be committed exactly once"
    assert all(c >= 1 for c in counts), "an iteration that commits nothing is a wasted forward pass"
    assert counts == sorted(counts, reverse=True), "the remainder belongs on the early iterations"


def test_max_new_tokens_must_divide_into_blocks():
    with pytest.raises(OLMoConfigurationError, match="multiple of block_length"):
        DiffusionSamplingConfig(mask_token_id=0, max_new_tokens=100, block_length=32)


def test_steps_cannot_exceed_the_block_length():
    with pytest.raises(OLMoConfigurationError, match="steps_per_block"):
        DiffusionSamplingConfig(
            mask_token_id=0, max_new_tokens=64, block_length=32, steps_per_block=33
        )


def test_select_commits_takes_the_most_confident_positions():
    # One row, four positions; logits engineered so position 2 is the most confident and 0 the least.
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 5] = 1.0
    logits[0, 1, 5] = 3.0
    logits[0, 2, 5] = 9.0
    logits[0, 3, 5] = 2.0
    still_masked = torch.ones(1, 4, dtype=torch.bool)

    commit_mask, token_ids, confidence = select_commits(logits, still_masked, n_commit=2)

    assert commit_mask[0].tolist() == [False, False, True, True] or commit_mask[0].tolist() == [
        False,
        True,
        True,
        False,
    ]
    # Position 2 is the most confident and must always be in.
    assert commit_mask[0, 2]
    assert (token_ids == 5).all(), "the argmax token is the same everywhere by construction"
    assert confidence[0, 2] == confidence[0].max()


def test_select_commits_never_reselects_a_resolved_position():
    """A committed position is final; reselecting it would overwrite a decision with a stale one."""
    logits = torch.zeros(1, 4, 10)
    logits[0, 0, 5] = 100.0  # by far the most confident, but already resolved
    still_masked = torch.tensor([[False, True, True, True]])

    commit_mask, _, _ = select_commits(logits, still_masked, n_commit=2)

    assert not commit_mask[0, 0]
    assert commit_mask[0].sum() == 2


def test_select_commits_handles_fewer_remaining_than_requested():
    logits = torch.zeros(2, 4, 10)
    still_masked = torch.tensor([[True, False, False, False], [True, True, True, False]])

    commit_mask, _, _ = select_commits(logits, still_masked, n_commit=3)

    # Never more than what is actually still masked in that row.
    assert (commit_mask & ~still_masked).sum() == 0
    assert commit_mask[0].sum() <= 1


def test_random_remasking_differs_from_confidence_remasking():
    """The control that shows confidence ordering is doing something."""
    torch.manual_seed(0)
    logits = torch.randn(1, 64, 50)
    still_masked = torch.ones(1, 64, dtype=torch.bool)

    by_conf, _, _ = select_commits(
        logits, still_masked, 8, strategy=RemaskingStrategy.low_confidence
    )
    g = torch.Generator().manual_seed(1)
    by_rand, _, _ = select_commits(
        logits, still_masked, 8, strategy=RemaskingStrategy.random, generator=g
    )
    assert not torch.equal(by_conf, by_rand)


def _policy(**kwargs) -> EarlySkippingPolicy:
    cfg_kwargs = {"enabled": True, "skip_layers": 8, "refresh_interval": 4}
    cfg_kwargs.update(kwargs)
    return EarlySkippingPolicy(
        config=EarlySkippingConfig(**cfg_kwargs),
        n_layers=16,
        recurrent_layers=tuple(i for i in range(16) if i % 4 != 3),
    )


def test_disabled_policy_skips_nothing():
    policy = EarlySkippingPolicy(config=EarlySkippingConfig(enabled=False), n_layers=16)
    assert policy.skippable_layers() == ()


def test_refresh_iterations_skip_nothing():
    """The paper's guard against accumulating drift, which nothing downstream would detect."""
    policy = _policy(refresh_interval=4)
    seen = []
    for _ in range(8):
        seen.append(len(policy.skippable_layers()) > 0)
        policy.advance()
    # Iterations 0 and 4 are full; the rest may skip.
    assert seen == [False, True, True, True, False, True, True, True]


def test_first_iteration_treats_everything_as_important():
    """With no previous state there is nothing to reuse, so nothing may be skipped on its merits."""
    policy = _policy()
    hidden = torch.randn(2, 12, 32)
    importance = policy.importance(0, hidden, confidence=None)
    torch.testing.assert_close(importance, torch.ones(2, 12))


def test_unchanged_positions_rank_as_least_important():
    policy = _policy(variation_weight=1.0)
    hidden = torch.randn(1, 6, 32)
    policy.record(0, hidden)

    moved = hidden.clone()
    moved[0, 3] += 10.0  # only position 3 changed

    importance = policy.importance(0, moved, confidence=None)
    assert importance[0, 3] == importance[0].max()
    assert importance[0, 3] > 0
    others = [i for i in range(6) if i != 3]
    assert all(importance[0, i] == 0 for i in others)


def test_confidence_lowers_importance():
    """High confidence is what licenses reuse, so it must reduce importance, not raise it."""
    policy = _policy(variation_weight=0.0)
    hidden = torch.randn(1, 4, 16)
    policy.record(0, hidden)
    confident = torch.tensor([[0.99, 0.99, 0.01, 0.99]])

    importance = policy.importance(0, hidden, confidence=confident)
    assert (
        importance[0, 2] == importance[0].max()
    ), "the uncertain position must be the important one"


def test_protected_positions_are_never_skipped():
    """The current block is being decided this iteration; reusing its state reuses the answer."""
    policy = _policy(skip_fraction=0.9)
    importance = torch.zeros(1, 10)  # everything looks skippable
    protected = torch.zeros(1, 10, dtype=torch.bool)
    protected[0, 4:7] = True

    skip = policy.positions_to_skip(importance, protected=protected)

    assert not skip[0, 4:7].any()
    assert skip.sum() > 0, "protecting three positions should not disable skipping entirely"


def test_skip_fraction_zero_skips_nothing():
    policy = _policy(skip_fraction=0.0)
    assert not policy.positions_to_skip(torch.zeros(1, 10)).any()


def test_reset_forgets_previous_states():
    policy = _policy()
    policy.record(0, torch.randn(1, 4, 8))
    policy.advance()
    policy.reset()

    assert policy.is_refresh_iteration(), "a reset must leave the next iteration a full one"
    # And importance falls back to "everything matters".
    torch.testing.assert_close(policy.importance(0, torch.randn(1, 4, 8), None), torch.ones(1, 4))


def test_budget_reports_the_split_and_not_a_product():
    """The recurrent scans early-skipping cannot touch are the ones making the backbone cheap.

    A single speedup number over this architecture invites multiplying ES-dLLM's gain by the
    linear-attention gain, and this is what says why that product is wrong.
    """
    policy = _policy(skip_layers=8)
    policy.advance()  # off a refresh iteration, so skipping is live
    budget = policy.describe_budget()

    # 12 of 16 blocks are recurrent in the 3:1 hybrid.
    assert budget["recurrent_scan_unskippable"] == pytest.approx(12 / 16)
    # Feed-forward is position-independent, so all 8 shallow blocks are reachable.
    assert budget["feed_forward_skippable"] == pytest.approx(8 / 16)
    # Only the attention blocks among those 8 are query-skippable: indices 3 and 7.
    assert budget["attention_query_skippable"] == pytest.approx(2 / 16)
    assert budget["attention_query_skippable"] < budget["feed_forward_skippable"]
