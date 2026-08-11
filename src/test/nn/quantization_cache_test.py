import weakref

import pytest
import torch

import olmo_core.nn.quantization as quantization
from olmo_core.nn.quantization import (
    QuantConfig,
    QuantLinear,
    TWNQuantCache,
    reset_twn_quant_caches,
    twn_quantize,
    twn_quantize_ste,
)


def test_cache_matches_uncached_bitwise():
    w = torch.randn(64, 128, dtype=torch.float32)
    cache = TWNQuantCache()
    assert torch.equal(cache.quantize(w, in_dim=-1), twn_quantize(w, in_dim=-1))


def test_cache_hit_returns_same_values_across_calls():
    w = torch.randn(32, 64)
    cache = TWNQuantCache()
    first = cache.quantize(w, in_dim=-1)
    second = cache.quantize(w, in_dim=-1)
    assert torch.equal(first, second)


def test_cache_invalidates_when_latent_weight_changes():
    w = torch.randn(32, 64)
    cache = TWNQuantCache()
    before = cache.quantize(w, in_dim=-1).clone()

    # An optimizer updates the weight in place, which bumps the version counter.
    with torch.no_grad():
        w.add_(torch.randn_like(w))

    after = cache.quantize(w, in_dim=-1)
    assert not torch.equal(before, after)
    assert torch.equal(after, twn_quantize(w, in_dim=-1))


def test_cache_invalidates_when_the_parameter_storage_is_swapped_wholesale():
    # `.data = <fresh tensor>` keeps the Parameter object and restarts the version counter at
    # zero, so an entry stored at version zero matches on both halves of the key while the
    # latent weight underneath it has changed. `Module._apply` -- `.to()`, `.cuda()`, dtype
    # casts -- swaps storage exactly this way, so the collision is reachable in normal use and
    # would silently train against weights that no longer exist.
    param = torch.nn.Parameter(torch.randn(32, 64))
    cache = TWNQuantCache()
    before = cache.quantize(param, in_dim=-1).clone()
    assert param._version == 0

    param.data = torch.randn(32, 64) * 5.0
    assert param._version == 0

    after = cache.quantize(param, in_dim=-1)
    assert not torch.equal(before, after)
    assert torch.equal(after, twn_quantize(param, in_dim=-1))


def test_cache_releases_stale_weight_before_allocating_replacement(monkeypatch):
    w = torch.randn(32, 64)
    cache = TWNQuantCache()
    cache.quantize(w, in_dim=-1)
    stale = weakref.ref(cache._quantized)

    with torch.no_grad():
        w.add_(1)

    original_quantize = quantization.twn_quantize

    def assert_stale_released(*args, **kwargs):
        assert stale() is None
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(quantization, "twn_quantize", assert_stale_released)
    cache.quantize(w, in_dim=-1)


def test_cache_invalidates_for_a_different_tensor():
    cache = TWNQuantCache()
    a = torch.randn(16, 32)
    b = torch.randn(16, 32)
    assert torch.equal(cache.quantize(a, in_dim=-1), twn_quantize(a, in_dim=-1))
    assert torch.equal(cache.quantize(b, in_dim=-1), twn_quantize(b, in_dim=-1))


@pytest.mark.parametrize("cached", [False, True])
def test_gradient_is_identity_ste_either_way(cached: bool):
    w = torch.randn(8, 16, requires_grad=True)
    x = torch.randn(4, 16)

    if cached:
        q = TWNQuantCache().quantize(w, in_dim=-1)
    else:
        q = twn_quantize_ste(w, in_dim=-1)
    (q @ x.T).sum().backward()

    assert w.grad is not None
    # Identity STE: dL/dW_latent is dL/dW_q, which for this loss is x summed over rows.
    torch.testing.assert_close(w.grad, x.sum(dim=0).expand_as(w))


def test_cached_gradient_matches_uncached_across_two_microbatches():
    """A cache hit on the second microbatch must accumulate the same gradient."""
    torch.manual_seed(0)
    batches = [torch.randn(4, 16) for _ in range(2)]

    def run(cached: bool) -> torch.Tensor:
        w = torch.nn.Parameter(torch.randn(8, 16, generator=torch.Generator().manual_seed(1)))
        cache = TWNQuantCache()
        for x in batches:
            q = cache.quantize(w, in_dim=-1) if cached else twn_quantize_ste(w, in_dim=-1)
            (q @ x.T).sum().backward()
        assert w.grad is not None
        return w.grad

    torch.testing.assert_close(run(cached=True), run(cached=False))


def test_quant_linear_cached_matches_uncached():
    torch.manual_seed(0)
    x = torch.randn(4, 32)
    plain = QuantLinear(32, 16, enabled=True, cache_quantized_weight=False)
    cached = QuantLinear(32, 16, enabled=True, cache_quantized_weight=True)
    cached.load_state_dict(plain.state_dict())

    assert plain.quant_cache is None
    assert cached.quant_cache is not None
    for _ in range(3):
        torch.testing.assert_close(cached(x), plain(x))


def test_cache_counts_hits_and_misses_so_a_useless_cache_is_visible():
    """A cache that never hits must be observable, because it is not free.

    The entry holds the latent tensor alive. That costs nothing for a persistent parameter, but
    under FSDP2 the weight the forward sees is a freshly materialized all-gather buffer -- a new
    tensor with new storage on every unshard -- so the key cannot match and the memo degrades to
    pure overhead plus a pinned stale buffer. The flag that enables caching is documented as a
    speedup, so the configuration where it is the opposite needs to be measurable rather than
    inferred.
    """
    cache = TWNQuantCache()
    weight = torch.randn(32, 64)
    for _ in range(4):
        cache.quantize(weight, in_dim=-1)
    assert (cache.hits, cache.misses) == (3, 1)

    # Now the FSDP2 shape: a different tensor object every call, as an all-gather produces.
    resharding = TWNQuantCache()
    for _ in range(4):
        resharding.quantize(torch.randn(32, 64), in_dim=-1)
    assert (resharding.hits, resharding.misses) == (0, 4)


def test_caching_leaves_a_multi_step_training_run_bitwise_unchanged():
    # The single-call tests above compare one forward at a time, which cannot see an entry that
    # outlives the weight it came from. Caching only pays across the microbatches of a gradient
    # accumulation window, so that window is also where a stale entry would do its damage --
    # silently, as a slightly wrong weight rather than an error. Run the loop and require the
    # trajectory to be identical, since caching is meant to buy speed and nothing else.
    def run(*, cached: bool, steps: int = 6, accum: int = 4):
        torch.manual_seed(0)
        layer = QuantLinear(64, 32, enabled=True, cache_quantized_weight=cached)
        opt = torch.optim.AdamW(layer.parameters(), lr=1e-2)
        torch.manual_seed(123)
        batches = [torch.randn(16, 64) for _ in range(steps * accum)]
        losses = []
        for step in range(steps):
            opt.zero_grad()
            for micro in range(accum):
                loss = layer(batches[step * accum + micro]).square().mean() / accum
                loss.backward()
                losses.append(loss.item())
            opt.step()
        return losses, layer.weight.detach().clone()

    uncached_losses, uncached_weight = run(cached=False)
    cached_losses, cached_weight = run(cached=True)

    assert cached_losses == uncached_losses
    assert torch.equal(cached_weight, uncached_weight)


def test_quant_linear_disabled_is_untouched_by_caching():
    x = torch.randn(4, 32)
    control = QuantLinear(32, 16, enabled=False, cache_quantized_weight=True)
    torch.testing.assert_close(control(x), torch.nn.functional.linear(x, control.weight))


def test_reset_clears_reachable_caches():
    model = torch.nn.Sequential(
        QuantLinear(8, 8, enabled=True, cache_quantized_weight=True),
        QuantLinear(8, 8, enabled=True, cache_quantized_weight=True),
    )
    model(torch.randn(2, 8))
    assert reset_twn_quant_caches(model) == 2


def test_config_defaults_to_uncached():
    assert QuantConfig().cache_quantized_weight is False
