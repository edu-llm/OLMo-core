import pytest
import torch

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
