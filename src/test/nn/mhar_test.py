import pytest
import torch

from olmo_core.nn.residual_stream import MHARConfig, MHARRoutingSite

D_MODEL = 64
BATCH, SEQ = 2, 8


def sources(n: int, d_model: int = D_MODEL) -> list:
    return [torch.randn(BATCH, SEQ, d_model) for _ in range(n)]


def test_zero_init_query_makes_routing_a_uniform_average():
    """
    Zero-initialized queries are the paper's own choice and they are load-bearing: every depth
    softmax starts as a uniform average over its sources. A random query makes the softmax
    arbitrarily peaked at step zero, and the paper's web-corpus tables were measured before
    they fixed that.
    """
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=8)
    srcs = sources(5)

    routed = site(srcs)
    torch.testing.assert_close(routed, torch.stack(srcs).mean(dim=0), atol=1e-5, rtol=1e-5)


def test_a_single_source_routes_to_itself():
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=4)
    only = sources(1)
    torch.testing.assert_close(site(only), only[0], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("n_heads", [1, 2, 4, 8, 16])
def test_head_count_is_a_reshape_and_never_changes_the_parameter_count(n_heads: int):
    """
    The one claim that is genuinely zero-parameter. The H queries are a reshape of the same
    d_model numbers, so H=8 is iso-parameter with H=1 -- which is what makes the head sweep a
    clean control rather than a capacity comparison.
    """
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=n_heads)
    assert sum(p.numel() for p in site.parameters()) == 2 * D_MODEL
    assert MHARConfig(n_route_heads=n_heads).num_params(D_MODEL) == 2 * D_MODEL


def test_the_heads_route_independently():
    """
    The whole point of the paper: one query forces every feature subspace to read the depth
    history through one distribution. With H heads the softmaxes are independent, so a query
    that differs between heads must produce different mixtures in different slices.
    """
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=2, zero_init_query=False)
    head = D_MODEL // 2
    with torch.no_grad():
        site.mhar_query.zero_()
        # Head 0 reads the first coordinate strongly; head 1 stays uniform.
        site.mhar_query[0] = 10.0

    srcs = sources(4)
    routed = site(srcs)
    uniform = torch.stack(srcs).mean(dim=0)

    # The untouched head is still the uniform average; the driven head is not.
    torch.testing.assert_close(routed[..., head:], uniform[..., head:], atol=1e-5, rtol=1e-5)
    assert not torch.allclose(routed[..., :head], uniform[..., :head], atol=1e-3)


def test_routing_weights_are_a_convex_combination_over_sources():
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=4, zero_init_query=False)
    srcs = sources(6)
    routed = site(srcs)

    stacked = torch.stack(srcs)
    lo, hi = stacked.min(dim=0).values, stacked.max(dim=0).values
    assert (routed >= lo - 1e-4).all() and (routed <= hi + 1e-4).all()


def test_keys_are_normalized_over_the_full_row_and_values_are_not():
    """
    The authors' reference and their fused kernel both normalize over the whole d_model row and
    then slice into heads -- against their own paper figure, which normalizes per head. And the
    values stay raw. Scaling one source by a constant must therefore move the output, because
    only the key side is scale-invariant.
    """
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=4, zero_init_query=False)
    srcs = sources(3)
    scaled = [srcs[0] * 7.0, srcs[1], srcs[2]]

    assert not torch.allclose(site(srcs), site(scaled), atol=1e-3)


def test_gradients_reach_the_query_and_the_key_gain():
    site = MHARRoutingSite(d_model=D_MODEL, n_route_heads=4, zero_init_query=False)
    site(sources(4)).sum().backward()

    for name, param in site.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name


def test_it_refuses_a_head_count_that_does_not_divide_the_width():
    with pytest.raises(ValueError, match="not divisible"):
        MHARRoutingSite(d_model=100, n_route_heads=8)


def test_it_refuses_an_empty_source_list():
    with pytest.raises(ValueError, match="at least one source"):
        MHARRoutingSite(d_model=D_MODEL)([])


def test_the_paper_s_own_parameter_counts_reproduce():
    """
    (2L+1) sites x 2 x d_model. The paper reports +100K at 350M (d=1024, L=24) and +187K at 1B
    (d=1280, L=36); reproducing both is the check that the site count and per-site width are
    right, since either being wrong would still give a plausible-looking number.
    """
    for d_model, n_layers, expected in ((1024, 24, 100_352), (1280, 36, 186_880)):
        sites = 2 * n_layers + 1
        assert sites * MHARConfig().num_params(d_model) == expected

    # And our own shape, for the record.
    assert (2 * 16 + 1) * MHARConfig().num_params(1024) == 67_584
