import torch

from olmo_core.nn.flash_pd_ssm import (
    SparseAffineTransition,
    apply_sparse_affine,
    column_one_hot_to_destination,
    compose_sparse_affine,
    destination_to_column_one_hot,
    selected_transition_destination,
    selected_transition_matrix,
    slope_annealed_hardmax,
    sparse_affine_to_dense,
)


def _complex_randn(*shape: int, dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return torch.complex(
        torch.randn(*shape, dtype=dtype),
        torch.randn(*shape, dtype=dtype),
    )


def _random_transition(batch: int, heads: int, state: int) -> SparseAffineTransition:
    return SparseAffineTransition(
        destination=torch.randint(state, (batch, heads, state)),
        scale=_complex_randn(batch, heads, state),
        bias=_complex_randn(batch, heads, state),
    )


def test_sparse_affine_composition_is_associative():
    torch.manual_seed(0)
    a = _random_transition(2, 3, 5)
    b = _random_transition(2, 3, 5)
    c = _random_transition(2, 3, 5)

    left = compose_sparse_affine(c, compose_sparse_affine(b, a))
    right = compose_sparse_affine(compose_sparse_affine(c, b), a)

    torch.testing.assert_close(left.destination, right.destination)
    torch.testing.assert_close(left.scale, right.scale)
    torch.testing.assert_close(left.bias, right.bias)


def test_sparse_affine_application_and_composition_match_dense_matrices():
    torch.manual_seed(1)
    earlier = _random_transition(2, 2, 7)
    later = _random_transition(2, 2, 7)
    state = _complex_randn(2, 2, 7)

    sparse_out = apply_sparse_affine(later, apply_sparse_affine(earlier, state))
    later_matrix, later_bias = sparse_affine_to_dense(later)
    earlier_matrix, earlier_bias = sparse_affine_to_dense(earlier)
    dense_out = (
        torch.einsum(
            "...ij,...j->...i",
            later_matrix,
            torch.einsum("...ij,...j->...i", earlier_matrix, state) + earlier_bias,
        )
        + later_bias
    )
    torch.testing.assert_close(sparse_out, dense_out)

    composed = compose_sparse_affine(later, earlier)
    composed_matrix, composed_bias = sparse_affine_to_dense(composed)
    torch.testing.assert_close(composed_matrix, later_matrix @ earlier_matrix)
    torch.testing.assert_close(
        composed_bias,
        torch.einsum("...ij,...j->...i", later_matrix, earlier_bias) + later_bias,
    )


def test_column_sparse_transition_preserves_many_to_one_collisions():
    """Two source states entering one destination must sum rather than fan out."""
    transition = SparseAffineTransition(
        destination=torch.tensor([0, 0]),
        scale=torch.tensor([1.0, 1.0]),
        bias=torch.zeros(2),
    )

    actual = apply_sparse_affine(transition, torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(actual, torch.tensor([5.0, 0.0]))


def test_column_sparse_composition_uses_source_to_destination_maps():
    """Sparse composition must match p_r[p_l] and collision-preserving bias scatter."""
    earlier = SparseAffineTransition(
        destination=torch.tensor([1, 1, 2]),
        scale=torch.tensor([2.0, 3.0, 5.0]),
        bias=torch.tensor([7.0, 11.0, 13.0]),
    )
    later = SparseAffineTransition(
        destination=torch.tensor([2, 0, 2]),
        scale=torch.tensor([17.0, 19.0, 23.0]),
        bias=torch.tensor([29.0, 31.0, 37.0]),
    )

    composed = compose_sparse_affine(later, earlier)

    torch.testing.assert_close(composed.destination, torch.tensor([0, 0, 2]))
    torch.testing.assert_close(composed.scale, torch.tensor([38.0, 57.0, 115.0]))
    torch.testing.assert_close(composed.bias, torch.tensor([238.0, 31.0, 455.0]))


def test_destination_vector_is_column_one_hot_transition_storage():
    destination = torch.tensor([[[2, 0, 2, 1]]])
    matrix = destination_to_column_one_hot(destination)

    assert matrix.shape == (1, 1, 4, 4)
    torch.testing.assert_close(
        matrix.sum(dim=-2),
        torch.ones_like(destination, dtype=matrix.dtype),
    )
    assert torch.all((matrix == 0) | (matrix == 1))
    torch.testing.assert_close(column_one_hot_to_destination(matrix), destination)


def test_slope_annealed_ste_is_discrete_forward_with_finite_gradients():
    torch.manual_seed(2)
    logits = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)
    weights = torch.randn_like(logits)

    selected = slope_annealed_hardmax(logits, dim=-1, temperature=0.4)
    assert torch.all((selected == 0) | (selected == 1))
    torch.testing.assert_close(selected.sum(dim=-1), torch.ones(3, dtype=torch.float64))

    (selected * weights).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_dictionary_and_timestep_selectors_use_ste():
    torch.manual_seed(3)
    batch, time, heads, dictionary_size, state = 2, 4, 2, 3, 5
    dictionary = torch.randn(
        heads,
        dictionary_size,
        state,
        state,
        dtype=torch.float64,
        requires_grad=True,
    )
    selector = torch.randn(
        batch,
        time,
        heads,
        dictionary_size,
        dtype=torch.float64,
        requires_grad=True,
    )

    transition = selected_transition_matrix(
        dictionary,
        selector,
        temperature=0.6,
    )
    assert transition.shape == (batch, time, heads, state, state)
    assert torch.all((transition == 0) | (transition == 1))
    torch.testing.assert_close(
        transition.sum(dim=-2),
        torch.ones(batch, time, heads, state, dtype=torch.float64),
    )

    weights = torch.randn_like(transition)
    (transition * weights).sum().backward()
    assert dictionary.grad is not None and selector.grad is not None
    assert torch.isfinite(dictionary.grad).all()
    assert torch.isfinite(selector.grad).all()
    assert torch.count_nonzero(dictionary.grad) > 0
    assert torch.count_nonzero(selector.grad) > 0


def test_compact_hard_selection_matches_dense_forward_transition():
    torch.manual_seed(4)
    dictionary = torch.randn(2, 3, 5, 5)
    selector = torch.randn(2, 7, 2, 3)

    destination = selected_transition_destination(dictionary, selector)
    compact_matrix = destination_to_column_one_hot(destination)
    dense_matrix = selected_transition_matrix(
        dictionary,
        selector,
        temperature=1.0,
    )

    torch.testing.assert_close(compact_matrix, dense_matrix)
