"""
The one place that maps between a model size and a corpus size.

:math:`\\rho` = demanded fact bits / model capacity is the experiment's independent variable, so a
cell sitting at a different :math:`\\rho` than its label does not add noise to the trend, it
destroys it. Everything that needs the mapping calls :func:`solve` or :func:`rho_of` here, and no
caller computes an entity count of its own. :func:`check` is what a config runs to prove the two
agree.

Three things about this arithmetic are worth knowing before you use it.

**Capacity is non-embedding parameters only.** A 32k embedding table is 39% of the smallest model
in the ladder, and none of it stores facts in the sense Allen-Zhu measures. Pass the count read
off a built model, never a round-number label -- see :func:`factcrowd.ladder.sizes.build`.

**Bits per entity is computed, not assumed.** It comes from
:attr:`factcrowd.corpus.entities.EntityTable.bits_per_entity`, which sums ``log2(len(pool))`` over
the declared attribute pools, so it is exact by construction. This module takes it as a required
argument rather than defaulting to 47.6, because a default is how the corpus and the arithmetic
come to disagree about the same number. :data:`BIOS_BITS_PER_ENTITY` exists only as the published
value to check the bioS schema against.

**:math:`\\rho` is nominal until the bit-counter confirms it.** :data:`R_E_AT_200_EXPOSURES` is an
assumed capacity constant with a real uncertainty band, so a cell labelled :math:`\\rho=1` may
truly sit anywhere in roughly 0.86 to 1.2. That shifts the whole x-axis together and so does not
break a trend measured across a 16x sweep, but it is why results are reported against *achieved*
R(F) from :func:`achieved_r` rather than against the label.
"""

import math
from typing import NamedTuple, Optional

from olmo_core.exceptions import OLMoConfigurationError

__all__ = [
    "BIOS_BITS_PER_ENTITY",
    "R_E_AT_200_EXPOSURES",
    "R_E_BAND",
    "EXPOSURES",
    "TOKENS_PER_BIO",
    "CorpusSize",
    "capacity_bits",
    "demanded_bits",
    "resolve_r_e",
    "r_e_loglinear",
    "solve",
    "rho_of",
    "check",
    "achieved_r",
]


BIOS_BITS_PER_ENTITY = 47.6
"""
Bits per person in Physics of Language Models 3.3's bioS schema, ignoring names.

Published reference value only. Live code uses the figure computed from the pools actually in use;
a test asserts the bioS schema reproduces this one, which is what makes our bit-counts comparable
to theirs.
"""

R_E_AT_200_EXPOSURES = 1.2
"""
Assumed capacity in bits per non-embedding parameter at 200 exposures per fact.

Physics 3.3 measures ~2 bits/param at 1000 exposures and ~1 bit/param at 100. 200 exposures sits
between them and 1.2 is the programme's declared interpolation. Note that log-linear
interpolation between those two anchors gives 1.30 instead (:func:`r_e_loglinear`), an 8%
disagreement that moves every entity count by 8%. It shifts the whole :math:`\\rho` axis
together, so it cannot manufacture or hide a trend -- but it is a real open choice, not a
rounding, and P9 is the prediction that settles it from data.
"""

R_E_BAND = (1.0, 1.4)
"""
The uncertainty band on :data:`R_E_AT_200_EXPOSURES`, for sensitivity analysis.

Worth running the grid's :math:`\\rho` labels through both ends before reporting: it is the
difference between "the hinge landed at :math:`\\rho=1`" and "the hinge landed within 20% of
:math:`\\rho=1`", and only the second is defensible from one interpolated constant.
"""

EXPOSURES = 200
"""
Exposures per fact, fixed for every cell in the grid.

Above our measured collapse threshold (between 49 and 196) and above the ~35-exposure floor where
capacity is roughly zero. Fixing it is what decouples entity count from exposure starvation, which
a previous sweep confounded by holding total tokens fixed while raising entity count.
"""

TOKENS_PER_BIO = 100
"""
Measured tokens per rendered biography, used for token-budget arithmetic only.

The real per-cell token count comes from the rendered stream and is recomputed, not trusted. This
constant sizes the budget in advance; :func:`solve`'s ``fact_tokens`` is therefore an estimate and
is documented as one.
"""


class CorpusSize(NamedTuple):
    """
    What a :math:`(P, \\rho)` pair demands of the corpus.

    ``achieved_rho`` is carried alongside the request because ``n_entities`` is an integer: the
    corpus can only approximate a requested :math:`\\rho`, and the residual belongs in the open
    where :func:`check` can see it rather than hidden inside a rounding.
    """

    n_entities: int
    """Number of entities the fact slice must cover."""

    fact_tokens: int
    """Estimated tokens in the fact slice, at :data:`TOKENS_PER_BIO` per exposure."""

    achieved_rho: float
    """The :math:`\\rho` that ``n_entities`` actually realises, after integer rounding."""


def capacity_bits(non_embedding_params: int, r_e: float) -> float:
    """
    The fact capacity of a model, in bits.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param r_e: Bits stored per non-embedding parameter, e.g. :data:`R_E_AT_200_EXPOSURES`.

    :returns: Capacity in bits.

    :raises OLMoConfigurationError: If either argument is not positive.
    """
    if non_embedding_params <= 0:
        raise OLMoConfigurationError(
            f"'non_embedding_params' must be positive, got {non_embedding_params}"
        )
    if r_e <= 0:
        raise OLMoConfigurationError(f"'r_e' must be positive, got {r_e}")
    return non_embedding_params * r_e


def demanded_bits(n_entities: int, bits_per_entity: float) -> float:
    """
    The fact bits a corpus makes available.

    :param n_entities: Number of entities in the fact slice.
    :param bits_per_entity: Exact bits per entity, from the declared attribute pools.

    :returns: Demanded bits.

    :raises OLMoConfigurationError: If either argument is negative, or ``bits_per_entity`` is zero.
    """
    if n_entities < 0:
        raise OLMoConfigurationError(f"'n_entities' must not be negative, got {n_entities}")
    if bits_per_entity <= 0:
        raise OLMoConfigurationError(f"'bits_per_entity' must be positive, got {bits_per_entity}")
    return n_entities * bits_per_entity


def r_e_loglinear(exposures: int) -> float:
    """
    Interpolate capacity per parameter log-linearly between Physics 3.3's two anchors.

    1.0 bits/param at 100 exposures and 2.0 at 1000, linear in ``log10(exposures)``. Offered for
    the sensitivity check described on :data:`R_E_AT_200_EXPOSURES`, not as the grid's default --
    at 200 exposures it returns 1.301 where the programme declared 1.2.

    :param exposures: Exposures per fact.

    :returns: Bits per non-embedding parameter.

    :raises OLMoConfigurationError: If ``exposures`` is not positive.
    """
    if exposures <= 0:
        raise OLMoConfigurationError(f"'exposures' must be positive, got {exposures}")
    return 1.0 + (math.log10(exposures) - 2.0)


def resolve_r_e(exposures: int, r_e: Optional[float] = None) -> float:
    """
    Resolve the capacity constant for an exposure count, refusing to guess.

    An explicit ``r_e`` is returned unchanged. Otherwise only :data:`EXPOSURES` has a declared
    value, and any other exposure count raises rather than silently reusing 1.2 -- because
    capacity per parameter is a function of exposures, and a sweep that changed exposures while
    keeping 1.2 would put every cell at an unknown :math:`\\rho`.

    :param exposures: Exposures per fact.
    :param r_e: An explicit capacity constant, or ``None`` to resolve from ``exposures``.

    :returns: Bits per non-embedding parameter.

    :raises OLMoConfigurationError: If ``r_e`` is ``None`` and ``exposures`` is not
        :data:`EXPOSURES`.
    """
    if r_e is not None:
        return r_e
    if exposures != EXPOSURES:
        raise OLMoConfigurationError(
            f"no declared capacity constant for {exposures} exposures (only {EXPOSURES}). "
            f"Capacity per parameter varies with exposures, so pass 'r_e' explicitly -- "
            f"r_e_loglinear({exposures}) = {r_e_loglinear(exposures):.3f} is one defensible "
            f"choice. Reusing {R_E_AT_200_EXPOSURES} would put every cell at an unknown rho."
        )
    return R_E_AT_200_EXPOSURES


def solve(
    non_embedding_params: int,
    rho: float,
    *,
    bits_per_entity: float,
    exposures: int = EXPOSURES,
    tokens_per_bio: int = TOKENS_PER_BIO,
    r_e: Optional[float] = None,
) -> CorpusSize:
    """
    Size the fact slice for a model and a target :math:`\\rho`.

    This is the forward direction of the experiment's central mapping, and the only sanctioned way
    to obtain an entity count. A config states :math:`\\rho`; ``n_entities`` is derived here so the
    two cannot disagree.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param rho: Target oversubscription ratio, demanded bits over capacity.
    :param bits_per_entity: Exact bits per entity, from the declared attribute pools.
    :param exposures: Exposures per fact. Defaults to :data:`EXPOSURES`.
    :param tokens_per_bio: Tokens per rendered biography, for the token estimate only.
    :param r_e: Capacity per non-embedding parameter. Resolved from ``exposures`` when omitted.

    :returns: The :class:`CorpusSize` this cell demands.

    :raises OLMoConfigurationError: If ``rho`` is not positive, if any size argument is
        non-positive, or if ``r_e`` cannot be resolved for ``exposures``.
    """
    if rho <= 0:
        raise OLMoConfigurationError(f"'rho' must be positive, got {rho}")
    if bits_per_entity <= 0:
        # Checked here as well as in demanded_bits: solve() divides by it directly, so without
        # this the caller gets a ZeroDivisionError from inside the arithmetic instead of a
        # configuration error naming the field they got wrong.
        raise OLMoConfigurationError(f"'bits_per_entity' must be positive, got {bits_per_entity}")
    if tokens_per_bio <= 0:
        raise OLMoConfigurationError(f"'tokens_per_bio' must be positive, got {tokens_per_bio}")
    if exposures <= 0:
        raise OLMoConfigurationError(f"'exposures' must be positive, got {exposures}")

    resolved_r_e = resolve_r_e(exposures, r_e)
    target_bits = rho * capacity_bits(non_embedding_params, resolved_r_e)
    n_entities = round(target_bits / bits_per_entity)
    if n_entities < 1:
        raise OLMoConfigurationError(
            f"rho={rho} against {non_embedding_params:,} non-embedding params at "
            f"{bits_per_entity:.2f} bits/entity rounds to {n_entities} entities. The cell is too "
            f"small to be a corpus; raise rho or the model size."
        )

    return CorpusSize(
        n_entities=n_entities,
        fact_tokens=n_entities * exposures * tokens_per_bio,
        achieved_rho=rho_of(
            non_embedding_params, n_entities, bits_per_entity=bits_per_entity, r_e=resolved_r_e
        ),
    )


def rho_of(
    non_embedding_params: int,
    n_entities: int,
    *,
    bits_per_entity: float,
    exposures: int = EXPOSURES,
    r_e: Optional[float] = None,
) -> float:
    """
    The inverse of :func:`solve`: what :math:`\\rho` an entity count actually realises.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param n_entities: Number of entities in the fact slice.
    :param bits_per_entity: Exact bits per entity, from the declared attribute pools.
    :param exposures: Exposures per fact. Defaults to :data:`EXPOSURES`.
    :param r_e: Capacity per non-embedding parameter. Resolved from ``exposures`` when omitted.

    :returns: The realised :math:`\\rho`.

    :raises OLMoConfigurationError: If any argument is out of range, or if ``r_e`` cannot be
        resolved for ``exposures``.
    """
    resolved_r_e = resolve_r_e(exposures, r_e)
    return demanded_bits(n_entities, bits_per_entity) / capacity_bits(
        non_embedding_params, resolved_r_e
    )


def check(
    non_embedding_params: int,
    rho: float,
    n_entities: int,
    *,
    bits_per_entity: float,
    exposures: int = EXPOSURES,
    r_e: Optional[float] = None,
    tolerance: float = 0.01,
    label: str = "cell",
) -> None:
    """
    Prove that a stated :math:`\\rho` and a stated entity count describe the same cell.

    A cell whose label and corpus disagree is worse than a missing cell: it lands on the trend
    plot at the wrong x. The tolerance is about *specification* disagreement -- integer rounding of
    ``n_entities`` costs well under 0.001% at grid sizes, so anything above a small fraction of a
    percent means two numbers were chosen independently.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param rho: The :math:`\\rho` the cell claims.
    :param n_entities: The entity count the cell will actually generate.
    :param bits_per_entity: Exact bits per entity, from the declared attribute pools.
    :param exposures: Exposures per fact. Defaults to :data:`EXPOSURES`.
    :param r_e: Capacity per non-embedding parameter. Resolved from ``exposures`` when omitted.
    :param tolerance: Maximum tolerated relative disagreement. Defaults to 1%.
    :param label: Name used in the error message, e.g. a ``cell_id``.

    :raises OLMoConfigurationError: If the two disagree by more than ``tolerance``.
    """
    realised = rho_of(
        non_embedding_params,
        n_entities,
        bits_per_entity=bits_per_entity,
        exposures=exposures,
        r_e=r_e,
    )
    relative_error = abs(realised - rho) / rho
    if relative_error > tolerance:
        expected = solve(
            non_embedding_params,
            rho,
            bits_per_entity=bits_per_entity,
            exposures=exposures,
            r_e=r_e,
        )
        raise OLMoConfigurationError(
            f"{label}: rho and n_entities disagree by {relative_error:.1%}, over the "
            f"{tolerance:.1%} tolerance. The config says rho={rho} with "
            f"{n_entities:,} entities, but {n_entities:,} entities against "
            f"{non_embedding_params:,} non-embedding params at {bits_per_entity:.2f} bits/entity "
            f"is rho={realised:.4f}; rho={rho} wants {expected.n_entities:,} entities. Derive "
            f"n_entities from rho with solve() rather than stating both."
        )


def achieved_r(achieved_bits: float, non_embedding_params: int) -> float:
    """
    Achieved R(F): bits the model actually stored, per non-embedding parameter.

    The x-axis of every plot that matters. ``achieved_bits`` comes from the Allen-Zhu bit-counter,
    so this is the measured counterpart to the nominal :math:`\\rho` a config declares -- and the
    reason a nominal-only result would measure nothing.

    :param achieved_bits: Stored bits, summed over value tokens by the bit-counter.
    :param non_embedding_params: Non-embedding parameter count, read off a built model.

    :returns: Achieved bits per non-embedding parameter.

    :raises OLMoConfigurationError: If ``achieved_bits`` is negative or the parameter count is not
        positive.
    """
    if achieved_bits < 0:
        raise OLMoConfigurationError(f"'achieved_bits' must not be negative, got {achieved_bits}")
    if non_embedding_params <= 0:
        raise OLMoConfigurationError(
            f"'non_embedding_params' must be positive, got {non_embedding_params}"
        )
    return achieved_bits / non_embedding_params
