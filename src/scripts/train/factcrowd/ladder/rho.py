"""
The one place that maps between a model size and a corpus size.

**The independent variable is demanded fact bits per parameter.** An earlier revision used
:math:`\\rho` = demanded bits / (R_E · P), which put an assumed constant on both sides of the
equation: R_E defined the x-axis *and* was the predicted outcome, so "the knee sits at
:math:`\\rho=1` by construction" was a tautology and :func:`check` was comparing a quantity to
itself. R_E now appears only in :func:`rho_from_demand`, which converts a demand into the
interpretive :math:`\\rho` scale, and never in placing a cell.

Everything that needs the mapping calls :func:`solve` or :func:`demand_per_param` here, and no
caller computes an entity count of its own. :func:`check` is what a config runs to prove that a
stated demand and a stated entity count describe the same cell.

Four things about this arithmetic are worth knowing before you use it.

**Demand includes a name term.** Physics 3.3's bioS demand is ``N·[log2(N0/N) + log2(S0)]``, where
``N0`` is the size of the name universe. Knowing *that* a given name exists is information, even
though the name is a key rather than a value. The term is +16.4% of *attribute* bits at 714k entities
against a 160M name space and +9.8% at 6.4M (14.1% and 8.9% of total demand), so it bends the trend
rather than shifting it -- and without it
achieved R(F) can exceed R^max, which Remark 4.2 forbids. It also makes demand **nonlinear in N**,
which is why :func:`solve` bisects instead of dividing.

The term behaves differently on the two axes, and the difference is worth holding onto. On the count
axis ``N`` varies, so the term **bends** the curve. On the entropy axis ``N`` is fixed, so it is a
**constant offset** -- at 28M with N = 714,331 it is 0.197 bits/param, which is why that axis's
``b=0`` cell sits at 0.197 rather than at zero. There is no zero-demand cell as long as entities have
distinct names, and saying otherwise would misplace the sweep's intercept.

**Capacity has two defensible bases and they diverge with model size.** Non-embedding parameters is
the primary basis for placing cells: nothing in a tied embedding table is what Allen-Zhu's law is
about. Total parameters is what the paper's own wording suggests ("P … total number of parameters",
excluding only *unused* embedding rows), and for a 32k tied vocab with a BPE trained on our own
corpus nearly every row is used. The two differ **1.67x at 13M falling to 1.22x at 113M, monotone in
model size**, so a design that silently picks one loses the cross-size comparability the size axis
exists for. :class:`Demand` carries both and every plot says which it is. (An earlier draft said
1.67x; the arithmetic gives 1.6504 at a tied 32k vocab and d_model=256.)

**Bits per entity is computed, not assumed.** It comes from
:attr:`factcrowd.corpus.entities.Schema.bits_per_entity`, which sums ``log2(len(pool))`` over the
declared pools, so it is exact by construction. This module takes it as a required argument rather
than defaulting to 47.6, because a default is how a corpus and the arithmetic describing it come to
disagree about the same number. :data:`BIOS_BITS_PER_ENTITY` is here only as the published value to
check a schema against.

**R_E is unknown to about 1.8x.** See :data:`R_E_AT_200_EXPOSURES`. This is why results are reported
against *achieved* R(F) from :func:`achieved_r`, and why M0 measures R_E in our own setup before the
grid places a cell.
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
    "Demand",
    "CorpusSize",
    "capacity_bits",
    "name_bits",
    "demanded_bits",
    "demand_per_param",
    "demand",
    "solve",
    "rho_from_demand",
    "resolve_r_e",
    "r_e_loglinear",
    "check",
    "achieved_r",
]


BIOS_BITS_PER_ENTITY = 47.6
"""
Bits per person in Physics of Language Models 3.3's bioS schema, from its attribute pools alone.

Published reference value only; live code uses the figure computed from the pools actually in use. A
test asserts our bioS schema reproduces this one, which is what makes our bit-counts comparable to
theirs. Note that this is the ``log2(S0)`` half of the demand formula -- the name term is separate
and is added by :func:`demanded_bits`.
"""

R_E_AT_200_EXPOSURES = 1.2
"""
Assumed capacity in bits per parameter at 200 exposures per fact. **Interpretation only.**

Physics 3.3 measures ~2 bits/param at 1000 exposures and ~1 bit/param at 100, and there is **no
200-exposure run**. Log interpolation between those anchors gives 1.30 and linear gives 1.11, so
this constant is a choice inside a band rather than a measurement. Two further reasons to distrust
it: the paper reports gated MLPs at **1.3x lower capacity** than GPT-2 at 100 exposures even with a
tuned learning rate, and ``olmo2_*`` is SwiGLU; and Morris (``arXiv:2505.24832``) reports 3.6
bits/param by a different method. Treat the true value as unknown to roughly 1.8x, plan on 0.9-1.3,
and **measure it once in our own setup** at exposures in {50, 100, 200, 400, 1000} before the grid
places a cell.

Because this appears only in :func:`rho_from_demand`, getting it wrong relabels the x-axis and
cannot bend it. That is the whole reason demand per parameter, not :math:`\\rho`, is what a cell is
placed by.
"""

R_E_BAND = (0.9, 1.4)
"""
The uncertainty band on :data:`R_E_AT_200_EXPOSURES`, for sensitivity analysis.

Widened at the bottom from an earlier 1.0 to account for the SwiGLU penalty noted above. Run the
grid's :math:`\\rho` labels through both ends before reporting: it is the difference between "the
knee landed at :math:`\\rho=1`" and "the knee landed within 30% of :math:`\\rho=1`", and only the
second is defensible from an interpolated constant.
"""

EXPOSURES = 200
"""
Exposures per fact, fixed for every cell in the grid.

Fixing it is what decouples entity count from exposure starvation. A previous sweep held total
tokens fixed while raising entity count, so exposures fell 196 -> 49 -> 12, storage collapsed from
33.1 to 0.20 bits/entity, and nothing could be attributed to entity count rather than to starvation.
"""

TOKENS_PER_BIO = 100
"""
Assumed tokens per rendered biography, for budget arithmetic only.

**Provisional until the attribute vocabulary is real.** Bit accounting depends only on pool sizes
and so is already exact, but token counts depend on the actual strings, and the default schema still
carries placeholder values. The real per-cell token count is recomputed from the rendered stream;
this constant only sizes a budget in advance, which is why :attr:`CorpusSize.fact_tokens` is
documented as an estimate.
"""


class Demand(NamedTuple):
    """
    What a corpus asks of a model, on both parameter bases.

    Carrying both is not indecision. The bases diverge monotonically with model size -- 1.650x at
    13M to 1.217x at 113M -- so a cross-size comparison made on one basis and read as the other is
    wrong by an amount that looks like a trend.
    """

    bits: float
    """Total demanded bits, including the name term."""

    attribute_bits: float
    """The ``N·log2(S0)`` part: bits carried by attribute values alone."""

    name_bits: float
    """The ``N·log2(N0/N)`` part: bits carried by which names exist."""

    per_non_embedding_param: float
    """Demanded bits per non-embedding parameter. The primary axis."""

    per_total_param: float
    """Demanded bits per total parameter, reported alongside."""


class CorpusSize(NamedTuple):
    """
    What a ``(P, demand)`` pair asks of the corpus.

    ``achieved_demand_per_param`` is carried beside the request because ``n_entities`` is an integer:
    the corpus can only approximate a requested demand, and the residual belongs in the open where
    :func:`check` can see it rather than hidden inside a rounding.
    """

    n_entities: int
    """Number of entities the fact slice must cover."""

    fact_tokens: int
    """Estimated tokens in the fact slice, at :data:`TOKENS_PER_BIO` per exposure."""

    achieved_demand_per_param: float
    """The demand per non-embedding parameter that ``n_entities`` actually realises."""


def _require_positive(name: str, value: float) -> None:
    """Raise a configuration error if ``value`` is not strictly positive."""
    if value <= 0:
        raise OLMoConfigurationError(f"'{name}' must be positive, got {value}")


def _require_non_negative(name: str, value: float) -> None:
    """Raise a configuration error if ``value`` is negative."""
    if value < 0:
        raise OLMoConfigurationError(f"'{name}' must not be negative, got {value}")


def capacity_bits(params: int, r_e: float) -> float:
    """
    The fact capacity of a model in bits, on whichever parameter basis the caller passes.

    Excluding embeddings, or not, is the caller's decision and this function cannot check it. A total
    count at ``d_model=256`` overstates a non-embedding basis by 65%, and every entity count derived
    from it with the same factor -- so read the count off a built model via
    :func:`factcrowd.ladder.sizes.build` and say which basis it is.

    :param params: Parameter count on the chosen basis.
    :param r_e: Bits stored per parameter, e.g. :data:`R_E_AT_200_EXPOSURES`.

    :returns: Capacity in bits.

    :raises OLMoConfigurationError: If either argument is not positive.
    """
    _require_positive("params", params)
    _require_positive("r_e", r_e)
    return params * r_e


def name_bits(n_entities: int, name_space: int) -> float:
    """
    The ``N·log2(N0/N)`` term: bits carried by *which* names exist out of the possible names.

    Zero when the corpus uses every available name, and larger the sparser the selection -- which is
    the right shape, because a name drawn from a wider universe is more surprising.

    A consequence worth stating: the size of the name pools is now a load-bearing design parameter
    rather than an arbitrary "must be at least N" choice. Widening them raises demand at fixed
    entity count.

    :param n_entities: Number of entities in the corpus.
    :param name_space: Number of distinct names the name pools can express.

    :returns: Bits contributed by name selection.

    :raises OLMoConfigurationError: If ``n_entities`` exceeds ``name_space``, or ``name_space`` is not
        positive.
    """
    _require_non_negative("n_entities", n_entities)
    _require_positive("name_space", name_space)
    if n_entities == 0:
        # The reasoning-only control. N*log2(N0/N) has the limit 0 as N -> 0, but the expression
        # itself divides by N, so the limit has to be written down rather than evaluated.
        return 0.0
    if n_entities > name_space:
        raise OLMoConfigurationError(
            f"{n_entities:,} entities exceeds a name space of {name_space:,}, so two entities would "
            f"share a name and the corpus would assert contradictory facts about one key"
        )
    return n_entities * math.log2(name_space / n_entities)


def demanded_bits(n_entities: int, bits_per_entity: float, *, name_space: Optional[int]) -> float:
    """
    Total fact bits a corpus makes available, attribute bits plus the name term.

    :param n_entities: Number of entities in the fact slice.
    :param bits_per_entity: Exact bits per entity from the declared attribute pools -- the
        ``log2(S0)`` half.
    :param name_space: Size of the name universe, for the ``log2(N0/N)`` half. **Required, with no
        default**, for the same reason ``bits_per_entity`` is: a default is how a corpus and the
        arithmetic describing it come to disagree about the same number, and this is the specific
        disagreement revision 2 exists to fix. Pass ``None`` to drop the name term deliberately --
        which understates demand by 8-24% of attribute bits and permits achieved R(F) > R^max, so do
        it only to reproduce an attribute-only published figure.

    :returns: Demanded bits.

    :raises OLMoConfigurationError: If any argument is out of range.
    """
    _require_non_negative("n_entities", n_entities)
    # Zero entities is the reasoning-only control: no facts, so no demand, on either half of the sum.
    # Zero *bits per entity* is legal here and nowhere else. The entropy axis's b=0 cell has singleton
    # value pools, so
    # its attribute bits really are zero -- but its demand is not, because distinct names still carry
    # the name term. solve() stays strict: on the linear path it divides by this, and on the name-term
    # path b=0 falls below the monotonicity threshold, so neither path can serve it.
    _require_non_negative("bits_per_entity", bits_per_entity)
    total = n_entities * bits_per_entity
    if name_space is not None:
        total += name_bits(n_entities, name_space)
    return total


def demand(
    n_entities: int,
    *,
    bits_per_entity: float,
    non_embedding_params: int,
    total_params: int,
    name_space: Optional[int],
) -> Demand:
    """
    The full demand picture for a cell, on both parameter bases.

    This is what a :class:`~factcrowd.specs.CellSpec` reports and what the analysis plots against.

    :param n_entities: Number of entities in the fact slice.
    :param bits_per_entity: Exact bits per entity from the declared attribute pools.
    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param total_params: Total parameter count, read off the same model.
    :param name_space: Size of the name universe. See :func:`demanded_bits`.

    :returns: The :class:`Demand`.

    :raises OLMoConfigurationError: If any argument is out of range, or if ``total_params`` is below
        ``non_embedding_params``, which would mean the two were computed from different models.
    """
    _require_positive("non_embedding_params", non_embedding_params)
    _require_positive("total_params", total_params)
    if total_params < non_embedding_params:
        raise OLMoConfigurationError(
            f"'total_params' ({total_params:,}) is below 'non_embedding_params' "
            f"({non_embedding_params:,}), so they cannot describe the same model"
        )

    _require_non_negative("n_entities", n_entities)  # zero is the reasoning-only control
    _require_non_negative("bits_per_entity", bits_per_entity)
    attribute_bits = n_entities * bits_per_entity
    names = name_bits(n_entities, name_space) if name_space is not None else 0.0
    total_bits = attribute_bits + names
    return Demand(
        bits=total_bits,
        attribute_bits=attribute_bits,
        name_bits=names,
        per_non_embedding_param=total_bits / non_embedding_params,
        per_total_param=total_bits / total_params,
    )


def demand_per_param(
    n_entities: int,
    params: int,
    *,
    bits_per_entity: float,
    name_space: Optional[int],
) -> float:
    """
    Demanded bits per parameter -- the experiment's independent variable.

    :param n_entities: Number of entities in the fact slice.
    :param params: Parameter count on the chosen basis.
    :param bits_per_entity: Exact bits per entity from the declared attribute pools.
    :param name_space: Size of the name universe. See :func:`demanded_bits`.

    :returns: Demanded bits per parameter.

    :raises OLMoConfigurationError: If any argument is out of range.
    """
    _require_positive("params", params)
    return demanded_bits(n_entities, bits_per_entity, name_space=name_space) / params


def solve(
    non_embedding_params: int,
    demand_bits_per_param: float,
    *,
    bits_per_entity: float,
    name_space: Optional[int],
    exposures: int = EXPOSURES,
    tokens_per_bio: int = TOKENS_PER_BIO,
    tolerance: float = 0.01,
) -> CorpusSize:
    """
    Size the fact slice for a model and a target demand per parameter.

    The only sanctioned way to obtain an entity count. A config states a demand; ``n_entities`` is
    derived here so the two cannot disagree.

    Demand is **nonlinear in N** once the name term is included, so this bisects rather than
    dividing. The derivative is ``bits_per_entity + log2(N0/N) - 1/ln2``, which is minimised at
    ``N = name_space`` where it equals ``bits_per_entity - 1/ln2``. **So monotonicity -- and with it
    the validity of the bisection -- requires ``bits_per_entity > 1/ln2 = 1.442695``**, and that is
    checked rather than assumed. Below the threshold demand rises, peaks and falls, so a bisection
    would both return non-closest answers and refuse reachable targets; at ``bits_per_entity = 1.0``
    and a 160M name space the broken region is a quarter of the range. Any real schema here clears
    the threshold by an order of magnitude -- the entropy axis's smallest non-zero cell is 24
    bits/entity -- but a one-attribute debug schema would not.

    On an exact tie between two entity counts the higher is kept. Which one is arbitrary; that it is
    deterministic is not.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param demand_bits_per_param: Target demanded bits per non-embedding parameter.
    :param bits_per_entity: Exact bits per entity from the declared attribute pools.
    :param name_space: Size of the name universe. See :func:`demanded_bits`.
    :param exposures: Exposures per fact. Defaults to :data:`EXPOSURES`.
    :param tokens_per_bio: Tokens per rendered biography, for the token estimate only.
    :param tolerance: Maximum tolerated relative gap between the requested demand and what an
        integer entity count can realise. Defaults to 1%, matching :func:`check`, so that
        ``check(solve(...))`` never fails -- which is the invariant that keeps a cell's label and its
        corpus describing the same thing.

    :returns: The :class:`CorpusSize` this cell demands.

    :raises OLMoConfigurationError: If any argument is out of range, if ``bits_per_entity`` is at or
        below ``1/ln2`` while a ``name_space`` is given, or if no integer entity count realises the
        target within ``tolerance``.
    """
    _require_positive("non_embedding_params", non_embedding_params)
    _require_positive("demand_bits_per_param", demand_bits_per_param)
    _require_positive("bits_per_entity", bits_per_entity)
    _require_positive("exposures", exposures)
    _require_positive("tokens_per_bio", tokens_per_bio)

    target_bits = demand_bits_per_param * non_embedding_params
    monotonicity_floor = 1.0 / math.log(2.0)

    def bits_at(n: int) -> float:
        return demanded_bits(n, bits_per_entity, name_space=name_space)

    if name_space is not None and bits_per_entity <= monotonicity_floor:
        raise OLMoConfigurationError(
            f"'bits_per_entity' is {bits_per_entity:.4g}, at or below the 1/ln2 = "
            f"{monotonicity_floor:.6f} threshold where demand stops being monotone in N once the "
            f"name term is included. The derivative is bits_per_entity + log2(N0/N) - 1/ln2, so "
            f"below the threshold demand rises, peaks near N0/e and falls -- and a bisection would "
            f"then return non-closest answers and refuse reachable targets. Either use a schema "
            f"carrying more bits per entity, or pass name_space=None to drop the name term and use "
            f"the linear path."
        )

    if name_space is None:
        n_entities = round(target_bits / bits_per_entity)
    else:
        if bits_at(name_space) < target_bits:
            raise OLMoConfigurationError(
                f"demand of {demand_bits_per_param:.3f} bits/param against "
                f"{non_embedding_params:,} params needs {target_bits:,.0f} bits, more than the "
                f"{bits_at(name_space):,.0f} a name space of {name_space:,} can carry. Widen the "
                f"name pools or lower the demand."
            )
        low, high = 1, name_space
        while low < high:
            mid = (low + high) // 2
            if bits_at(mid) < target_bits:
                low = mid + 1
            else:
                high = mid
        # ``low`` is the smallest N at or above target; the neighbour below may be closer.
        n_entities = low
        if low > 1 and abs(bits_at(low - 1) - target_bits) < abs(bits_at(low) - target_bits):
            n_entities = low - 1

    if n_entities < 1:
        raise OLMoConfigurationError(
            f"a demand of {demand_bits_per_param:.4g} bits/param against "
            f"{non_embedding_params:,} params at {bits_per_entity:.2f} bits/entity rounds to "
            f"{n_entities} entities. The cell is too small to be a corpus; raise the demand or the "
            f"model size."
        )

    achieved = demand_per_param(
        n_entities,
        non_embedding_params,
        bits_per_entity=bits_per_entity,
        name_space=name_space,
    )
    # The bisection brackets in [1, name_space], so it *always* returns something -- including 1 for
    # an unreachably small target, which would be a cell running thousands of times off its label
    # without a word of complaint. Checking what was achieved is what turns that into a refusal, and
    # it makes check(solve(...)) hold by construction rather than by coincidence.
    residual = abs(achieved - demand_bits_per_param) / demand_bits_per_param
    if residual > tolerance:
        cause = (
            f"the closest integer count is {n_entities:,}, and at that size one entity moves the "
            f"demand by more than {tolerance:.1%} -- integer granularity, not a range problem"
            if n_entities < 50
            else "the reachable range for this name space does not contain the target"
        )
        raise OLMoConfigurationError(
            f"no entity count realises a demand of {demand_bits_per_param:.4g} bits/param within "
            f"{tolerance:.1%}: {n_entities:,} entities gives {achieved:.4g}, off by "
            f"{residual:.1%}. {cause}. Raise 'tolerance' if an approximate cell is acceptable, or "
            f"state n_entities directly."
        )

    return CorpusSize(
        n_entities=n_entities,
        fact_tokens=n_entities * exposures * tokens_per_bio,
        achieved_demand_per_param=achieved,
    )


def rho_from_demand(demand_bits_per_param: float, r_e: Optional[float] = None) -> float:
    """
    Convert a demand into the interpretive :math:`\\rho` scale. **Never used to place a cell.**

    :math:`\\rho` = 1 is where Allen-Zhu's law says the knee should be, so this is how a demand is
    read against the literature. It is a presentation transform on an assumed constant, and getting
    R_E wrong relabels the axis without bending it.

    :param demand_bits_per_param: Demanded bits per parameter.
    :param r_e: Capacity per parameter. Defaults to :data:`R_E_AT_200_EXPOSURES`.

    :returns: The oversubscription ratio :math:`\\rho`.

    :raises OLMoConfigurationError: If ``r_e`` is not positive.
    """
    resolved = R_E_AT_200_EXPOSURES if r_e is None else r_e
    _require_positive("r_e", resolved)
    return demand_bits_per_param / resolved


def r_e_loglinear(exposures: int) -> float:
    """
    Interpolate capacity per parameter log-linearly between Physics 3.3's two anchors.

    1.0 bits/param at 100 exposures and 2.0 at 1000, linear in ``log10(exposures)``. At 200 it
    returns 1.301 where the programme declared 1.2 -- offered for the sensitivity check described on
    :data:`R_E_AT_200_EXPOSURES`, not as a default.

    :param exposures: Exposures per fact.

    :returns: Bits per parameter.

    :raises OLMoConfigurationError: If ``exposures`` is not positive.
    """
    _require_positive("exposures", exposures)
    return 1.0 + (math.log10(exposures) - 2.0)


def resolve_r_e(exposures: int, r_e: Optional[float] = None) -> float:
    """
    Resolve the interpretive capacity constant for an exposure count, refusing to guess.

    An explicit ``r_e`` is returned unchanged. Otherwise only :data:`EXPOSURES` has a declared value,
    and any other exposure count raises rather than silently reusing 1.2 -- because capacity per
    parameter is a function of exposures, and a sweep that changed exposures while keeping 1.2 would
    read every cell against the wrong reference.

    :param exposures: Exposures per fact.
    :param r_e: An explicit capacity constant, or ``None`` to resolve from ``exposures``.

    :returns: Bits per parameter.

    :raises OLMoConfigurationError: If ``r_e`` is ``None`` and ``exposures`` is not
        :data:`EXPOSURES`.
    """
    if r_e is not None:
        return r_e
    if exposures != EXPOSURES:
        raise OLMoConfigurationError(
            f"no declared capacity constant for {exposures} exposures (only {EXPOSURES}). "
            f"Capacity per parameter varies with exposures, so pass 'r_e' explicitly -- "
            f"r_e_loglinear({exposures}) = {r_e_loglinear(exposures):.3f} is one defensible choice. "
            f"Reusing {R_E_AT_200_EXPOSURES} would read every cell against the wrong reference."
        )
    return R_E_AT_200_EXPOSURES


def check(
    non_embedding_params: int,
    demand_bits_per_param: float,
    n_entities: int,
    *,
    bits_per_entity: float,
    name_space: Optional[int],
    tolerance: float = 0.01,
    label: str = "cell",
) -> None:
    """
    Prove that a stated demand and a stated entity count describe the same cell.

    A cell whose label and corpus disagree is worse than a missing cell: it lands on the trend plot
    at the wrong x. The tolerance is about *specification* disagreement -- integer rounding of
    ``n_entities`` costs well under 0.001% at grid sizes -- so anything above a small fraction of a
    percent means two numbers were chosen independently.

    :param non_embedding_params: Non-embedding parameter count, read off a built model.
    :param demand_bits_per_param: The demand the cell claims.
    :param n_entities: The entity count the cell will actually generate.
    :param bits_per_entity: Exact bits per entity from the declared attribute pools.
    :param name_space: Size of the name universe. See :func:`demanded_bits`.
    :param tolerance: Maximum tolerated relative disagreement. Defaults to 1%.
    :param label: Name used in the error message, e.g. a ``cell_id``.

    :raises OLMoConfigurationError: If the two disagree by more than ``tolerance``.
    """
    _require_positive("demand_bits_per_param", demand_bits_per_param)
    realised = demand_per_param(
        n_entities,
        non_embedding_params,
        bits_per_entity=bits_per_entity,
        name_space=name_space,
    )
    relative_error = abs(realised - demand_bits_per_param) / demand_bits_per_param
    if relative_error > tolerance:
        # solve() can itself refuse -- an unreachable demand, or a schema below the monotonicity
        # threshold -- and letting that propagate would replace the disagreement report with an
        # unrelated error, losing both the label and the two numbers a reader needs.
        try:
            expected = solve(
                non_embedding_params,
                demand_bits_per_param,
                bits_per_entity=bits_per_entity,
                name_space=name_space,
            )
            wanted = f"{expected.n_entities:,} entities"
        except OLMoConfigurationError as error:
            wanted = f"no reachable entity count ({error})"
        raise OLMoConfigurationError(
            f"{label}: demand and n_entities disagree by {relative_error:.1%}, over the "
            f"{tolerance:.1%} tolerance. The config says {demand_bits_per_param:.4g} bits/param "
            f"with {n_entities:,} entities, but {n_entities:,} entities against "
            f"{non_embedding_params:,} params at {bits_per_entity:.2f} bits/entity is "
            f"{realised:.4g} bits/param; {demand_bits_per_param:.4g} wants {wanted}. Derive "
            f"n_entities from the demand with solve() rather than stating both."
        )


def achieved_r(achieved_bits: float, params: int) -> float:
    """
    Achieved R(F): bits the model actually stored, per parameter.

    The x-axis of every plot that matters. ``achieved_bits`` comes from the Allen-Zhu bit-counter, so
    this is the measured counterpart to the demand a config declares -- and the reason a
    demand-only result would measure nothing.

    :param achieved_bits: Stored bits, summed over value tokens by the bit-counter.
    :param params: Parameter count on the same basis the demand was reported against.

    :returns: Achieved bits per parameter.

    :raises OLMoConfigurationError: If ``achieved_bits`` is negative or ``params`` is not positive.
    """
    if achieved_bits < 0:
        raise OLMoConfigurationError(f"'achieved_bits' must not be negative, got {achieved_bits}")
    _require_positive("params", params)
    return achieved_bits / params
