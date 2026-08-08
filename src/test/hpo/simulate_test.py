from olmo_core.hpo.ftpfn import ObservedCurve, QueryPoint, assemble_posterior_input
from olmo_core.hpo.objective import CENormalizer
from olmo_core.hpo.simulate import OracleFTPFN, RandomProposer, SyntheticObjective
from olmo_core.hpo.types import CurvePoint


def test_curve_decreases_with_tokens_toward_asymptote():
    obj = SyntheticObjective(optimum=(0.5, 0.5), floor_ce=2.0, ceil_ce=6.0)
    early = obj.ce((0.5, 0.5), t=0.1)
    late = obj.ce((0.5, 0.5), t=1.0)
    assert early > late  # learning reduces CE
    assert late >= obj.floor_ce - 1e-9


def test_config_closer_to_optimum_has_lower_asymptote():
    obj = SyntheticObjective(optimum=(0.5, 0.5))
    near = obj.ce((0.5, 0.5), t=1.0)
    far = obj.ce((0.0, 0.0), t=1.0)
    assert near < far


def test_curve_is_deterministic():
    obj = SyntheticObjective(optimum=(0.3,), noise=0.1, seed=7)
    a = obj.ce((0.42,), t=0.5)
    b = obj.ce((0.42,), t=0.5)
    assert a == b


def test_oracle_posterior_prefers_configs_nearer_optimum():
    obj = SyntheticObjective(optimum=(0.5,), floor_ce=2.0, ceil_ce=6.0)
    norm = CENormalizer(ce_at_zero=6.0, ce_at_one=2.0)
    post = OracleFTPFN(obj, norm)
    observed = [ObservedCurve(1, (0.5,), (CurvePoint(1024, obj.ce((0.5,), 0.25)),))]
    queries = [QueryPoint(1, (0.5,), t=1.0), QueryPoint(0, (0.05,), t=1.0)]
    x = assemble_posterior_input(observed, queries, target_tokens=4096, normalizer=norm)
    pis = post.pi(x, threshold=0.5)
    assert pis[0] > pis[1]  # the near-optimum config is more promising


def test_random_proposer_is_seeded_and_in_unit_cube():
    p1 = RandomProposer(ndim=3, seed=0)
    p2 = RandomProposer(ndim=3, seed=0)
    a = p1.ask(5)
    b = p2.ask(5)
    assert a == b  # same seed -> same proposals
    assert all(len(v) == 3 and all(0.0 <= c <= 1.0 for c in v) for v in a)
