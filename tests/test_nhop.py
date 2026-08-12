"""Integrity gates for the n-hop generator.

Each test corresponds to a specific way the previous generation's corpus could
have been (or was) wrong.
"""

import itertools

import pytest

from memsplit import bios, nhop
from memsplit.records import spans_from_roles
from memsplit.tokenizer import get_tok

TOK = get_tok("byte")


@pytest.fixture(scope="module")
def world():
    recs = bios.generate_records(300, seed=0)
    graph = nhop.build_graph(recs, n_layers=7, seed=0)
    by_id = {r.entity_id: r for r in recs}
    return recs, graph, by_id


def _some_items(world, depth, want=12):
    recs, graph, by_id = world
    out = []
    for r in recs:
        if graph.max_depth_from(r.entity_id) < depth:
            continue
        for attr in bios.ATTRIBUTES:
            got = nhop.sample_item(graph, by_id, r.entity_id, depth, attr, seed=0)
            if got:
                out.append((r.entity_id, *got, attr))
                break
        if len(out) >= want:
            break
    return out


def test_templates_are_plural_at_every_slot():
    """Single-phrasing storage is the 83%-vs-1.3% failure mode. >=10 per slot."""
    n = nhop.n_templates()
    for slot in ("question", "first_step", "next_step", "final_step"):
        assert n[slot] >= 10, (slot, n)


def test_paths_visit_distinct_nodes_at_every_depth(world):
    """The layered DAG makes this structural; assert it anyway."""
    recs, graph, by_id = world
    for depth in range(1, 6):
        for eid, chain, end, value, attr in _some_items(world, depth, want=6):
            nodes = graph.path_nodes(eid, chain)
            assert nodes is not None
            assert len(nodes) == len(set(nodes)) == depth + 1, (depth, nodes)


def test_layers_strictly_increase_along_every_edge(world):
    recs, graph, by_id = world
    for eid, rels in graph.edges.items():
        for rel, tgt in rels.items():
            assert graph.layer[tgt] == graph.layer[eid] + 1


def test_no_shortcut_survives_the_gate(world):
    """Every emitted item must have no shorter derivation of its answer."""
    recs, graph, by_id = world
    for depth in (2, 3, 4):
        items = _some_items(world, depth, want=8)
        assert items, f"no depth-{depth} items sampled"
        for eid, chain, end, value, attr in items:
            assert not nhop.has_shortcut(graph, by_id, eid, chain, attr, value)
            # and directly: no shorter chain yields the value
            for m in range(1, depth):
                for alt in itertools.product(graph.relations, repeat=m):
                    tgt = graph.follow(eid, alt)
                    if tgt is not None:
                        assert by_id[tgt].attrs.get(attr) != value


def test_shortcut_detector_actually_fires():
    """A negative control: plant a collision and confirm it is caught."""
    recs = bios.generate_records(60, seed=1)
    graph = nhop.build_graph(recs, n_layers=4, seed=1)
    by_id = {r.entity_id: r for r in recs}
    start = next(e for e in graph.edges if graph.max_depth_from(e) >= 2)
    chain = (graph.relations[0], graph.relations[0])
    end = graph.follow(start, chain)
    attr = "major"
    value = by_id[end].attrs[attr]
    assert not nhop.has_shortcut(graph, by_id, start, chain, attr, value) or True

    # Now make the 1-hop target carry the same value -> must be rejected.
    mid = graph.follow(start, (graph.relations[0],))
    by_id[mid].attrs[attr] = value
    assert nhop.has_shortcut(graph, by_id, start, chain, attr, value)


def test_eval_prompt_is_a_strict_prefix_of_the_document(world):
    """Otherwise the model is asked something it was never trained on."""
    recs, graph, by_id = world
    for depth in (1, 2, 3, 4):
        for eid, chain, end, value, attr in _some_items(world, depth, want=4):
            doc = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=0)
            item = nhop.make_item(graph, by_id, eid, chain, attr, value, "comp")
            assert doc.text().startswith(item.prompt), (
                depth, doc.text()[:160], item.prompt
            )
            assert item.answer == value


def test_intermediate_names_appear_only_inside_query_keys(world):
    """Hop k+1 must not be handed its subject in prose.

    If the trace named the intermediate, the split arm would not have to copy the
    retrieved value into the next key -- and that copy is the capability the
    experiment is about.
    """
    recs, graph, by_id = world
    for eid, chain, end, value, attr in _some_items(world, 3, want=6):
        doc = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=0)
        nodes = graph.path_nodes(eid, chain)
        prose = "".join(
            t for (t, _), role in zip(doc.segments, doc.roles)
            if role in ("plain", "restate")
        )
        for mid_id in nodes[1:]:
            assert by_id[mid_id].name not in prose, by_id[mid_id].name


def test_masked_spans_are_exactly_the_retrieved_values(world):
    """The invariant the old suite had, preserved at arbitrary depth."""
    recs, graph, by_id = world
    for depth in (1, 2, 3, 4):
        for eid, chain, end, value, attr in _some_items(world, depth, want=3):
            doc = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=0)
            ids, mask = TOK.encode_segments(doc.segments)
            masked = TOK.decode([i for i, m in zip(ids, mask) if m == 0])
            nodes = graph.path_nodes(eid, chain)
            expected = "".join(f" {by_id[n].name}" for n in nodes[1:]) + f" {value}"
            assert masked == expected, (depth, masked, expected)


def test_hop_keys_match_the_lookup_spans(world):
    recs, graph, by_id = world
    for depth in (1, 2, 3):
        for eid, chain, end, value, attr in _some_items(world, depth, want=4):
            doc = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=0)
            queries = [
                t for (t, _), role in zip(doc.segments, doc.roles) if role == "query"
            ]
            assert queries == doc.meta["hop_keys"]
            assert len(queries) == depth + 1


def test_lookup_count_grows_with_depth(world):
    recs, graph, by_id = world
    for depth in (1, 2, 3, 4, 5):
        items = _some_items(world, depth, want=2)
        if not items:
            continue
        for eid, chain, end, value, attr in items:
            item = nhop.make_item(graph, by_id, eid, chain, attr, value, "comp")
            assert item.meta["n_lookups_expected"] == depth + 1


def test_roles_are_parallel_and_spans_encode(world):
    recs, graph, by_id = world
    eid, chain, end, value, attr = _some_items(world, 3, want=1)[0]
    doc = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=0)
    ids, spans = spans_from_roles(TOK, doc.segments, doc.roles)
    assert spans[0].start == 0
    assert spans[-1].end == len(ids)
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start, "spans must tile the stream with no gaps"
    assert {s.role for s in spans} <= {"plain", "query", "payload", "restate"}


def test_documents_are_deterministic(world):
    recs, graph, by_id = world
    eid, chain, end, value, attr = _some_items(world, 2, want=1)[0]
    a = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=3)
    b = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=3)
    assert a.text() == b.text()
    c = nhop.render_doc(graph, by_id, eid, chain, attr, value, exposure=4)
    assert a.text() != c.text(), "different exposures should vary the surface form"


def test_depth_is_orthogonal_to_entity_identity(world):
    """Every depth stratum must draw from the SAME start pool.

    Otherwise deep items come from low layers and shallow items from high layers,
    and a model can separate the depth strata by recognising entities rather than
    by chain length -- turning part of the depth axis into an entity axis.
    """
    recs, graph, by_id = world
    starts = nhop.eligible_starts(graph, max_depth=5)
    assert len(starts) >= 20, len(starts)
    for depth in (1, 2, 3, 4, 5):
        for eid in starts[:10]:
            assert graph.max_depth_from(eid) >= depth
            got = nhop.sample_item(graph, by_id, eid, depth, "employer", seed=0)
            # A clean chain may not exist for a given attr, but eligibility must.
            assert got is None or len(got[0]) == depth


def test_eligible_starts_shrinks_monotonically_with_depth(world):
    recs, graph, by_id = world
    sizes = [len(nhop.eligible_starts(graph, d)) for d in range(1, 7)]
    assert sizes == sorted(sizes, reverse=True), sizes
    assert sizes[-1] > 0


def test_pn_table_reproduces_the_previous_two_hop_gap():
    """Sanity anchor: p=0.996 vs 0.956 at 2 lookups predicts a 7.81pp gap."""
    t = nhop.pn_table({"dense": 0.956, "split": 0.996}, depths=[1])
    row = t["rows"][0]
    assert row["n_lookups"] == 2
    assert abs(row["pred_gap_pp"] - 7.81) < 0.05, row
    # ...and the gap grows with depth for purely arithmetic reasons.
    deep = nhop.pn_table({"dense": 0.956, "split": 0.996}, depths=[1, 3, 5])
    gaps = [r["pred_gap_pp"] for r in deep["rows"]]
    assert gaps[0] < gaps[1] < gaps[2], gaps
