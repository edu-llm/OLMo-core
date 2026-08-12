"""Mask-ledger invariants, including the ones the old corpus violated."""

import numpy as np
import pytest

from memsplit import bios, masking, nhop
from memsplit.records import spans_from_roles
from memsplit.tokenizer import get_tok

TOK = get_tok("byte")


def _doc(depth: int = 2, attr: str = "employer"):
    recs = bios.generate_records(300, seed=0)
    graph = nhop.build_graph(recs, n_layers=7, seed=0)
    by_id = {r.entity_id: r for r in recs}
    for eid in nhop.eligible_starts(graph, depth):
        got = nhop.sample_item(graph, by_id, eid, depth, attr, seed=0)
        if got:
            chain, end, value = got
            return nhop.render_doc(graph, by_id, eid, chain, attr, value, 0)
    raise AssertionError("no item")


def _plan(depth=2, **kw):
    doc = _doc(depth)
    ids, spans = spans_from_roles(TOK, doc.segments, doc.roles)
    return doc, ids, spans, masking.derive_weights(spans, len(ids), seed=0, **kw)


def test_one_stream_all_conditions_same_length():
    """The whole point: every arm indexes the SAME token stream."""
    doc, ids, spans, plan = _plan()
    assert plan.n_tokens == len(ids)
    for cond in masking.CONDITIONS:
        assert plan.weights[cond].shape == (len(ids),)


def test_dense_supervises_everything():
    _, _, _, plan = _plan()
    assert plan.weights["dense"].sum() == plan.n_tokens


def test_split_masks_exactly_the_payload_spans():
    doc, ids, spans, plan = _plan(depth=3)
    payload = np.zeros(len(ids), dtype=bool)
    for s in spans:
        if s.role == "payload":
            payload[s.start : s.end] = True
    assert ((plan.weights["split"] == 0) == payload).all()


def test_controls_supervise_payloads_and_mask_the_same_count():
    """Equal mass is what separates 'masking facts' from 'masking anything'."""
    doc, ids, spans, plan = _plan(depth=3)
    n_pay = plan.diagnostics["n_payload_tokens"]
    for cond in ("random_contig", "random_scatter"):
        w = plan.weights[cond]
        for s in spans:
            if s.role == "payload":
                assert (w[s.start : s.end] == 1).all(), f"{cond} masked a payload"
        assert int((w == 0).sum()) == n_pay, (cond, int((w == 0).sum()), n_pay)


def test_controls_never_touch_query_spans():
    """Queries are the skill under test; masking them changes the experiment."""
    doc, ids, spans, plan = _plan(depth=3)
    for cond in ("random_contig", "random_scatter"):
        for s in spans:
            if s.role == "query":
                assert (plan.weights[cond][s.start : s.end] == 1).all()


def test_controls_never_land_in_a_cue_window():
    """24.5% of control mass previously landed here, biasing it toward treatment."""
    doc, ids, spans, plan = _plan(depth=3)
    cue = masking._cue_tokens(spans, len(ids))
    for cond in ("random_contig", "random_scatter"):
        masked = set(np.where(plan.weights[cond] == 0)[0].tolist())
        assert not (masked & cue), sorted(masked & cue)[:8]


def test_restate_tokens_are_supervised_in_every_condition():
    """The tail is an in-context copy, and it IS supervised. Report it, don't hide it."""
    doc, ids, spans, plan = _plan(depth=2)
    restate = [s for s in spans if s.role == "restate"]
    assert restate, "expected a restatement tail"
    for cond in masking.CONDITIONS:
        for s in restate:
            assert (plan.weights[cond][s.start : s.end] == 1).all()
    assert plan.diagnostics["n_restate_tokens"] > 0


def test_mask_restatements_is_an_ablatable_axis():
    """The tail is 31% of value-token mass; make it a knob, not an assumption."""
    doc, ids, spans, off = _plan(depth=3)
    _, _, spans2, on = _plan(depth=3, mask_restatements=True)

    restate = [s for s in spans if s.role == "restate"]
    assert restate

    for s in restate:
        assert (off.weights["split"][s.start : s.end] == 1).all()
        assert (on.weights["split"][s.start : s.end] == 0).all()

    assert on.diagnostics["n_payload_tokens"] > off.diagnostics["n_payload_tokens"]
    assert off.diagnostics["mask_restatements"] is False
    assert on.diagnostics["mask_restatements"] is True
    # Controls still match the (now larger) treatment mass.
    assert on.diagnostics["count_matched_scatter"]


def test_restated_mass_is_reported_and_excludes_connective_prose():
    """Tagging the whole tail `restate` would inflate an honesty number."""
    doc, ids, spans, plan = _plan(depth=3)
    restate_text = "".join(
        t for (t, _), r in zip(doc.segments, doc.roles) if r == "restate"
    )
    value = doc.meta["value"]
    # Exactly two restatements of the value, and no punctuation or stock phrasing.
    assert restate_text == value * 2, restate_text
    share = plan.diagnostics["n_restate_tokens"] / (
        plan.diagnostics["n_restate_tokens"] + plan.diagnostics["n_payload_tokens"]
    )
    assert 0.15 < share < 0.45, share


def test_scattered_control_prefers_hard_tokens_when_given_a_table():
    """With a difficulty table the scattered control should pick the hard tail."""
    doc, ids, spans, plan_u = _plan(depth=3)
    nll = np.linspace(0.0, 4.0, len(ids))
    _, ids2, spans2, plan_d = _plan(depth=3, token_nll=nll)
    hard = np.where(plan_d.weights["random_scatter"] == 0)[0]
    unif = np.where(plan_u.weights["random_scatter"] == 0)[0]
    assert nll[hard].mean() > nll[unif].mean(), (nll[hard].mean(), nll[unif].mean())
    assert "mean_nll_scatter" in plan_d.diagnostics


def test_difficulty_gap_matches_the_published_tolerance():
    """20% is the preregistered tolerance; the helper must agree with it."""
    assert masking.difficulty_gap(1.9598, 1.2712) == pytest.approx(0.351, abs=0.002)
    assert masking.difficulty_gap(1.9584, 1.7677) == pytest.approx(0.097, abs=0.002)
    assert masking.difficulty_gap(1.9584, 1.8644) == pytest.approx(0.048, abs=0.002)
    assert masking.difficulty_gap(1.9598, 1.2712) > 0.20   # contiguous fails
    assert masking.difficulty_gap(1.9584, 1.7677) < 0.20   # scattered passes


def test_undersupply_raises_rather_than_degrading_silently():
    """A short control that looks matched in the manifest is worse than none."""
    from memsplit.records import Span

    spans = [
        Span(0, 2, "plain"),
        Span(2, 40, "payload"),   # far more payload than eligible prose
        Span(40, 42, "plain"),
    ]
    with pytest.raises(masking.ControlUndersupply):
        masking.derive_weights(spans, 42, seed=0, strict=True)

    plan = masking.derive_weights(spans, 42, seed=0, strict=False)
    assert plan.diagnostics["count_matched_scatter"] is False


def test_aggregate_report_surfaces_the_referee_numbers():
    diags = []
    recs = bios.generate_records(300, seed=0)
    graph = nhop.build_graph(recs, n_layers=7, seed=0)
    by_id = {r.entity_id: r for r in recs}
    made = 0
    for eid in nhop.eligible_starts(graph, 3):
        got = nhop.sample_item(graph, by_id, eid, 3, "employer", seed=0)
        if not got:
            continue
        chain, end, value = got
        doc = nhop.render_doc(graph, by_id, eid, chain, "employer", value, 0)
        ids, spans = spans_from_roles(TOK, doc.segments, doc.roles)
        nll = np.full(len(ids), 1.0)
        for s in spans:
            if s.role == "payload":
                nll[s.start : s.end] = 2.0
        diags.append(
            masking.derive_weights(spans, len(ids), seed=eid, token_nll=nll).diagnostics
        )
        made += 1
        if made >= 6:
            break
    rep = masking.aggregate_report(diags)
    assert rep["n_docs"] == 6
    assert rep["count_matched_contig"] and rep["count_matched_scatter"]
    assert 0.0 < rep["masked_token_frac_split"] < 0.5
    assert rep["difficulty_table_used"]
    assert "difficulty_gap_scatter" in rep
    assert rep["restate_share_of_value_tokens"] > 0
