"""Source-label exception evidence and separately frozen descriptive outputs.

No methods assessment, candidate census or reporting result approves a source
row or dependency membership. Original labels and primary scoring stay fixed.
"""
from __future__ import annotations

import collections
import copy
import itertools
from pathlib import Path
import hashlib
import json

from .common import digest, require
from .data import normalize

REPORTING_SCHEMA = "spacing-confirmation-lane-reporting-policy-v2"
PROPERTY_REVIEW_SCHEMA = "spacing-confirmation-source-label-property-review-v1"
REPORTING_POLICY_SHA256 = "bfe857888ed8a9b8fd1f771e88468344333c760fb38f0176432850bb3b98faee"
REPORTING_SPECIFICATION_SHA256 = "d4dd9ffe1746ca955b022005b3bef04d7495a59adbc7c9060845256cd3ca70b0"
ATTRIBUTES = ("subject", "relation", "modality", "qualifiers", "scope", "answer_slot")


def _reporting_policy_definition():
    """Prospective specification with no real membership or approval claims."""
    value = {
        "schema": REPORTING_SCHEMA,
        "normalization": "data.normalize: Unicode casefold, whitespace, terminal .!?,;: only",
        "K1": {
            "population": "Every accepted lane, final source group, role and registered replicate",
            "counts": "Scored canonical duplicate roots per original unit, never duplicate members or expanded forms",
            "weights": "Old/new: unit root count divided by all scored roots of that role in its final group. Control/QA: each raw original unit is a separate primary aggregation unit; report its identity and full within-unit weight, with no final-group merge assumption. Also disclose the unchanged event-level weight of each primary aggregation unit",
            "invariance": "For a within-group pair with one distinct original normalized label per unit, equal root counts give equal pair contributions when all pair roots receive the same collapsed label and that common label switches between the two originals; unequal counts disclose unequal weights",
            "limits": "No arbitrary target-dependent prediction-swap invariance, answer-NLL invariance or semantic validity is asserted; multi-label units get no collapsed-label invariance claim",
            "cross_group": "No within-group invariance claim for pairs in different primary aggregation units, including raw-unit control/QA roles",
            "primary_reweighted": False,
        },
        "K2": {
            "population": "Every unordered unit pair in every accepted lane, both target sides separately, every saved arm/endpoint/role/section",
            "categories": ["own_label", "sibling_label", "other"],
            "classification": "Compare existing-normalized saved prediction to the exact original labels of its own unit, then this pair's sibling unit, else other",
            "denominator": "Available saved predictions on that target side; disclose missing prediction IDs and full table undefined if any are missing",
            "gold_check": "Each scored row's original gold must belong to its mapped original lane unit",
            "pooling": "No pooled lane score or either-label success metric",
            "missing_section": "Explicitly unavailable; never fabricate absent generation",
        },
        "K5": {
            "score": "Lane-only complement excludes all original units outside the independently accepted narrow lane union",
            "merged_group_qa": "Exclude a merged acquisition QA row if any constituent unit is excluded",
            "weighting": "A2 forms/duplicate cluster first, then source group, dependence block, event, role",
            "inference": "Descriptive paired independently trained replicate mean and 95% interval; retain undefined pairs",
            "A1": "Compare saved unchanged primary sign and interval zero coverage for every contrast/endpoint/role/section/metric",
        },
        "primary_scoring_changed": False, "gold_or_aliases_changed": False,
        "acquisition_gate_or_E_changed": False, "membership_approval_claimed": False,
    }
    value["sha256"] = digest(value)
    return value


def reporting_policy():
    """Read the separate prospective freeze, still requiring fresh final review."""
    return reporting_specification()["reporting_policy"]


def reporting_specification():
    """Return the checked outer timed specification for preregistration binding."""
    from .confirmation_dependency import _sealed, SUPPORTED_ALGORITHM_SHA256
    path = Path(__file__).resolve().parents[1] / "configs/confirmation-lane-reporting-policy-v2-20261008.json"
    artifact = json.loads(path.read_text())
    _sealed(artifact)
    policy = artifact["reporting_policy"]
    _sealed(policy)
    require(artifact.get("schema") == "spacing-confirmation-lane-reporting-specification-freeze-v2" and
            artifact.get("sha256") == REPORTING_SPECIFICATION_SHA256 and
            artifact.get("algorithm_sha256") == SUPPORTED_ALGORITHM_SHA256 and
            artifact.get("confirmation_training_started") is False and artifact.get("approval_claimed") is False and
            artifact.get("fresh_final_review_required") is True and
            policy.get("sha256") == REPORTING_POLICY_SHA256 and policy == _reporting_policy_definition(),
            "Separate prospective lane reporting specification changed or invents approval")
    return artifact


def verify_exception_protocol(protocol):
    from .confirmation_dependency import _sealed
    _sealed(protocol)
    require(protocol.get("schema") == "spacing-sourceqa-prospective-review-protocol-v1" and
            protocol.get("status") == "operator_adopted_criteria_with_launch_obligations_unfulfilled" and
            protocol.get("source_approval_created_by_this_document") is False and
            protocol.get("confirmation_ready") is False and protocol.get("main_launched") is False,
            "Exception protocol must be adopted criteria, never methods-only or row approval")
    require(set(protocol.get("required_controls", {})) >= {
        "K1_probe_balance", "K2_lane_confusion_tables", "K3_surface_correlate_flag",
        "K4_block_and_mask", "K5_A1_A2", "K6_census", "K7_freeze_order",
        "K8_independent_final_reviews", "K9_disclosure"}, "Exception protocol lacks its full uniform controls")
    require(all(protocol.get("exception_rules", {}).get(k) for k in
                ("LE1_eligibility_any_of", "LE1_exclusions", "LE2_membership", "LE3_certification")),
            "Exception protocol lacks property-defined eligibility and discriminator rules")
    repo = Path(__file__).resolve().parents[3]
    for name in ("data.py", "acquisition.py", "encoding.py"):
        relative = "experiments/spacing_rerun/spacing_rerun/" + name
        require(protocol.get("frozen_mechanical_files", {}).get(relative) ==
                hashlib.sha256((repo / relative).read_bytes()).hexdigest(),
                "Exception protocol frozen normalization/mechanical code differs")


def build_exception_lane(candidate, membership, chunks):
    """Bind a proposed narrow lane independently of its broader dependence edge."""
    return _build_exception_lane(candidate, membership, chunks, require_source_approval=True)


def build_exception_lane_candidate(candidate, membership, chunks):
    """Bind geometry to genuine row states without claiming source approval.

    Property review, custody and final membership bind this stable candidate
    digest. Later positive source decisions are bound separately by the policy.
    """
    lane = _build_exception_lane(candidate, membership, chunks, require_source_approval=False)
    from .confirmation import decisions_for
    receipts = {r["sha256"]: r for c in chunks for v in list(c.get("history", [])) + [c] for r in v.get("reviews", [])}
    states = []
    for ref in candidate["source_record_review_bindings"]:
        decision = next(d for d in decisions_for(receipts[ref["source_receipt_sha256"]], "acquisition_records")
                        if d["id"] == ref["record_id"] and d["reviewed_content_sha256"] == ref["reviewed_content_sha256"])
        states.append(dict(ref, actual_decision=copy.deepcopy(decision)))
    return dict(lane, schema="spacing-confirmation-exception-lane-candidate-v1", candidate_only=True,
                source_approval_verified=False, source_decision_states=states)


def _build_exception_lane(candidate, membership, chunks, *, require_source_approval):
    from .confirmation import decisions_for
    require(set(candidate) == {"id", "event", "sub_kind", "eligibility", "source_unit_ids",
                              "acquisition_record_ids", "dependence_recommendation_ids",
                              "source_record_review_bindings", "disclosure_notes"},
            "Lane candidate requires exact narrow members, broader edges and actual row review bindings")
    require(candidate["id"] and candidate["disclosure_notes"] and
            candidate["eligibility"] in ("LE1a", "LE1b", "LE1c") and
            candidate["sub_kind"] == {"LE1a": "conflict", "LE1b": "alias", "LE1c": "original_label_nonuniqueness"}[candidate["eligibility"]],
            "Lane eligibility/sub-kind or nonuniqueness disclosure differs")
    units = candidate["source_unit_ids"]
    require(len(units) >= 2 and units == sorted(set(units)) and set(units) <= set(membership["sources"]) and
            all(membership["sources"][u]["event"] == candidate["event"] for u in units),
            "Lane original members repeat, disappear or cross events")
    labels = {u: membership["sources"][u]["canonical_answer_labels"] for u in units}
    require(all(labels[u] and all(isinstance(v, str) and normalize(v) for v in labels[u]) for u in units) and
            all(not set(map(normalize, labels[a])) & set(map(normalize, labels[b]))
                for a, b in itertools.combinations(units, 2)),
            "Lane labels must be distinct original canonical labels after frozen normalization")
    if candidate["eligibility"] == "LE1a":
        require(set(units) & set(membership["preserved_source_unit_ids"]),
                "Conflict lane lacks an original preserved conflict/ambiguity disposition")
    edges = {e["recommendation_id"]: e for e in membership["edges"]}
    edge_ids = candidate["dependence_recommendation_ids"]
    require(edge_ids and edge_ids == sorted(set(edge_ids)) and set(edge_ids) <= set(edges) and
            all(edges[e]["event"] == candidate["event"] for e in edge_ids),
            "Lane lacks exact same-event dependence finding")
    whole_edge_units = sorted({u for e in edge_ids for u in edges[e]["original_source_unit_ids"]})
    require(set(units) <= set(whole_edge_units) and set(whole_edge_units) <= set(membership["flagged_source_unit_ids"]),
            "Lane requires the whole dependence edge mask, including broader nonlane members")
    blocks = sorted({membership["group_block"][membership["unit_group"][u]] for u in whole_edge_units})
    require(len(blocks) == 1, "Lane and broader dependence edges must share one final block")
    current = {r["id"]: r for c in chunks for r in c["review_packet"]["acquisition_records"]}
    records = candidate["acquisition_record_ids"]
    require(records and records == sorted(set(records)) and set(records) <= set(current),
            "Lane omits or repeats exact current QA identities")
    attributed = {}
    for rid in records:
        row = current[rid]
        own = [u for u in units if u in membership["groups"][row["unit_id"]]["source_unit_ids"] and
               normalize(row["answer"]) in set(map(normalize, labels[u]))]
        require(row["event"] == candidate["event"] and len(own) == 1,
                "Lane QA original own-label attribution is absent or ambiguous")
        attributed[rid] = own[0]
    require(set(attributed.values()) == set(units), "Lane QA membership lost an original lane unit")
    receipts = {r["sha256"]: r for c in chunks for v in list(c.get("history", [])) + [c] for r in v.get("reviews", [])}
    refs = candidate["source_record_review_bindings"]
    require(refs and len({(r["record_id"], r["source_receipt_sha256"]) for r in refs}) == len(refs) and
            {r["record_id"] for r in refs} == set(records), "Every lane QA needs actual exact-content review evidence")
    for ref in refs:
        require(set(ref) == {"record_id", "source_receipt_sha256", "reviewed_content_sha256"} and
                ref["reviewed_content_sha256"] == digest(current[ref["record_id"]]) and
                ref["source_receipt_sha256"] in receipts and any(
                    d.get("id") == ref["record_id"] and isinstance(d.get("status"), str) and d["status"] and
                    (not require_source_approval or d["status"] == "approved") and
                    d.get("reviewed_content_sha256") == ref["reviewed_content_sha256"]
                    for d in decisions_for(receipts[ref["source_receipt_sha256"]], "acquisition_records")),
                "Lane QA lacks the required genuine exact-current source decision")
    value = copy.deepcopy(candidate)
    value.update(original_labels=copy.deepcopy(labels),
                 source_unit_bindings={u: membership["sources"][u]["sha256"] for u in units},
                 source_group_ids=sorted({membership["unit_group"][u] for u in units}),
                 acquisition_record_own_units=attributed,
                 whole_edge_source_unit_ids=whole_edge_units, dependency_block_ids=blocks)
    return value


def validate_lane_source_resolutions(lanes, bindings, chunks):
    """Final eligibility requires positive current decisions outside candidate geometry."""
    from .confirmation import decisions_for
    receipts = {r["sha256"]: r for c in chunks for v in list(c.get("history", [])) + [c] for r in v.get("reviews", [])}
    current = {r["id"]: r for c in chunks for r in c["review_packet"]["acquisition_records"]}
    needed = {r for lane in lanes if lane.get("candidate_only") is True for r in lane["acquisition_record_ids"]}
    require({b.get("record_id") for b in bindings} == needed and len(bindings) == len(needed),
            "Candidate lanes lack exact separate final source approval bindings")
    for ref in bindings:
        require(set(ref) == {"record_id", "source_receipt_sha256", "reviewed_content_sha256"} and
                ref["reviewed_content_sha256"] == digest(current[ref["record_id"]]) and
                ref["source_receipt_sha256"] in receipts and any(
                    d.get("id") == ref["record_id"] and d.get("status") == "approved" and
                    d.get("reviewed_content_sha256") == ref["reviewed_content_sha256"]
                    for d in decisions_for(receipts[ref["source_receipt_sha256"]], "acquisition_records")),
                "Candidate lane has no genuine resolved current source approval")


def validate_lane_inputs(lanes, protocol, property_reviews, membership, chunks, policy_authors, *, original_trace_input_files_by_proof=None):
    """Validate evidence without supplying the later final membership decision."""
    from .confirmation import identity
    from .confirmation_dependency import _completed_review, _sealed, source_author_channels, policy_author_channels, review_channels
    from .sourceqa_input_context import sourceqa_validator_keywords
    require(len({v["id"] for v in lanes}) == len(lanes), "Repeated narrow lane identity")
    if protocol is not None:
        verify_exception_protocol(protocol)
    require(not lanes or protocol is not None, "Exception lanes lack the adopted uniform protocol")
    require(len(property_reviews) == len(lanes) and
            {v["lane_id"] for v in property_reviews} == {v["id"] for v in lanes},
            "Every lane needs one independent exact-content property review, never methods-only approval")
    author_ids = {identity(a) for a in policy_authors}
    author_ids.update(identity(v["authored"]["author"]) for c in chunks
                      for v in list(c.get("history", [])) + [c])
    reviews = {v["lane_id"]: v for v in property_reviews}
    author_channels = source_author_channels(chunks, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof)) | policy_author_channels(policy_authors)
    bindings = []
    for lane in lanes:
        candidate = {k: v for k, v in lane.items() if k not in {
            "original_labels", "source_unit_bindings", "source_group_ids", "acquisition_record_own_units",
            "whole_edge_source_unit_ids", "dependency_block_ids", "schema", "candidate_only",
            "source_approval_verified", "source_decision_states"}}
        builder = build_exception_lane_candidate if lane.get("candidate_only") is True else build_exception_lane
        require(lane == builder(candidate, membership, chunks), "Narrow lane labels, QA, source or broader edge changed")
        review = reviews[lane["id"]]
        receipt = review["receipt"]
        _sealed(receipt)
        require(receipt.get("schema") == PROPERTY_REVIEW_SCHEMA and receipt.get("status") == "approved" and
                receipt.get("lane_sha256") == digest(lane) and receipt.get("protocol_sha256") == protocol["sha256"] and
                receipt.get("unresolved_items") == [] and receipt.get("eligibility") == lane["eligibility"] and
                receipt.get("sub_kind") == lane["sub_kind"] and receipt.get("eligibility_notes") and
                receipt.get("ordinary_qa_defects_resolved") is True,
                "Lane property review is provisional, methods-only or binds different exact content")
        require(all(receipt.get(k) is False for k in
                    ("model_outcomes_seen", "evaluation_wording_seen", "author_rationales_seen", "human_review_complete")),
                "Lane property review lost independent source-only blinding")
        attributes = receipt.get("attribute_certifications", {})
        require(set(attributes) == set(ATTRIBUTES) and all(
            attributes[k].get("no_faithful_nonleaking_semantic_discriminator") is True and attributes[k].get("notes")
            for k in ATTRIBUTES), "Lane requires independent attribute-wise no-discriminator certification")
        require(receipt.get("own_attribution_preserved") is True and
                receipt.get("nonuniqueness_not_created_by_qa") is True and
                receipt.get("original_labels_only") is True and
                receipt.get("surface_correlates_assessed") is True and
                isinstance(receipt.get("surface_correlate_certifications"), list),
                "Lane property review lost required attribution, original labels or correlate disclosure")
        require(identity(receipt["reviewer"]) not in author_ids,
                "Lane property reviewer is a source or policy author")
        binding = _completed_review(receipt, review["evidence"])
        require(not review_channels(binding) & author_channels,
                "Lane property reviewer shares an actual source/policy-author task channel")
        bindings.append(dict(binding, lane_id=lane["id"], lane_sha256=digest(lane)))
    required_surface_correlates(lanes, property_reviews,
                               [r["id"] for c in chunks for r in c["review_packet"]["acquisition_records"]])
    return sorted(bindings, key=lambda b: b["lane_id"])


def required_surface_correlates(lanes, property_reviews, acquisition_record_ids):
    """Derive K3 identifier obligations from exact independent property receipts."""
    required = {rid: set() for rid in acquisition_record_ids}
    require(len(required) == len(acquisition_record_ids), "Repeated current QA in correlate binding")
    by_lane = {v["lane_id"]: v["receipt"] for v in property_reviews}
    seen = set()
    for lane in lanes:
        for certification in by_lane[lane["id"]]["surface_correlate_certifications"]:
            cid = certification.get("id")
            rows, units = certification.get("acquisition_record_ids"), certification.get("source_unit_ids")
            require(isinstance(cid, str) and cid and cid not in seen and
                    isinstance(rows, list) and rows and rows == sorted(set(rows)) and
                    set(rows) <= set(lane["acquisition_record_ids"]) and
                    isinstance(units, list) and units == sorted({lane["acquisition_record_own_units"][rid] for rid in rows}) and
                    certification.get("original_source_backed") is True and certification.get("notes"),
                    "L5 certificate lacks exact source-backed lane QA/own-unit identifiers")
            seen.add(cid)
            for rid in rows:
                require(rid in required, "L5 certificate refers to absent current QA")
                required[rid].add(cid)
    return {rid: sorted(ids) for rid, ids in sorted(required.items())}


def balance_disclosures(lanes, binding, membership):
    """K1 count actual scored roots and disclose unchanged primary weights."""
    maps = {**binding["record_mappings"], **binding["teaching_root_mappings"]}
    counts = collections.Counter((m["role"], m["source_group_id"], m["source_unit_ids"][0]) for m in maps.values())
    group_totals = collections.Counter((m["role"], m["source_group_id"]) for m in maps.values())
    primary_groups = collections.defaultdict(set)
    for meta in maps.values():
        primary_groups[meta["role"], meta["event"]].add(
            meta["source_unit_ids"][0] if meta["role"] in ("control", "qa") else meta["source_group_id"])
    label_counts = collections.Counter()
    for meta in maps.values():
        sid = meta["source_unit_ids"][0]
        source = membership["sources"][sid]
        probe = next(p for p in source["canonical_probes"] if p["opaque_probe_id"] == meta["canonical_probe_id"])
        label = probe.get("source_answer_label")
        if label is None:
            require(len(source["canonical_answer_labels"]) == 1, "Multi-label source root lacks original label provenance")
            label = source["canonical_answer_labels"][0]
        require(normalize(label) in set(map(normalize, source["canonical_answer_labels"])),
                "K1 canonical root label differs from original source labels")
        label_counts[meta["role"], meta["source_group_id"], sid, normalize(label)] += 1
    result = []
    for lane in lanes:
        for role in ("old", "new", "control", "qa"):
            groups = []
            for gid in lane["source_group_ids"]:
                members = [u for u in lane["source_unit_ids"] if membership["unit_group"][u] == gid]
                total = group_totals[role, gid]
                unit_counts = {u: counts[role, gid, u] for u in members}
                single_label = {u: len(set(map(normalize, membership["sources"][u]["canonical_answer_labels"]))) == 1 for u in members}
                primary_ids = {u: u if role in ("control", "qa") else gid for u in members}
                event = membership["groups"][gid]["event"]
                event_primary_count = len(primary_groups[role, event])
                group_weights = {u: (1.0 if role in ("control", "qa") else n / total)
                                 if n and total else None for u, n in unit_counts.items()}
                groups.append({"source_group_id": gid, "scored_canonical_roots_per_lane_unit": unit_counts,
                               "scored_roots_per_unit_original_normalized_label": {u: {label: label_counts[role, gid, u, label]
                                   for label in sorted(set(map(normalize, membership["sources"][u]["canonical_answer_labels"]))) } for u in members},
                               "all_group_scored_canonical_roots": total,
                               "primary_aggregation_unit_ids": primary_ids,
                               "primary_aggregation_units_in_event": event_primary_count,
                               "unchanged_primary_group_weights": group_weights,
                               "unchanged_primary_event_weights": {u: w / event_primary_count if w is not None and event_primary_count else None
                                                                   for u, w in group_weights.items()},
                               "pair_label_swap_disclosures": [{"source_unit_ids": [a, b],
                                   "equal_count_label_swap_invariant": unit_counts[a] == unit_counts[b] if unit_counts[a] and unit_counts[b] and single_label[a] and single_label[b] and primary_ids[a] == primary_ids[b] else None,
                                   "invariance_condition": "All pair roots share one collapsed prediction label, switched between the two distinct originals; no arbitrary prediction or NLL invariance claim",
                                   "status": "no_scored_roots" if not unit_counts[a] or not unit_counts[b] else
                                             "different_primary_aggregation_units" if primary_ids[a] != primary_ids[b] else
                                             "multi_label_no_invariance_claim" if not single_label[a] or not single_label[b] else
                                             "equal_counts" if unit_counts[a] == unit_counts[b] else "unequal_weights"}
                                   for a, b in itertools.combinations(members, 2)]})
            result.append({"lane_id": lane["id"], "role": role, "groups": groups,
                           "cross_group_pairs": [[a, b] for a, b in itertools.combinations(lane["source_unit_ids"], 2)
                                                 if membership["unit_group"][a] != membership["unit_group"][b]],
                           "primary_reweighted": False})
    return result


def prediction_tables(lanes, rows, mapping, gold_labels):
    """K2 all unit pairs with both own targets reported independently."""
    result = []
    for lane in lanes:
        labels = {u: set(map(normalize, vs)) for u, vs in lane["original_labels"].items()}
        for a, b in itertools.combinations(lane["source_unit_ids"], 2):
            sides = []
            for own, sibling in ((a, b), (b, a)):
                counts = dict.fromkeys(("own_label", "sibling_label", "other"), 0)
                missing, records = [], []
                for row in rows:
                    meta = mapping[row["id"]]
                    gold = normalize(gold_labels[row["id"]])
                    require(len(meta["source_unit_ids"]) != 1 or own not in meta["source_unit_ids"] or gold in labels[own],
                            "Saved lane target gold differs from its exact original own labels")
                    if own not in meta["source_unit_ids"] or gold not in labels[own]:
                        continue
                    prediction = row.get("prediction")
                    if not isinstance(prediction, str):
                        missing.append(row["id"])
                        continue
                    normalized = normalize(prediction)
                    category = "own_label" if normalized in labels[own] else "sibling_label" if normalized in labels[sibling] else "other"
                    counts[category] += 1
                    records.append({"id": row["id"], "prediction_category": category})
                total = sum(counts.values())
                sides.append({"own_source_unit_id": own, "sibling_source_unit_id": sibling,
                              "counts": counts if not missing else None, "available_prediction_counts": counts,
                              "frequencies": {k: v / total for k, v in counts.items()} if total and not missing else None,
                              "available_prediction_frequencies": {k: v / total for k, v in counts.items()} if total else None,
                              "available_predictions": total, "missing_prediction_row_ids": missing, "records": records,
                              "status": "undefined_missing_saved_predictions" if missing else "complete" if total else "no_scored_rows"})
            result.append({"lane_id": lane["id"], "source_unit_pair": [a, b], "target_sides": sides,
                           "descriptive_only": True, "either_label_success_metric_computed": False,
                           "gold_or_aliases_changed": False})
    return result


def validate_final_lane_decisions(receipt, lanes):
    """Fresh membership decision explicitly resolves narrow versus broad sets."""
    from .confirmation_dependency import _decisions
    _decisions(receipt.get("lane_decisions"), lanes, "source-label exception lanes")
    decisions = {d["id"]: d for d in receipt["lane_decisions"]}
    for lane in lanes:
        decision = decisions[lane["id"]]
        require(all(decision.get(key) == lane[key] for key in
                    ("source_unit_ids", "acquisition_record_ids", "whole_edge_source_unit_ids", "dependency_block_ids")) and
                decision.get("narrow_lane_vs_broader_dependence_resolved") is True,
                "Final review must explicitly resolve narrow lane versus broader dependence membership")
    require(receipt.get("lane_reporting_policy_approved") is True,
            "Separate descriptive lane reporting policy lacks actual final review")


def complement_contrast(replicates, lanes, *, contrast, endpoint, role, variant, metric):
    """K5 separate A2 lane-only complement and explicit A1 divergence."""
    from .confirmation_dependency import score, paired_summary, divergence
    lane_units = {u for lane in lanes for u in lane["source_unit_ids"]}
    primary, secondary, scores = [], [], []
    for rep in replicates:
        excluded = {u for m in rep["mapping"].values() for u in m["source_unit_ids"]} - lane_units
        a = score(rep["a_rows"], rep["mapping"], metric, excluded_units=excluded)
        b = score(rep["b_rows"], rep["mapping"], metric, excluded_units=excluded)
        primary.append({"replicate_id": rep["replicate_id"], "a": rep["primary_a"], "b": rep["primary_b"]})
        secondary.append({"replicate_id": rep["replicate_id"], "a": a["event_macro"], "b": b["event_macro"]})
        scores.append({"replicate_id": rep["replicate_id"], "a": a, "b": b})
    p, s = paired_summary(primary), paired_summary(secondary)
    return {"contrast": contrast, "endpoint": endpoint, "role": role, "variant": variant, "metric": metric,
            "lane_source_unit_ids": sorted(lane_units), "primary": p, "lane_only_complement": s,
            "primary_secondary_divergence": divergence(p, s), "replicate_scores": scores,
            "weighting": "equal_block", "diagnostic_only": True}
