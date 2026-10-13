"""Prospective confirmation diagnostics; never modify primary scores or eligibility."""
from __future__ import annotations

from .sourceqa_input_context import sourceqa_validator_keywords
import collections
import copy
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from .common import digest, require

SUPPORTED_ALGORITHM_SHA256 = "91e7e9a7015fe83dae9be1c5b8ab28b6322f0971354ea75e802d2da9bac62503"
FINAL_INPUT_SCHEMA = "spacing-confirmation-dependency-final-policy-input-v2"
FINAL_MEMBERSHIP_SCHEMA = "spacing-confirmation-dependency-final-membership-v1"
FINAL_REVIEW_SCHEMA = "spacing-confirmation-dependency-final-policy-review-v2"
FROZEN_POLICY_SCHEMA = "spacing-confirmation-dependency-frozen-policy-v2"
LEGACY_MODEL_ALIASES = frozenset({
    ("claude opus 5.5", "claude-opus-5-5"),
    ("claude-opus-5.5", "claude-opus-5-5"),
    ("opus 5.5", "claude-opus-5-5"),
})


def algorithm_v3():
    """The exact supported specification, still requiring actual fresh review."""
    path = Path(__file__).resolve().parents[1] / "configs/confirmation-dependency-algorithm-v3-r2-20261008.json"
    artifact = json.loads(path.read_text())
    _sealed(artifact)
    require(digest(artifact["algorithm"]) == SUPPORTED_ALGORITHM_SHA256,
            "Supported V3 algorithm file changed")
    return artifact["algorithm"]


def _sealed(value):
    require(value.get("sha256") == digest({k: v for k, v in value.items() if k != "sha256"}),
            "Source-only dependency input seal differs")


def membership_proposal(packets, recommendations, flagged_units=()):
    """Project exact original units into current groups without claiming approval.

    Recommendations retain their old receipt provenance. Overlaps form proposed
    within-event components only; an independent reviewer must accept every edge
    before the final membership binding.
    """
    groups, sources, parent, bindings = {}, {}, {}, []
    for packet in packets:
        _sealed(packet)
        require(packet.get("author_rationales_included") is False,
                "Membership proposal needs exact source-only blind packets")
        bindings.append({"chunk_index": packet["chunk_index"], "packet_sha256": packet["sha256"],
                         "authored_chunk_sha256": packet["authored_chunk_sha256"]})
        for group in packet["groups"]:
            require(group["id"] not in groups, "Duplicate current group")
            groups[group["id"]] = group
            parent[group["id"]] = group["id"]
        for source in packet["source_units"]:
            require(source["unit_id"] not in sources, "Duplicate original source unit")
            sources[source["unit_id"]] = source
    unit_group = {}
    for gid, group in groups.items():
        for sid in group["source_unit_ids"]:
            require(sid in sources and sid not in unit_group and
                    sources[sid]["event"] == group["event"], "Missing, repeated or cross-event source membership")
            unit_group[sid] = gid
    require(set(unit_group) == set(sources), "Current grouping lost original units")
    def root(gid):
        while parent[gid] != gid:
            parent[gid] = parent[parent[gid]]
            gid = parent[gid]
        return gid
    edges = []
    for rec in recommendations:
        for record in rec["records"]:
            for prior in record["sources"]:
                require(prior["unit_id"] in sources and all(
                    sources[prior["unit_id"]][key] == prior[key]
                    for key in ("source_assertion", "canonical_answer_labels", "sha256")),
                    "Recommendation original assertion/labels/content binding differs")
        sids = sorted({sid for record in rec["records"] for sid in record["group"]["source_unit_ids"]})
        require(sids and set(sids) <= set(sources), "Recommendation lost original unit provenance")
        gids = sorted({unit_group[sid] for sid in sids})
        require({groups[gid]["event"] for gid in gids} == {rec["event"]},
                "Recommendation crosses or changes event")
        for gid in gids[1:]:
            left, right = root(gids[0]), root(gid)
            parent[max(left, right)] = min(left, right)
        edges.append({"recommendation_id": rec["id"], "event": rec["event"],
                      "original_source_unit_ids": sids, "current_group_ids": gids,
                      "original_recommendation": rec, "independently_accepted": False})
    components = collections.defaultdict(list)
    for gid in sorted(groups):
        components[root(gid)].append(gid)
    group_block, blocks = {}, []
    for gids in components.values():
        event = groups[gids[0]]["event"]
        identity = "dep-" + digest({"event": event, "group_ids": gids})
        blocks.append({"id": identity, "event": event, "group_ids": gids})
        group_block.update({gid: identity for gid in gids})
    records, duplicate_owner = {}, {}
    for sid, source in sorted(sources.items()):
        gid = unit_group[sid]
        for probe in source["canonical_probes"]:
            pid = probe["opaque_probe_id"]
            members = sorted(set(probe.get("opaque_duplicate_cluster_members", [pid])))
            require(pid in members and members, "Duplicate cluster omits canonical root")
            cluster = "duplicate-" + digest(members)
            for member in members:
                require(member not in duplicate_owner, "Overlapping duplicate clusters or repeated probe")
                duplicate_owner[member] = cluster
                records[member] = {"event": source["event"], "source_unit_ids": [sid],
                                   "source_group_id": gid, "dependency_block_id": group_block[gid],
                                   "duplicate_cluster_id": cluster, "canonical_probe_id": pid,
                                   "original_duplicate_cluster_members": members,
                                   "is_canonical_duplicate_root": member == pid, "kind": "transfer"}
    flagged = sorted(set(flagged_units))
    require(set(flagged) <= set(sources), "Flag mask references absent original units")
    preserved = sorted({d["unit_id"] for p in packets
                        for d in p["source_preservation_protocol"]["source_dispositions"]
                        if d["status"] in ("preserve_source_ambiguity", "preserve_source_conflict")})
    require(set(preserved) <= set(sources), "Preservation mask references absent units")
    proposal = {"schema": "spacing-confirmation-dependency-membership-proposal-v1",
                "source_bindings": sorted(bindings, key=lambda b: b["chunk_index"]),
                "groups": groups, "sources": sources, "blocks": sorted(blocks, key=lambda b: b["id"]),
                "unit_group": unit_group, "group_block": group_block, "record_mappings": records,
                "proposed_edges": edges, "proposed_flagged_source_unit_ids": flagged,
                "preserved_source_unit_ids": preserved, "final_membership_frozen": False,
                "membership_approval_claimed": False, "confirmation_ready": False}
    proposal["sha256"] = digest(proposal)
    return proposal


def source_binding_snapshot(chunks, catalog):
    """Bind all final source content, memberships, receipts and retained history."""
    _sealed(catalog)
    require(len(chunks) == 8 and sorted(c["review_packet"]["chunk_index"] for c in chunks) == list(range(1, 9)),
            "Final dependency policy requires all eight exact source chunks")
    original = {u["unit_id"]: u for u in catalog["units"]}
    covered, bindings = set(), []
    for chunk in sorted(chunks, key=lambda c: c["review_packet"]["chunk_index"]):
        packet = chunk["review_packet"]
        _sealed(packet)
        _sealed(chunk["authored"])
        _sealed(chunk["author_packet"])
        require(packet["source_catalog_sha256"] == catalog["sha256"] and
                packet["authored_chunk_sha256"] == chunk["authored"]["sha256"],
                "Final source packet/catalog/author binding changed")
        for unit in packet["source_units"]:
            sid = unit["unit_id"]
            require(sid in original and sid not in covered and unit == {
                k: original[sid][k] for k in ("unit_id", "event", "source_assertion", "canonical_answer_labels", "canonical_probes", "sha256")},
                    "Final original-source binding changed or repeated")
            covered.add(sid)
        versions = list(chunk.get("history", [])) + [chunk]
        receipts = []
        for version in versions:
            for receipt in version.get("reviews", []):
                _sealed(receipt)
                receipts.append({"receipt_sha256": receipt["sha256"],
                                 "review_packet_sha256": version["review_packet"]["sha256"]})
        bindings.append({"chunk_index": packet["chunk_index"],
                         "author_packet_sha256": chunk["author_packet"]["sha256"],
                         "authored_chunk_sha256": chunk["authored"]["sha256"],
                         "packet_sha256": packet["sha256"],
                         "source_unit_bindings": {u["unit_id"]: u["sha256"] for u in packet["source_units"]},
                         "groups_sha256": digest(packet["groups"]),
                         "acquisition_records_sha256": digest(packet["acquisition_records"]),
                         "preservation_sha256": digest(packet["source_preservation_protocol"]),
                         "history_sha256": digest([{"authored_sha256": v["authored"]["sha256"],
                                                    "packet_sha256": v["review_packet"]["sha256"],
                                                    "receipt_sha256": [r["sha256"] for r in v.get("reviews", [])]}
                                                   for v in versions]),
                         "semantic_receipts": receipts,
                         "author_task_provenance_sha256": digest(chunk.get("author_task_provenance", []))})
    require(covered == set(original), "Final all-eight binding lost original source units")
    return bindings


def _final_membership(packets, recommendations, flagged_units):
    value = membership_proposal(packets, recommendations, flagged_units)
    value.pop("sha256")
    value["schema"] = FINAL_MEMBERSHIP_SCHEMA
    for key in ("final_membership_frozen", "membership_approval_claimed", "confirmation_ready"):
        value.pop(key)
    value["edges"] = value.pop("proposed_edges")
    for edge in value["edges"]:
        edge.pop("independently_accepted")
    value["flagged_source_unit_ids"] = value.pop("proposed_flagged_source_unit_ids")
    value["sha256"] = digest(value)
    return value


def _bind_prepared_roots(records, membership):
    require(len({r["id"] for r in records}) == len(records), "Repeated prepared canonical identity across roles")
    transfer = [r for r in records if r["role"] != "qa"]
    binding = bind_canonical_records(transfer, membership)
    teaching = {}
    for row in records:
        if row["role"] != "qa":
            continue
        meta = membership["record_mappings"].get(row["opaque_probe_id"])
        require(meta and meta["is_canonical_duplicate_root"] and meta["event"] == row["event"] and
                row["unit_id"] == meta["source_unit_ids"][0] and row["id"] not in teaching,
                "Teaching root metadata differs from exact raw source provenance")
        teaching[row["id"]] = dict(meta, opaque_probe_id=row["opaque_probe_id"], role="qa")
    all_maps = list(binding["record_mappings"].values()) + list(teaching.values())
    expected = {pid for pid, meta in membership["record_mappings"].items() if meta["is_canonical_duplicate_root"]}
    require(len(records) == len(all_maps) == len(expected) and {m["opaque_probe_id"] for m in all_maps} == expected,
            "Prepared transfer and teaching roots must cover every catalog duplicate root exactly")
    binding["teaching_root_mappings"] = teaching
    binding["transfer_and_teaching_cover_catalog_roots"] = True
    binding["sha256"] = digest({k: v for k, v in binding.items() if k != "sha256"})
    return binding


def prepared_source_metadata(manifest):
    """Extract explicit audited ID provenance, never evaluation wording."""
    _sealed(manifest)
    probes = manifest["confirmation_audit"]["probe_map"]["probes"]
    by_original = {m["original_probe_id"]: opaque for opaque, m in probes.items()}
    require(len(by_original) == len(probes), "Audited opaque-to-original mapping is not unique")
    records = []
    for fact in manifest["split"]["facts"]:
        opaque = by_original.get(fact["id"])
        require(opaque and probes[opaque]["event"] == fact["event"] and fact.get("unit_id"),
                "Prepared fact lacks exact audited source-root metadata")
        records.append({"id": fact["id"], "opaque_probe_id": opaque, "event": fact["event"],
                        "role": fact["role"], "unit_id": fact["unit_id"]})
    return records


def training_record_provenance(manifest, examples, membership):
    """Freeze attribution and token metadata for every actual packed example.

    Declaration input tokens include prompts, EOS and filler. Answer targets
    count only gold QA/acquisition answer tokens and exclude EOS and filler.
    Generic replacement and filler presentations have no source attribution.
    """
    from .data import normalize
    require(digest(examples) == manifest["examples_sha256"], "Actual prepared example content differs from manifest")
    roots = _bind_prepared_roots(prepared_source_metadata(manifest), membership)
    root_maps = dict(roots["record_mappings"], **roots["teaching_root_mappings"])
    acquisition = {r["id"]: r for r in manifest["acquisition_qa_pool"]["records"]}
    teaching = {r["id"]: r for r in manifest["qa_teaching_pool"]["records"]}
    mappings = {}
    for key, example in examples.items():
        prefix, _, rid = key.partition("/")
        group = None
        if prefix in ("old", "new"):
            require(rid in membership["groups"], "Declaration example references absent final source group")
            group = rid
            sids, kind = membership["groups"][rid]["source_unit_ids"], "declaration"
            roles = {m["role"] for m in root_maps.values() if m["source_group_id"] == rid}
            require(roles == {prefix}, "Declaration provenance changes prepared role")
        elif prefix == "acq":
            target_id, variant = rid.rsplit("/", 1)
            record = acquisition.get(target_id)
            require(record and variant.isdigit() and int(variant) < len(record["questions"]),
                    "Acquisition example references absent target or form")
            group = record["unit_id"]
            require(group in membership["groups"], "Acquisition example source group differs")
            sids = [sid for sid in membership["groups"][group]["source_unit_ids"]
                    if normalize(record["answer"]) in {normalize(a) for a in membership["sources"][sid]["canonical_answer_labels"]}]
            require(sids, "Acquisition answer has no exact original-source label attribution")
            kind = "acquisition_qa"
        elif prefix == "qa":
            record = teaching.get(rid)
            require(record and record["canonical_probe_id"] in roots["teaching_root_mappings"],
                    "Teaching example lacks explicitly audited canonical-root provenance")
            meta = roots["teaching_root_mappings"][record["canonical_probe_id"]]
            sids, group, kind = meta["source_unit_ids"], meta["source_group_id"], "teaching_qa"
        else:
            require(key == "filler" or prefix == "gen" and rid in membership["groups"],
                    "Unknown unrelated training example")
            sids, kind = [], "generic"
        input_tokens = sum(example["attention_mask"])
        answer_tokens = example["content_tokens"] - 1 if kind in ("teaching_qa", "acquisition_qa") else 0
        require(input_tokens > 0 and 0 <= answer_tokens <= example["loss_tokens"], "Invalid actual packed-example token counts")
        mappings[key] = {"source_unit_ids": sorted(sids), "source_group_id": group, "kind": kind,
                         "canonical_probe_id": record["canonical_probe_id"] if kind == "teaching_qa" else None,
                         "input_tokens": input_tokens, "answer_target_tokens": answer_tokens,
                         "loss_tokens": example["loss_tokens"], "content_sha256": example["content_sha256"]}
    value = {"schema": "spacing-confirmation-training-source-provenance-v1", "manifest_sha256": manifest["sha256"],
             "examples_sha256": manifest["examples_sha256"], "records": mappings,
             "qa_answer_tokens_exclude_eos_and_filler": True, "generic_source_attribution": "empty",
             "outcome_based_filters": False}
    value["sha256"] = digest(value)
    return value


def _validate_training_binding(provenance, membership, roots, chunks, manifest_sha256):
    from .data import normalize
    _sealed(provenance)
    require(provenance.get("schema") == "spacing-confirmation-training-source-provenance-v1" and
            provenance.get("manifest_sha256") == manifest_sha256 and provenance.get("outcome_based_filters") is False and
            provenance.get("qa_answer_tokens_exclude_eos_and_filler") is True and provenance.get("generic_source_attribution") == "empty",
            "Final training provenance changed manifest, token convention or outcome filtering")
    qa = {r["id"]: r for c in chunks for r in c["review_packet"]["acquisition_records"]}
    group_roles = collections.defaultdict(set)
    for maps in (roots["record_mappings"], roots["teaching_root_mappings"]):
        for meta in maps.values():
            group_roles[meta["source_group_id"]].add(meta["role"])
    for key, meta in provenance["records"].items():
        prefix, _, rid = key.partition("/")
        if prefix in ("old", "new"):
            require(rid in membership["groups"] and meta["kind"] == "declaration" and meta["source_group_id"] == rid,
                    "Frozen declaration group provenance changed")
            require(group_roles[rid] == {prefix}, "Frozen declaration changed its prepared training role")
            expected = membership["groups"][rid]["source_unit_ids"]
        elif prefix == "acq":
            target, variant = rid.rsplit("/", 1)
            record = qa.get(target)
            require(record and variant.isdigit() and int(variant) < len(record["questions"]) and meta["kind"] == "acquisition_qa" and
                    meta["source_group_id"] == record["unit_id"], "Frozen acquisition target/form provenance changed")
            require(group_roles[record["unit_id"]] == {"old"}, "Frozen acquisition QA entered a non-acquisition role")
            expected = [sid for sid in membership["groups"][record["unit_id"]]["source_unit_ids"]
                        if normalize(record["answer"]) in {normalize(a) for a in membership["sources"][sid]["canonical_answer_labels"]}]
        elif prefix == "qa":
            root = roots["teaching_root_mappings"].get(meta.get("canonical_probe_id"))
            require(root and meta["kind"] == "teaching_qa" and meta["source_group_id"] == root["source_group_id"],
                    "Frozen teaching form lost exact canonical-root provenance")
            expected = root["source_unit_ids"]
        else:
            require(meta["kind"] == "generic" and (key == "filler" or prefix == "gen" and rid in membership["groups"]),
                    "Frozen unrelated example kind changed")
            require(meta["source_group_id"] is None and (key == "filler" or group_roles[rid] == {"old"}),
                    "Frozen generic replacement credits source content or changes role")
            expected = []
        require(meta["source_unit_ids"] == sorted(expected) and all(isinstance(meta[k], int) and not isinstance(meta[k], bool) and meta[k] >= 0
                    for k in ("input_tokens", "answer_target_tokens", "loss_tokens")) and
                meta["input_tokens"] >= meta["loss_tokens"] > 0 and meta["answer_target_tokens"] <= meta["loss_tokens"] and
                (meta["answer_target_tokens"] > 0 if meta["kind"] in ("acquisition_qa", "teaching_qa") else meta["answer_target_tokens"] == 0) and
                isinstance(meta["content_sha256"], str) and len(meta["content_sha256"]) == 64,
                "Frozen source attribution or actual packed-example tokens changed")


def _original_review_write_trace_binding(receipt, evidence, status, receipt_file_sha):
    """Bind an abbreviated legacy summary to retained original child tool output.

    The complete original read response, item identity and executed output must
    be retained. A current path/hash or a parent completion claim is insufficient.
    """
    from .sourceqa_protocol import completed_output_binding, json_pointer
    artifact = evidence.get("completion_trace_file", {})
    require(isinstance(artifact.get("utf8"), str) and
            hashlib.sha256(artifact["utf8"].encode()).hexdigest() == artifact.get("file_sha256"),
            "Original review write trace lacks exact retained read-response bytes")
    raw = json.loads(artifact["utf8"])
    trace = json_pointer(raw, artifact.get("trace_json_pointer", "/structuredContent"))
    require(isinstance(trace, dict) and raw.get("isError") is not True and
            any(c.get("type") == "text" and json.loads(c["text"]) == trace for c in raw.get("content", [])),
            "Original write trace raw and structured read responses differ")
    require(completed_output_binding(receipt, evidence, status), "Original child write trace is not completion-bound")
    require(trace.get("thread", {}).get("threadId") == status["childThreadId"],
            "Original review write trace belongs to a different actual child thread")
    output_path = evidence.get("receipt_file", {}).get("path")
    require(isinstance(output_path, str) and output_path.startswith("/"),
            "Original write trace requires its exact declared receipt output path")
    items = [item for item in trace.get("items", []) if receipt["sha256"] in item.get("text", "").splitlines() and
             any(line.split(maxsplit=1) == [receipt_file_sha, output_path] for line in item.get("text", "").splitlines())]
    require(len(items) == 1, "Original review write trace lacks its exact actual tool item")
    item = items[0]
    require(item.get("sourceThreadId") == status["childThreadId"] and item.get("runId") == status["childRunId"] and
            item.get("type") == "command_execution" and item.get("status") == "completed" and
            item.get("visibility") == "local" and item.get("textTruncated") is False and
            item.get("nextTextOffset") is None and isinstance(item.get("text"), str) and
            any(run.get("runId") == status["childRunId"] and run.get("status") == "completed" and
                run.get("model") == status.get("model") for run in trace.get("recentRuns", [])),
            "Original review write trace is truncated, failed, stale or from another actual run/model")
    require(item.get("itemId"), "Original review write trace lacks a durable actual item identity")
    lines = item["text"].splitlines()
    require(receipt["sha256"] in lines and
            any(line.split(maxsplit=1) == [receipt_file_sha, output_path] for line in lines) and
            any(output_path[-45:] + " True" == line for line in lines) and
            "shasum -a 256 " + output_path in item["text"] and
            "chmod 444 " + output_path in item["text"],
            "Original child write output does not bind full receipt seal, exact raw file hash/path and verified read-only file")
    return {"read_response_file_sha256": artifact["file_sha256"], "item_id": item["itemId"],
            "item_sha256": digest(item), "child_thread_id": item["sourceThreadId"],
            "child_run_id": item["runId"], "receipt_path": output_path,
            "receipt_sha256": receipt["sha256"], "receipt_file_sha256": receipt_file_sha}


def _completed_review(receipt, evidence, *, require_named_task=True):
    """Check retained actual bytes and task result, not a parent's completion flag."""
    from .confirmation import identity, raw_artifact
    _sealed(receipt)
    receipt_file_sha = raw_artifact(receipt, evidence.get("receipt_file"))
    status_file = evidence.get("task_status_file", {})
    require(isinstance(status_file.get("utf8"), str) and
            hashlib.sha256(status_file["utf8"].encode()).hexdigest() == status_file.get("file_sha256"),
            "Completed review lacks exact actual task-status bytes")
    snapshot = json.loads(status_file["utf8"])
    status = evidence.get("task_status")
    client_id = evidence.get("client_request_id")
    if "status_json_pointer" in status_file:
        from urllib.parse import unquote
        from .sourceqa_protocol import json_pointer
        rows = [json_pointer(snapshot, status_file["status_json_pointer"])]
        require(rows == [status] and client_id and unquote(status["taskId"]).endswith(":delegate-task:" + client_id),
                "Direct actual status response differs from its exact client/task identity")
    else:
        rows = [r for r in snapshot.get("task_statuses", [])
                if r.get("clientRequestId") == client_id and r.get("result") == status]
    require(client_id and len(rows) == 1 and status.get("status") == status.get("latestTerminalStatus") == "completed" and
            status.get("hasPendingChildRuns") is False and status.get("taskId") and
            status.get("childThreadId") and status.get("childRunId") and
            (not status.get("latestTerminalRunId") or status["latestTerminalRunId"] == status["childRunId"]) and
            status.get("latestTerminalSummary"),
            "Review lacks its exact independently completed latest task result")
    trace_binding = None
    if receipt["sha256"] not in status["latestTerminalSummary"]:
        require(not require_named_task, "Fresh named review lacks its full receipt seal in the actual terminal summary")
        trace_binding = _original_review_write_trace_binding(receipt, evidence, status, receipt_file_sha)
    reviewer = receipt.get("reviewer") or receipt.get("reviewer_identity")
    require(isinstance(reviewer, dict), "Completed review lacks actual reviewer identity")
    names = sorted(_declared_identity_aliases(reviewer))
    if names and require_named_task:
        actual_aliases = {client_id, status["taskId"], status["childRunId"], status["childThreadId"],
                          status.get("latestTerminalRunId")}
        require(identity(reviewer) in (client_id, status["taskId"]) and set(names) <= actual_aliases,
                "Completed task has contradictory explicit named reviewer identities")
    require(not require_named_task or names, "Final policy review requires an explicitly named actual task")
    model = str(reviewer.get("model", reviewer.get("model_id", ""))).strip().casefold()
    actual = str(status.get("model", "")).strip().casefold()
    # [1m] declares a context-window option, not a different concrete model.
    normalized_model, normalized_actual = model.removesuffix("[1m]"), actual.removesuffix("[1m]")
    alias = evidence.get("legacy_model_alias")
    disclosed_alias = False
    if normalized_model != normalized_actual:
        require(not require_named_task and (normalized_model, normalized_actual) in LEGACY_MODEL_ALIASES and
                isinstance(alias, dict) and alias.get("receipt_sha256") == receipt["sha256"] and
                alias.get("declared_model") == reviewer.get("model", reviewer.get("model_id")) and
                alias.get("actual_model") == status.get("model") and isinstance(alias.get("notes"), str) and alias["notes"].strip(),
                "Concrete completed review model differs without a supported disclosed legacy label alias")
        disclosed_alias = True
    require(model and actual and (normalized_model == normalized_actual or disclosed_alias),
            "Completed review model differs from its actual task")
    return {"receipt_sha256": receipt["sha256"], "receipt_file_sha256": receipt_file_sha,
            "task_status_sha256": digest(status), "task_status_file_sha256": status_file["file_sha256"],
            "task_id": status["taskId"], "child_run_id": status["childRunId"],
            "child_thread_id": status["childThreadId"], "latest_terminal_run_id": status.get("latestTerminalRunId"),
            "client_request_id": client_id,
            "declared_model": reviewer.get("model", reviewer.get("model_id")), "actual_model": status["model"],
            "legacy_model_alias": copy.deepcopy(alias) if disclosed_alias else None,
            "original_write_trace_binding": trace_binding,
            "declared_reviewer_identity": identity(reviewer) if names else None,
            "declared_reviewer_identities": names,
            "declared_identity_matches_actual_task": bool(names and identity(reviewer) in (client_id, status["taskId"]))}


def review_channels(binding, *, include_declared_identity=True):
    """Retain actual task/request/thread/run aliases for independence checks."""
    keys = ("task_id", "client_request_id", "child_run_id", "child_thread_id", "latest_terminal_run_id")
    if include_declared_identity:
        keys += ("declared_reviewer_identity",)
    aliases = {binding[k] for k in keys if binding.get(k)}
    if include_declared_identity:
        aliases.update(binding.get("declared_reviewer_identities", []))
    return {prefix + alias for alias in aliases for prefix in ("", "named:", "task:", "request:", "run:", "thread:")}


def policy_author_channels(authors):
    from .confirmation import identity
    aliases = set()
    for author in authors:
        require(isinstance(author, dict), "Policy author needs an explicit identity/provenance object")
        aliases.add(identity(author))
        aliases.update(_declared_identity_aliases(author))
        aliases.update(_policy_author_execution_aliases(author))
    return {prefix + alias for alias in aliases for prefix in ("", "named:", "task:", "request:", "run:", "thread:")}


def _declared_identity_aliases(person):
    """Keep every explicit person declaration, separately from execution IDs."""
    require(isinstance(person, dict), "Declared reviewer/author identity must be an object")
    aliases = set()
    for key in ("actual_identity", "agent_id", "identity", "review_task_id", "review_name", "name", "client_label"):
        if key in person:
            value = person[key]
            require(isinstance(value, str) and value.strip() and value == value.strip(),
                    "Declared reviewer/author name is not a canonical nonempty string")
            aliases.add(value)
    return aliases


def _policy_author_execution_aliases(author):
    """Normalize the supported direct, task_record and result-envelope shapes.

    Identity names and canonical native task names are separate channels. Other
    snake/native spellings of the same execution field must agree across every
    supplied layer. Unknown provenance fields/containers fail instead of being
    silently dropped. This creates no independent task-authenticity approval.
    """
    fields = {
        "task": ("task_id", "actual_task_id", "taskId", "child_node_id", "childNodeId"),
        "request": ("client_request_id", "clientRequestId"),
        "run": ("child_run_id", "childRunId", "latest_terminal_run_id", "latestTerminalRunId"),
        "thread": ("child_thread_id", "childThreadId"),
        "canonical": ("canonical_task_name",),
    }
    execution_keys = {key for values in fields.values() for key in values}
    identity_keys = {"actual_identity", "agent_id", "identity", "review_task_id", "review_name", "name", "client_label",
                     "provider", "harness", "model", "model_id", "model_family", "human", "role", "notes", "identity_basis"}
    status_keys = {"schema", "provider", "harness", "model", "model_id", "status", "latestTerminalStatus",
                   "latest_terminal_status", "hasPendingChildRuns", "has_pending_child_runs", "workState", "work_state",
                   "childNodeId", "child_node_id", "providerInstanceId", "provider_instance_id", "latestTerminalSummary",
                   "latest_terminal_summary", "latestTerminalResultContextTransferId", "resultContextTransferId",
                   "summary", "waitTimedOut", "actual_result", "actual_result_text", "result_source", "notes"}
    require(set(author) <= identity_keys | execution_keys | status_keys | {"task_record"},
            "Unsupported policy-author identity/provenance field or envelope")
    layers = [author]
    if "task_record" in author:
        record = author["task_record"]
        require(isinstance(record, dict) and record and set(record) <= execution_keys | status_keys | {"result"},
                "Unsupported policy-author task_record provenance shape")
        layers.append(record)
        if "result" in record:
            result = record["result"]
            require(isinstance(result, dict) and result and set(result) <= execution_keys | status_keys,
                    "Unsupported policy-author task_record.result provenance shape")
            layers.append(result)
        require(any(key in layer for layer in layers[1:] for key in execution_keys),
                "Policy-author task_record has no supported execution identity")
    for layer in layers:
        for key in status_keys - {"actual_result"}:
            if key in layer:
                value = layer[key]
                boolean = key in {"hasPendingChildRuns", "has_pending_child_runs", "waitTimedOut"}
                require(type(value) is bool if boolean else value is None or isinstance(value, str),
                        "Unsupported policy-author status metadata shape: " + key)
        native_keys = {"actual_result", "actual_result_text", "result_source"}
        if native_keys & set(layer):
            payload = layer.get("actual_result")
            require(layer.get("schema") == "spacing-native-author-task-evidence-v1" and
                    isinstance(payload, dict) and payload and
                    not set(payload) & (execution_keys | {"task_record", "result"}) and
                    layer.get("canonical_task_name") and layer.get("actual_result_text") and layer.get("result_source"),
                    "Unsupported policy-author native output payload/provenance shape")
    aliases = set()
    for field, keys in fields.items():
        values = [layer[key] for layer in layers for key in keys if key in layer]
        require(all(isinstance(value, str) and value.strip() and value == value.strip() for value in values),
                "Policy-author execution alias is not a canonical nonempty string: " + field)
        require(len(set(values)) <= 1, "Contradictory policy-author execution aliases: " + field)
        aliases.update(values)
    return aliases


def declared_source_reviewer_channels(chunks):
    """Filter only the two documented generic legacy runtime declarations."""
    aliases = set()
    for chunk in chunks:
        for version in list(chunk.get("history", [])) + [chunk]:
            for receipt in version.get("reviews", []):
                reviewer = receipt.get("reviewer", receipt.get("reviewer_identity"))
                aliases.update(_declared_identity_aliases(reviewer) - {"/root", "root"})
    return {prefix + alias for alias in aliases for prefix in ("", "named:", "task:", "request:", "run:", "thread:")}


def source_author_channels(chunks, *, original_trace_input_files_by_proof=None):
    """Resolve current/history author channels and their verified task aliases."""
    from .confirmation import author_context
    channels = set()
    for chunk in chunks:
        for version in list(chunk.get("history", [])) + [chunk]:
            context = author_context(version, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))
            channels.update(context["channels"])
            authored = version["authored"]
            for person in (authored["author"], authored.get("revision_author")):
                if person is not None:
                    channels.update(prefix + alias for alias in _declared_identity_aliases(person)
                                    for prefix in ("", "named:", "task:", "request:", "run:", "thread:"))
            for row in context["task_provenance"]:
                record = row["task_record"]
                aliases = {record[k] for k in ("task_id", "canonical_task_name", "client_request_id",
                                               "child_run_id", "child_thread_id") if record.get(k)}
                channels.update(prefix + alias for alias in aliases
                                for prefix in ("", "named:", "task:", "request:", "run:", "thread:"))
    return channels


def _source_review_bindings(chunks, evidence):
    receipts = {}
    for chunk in chunks:
        for version in list(chunk.get("history", [])) + [chunk]:
            for receipt in version.get("reviews", []):
                require(receipt["sha256"] not in receipts or receipts[receipt["sha256"]] == receipt,
                        "Retained semantic receipt seal has contradictory contents")
                receipts[receipt["sha256"]] = receipt
    by_sha = {e["receipt_sha256"]: e for e in evidence}
    require(len(by_sha) == len(evidence) and set(by_sha) == set(receipts),
            "Source review evidence must cover every retained actual semantic receipt exactly")
    # Original receipts sometimes use a runtime label such as /root or a
    # reviewer-local name. Preserve it. Exact output bytes, hash in the actual
    # task result, and model bind execution; validate_source_chunks separately
    # checks author/reviewer independence including generic runtime labels.
    return [_completed_review(receipts[sha], by_sha[sha], require_named_task=False) for sha in sorted(receipts)]


def build_final_policy_input(chunks, catalog, recommendations, flagged_units, prepared_records,
                            source_review_evidence, *, policy_authors, required_source_conditions=(),
                            exception_protocol=None, exception_lanes=(), lane_property_reviews=(),
                            protocol_coverage=None, lane_source_review_bindings=(), expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Create review input only. This function supplies no source or policy approval.

    prepared_records is the complete registered list of {replicate_id,
    manifest_sha256, records}. Each record has id, opaque_probe_id, event, role,
    unit_id and no evaluation wording. The caller retains complete source history.
    """
    membership = _final_membership([c["review_packet"] for c in chunks], recommendations, flagged_units)
    require(prepared_records and len({r["replicate_id"] for r in prepared_records}) == len(prepared_records),
            "Missing or repeated registered preparation")
    canonical = []
    for prepared in prepared_records:
        require(isinstance(prepared.get("manifest_sha256"), str) and len(prepared["manifest_sha256"]) == 64,
                "Final canonical binding lacks its actual prepared manifest")
        require(all(set(row) == {"id", "opaque_probe_id", "event", "role", "unit_id"}
                    for row in prepared["records"]), "Final policy input must contain source metadata only")
        binding = _bind_prepared_roots(prepared["records"], membership)
        provenance = prepared.get("training_provenance")
        require(provenance is not None, "Final policy must bind actual training-record provenance before training")
        _validate_training_binding(provenance, membership, binding, chunks, prepared["manifest_sha256"])
        canonical.append({"replicate_id": prepared["replicate_id"],
                          "manifest_sha256": prepared["manifest_sha256"], "binding": binding,
                          "training_provenance": provenance})
    from .confirmation import identity
    require(policy_authors and len({identity(a) for a in policy_authors}) == len(policy_authors),
            "Final policy must disclose its actual authors")
    from .confirmation_lane import reporting_policy, validate_lane_inputs, required_surface_correlates
    property_bindings = validate_lane_inputs(exception_lanes, exception_protocol, lane_property_reviews,
                                            membership, chunks, policy_authors,
                                            **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    lane_specification = reporting_policy()
    require(exception_protocol is not None or protocol_coverage is None,
            "Protocol coverage must bind an adopted exception protocol")
    if exception_protocol is not None:
        require(protocol_coverage is not None, "Adopted exception protocol lacks exact current coverage proof")
        conditional_ids = {rid for lane in exception_lanes for rid in lane["acquisition_record_ids"]}
        conditional_ids.update(rid for condition in required_source_conditions for rid in condition["acquisition_record_ids"])
        require(conditional_ids <= set(protocol_coverage.get("conditional_acquisition_record_ids", [])),
                "Protocol coverage routes a lane/source-condition QA as an unconditional record")
        from .sourceqa_protocol import verify_protocol_coverage
        verify_protocol_coverage(protocol_coverage, exception_protocol, chunks, catalog, dependency_context={
            "membership_sha256": membership["sha256"], "exception_lanes_sha256": digest(list(exception_lanes)),
            "required_source_conditions_sha256": digest(list(required_source_conditions)),
            "lane_reporting_policy_sha256": lane_specification["sha256"], "algorithm_sha256": SUPPORTED_ALGORITHM_SHA256}, **sourceqa_validator_keywords(expected_evaluator_metadata=expected_evaluator_metadata, original_trace_input_files_by_proof=original_trace_input_files_by_proof))
        expected_correlates = required_surface_correlates(exception_lanes, lane_property_reviews,
            [r["id"] for c in chunks for r in c["review_packet"]["acquisition_records"]])
        require(protocol_coverage["evaluation_custody"].get("required_surface_correlate_ids_by_record") == expected_correlates,
                "Evaluation custody L5 obligations differ from actual independent lane property certificates")
    value = {"schema": FINAL_INPUT_SCHEMA, "algorithm": algorithm_v3(),
             "algorithm_sha256": SUPPORTED_ALGORITHM_SHA256,
             "source_catalog_sha256": catalog["sha256"], "source_bindings": source_binding_snapshot(chunks, catalog),
             "membership": membership, "canonical_bindings": sorted(canonical, key=lambda r: r["replicate_id"]),
             "source_review_evidence_bindings": _source_review_bindings(chunks, source_review_evidence),
             "policy_authors": copy.deepcopy(policy_authors),
             "required_source_conditions": copy.deepcopy(list(required_source_conditions)),
             "lane_reporting_policy": lane_specification,
             "exception_protocol": copy.deepcopy(exception_protocol),
             "protocol_coverage": copy.deepcopy(protocol_coverage),
             "exception_lanes": copy.deepcopy(list(exception_lanes)),
             "lane_source_review_bindings": copy.deepcopy(list(lane_source_review_bindings)),
             "lane_property_reviews": copy.deepcopy(list(lane_property_reviews)),
             "lane_property_review_bindings": property_bindings,
             "evaluation_wording_used": False, "model_outcomes_used": False,
             "human_review_complete": False, "approval_claimed": False}
    value["sha256"] = digest(value)
    return value


def _decisions(actual, records, label):
    expected = {row["id"]: digest(row) for row in records}
    require(len(expected) == len(records) and isinstance(actual, list) and
            len({r["id"] for r in actual}) == len(actual) and {r["id"] for r in actual} == set(expected),
            "Final review omitted or repeated " + label)
    require(all(r.get("status") == "approved" and r.get("reviewed_content_sha256") == expected[r["id"]] and r.get("notes")
                for r in actual), "Final review has unapproved or changed " + label)


def _conditions_satisfied(policy, chunks):
    """Enforce named conditional approvals against actual edges and masks."""
    from .confirmation import decisions_for
    from .data import normalize
    membership = policy["membership"]
    edges = {e["recommendation_id"]: e for e in membership["edges"]}
    receipts = {r["sha256"]: r for c in chunks for v in list(c.get("history", [])) + [c]
                for r in v.get("reviews", [])}
    current_rows = {r["id"]: r for c in chunks for r in c["review_packet"]["acquisition_records"]}
    current = {rid: digest(r) for rid, r in current_rows.items()}
    condition_rows = policy["required_source_conditions"]
    require(len({r["id"] for r in condition_rows}) == len(condition_rows), "Repeated final source condition")
    covered = collections.defaultdict(set)
    for condition in condition_rows:
        receipt = receipts.get(condition.get("source_receipt_sha256"))
        require(receipt is not None and condition.get("notes") and condition.get("acquisition_record_ids") and
                condition.get("dependence_recommendation_ids") and condition.get("flagged_source_unit_ids"),
                "Source condition lost actual receipt, rows, dependence sets or flag mask")
        approved = {r["id"]: r["reviewed_content_sha256"] for r in decisions_for(receipt, "acquisition_records")
                    if r["status"] == "approved"}
        require(all(rid in current and approved.get(rid) == current[rid] for rid in condition["acquisition_record_ids"]),
                "Conditional source approval does not bind unchanged current approved content")
        require(set(condition["dependence_recommendation_ids"]) <= set(edges) and
                set(condition["flagged_source_unit_ids"]) <= set(membership["flagged_source_unit_ids"]),
                "Conditional source approval lacks final dependence edges or flag mask")
        own_events = {current_rows[rid]["event"] for rid in condition["acquisition_record_ids"]}
        require(len(own_events) == 1 and all(edges[eid]["event"] in own_events for eid in condition["dependence_recommendation_ids"]),
                "Source condition dependence edge is unrelated to its own QA event")
        own_by_record = {rid: {sid for sid in membership["groups"][current_rows[rid]["unit_id"]]["source_unit_ids"]
                              if normalize(current_rows[rid]["answer"]) in
                              {normalize(label) for label in membership["sources"][sid]["canonical_answer_labels"]}}
                         for rid in condition["acquisition_record_ids"]}
        own_units = set().union(*own_by_record.values())
        edge_units = {sid for eid in condition["dependence_recommendation_ids"] for sid in edges[eid]["original_source_unit_ids"]}
        require(all(own_by_record.values()) and own_units <= edge_units and edge_units <= set(condition["flagged_source_unit_ids"]),
                "Source condition lost its own original QA provenance or whole dependence-edge flag mask")
        covered[receipt["sha256"]].update(condition["acquisition_record_ids"])
    # These actual approvals explicitly require the final policy. A fresh final
    # reviewer also attests that all further source conditions are captured.
    known = {
        "895c38afe0d6aa96b90bf3ed407ddc9600bc7be3454bf7d9d9fb7f476872b4c3":
            {"acq-d231fb65e2da03d021974efa", "acq-d947a21372dcd621e62097cc", "acq-31efdee3ab00b91f73495292",
             "acq-3b856257a29d2b7fccd37713", "acq-90dc348805614376045f8e9a", "acq-684c7e6f98afe785d5bb3657"},
        "424793a49ba75167600a10d314a06a9b91eeb5b02c121bd3d115c53f5bb2af69":
            {"acq-d231fb65e2da03d021974efa", "acq-d947a21372dcd621e62097cc", "acq-31efdee3ab00b91f73495292",
             "acq-3b856257a29d2b7fccd37713", "acq-90dc348805614376045f8e9a", "acq-684c7e6f98afe785d5bb3657",
             "acq-23f31b23c3839fd2f612cf28", "acq-fa72d15366e058541dc3d072"},
    }
    for sha, ids in known.items():
        if sha in receipts:
            approved = {r["id"]: r["reviewed_content_sha256"] for r in decisions_for(receipts[sha], "acquisition_records")
                        if r["status"] == "approved"}
            retained = {rid for rid in ids if rid in current and approved.get(rid) == current[rid]}
            require(retained <= covered[sha], "Known conditional source04 approvals were omitted from final policy")
            for rid in retained:
                name = "chunk04/" + ("D8" if rid in {"acq-d231fb65e2da03d021974efa", "acq-d947a21372dcd621e62097cc"}
                                      else "D25" if rid in {"acq-23f31b23c3839fd2f612cf28", "acq-fa72d15366e058541dc3d072"}
                                      else "D7")
                require(name in edges and any(
                    c["source_receipt_sha256"] == sha and rid in c["acquisition_record_ids"] and
                    name in c["dependence_recommendation_ids"] and
                    set(edges[name]["original_source_unit_ids"]) <= set(c["flagged_source_unit_ids"])
                    for c in condition_rows), "Known source04 condition lost exact D7/D8/D25 edge or whole-unit mask")
    for sha, receipt in receipts.items():
        explicit = {r["id"] for r in decisions_for(receipt, "acquisition_records")
                    if r.get("status") == "approved" and r.get("conditions") and
                    r["id"] in current and r["reviewed_content_sha256"] == current[r["id"]]}
        require(explicit <= covered[sha], "Explicit current source approval conditions were omitted from final policy")


def validate_final_policy(policy, chunks, catalog, review_receipt, review_evidence, source_review_evidence, *, expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Production eligibility check. Policy approval never supplies source approval."""
    from .confirmation import identity, validate_source_chunks
    _sealed(policy)
    require(policy.get("schema") == FINAL_INPUT_SCHEMA and policy.get("algorithm_sha256") == SUPPORTED_ALGORITHM_SHA256 and
            digest(policy.get("algorithm")) == SUPPORTED_ALGORITHM_SHA256,
            "Final policy needs the exact supported prospective V3 algorithm")
    require(all(policy.get(k) is False for k in ("evaluation_wording_used", "model_outcomes_used", "human_review_complete", "approval_claimed")),
            "Final policy changes source-only visibility or invents approval")
    require(policy["source_catalog_sha256"] == catalog["sha256"] and policy["source_bindings"] == source_binding_snapshot(chunks, catalog),
            "Final policy all-eight source binding changed")
    membership = policy["membership"]
    _sealed(membership)
    require(membership.get("schema") == FINAL_MEMBERSHIP_SCHEMA,
            "Provisional dependency membership cannot be frozen")
    recommendations = [e["original_recommendation"] for e in membership["edges"]]
    require(membership == _final_membership([c["review_packet"] for c in chunks], recommendations,
                                          membership["flagged_source_unit_ids"]),
            "Final membership masks, duplicate roots, edges or partitions changed")
    prepared = []
    for row in policy["canonical_bindings"]:
        binding = row["binding"]
        _sealed(binding)
        records = [{"id": rid, "opaque_probe_id": meta["opaque_probe_id"], "event": meta["event"],
                    "role": meta["role"], "unit_id": meta["source_unit_ids"][0] if meta["role"] in ("control", "qa") else meta["source_group_id"]}
                   for maps in (binding["record_mappings"], binding["teaching_root_mappings"]) for rid, meta in maps.items()]
        require(binding == _bind_prepared_roots(records, membership), "Final canonical duplicate-root proof changed")
        prepared.append({"replicate_id": row["replicate_id"], "manifest_sha256": row["manifest_sha256"], "records": records,
                         "training_provenance": row["training_provenance"]})
    rebuilt = build_final_policy_input(chunks, catalog, recommendations, membership["flagged_source_unit_ids"], prepared,
                                      source_review_evidence, policy_authors=policy["policy_authors"],
                                      required_source_conditions=policy["required_source_conditions"],
                                      exception_protocol=policy["exception_protocol"], exception_lanes=policy["exception_lanes"],
                                      lane_property_reviews=policy["lane_property_reviews"], protocol_coverage=policy["protocol_coverage"],
                                      lane_source_review_bindings=policy["lane_source_review_bindings"], **sourceqa_validator_keywords(expected_evaluator_metadata=expected_evaluator_metadata, original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    require(rebuilt == policy, "Final source, scored roots or actual completed source-review evidence changed")
    # All exact current source groups and QA must have real independent content
    # approvals, with every same-content negative retained and resolved.
    _, _, _, source_authors = validate_source_chunks(chunks, catalog, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    from .confirmation_lane import validate_lane_source_resolutions
    validate_lane_source_resolutions(policy["exception_lanes"], policy["lane_source_review_bindings"], chunks)
    author_channels = set(source_authors) | source_author_channels(chunks, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof)) | policy_author_channels(policy["policy_authors"])
    # Legacy /root labels are retained verbatim, but are not execution aliases.
    # validate_source_chunks separately checks declared identities and requires
    # actual provenance where generic source author/reviewer labels overlap.
    require(all(not review_channels(binding, include_declared_identity=False) & author_channels
                for binding in policy["source_review_evidence_bindings"]),
            "Source semantic reviewer shares an actual source/policy-author task channel")
    declared_source_channels = declared_source_reviewer_channels(chunks)
    require(not declared_source_channels & policy_author_channels(policy["policy_authors"]),
            "Explicit named source semantic reviewer is a policy author")
    require(not declared_source_channels & (set(source_authors) | source_author_channels(chunks, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))),
            "Explicit named source semantic reviewer is a source or revision author")
    require(all(not review_channels(binding) & author_channels for binding in policy["lane_property_review_bindings"]),
            "Lane property reviewer shares a source-author channel")
    _conditions_satisfied(policy, chunks)
    _sealed(review_receipt)
    require(review_receipt.get("schema") == FINAL_REVIEW_SCHEMA and review_receipt.get("status") == "approved" and
            review_receipt.get("membership_status") == "final_approved" and
            review_receipt.get("unresolved_items") == [] and review_receipt.get("all_source_conditions_captured") is True,
            "Final policy or source-condition adjudication remains provisional")
    expected = {"policy_sha256": policy["sha256"], "algorithm_sha256": SUPPORTED_ALGORITHM_SHA256,
                "membership_sha256": membership["sha256"], "source_bindings_sha256": digest(policy["source_bindings"]),
                "canonical_bindings_sha256": digest(policy["canonical_bindings"]),
                "source_review_evidence_sha256": digest(policy["source_review_evidence_bindings"]),
                "lane_reporting_policy_sha256": policy["lane_reporting_policy"]["sha256"],
                "exception_protocol_sha256": policy["exception_protocol"]["sha256"] if policy["exception_protocol"] else None,
                "protocol_coverage_sha256": policy["protocol_coverage"]["sha256"] if policy["protocol_coverage"] else None,
                "exception_lanes_sha256": digest(policy["exception_lanes"]),
                "lane_property_review_bindings_sha256": digest(policy["lane_property_review_bindings"])}
    require(all(review_receipt.get(k) == v for k, v in expected.items()), "Final actual policy review binds different content")
    require(all(review_receipt.get(k) is False for k in ("model_outcomes_seen", "evaluation_wording_seen", "author_rationales_seen", "human_review_complete")),
            "Final independent policy review lost its blind source-only inputs")
    reviewer = identity(review_receipt["reviewer"])
    actual_review = _completed_review(review_receipt, review_evidence)
    require(reviewer not in {identity(a) for a in policy["policy_authors"]} and
            "named:" + reviewer not in source_authors and reviewer not in source_authors,
            "Final policy reviewer is a policy or source author")
    require(not review_channels(actual_review) & author_channels,
            "Final policy reviewer shares an actual source/policy-author task channel")
    edge_rows = [dict(e, id=e["recommendation_id"]) for e in membership["edges"]]
    _decisions(review_receipt.get("edge_decisions"), edge_rows, "dependence edges")
    masks = [{"id": "flagged", "source_unit_ids": membership["flagged_source_unit_ids"]},
             {"id": "preserved", "source_unit_ids": membership["preserved_source_unit_ids"]}]
    _decisions(review_receipt.get("mask_decisions"), masks, "source masks")
    _decisions(review_receipt.get("condition_decisions"), policy["required_source_conditions"], "conditional source approvals")
    from .confirmation_lane import validate_final_lane_decisions
    validate_final_lane_decisions(review_receipt, policy["exception_lanes"])
    require(all(not review_channels(actual_review) & review_channels(binding)
                for binding in policy["lane_property_review_bindings"]),
            "Final membership reviewer is the lane property reviewer")
    return {"policy_sha256": policy["sha256"], "algorithm_sha256": SUPPORTED_ALGORITHM_SHA256,
            "membership_sha256": membership["sha256"], "source_catalog_sha256": catalog["sha256"],
            "actual_final_review": actual_review, "source_review_evidence_sha256": digest(policy["source_review_evidence_bindings"]),
            "lane_reporting_policy_sha256": policy["lane_reporting_policy"]["sha256"],
            "protocol_coverage_sha256": policy["protocol_coverage"]["sha256"] if policy["protocol_coverage"] else None,
            "exception_lanes_sha256": digest(policy["exception_lanes"]),
            "lane_property_review_bindings_sha256": digest(policy["lane_property_review_bindings"]),
            "registered_manifest_sha256": {r["replicate_id"]: r["manifest_sha256"] for r in policy["canonical_bindings"]},
            "source_approval_verified_separately": True}


def freeze_final_policy(path, policy, chunks, catalog, review_receipt, review_evidence, source_review_evidence, *, expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Validate then exclusively create an immutable, genuinely timed freeze."""
    validation = validate_final_policy(policy, chunks, catalog, review_receipt, review_evidence, source_review_evidence, **sourceqa_validator_keywords(expected_evaluator_metadata=expected_evaluator_metadata, original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    value = {"schema": FROZEN_POLICY_SCHEMA, "policy": policy, "review_receipt": review_receipt,
             "review_evidence": review_evidence, "source_review_evidence": source_review_evidence,
             "validation": validation, "frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "final_membership_frozen": True, "policy_approval_verified": True,
             "source_approval_verified_separately": True, "confirmation_outcomes_seen": False,
             "primary_scoring_changed": False, "acquisition_eligibility_changed": False}
    value["sha256"] = digest(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    return value


def verify_frozen_policy(path, expected_sha256, chunks, catalog, *, manifest_sha256=None, expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Launch guard: revalidate the frozen specification against exact live inputs."""
    value = json.loads(Path(path).read_text())
    _sealed(value)
    require(value.get("schema") == FROZEN_POLICY_SCHEMA and value["sha256"] == expected_sha256 and
            value.get("final_membership_frozen") is True and value.get("policy_approval_verified") is True and
            value.get("source_approval_verified_separately") is True and
            all(value.get(k) is False for k in ("confirmation_outcomes_seen", "primary_scoring_changed", "acquisition_eligibility_changed")),
            "Frozen final policy changed or lacks genuine source/policy approval")
    frozen_time = datetime.datetime.fromisoformat(value["frozen_utc"])
    require(frozen_time.tzinfo is not None and frozen_time <= datetime.datetime.now(datetime.timezone.utc),
            "Frozen policy timestamp lacks timezone or is in the future")
    validation = validate_final_policy(value["policy"], chunks, catalog, value["review_receipt"],
                                       value["review_evidence"], value["source_review_evidence"], **sourceqa_validator_keywords(expected_evaluator_metadata=expected_evaluator_metadata, original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    require(value["validation"] == validation, "Frozen final validation bindings changed")
    if manifest_sha256 is not None:
        require(manifest_sha256 in validation["registered_manifest_sha256"].values(),
                "Launch manifest was not bound in the independently reviewed final policy")
    return value


def verify_preregistered_dependencies(prereg, frozen_policy):
    """The adopted-protocol study must bind all reporting and coverage seals."""
    keys = ("sourceqa_protocol_sha256", "sourceqa_protocol_coverage_sha256",
            "lane_reporting_policy_sha256", "lane_reporting_specification_sha256")
    policy = frozen_policy["policy"]
    if policy.get("exception_protocol") is None and not any(k in prereg for k in keys):
        return
    from .confirmation_lane import reporting_specification
    require(prereg.get("confirmation_dependency_policy_path") and
            prereg.get("confirmation_dependency_policy_sha256") == frozen_policy["sha256"] and
            policy.get("exception_protocol") is not None and policy.get("protocol_coverage") is not None,
            "Adopted-protocol preregistration lacks its canonical frozen dependency policy binding")
    expected = {"sourceqa_protocol_sha256": policy["exception_protocol"]["sha256"],
                "sourceqa_protocol_coverage_sha256": policy["protocol_coverage"]["sha256"],
                "lane_reporting_policy_sha256": policy["lane_reporting_policy"]["sha256"],
                "lane_reporting_specification_sha256": reporting_specification()["sha256"]}
    require(all(prereg.get(k) == v for k, v in expected.items()),
            "Preregistered protocol, coverage or separate lane reporting specification differs")


def acquisition_form_mappings(records, membership):
    """Each original target's forms average once before its source group."""
    mapping = {}
    for record in records:
        gid = record["unit_id"]
        require(gid in membership["groups"] and
                record["event"] == membership["groups"][gid]["event"], "QA source group differs")
        for variant in range(len(record["questions"])):
            key = f"{record['id']}/{variant}"
            require(key not in mapping, "Repeated acquisition QA form")
            mapping[key] = {"event": record["event"],
                            "target_id": record["id"], "question_variant": variant,
                            "original_answer_label": record["answer"],
                            "source_unit_ids": membership["groups"][gid]["source_unit_ids"],
                            "source_group_id": gid,
                            "dependency_block_id": membership["group_block"][gid],
                            "duplicate_cluster_id": "qa-target/" + record["id"],
                            "canonical_probe_id": record["id"], "kind": "acquisition_qa"}
    return mapping


def bind_canonical_records(records, membership):
    """Source-metadata proof that duplicate members are not separately scored.

    The final preparer supplies exact canonical-to-opaque IDs and role/unit
    membership. No evaluation wording or alias inference is used here.
    """
    mapping, by_role, opaque_seen = {}, collections.defaultdict(list), set()
    for record in records:
        rid, opaque = record["id"], record["opaque_probe_id"]
        require(rid not in mapping and opaque not in opaque_seen and record["role"] in ("old", "new", "control"),
                "Repeated canonical prediction or non-transfer role")
        opaque_seen.add(opaque)
        require(opaque in membership["record_mappings"], "Canonical opaque ID is absent")
        meta = membership["record_mappings"][opaque]
        require(meta["is_canonical_duplicate_root"] and record["event"] == meta["event"],
                "Canonical transfer prediction is not its frozen duplicate root/event")
        expected = meta["source_unit_ids"][0] if record["role"] == "control" else meta["source_group_id"]
        require(record["unit_id"] == expected, "Prepared role/unit differs from exact source provenance")
        mapping[rid] = dict(meta, opaque_probe_id=opaque, role=record["role"])
        by_role[record["role"]].append(rid)
    require(mapping, "Missing canonical transfer binding")
    binding = {"schema": "spacing-confirmation-scored-duplicate-root-binding-v1",
               "membership_sha256": membership["sha256"], "record_mappings": mapping,
               "canonical_row_ids_by_role": {role: sorted(ids) for role, ids in by_role.items()},
               "only_canonical_duplicate_roots_scored": True,
               "duplicate_members_separately_scored": False,
               "canonical_and_paraphrase_share_root_mapping": True,
               "evaluation_wording_used": False, "primary_aggregation_changed": False}
    binding["sha256"] = digest(binding)
    return binding


def teaching_form_mappings(records, membership):
    """Bind expanded teaching forms through explicit canonical roots only."""
    mapping = {}
    for record in records:
        root = record.get("canonical_probe_id")
        require(root in membership["record_mappings"] and
                membership["record_mappings"][root]["is_canonical_duplicate_root"],
                "Teaching form lacks exact canonical-root provenance")
        require(record["id"] not in mapping, "Repeated teaching form")
        mapping[record["id"]] = dict(membership["record_mappings"][root], kind="teaching_qa")
    return mapping


def _value(row, metric):
    value = row.get(metric)
    require(metric in ("exact_match", "loss") and isinstance(value, (int, float)) and
            not isinstance(value, bool) and math.isfinite(value), "Invalid or missing original metric")
    require((metric != "exact_match" or value in (0, 1)) and (metric != "loss" or value >= 0),
            "Invalid original EM or answer loss")
    return float(value)


def score(rows, mapping, metric, *, weighting="equal_block", excluded_units=(),
          excluded_blocks=(), excluded_events=()):
    """A2: forms/duplicate members -> group -> block -> event -> role.

    The secondary 'primary_formula' keeps raw row weighting and the actual
    prepared unit IDs, including raw control units. This function returns a
    diagnostic and never replaces a saved primary aggregate.
    """
    require(weighting in ("equal_block", "primary_formula"), "Unknown diagnostic weighting")
    excluded_units, excluded_blocks, excluded_events = map(set, (excluded_units, excluded_blocks, excluded_events))
    seen, roles, events_seen, retained, statuses = set(), set(), set(), [], []
    for row in rows:
        rid = row["id"]
        require(rid not in seen and rid in mapping, "Repeated score row or missing exact mapping")
        seen.add(rid)
        meta = mapping[rid]
        require(row["event"] == meta["event"] and meta["source_unit_ids"], "Prediction provenance differs")
        roles.add(row["role"])
        events_seen.add(row["event"])
        value = _value(row, metric)
        excluded = bool(set(meta["source_unit_ids"]) & excluded_units or
                        meta["dependency_block_id"] in excluded_blocks or row["event"] in excluded_events)
        statuses.append((row, meta, excluded))
        if excluded:
            continue
        retained.append((row, meta, value))
    require(len(roles) <= 1, "Score each diagnostic role separately")
    grouped = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    owners = {}
    for row, meta, value in retained:
        gid = meta["source_group_id"] if weighting == "equal_block" else row["unit_id"]
        cluster = meta["duplicate_cluster_id"] if weighting == "equal_block" else row["id"]
        if weighting == "equal_block":
            owner = row["event"], gid, meta["dependency_block_id"]
            require(cluster not in owners or owners[cluster] == owner,
                    "A duplicate/target cluster crosses source group, block or event")
            owners[cluster] = owner
        grouped[row["event"]][gid][cluster].append(value)
    group_means = {event: {gid: statistics.mean(statistics.mean(values) for values in clusters.values())
                          for gid, clusters in by_group.items()} for event, by_group in grouped.items()}
    group_blocks = {}
    for _, meta, _ in retained:
        gid, block = meta["source_group_id"], meta["dependency_block_id"]
        require(gid not in group_blocks or group_blocks[gid] == block,
                "One source group has conflicting frozen block IDs")
        group_blocks[gid] = block
    event_means, block_counts = {}, {}
    for event, means in group_means.items():
        blocks = collections.defaultdict(list)
        for gid, value in means.items():
            blocks[group_blocks[gid] if weighting == "equal_block" else gid].append(value)
        event_means[event] = statistics.mean(statistics.mean(values) for values in blocks.values())
        block_counts[event] = len(blocks)
    counts = {}
    for label, key in (("event", lambda row, meta: row["event"]),
                       ("source_group", lambda row, meta: meta["source_group_id"]),
                       ("dependency_block", lambda row, meta: meta["dependency_block_id"])):
        cells = collections.defaultdict(lambda: {"rows_total": 0, "rows_retained": 0, "rows_excluded": 0})
        for row, meta, excluded in statuses:
            cell = cells[key(row, meta)]
            cell["rows_total"] += 1
            cell["rows_excluded" if excluded else "rows_retained"] += 1
        counts[label] = dict(cells)
    return {"event_macro": statistics.mean(event_means.values()) if event_means else None,
            "by_event": event_means, "group_means": group_means, "block_counts_by_event": block_counts,
            "empty_event_ids": sorted(events_seen - set(event_means)), "rows_total": len(rows),
            "rows_retained": len(retained), "rows_excluded": len(rows) - len(retained),
            "duplicate_or_target_clusters_retained": len(owners) if weighting == "equal_block" else len(retained),
            "row_counts_by_level": counts, "role": next(iter(roles)) if roles else None,
            "weighting": weighting, "diagnostic_only": True}


def paired_summary(pairs, *, interval=True):
    """Descriptive independent-replicate inference; no tests or launch decisions."""
    require(len({p["replicate_id"] for p in pairs}) == len(pairs), "Repeated trained replicate")
    differences, missing = [], []
    for pair in pairs:
        a, b = pair["a"], pair["b"]
        if a is None or b is None:
            missing.append(pair["replicate_id"])
            continue
        require(all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                    for v in (a, b)), "Invalid paired score")
        differences.append({"replicate_id": pair["replicate_id"], "difference": float(a - b)})
    values = [d["difference"] for d in differences]
    result = {"n_registered": len(pairs), "n_pair": len(values), "missing_secondary_pair_ids": missing,
              "paired_differences": differences, "estimate": statistics.mean(values) if values else None,
              "ci95_descriptive": None, "inference_unit": "independently trained paired replicate",
              "conditional_on_fixed_event_pool": True, "diagnostic_only": True}
    if interval and len(values) >= 2:
        from .analysis import paired_interval
        result["ci95_descriptive"] = paired_interval(values, .95)
    return result


def divergence(primary, secondary):
    """A1: state sign and zero-coverage differences for every contrast/endpoint."""
    def sign(summary):
        value = summary["estimate"]
        return None if value is None else "positive" if value > 0 else "negative" if value < 0 else "zero"
    def covers(summary):
        interval = summary["ci95_descriptive"]
        return None if interval is None else interval[0] <= 0 <= interval[1]
    p_sign, s_sign, p_zero, s_zero = sign(primary), sign(secondary), covers(primary), covers(secondary)
    sign_diff = None if None in (p_sign, s_sign) else p_sign != s_sign
    zero_diff = None if None in (p_zero, s_zero) else p_zero != s_zero
    statement = (f"Mean paired sign: primary {p_sign or 'undefined'}, secondary {s_sign or 'undefined'}; "
                 f"sign differs: {sign_diff}. Descriptive 95% interval covers zero: "
                 f"primary {p_zero}, secondary {s_zero}; coverage differs: {zero_diff}. "
                 f"Complete pairs: primary {primary['n_pair']}, secondary {secondary['n_pair']}. "
                 "This comparison selects no score, replicate, dose, arm or inference claim.")
    return {"primary_mean_sign": p_sign, "secondary_mean_sign": s_sign,
            "mean_sign_differs": sign_diff, "primary_ci95_covers_zero": p_zero,
            "secondary_ci95_covers_zero": s_zero, "ci95_zero_coverage_differs": zero_diff,
            "complete_pair_counts_differ": primary["n_pair"] != secondary["n_pair"],
            "statement": statement, "descriptive_only": True}


def diagnostic_scores(rows, mapping, metric, flagged_units, preserved_units):
    """Emit all masks and complements irrespective of their result."""
    all_units = {sid for meta in mapping.values() for sid in meta["source_unit_ids"]}
    masks = {"all_source": set(), "flagged_exclusion": set(flagged_units),
             "preserved_exclusion": set(preserved_units),
             "flagged_only": all_units - set(flagged_units),
             "preserved_only": all_units - set(preserved_units)}
    return {f"{weighting}/{name}": score(rows, mapping, metric, weighting=weighting, excluded_units=mask)
            for weighting in ("primary_formula", "equal_block") for name, mask in masks.items()}


def influence_scores(rows, mapping, metric):
    """Complete saved-score deletions from cached means and denominators."""
    full = {w: score(rows, mapping, metric, weighting=w) for w in ("primary_formula", "equal_block")}
    block_means = collections.defaultdict(lambda: collections.defaultdict(list))
    group_block = {meta["source_group_id"]: meta["dependency_block_id"] for meta in mapping.values()}
    for event, group_means in full["equal_block"]["group_means"].items():
        for gid, value in group_means.items():
            block_means[event][group_block[gid]].append(value)
    block_means = {event: {block: statistics.mean(values) for block, values in means.items()}
                   for event, means in block_means.items()}
    block_event = {meta["dependency_block_id"]: meta["event"] for meta in mapping.values()}
    block_rows, event_rows = collections.Counter(), collections.Counter()
    for row in rows:
        block_rows[mapping[row["id"]]["dependency_block_id"]] += 1
        event_rows[row["event"]] += 1
    def deleted(weighting, *, block=None, event=None):
        baseline = full[weighting]
        means = dict(baseline["by_event"])
        counts = dict(baseline["block_counts_by_event"])
        removed = 0
        if block is not None:
            event = block_event[block]
            if block in block_means.get(event, {}):
                surviving = [v for bid, v in block_means[event].items() if bid != block]
                if surviving:
                    means[event], counts[event] = statistics.mean(surviving), len(surviving)
                else:
                    means.pop(event)
                    counts.pop(event)
                removed = block_rows[block]
        elif event is not None:
            means.pop(event, None)
            counts.pop(event, None)
            removed = event_rows[event]
        return {"event_macro": statistics.mean(means.values()) if means else None,
                "by_event": means, "block_counts_by_event": counts,
                "empty_event_ids": sorted(set(baseline["by_event"]) - set(means)),
                "rows_total": len(rows), "rows_retained": len(rows) - removed,
                "rows_excluded": removed, "weighting": weighting, "diagnostic_only": True}
    cases = []
    for block in sorted({meta["dependency_block_id"] for meta in mapping.values()}):
        cases.append({"kind": "leave_one_block_out", "item_id": block, "weighting": "equal_block",
                      "score": deleted("equal_block", block=block)})
    for event in sorted({meta["event"] for meta in mapping.values()}):
        for weighting in ("primary_formula", "equal_block"):
            cases.append({"kind": "leave_one_event_out", "item_id": event, "weighting": weighting,
                          "score": deleted(weighting, event=event)})
    return cases


def contrast_report(primary_pairs, secondary_pairs, *, contrast, endpoint, role, variant, metric):
    """Labels and A1 disclosures accompany every registered secondary."""
    primary = paired_summary(primary_pairs)
    registered = {p["replicate_id"] for p in primary_pairs}
    summaries = {}
    for name, pairs in secondary_pairs.items():
        require({p["replicate_id"] for p in pairs} == registered,
                "Secondary report lost a valid registered replicate")
        summary = paired_summary(pairs)
        summaries[name] = {"summary": summary, "primary_secondary_divergence": divergence(primary, summary)}
    require(summaries, "Missing planned secondary diagnostics")
    return {"contrast": contrast, "endpoint": endpoint, "role": role, "variant": variant, "metric": metric,
            "primary": primary, "secondaries": summaries, "primary_scores_replaced": False,
            "acquisition_eligibility_changed": False, "diagnostic_only": True}


def complete_contrast_report(replicates, *, contrast, endpoint, role, variant, metric,
                             flagged_units, preserved_units):
    """Keep saved primary inputs and emit every planned mask and influence case."""
    primary_pairs, secondaries, influence_pairs, replicate_scores = [], collections.defaultdict(list), collections.defaultdict(list), []
    for rep in replicates:
        primary_pairs.append({"replicate_id": rep["replicate_id"], "a": rep["primary_a"], "b": rep["primary_b"]})
        a = diagnostic_scores(rep["a_rows"], rep["mapping"], metric, flagged_units, preserved_units)
        b = diagnostic_scores(rep["b_rows"], rep["mapping"], metric, flagged_units, preserved_units)
        replicate_scores.append({"replicate_id": rep["replicate_id"], "a": a, "b": b})
        for arm, observed, saved in (("a", a, rep["primary_a"]), ("b", b, rep["primary_b"])):
            computed = observed["primary_formula/all_source"]["event_macro"]
            require((saved is None and computed is None) or
                    saved is not None and computed is not None and math.isclose(saved, computed, rel_tol=0, abs_tol=1e-12),
                    "Saved primary differs from unchanged rows/unit/event aggregation: " + arm)
        for label in sorted(a):
            secondaries[label].append({"replicate_id": rep["replicate_id"],
                                      "a": a[label]["event_macro"], "b": b[label]["event_macro"]})
        ia = {(c["kind"], c["item_id"], c["weighting"]): c for c in influence_scores(rep["a_rows"], rep["mapping"], metric)}
        ib = {(c["kind"], c["item_id"], c["weighting"]): c for c in influence_scores(rep["b_rows"], rep["mapping"], metric)}
        require(set(ia) == set(ib), "Paired arms lost a common influence item")
        for key in ia:
            influence_pairs[key].append({"replicate_id": rep["replicate_id"],
                                         "a": ia[key]["score"]["event_macro"], "b": ib[key]["score"]["event_macro"]})
    result = contrast_report(primary_pairs, secondaries, contrast=contrast, endpoint=endpoint,
                             role=role, variant=variant, metric=metric)
    registered = {p["replicate_id"] for p in primary_pairs}
    influence = []
    for key, pairs in sorted(influence_pairs.items()):
        require({p["replicate_id"] for p in pairs} == registered,
                "Replicate mapping lost a globally frozen influence item")
        summary = paired_summary(pairs, interval=False)
        summary.pop("ci95_descriptive")
        baseline_label = key[2] + "/all_source"
        baseline = result["secondaries"][baseline_label]["summary"]
        # Point influence compares exactly the same complete replicate IDs.
        ids = {p["replicate_id"] for p in summary["paired_differences"]}
        full_pairs = {p["replicate_id"]: p["difference"] for p in baseline["paired_differences"]}
        full_mean = statistics.mean(full_pairs[rid] for rid in sorted(ids)) if ids else None
        shift = summary["estimate"] - full_mean if full_mean is not None else None
        influence.append({"kind": key[0], "item_id": key[1], "weighting": key[2],
                          "paired_summary": summary, "full_contrast_same_complete_pairs": full_mean,
                          "deletion_minus_full_mean_shift": shift, "point_influence_only": True})
    maxima = {}
    for kind in ("leave_one_block_out", "leave_one_event_out"):
        observed = [case for case in influence if case["kind"] == kind and case["deletion_minus_full_mean_shift"] is not None]
        maximum = max((abs(c["deletion_minus_full_mean_shift"]) for c in observed), default=None)
        maxima[kind] = {"maximum_absolute_mean_shift": maximum,
                        "all_tied_cases": [{k: c[k] for k in ("item_id", "weighting")} for c in observed
                                           if abs(c["deletion_minus_full_mean_shift"]) == maximum]}
    result["influence"] = influence
    result["replicate_scores"] = replicate_scores
    result["maximum_absolute_influence"] = maxima
    result["no_retraining_or_new_evaluation"] = True
    return result


def executed_presentations(logs, provenance, stage2_steps, *, expected_rows=None):
    """Recover actual ordered presentations and clocks from existing exposure logs."""
    _sealed(provenance)
    records, presentations = provenance["records"], []
    input_clock = target_clock = loss_clock = 0
    require(logs and (expected_rows is None or len(logs) == len(expected_rows)), "Executed history is incomplete")
    for index, log in enumerate(logs):
        row = log["examples"]
        require(log["global_step"] == index + 1 and log["row_sha256"] == digest(row) and
                all(k in records for k in row) and
                log["example_content_sha256"] == [records[k]["content_sha256"] for k in row] and
                log["loss_tokens"] == sum(records[k]["loss_tokens"] for k in row),
                "Executed presentation history differs from frozen content or clocks")
        if expected_rows is not None:
            require(row == expected_rows[index], "Executed order differs from frozen declaration/QA schedule")
        require(log["phase"] in ("stage1", "stage2", "buffer"), "Unknown executed training phase")
        for slot, key in enumerate(row):
            meta = records[key]
            input_clock += meta["input_tokens"]
            target_clock += meta["answer_target_tokens"]
            loss_clock += meta["loss_tokens"]
            phase = ("acquisition" if log["phase"] == "stage1" else "buffer" if log["phase"] == "buffer"
                     else "review" if key.startswith(("old/", "gen/")) else "interference")
            require(log["phase"] != "buffer" or log["phase_step"] >= stage2_steps,
                    "Buffer presentation precedes frozen stage2 end")
            presentations.append({"id": f"{index + 1}/{slot}", "key": key, "phase": phase,
                                  "kind": meta["kind"], "source_unit_ids": meta["source_unit_ids"],
                                  "global_step": index + 1, "input_token_clock": input_clock,
                                  "answer_target_token_clock": target_clock, "loss_token_clock": loss_clock,
                                  "input_tokens": meta["input_tokens"], "answer_target_tokens": meta["answer_target_tokens"],
                                  "loss_tokens": meta["loss_tokens"]})
    return presentations


def exposure_covariates(presentations, membership, endpoint_global_step):
    """Union each block's presentations once, without any outcome-based filtering."""
    observed = [p for p in presentations if p["global_step"] <= endpoint_global_step]
    require(observed and observed[-1]["global_step"] == endpoint_global_step and
            len({p["id"] for p in observed}) == len(observed), "Endpoint history is missing or repeats a presentation")
    clocks = {k: observed[-1][k] for k in ("global_step", "input_token_clock", "answer_target_token_clock", "loss_token_clock")}
    units, blocks = collections.defaultdict(list), collections.defaultdict(list)
    for presentation in observed:
        sids = presentation["source_unit_ids"]
        require(len(sids) == len(set(sids)) and set(sids) <= set(membership["sources"]),
                "Executed presentation has unknown or repeated source attribution")
        for sid in sids:
            units[sid].append(presentation)
        for block in {membership["group_block"][membership["unit_group"][sid]] for sid in sids}:
            blocks[block].append(presentation)
    def summary(rows):
        result = {"presentations": len(rows), "input_tokens": sum(p["input_tokens"] for p in rows),
                  "answer_target_tokens": sum(p["answer_target_tokens"] for p in rows),
                  "loss_tokens": sum(p["loss_tokens"] for p in rows), "clocks": {}}
        for clock, endpoint in clocks.items():
            values = [p[clock] for p in rows]
            gaps = [b - a for a, b in zip(values, values[1:])]
            result["clocks"][clock] = {"first": values[0] if values else None, "last": values[-1] if values else None,
                                       "sorted_between_presentation_gaps": sorted(gaps),
                                       "gap_min": min(gaps) if gaps else None,
                                       "gap_median": statistics.median(gaps) if gaps else None,
                                       "gap_max": max(gaps) if gaps else None,
                                       "last_presentation_to_endpoint_gap": endpoint - values[-1] if values else None}
        return result
    def by_phase(rows):
        return {"all_phases": summary(rows), **{phase: summary([p for p in rows if p["phase"] == phase])
                for phase in ("acquisition", "review", "interference", "buffer")}}
    return {"endpoint_global_step": endpoint_global_step, "endpoint_clocks": clocks,
            "source_units": {sid: by_phase(units[sid]) for sid in sorted(membership["sources"])},
            "dependency_blocks": {b["id"]: by_phase(blocks[b["id"]]) for b in membership["blocks"]},
            "history_complete": True, "outcome_based_filters": False, "block_presentations_counted_once": True,
            "clock_convention": "Within a batch, cumulative token clocks follow frozen example order; all slots share its optimizer step. QA answer targets exclude EOS and filler. Declaration answer-target count is zero.",
            "overlap_disclosure": "A shared presentation credits each original unit in its provenance. Unit totals are attributed exposure and need not sum to disjoint totals. Each block uses the union of presentation IDs."}


def d8_prediction_table(rows, mapping, membership, gold_labels):
    """Fixed-label descriptive frequencies, with no either-gold success metric."""
    from .data import normalize
    labels = ("an elaborate trick", "optical illusion", "other")
    edges = [e for e in membership["edges"] if e["recommendation_id"] == "chunk04/D8"]
    require(len(edges) <= 1, "D8 dependence membership repeats")
    d8_units = set(edges[0]["original_source_unit_ids"]) if edges else set()
    counts = {gold: {prediction: 0 for prediction in labels} for gold in labels[:2]}
    records, missing = [], []
    for row in rows:
        meta = mapping[row["id"]]
        if not set(meta["source_unit_ids"]) & d8_units:
            continue
        gold = normalize(gold_labels[row["id"]])
        if gold not in labels[:2]:
            continue
        if not isinstance(row.get("prediction"), str):
            missing.append(row["id"])
            continue
        prediction = normalize(row["prediction"])
        category = prediction if prediction in labels[:2] else "other"
        counts[gold][category] += 1
        records.append({"id": row["id"], "target_label": gold, "prediction_category": category})
    return {"counts": counts if not missing else None, "available_prediction_counts": counts,
            "records": records, "rows": len(records), "missing_prediction_row_ids": missing,
            "status": "complete" if not missing else "undefined_missing_saved_predictions",
            "diagnostic_only": True,
            "existing_normalization_and_fixed_labels": True, "either_gold_success_metric_computed": False,
            "gold_or_aliases_changed": False, "disclosure": "A trick-only prediction is other under the unchanged fixed labels and existing normalization."}


def _global_mapping(actual, membership):
    """Unscored catalog metadata keeps influence cases fixed across role splits."""
    result = dict(actual)
    for group in membership["groups"].values():
        key = "unscored-frozen-group/" + group["id"]
        require(key not in result, "Reserved unscored catalog identity collides")
        result[key] = {"event": group["event"], "source_unit_ids": group["source_unit_ids"],
                       "source_group_id": group["id"], "dependency_block_id": membership["group_block"][group["id"]],
                       "duplicate_cluster_id": key, "kind": "unscored_catalog_metadata"}
    return result


def _saved_evaluation(evaluation, manifest, binding, membership):
    """Adapt real evaluator JSON, preserving its rows, metrics and saved aggregates."""
    from .evaluation import aggregate
    require(evaluation["manifest_sha256"] == manifest["sha256"], "Saved evaluation manifest differs from frozen preparation")
    sections = {}
    transfer = binding["record_mappings"]
    for variant, section in evaluation["variants"].items():
        rows = section["facts"]
        require(len({r["id"] for r in rows}) == len(rows) and {r["id"] for r in rows} == set(transfer),
                "Saved transfer rows omit or repeat frozen canonical roots")
        require(section["aggregate"] == aggregate(rows) and all(r["role"] == transfer[r["id"]]["role"] and
                r["event"] == transfer[r["id"]]["event"] and r["unit_id"] ==
                (transfer[r["id"]]["source_unit_ids"][0] if r["role"] == "control" else transfer[r["id"]]["source_group_id"])
                for r in rows), "Saved transfer aggregation, role or source-unit provenance changed")
        sections["transfer/" + variant] = {"rows": rows, "mapping": _global_mapping(transfer, membership),
                                           "aggregate": section["aggregate"], "roles": ["old", "new", "control"]}
    acquisition = evaluation.get("acquisition_qa_diagnostic")
    if acquisition is not None:
        mapping = acquisition_form_mappings(manifest["acquisition_qa_pool"]["records"], membership)
        rows = acquisition["facts"]
        require({r["id"] for r in rows} == set(mapping) and len(rows) == len(mapping) and
                acquisition["aggregate"] == aggregate(rows)["old"] and all(r["role"] == "old" and
                r.get("target_id") == mapping[r["id"]]["target_id"] and
                type(r.get("question_variant")) is int and r["question_variant"] == mapping[r["id"]]["question_variant"] and
                r["unit_id"] == mapping[r["id"]]["source_group_id"] and r["event"] == mapping[r["id"]]["event"] for r in rows),
                "Saved acquisition diagnostic target/forms, source provenance or aggregate changed")
        sections["acquisition_qa"] = {"rows": rows, "mapping": _global_mapping(mapping, membership),
                                       "aggregate": {"old": acquisition["aggregate"]}, "roles": ["old"]}
    teaching = evaluation.get("qa_teaching_diagnostic")
    if teaching is not None:
        # The current evaluator scores canonical QA-role facts, while training
        # uses all source forms. Do not manufacture missing form predictions.
        mapping = {rid: dict(meta, kind="teaching_qa") for rid, meta in binding["teaching_root_mappings"].items()}
        rows = teaching["facts"]
        require(len(rows) == len(mapping) and {r["id"] for r in rows} == set(mapping) and
                teaching["aggregate"] == aggregate(rows)["qa"] and all(r["role"] == "qa" and
                r["event"] == mapping[r["id"]]["event"] and r["unit_id"] == mapping[r["id"]]["source_unit_ids"][0] for r in rows),
                "Saved teaching rows, original form provenance or aggregate changed")
        sections["teaching_qa"] = {"rows": rows, "mapping": _global_mapping(mapping, membership),
                                    "aggregate": {"qa": teaching["aggregate"]}, "roles": ["qa"],
                                    "scope": "Saved canonical QA-role roots; expanded training forms have no separate saved predictions."}
    return sections


def _endpoint_labels(schedule, config):
    stage2, end = schedule["stage2_steps"], len(next(iter(schedule["arms"].values())))
    labels = {"baseline": 0, "stage2_end": stage2, "buffer_end": end}
    labels.update({f"fraction/{fraction}": round(stage2 * fraction) for fraction in config["stage2_eval_fractions"]})
    labels.update({f"delay/{delay}": stage2 + delay for delay in config["eval_delays"]})
    return labels


def primary_aggregate_binding(manifest_sha256, completion, primary_delay, endpoint_step, arms):
    """Retain the analyzer's existing primary aggregates without changing its output."""
    value = {"schema": "spacing-confirmation-primary-aggregate-binding-v1", "manifest_sha256": manifest_sha256,
             "completion_sha256": completion.get("sha256"), "primary_delay": primary_delay,
             "endpoint_stage2_step": endpoint_step, "arms": copy.deepcopy(arms)}
    value["sha256"] = digest(value)
    return value


def saved_bundle_diagnostics(path, manifest, schedule, examples, policy, replicate_id, *, acquisition_E=None,
                             completion=None, primary_binding=None):
    """Read existing real arm files after the primary completion validator passes."""
    from .schedule import ARMS, stage1_epoch
    from .units import schedule_records
    from .common import read_json
    from .confirmation_lane import balance_disclosures, prediction_tables
    membership = policy["membership"]
    prepared = next((r for r in policy["canonical_bindings"] if r["replicate_id"] == replicate_id), None)
    require(prepared and prepared["manifest_sha256"] == manifest["sha256"], "Saved replicate is absent from final policy")
    binding = _bind_prepared_roots(prepared_source_metadata(manifest), membership)
    provenance = training_record_provenance(manifest, examples, membership)
    require(binding == prepared["binding"] and provenance == prepared["training_provenance"],
            "Actual saved preparation changed frozen root or training provenance")
    path = Path(path)
    pins, consumed = {}, []
    if completion is not None:
        _sealed(completion)
        require(isinstance(completion.get("files"), list) and completion["files"],
                "Saved reporting lacks retained primary completion file identities")
        for pin in completion["files"]:
            require(isinstance(pin, dict), "Primary completion file identity is missing, duplicated or noncanonical")
            literal = pin.get("path")
            require(isinstance(literal, str) and literal and
                    Path(literal).is_absolute() and str(Path(literal).resolve()) == literal and literal not in pins and
                    isinstance(pin.get("sha256"), str) and len(pin["sha256"]) == 64,
                    "Primary completion file identity is missing, duplicated or noncanonical")
            pins[literal] = pin["sha256"]
    if primary_binding is not None:
        _sealed(primary_binding)
        require(completion is not None and primary_binding.get("schema") == "spacing-confirmation-primary-aggregate-binding-v1" and
                primary_binding["manifest_sha256"] == manifest["sha256"] and
                primary_binding["completion_sha256"] == completion["sha256"] and
                set(primary_binding["arms"]) == set(ARMS) and
                primary_binding["endpoint_stage2_step"] == schedule["stage2_steps"] + primary_binding["primary_delay"],
                "Retained primary aggregate binding differs from its verified preparation/completion/endpoint")
    def read_bytes(file, *, required_pin=True):
        raw = file.read_bytes()
        identity, raw_sha = str(file.resolve()), hashlib.sha256(raw).hexdigest()
        if completion is not None:
            require((not required_pin or identity in pins) and
                    (identity not in pins or pins[identity] == raw_sha),
                    "Consumed saved bytes differ from retained primary completion file identity")
        consumed.append({"path": identity, "file_sha256": raw_sha,
                         "primary_completion_pin_verified": identity in pins})
        return raw
    def logs(file):
        return [json.loads(line) for line in read_bytes(file).splitlines()]
    shared_logs = logs(path / "stage1/exposures.jsonl")
    acquisition_files = sorted((path / "stage1/evaluations").glob("acquisition-*.json"))
    shared_evaluations, acquisition_evidence = [], []
    for file in acquisition_files:
        raw = read_bytes(file, required_pin=acquisition_E is not None)
        evaluation = json.loads(raw)
        epoch = evaluation.get("epoch")
        require(type(epoch) is int and epoch > 0 and file.name == f"acquisition-E{epoch:03}.json",
                "Saved acquisition epoch or checkpoint filename differs from its exact identity")
        shared_evaluations.append(evaluation)
        acquisition_evidence.append({"path": str(file.resolve()), "epoch": epoch,
                                     "file_sha256": hashlib.sha256(raw).hexdigest(),
                                     "evaluation_sha256": digest(evaluation)})
    require(shared_evaluations, "Saved acquisition evaluations are missing")
    epochs = [e["epoch"] for e in shared_evaluations]
    require(len(set(epochs)) == len(epochs), "Saved acquisition checkpoint epochs repeat")
    dose = max(e["epoch"] for e in shared_evaluations)
    expected_epochs = [acquisition_E] if acquisition_E is not None else [
        epoch for epoch in manifest["config"].get("acquisition_exposures", epochs) if epoch <= dose]
    require(set(epochs) == set(expected_epochs) and
            (acquisition_E is None or type(acquisition_E) is int and dose == acquisition_E),
            "Saved acquisition checkpoint set differs from the executed scheduled milestones")
    training = schedule_records(manifest["split"]["facts"], manifest["unit_registry"], manifest["qa_teaching_pool"]["records"])
    shared_rows, epoch_steps = [], {}
    for epoch in range(dose):
        shared_rows.extend(stage1_epoch(training, epoch, manifest["config"], manifest["acquisition_qa_pool"]["records"]))
        epoch_steps[epoch + 1] = len(shared_rows)
    require(all(type(e.get("global_step")) is int and e["global_step"] == epoch_steps[e["epoch"]]
                for e in shared_evaluations), "Saved acquisition checkpoint clock differs from its exact executed epoch prefix")
    for evidence in acquisition_evidence:
        evidence["executed_epoch_prefix_steps"] = epoch_steps[evidence["epoch"]]
        evidence["executed_epoch_prefix_sha256"] = digest(shared_rows[:epoch_steps[evidence["epoch"]]])
    require(len(shared_logs) == len(shared_rows), "Saved shared acquisition history is incomplete")
    endpoints, covariates, tables, lane_tables, unavailable = {}, {}, {}, {}, []
    # All existing acquisition checkpoints are emitted for every arm. They
    # share actual state; this duplication is explicit and supplies no extra n.
    for arm in ARMS:
        arm_logs = logs(path / arm / "exposures.jsonl")
        presentations = executed_presentations(shared_logs + arm_logs, provenance, schedule["stage2_steps"],
                                              expected_rows=shared_rows + schedule["arms"][arm])
        files = sorted((path / arm / "evaluations").glob("stage2-*.json"))
        arm_evaluations = [json.loads(read_bytes(p)) for p in files]
        by_step = {e["stage2_step"]: e for e in arm_evaluations}
        labels = _endpoint_labels(schedule, manifest["config"])
        require(len(by_step) == len(arm_evaluations) and set(by_step) == set(labels.values()),
                "Saved arm evaluation schedule is incomplete or changed")
        evaluations = [(label, by_step[step]) for label, step in labels.items()]
        evaluations += [(f"acquisition/E{e['epoch']}", e) for e in shared_evaluations]
        for label, evaluation in evaluations:
            key = arm + "/" + label
            sections = _saved_evaluation(evaluation, manifest, binding, membership)
            if primary_binding is not None and not label.startswith("acquisition/") and \
                    evaluation["stage2_step"] == primary_binding["endpoint_stage2_step"]:
                require(sections["transfer/canonical"]["aggregate"] == primary_binding["arms"][arm],
                        "Saved secondary endpoint aggregates differ from the original primary analysis aggregates")
                sections["transfer/canonical"]["aggregate"] = copy.deepcopy(primary_binding["arms"][arm])
            endpoints[key] = sections
            covariates[key] = exposure_covariates(presentations, membership, evaluation["global_step"])
            for section_name, section in sections.items():
                golds = {}
                if section_name.startswith("transfer/"):
                    by_fact = {f["id"]: f for f in manifest["split"]["facts"]}
                    golds = {r["id"]: by_fact[r["id"]]["answer"] for r in section["rows"]}
                elif section_name == "acquisition_qa":
                    golds = {r["id"]: section["mapping"][r["id"]]["original_answer_label"] for r in section["rows"]}
                else:
                    by_fact = {f["id"]: f for f in manifest["split"]["facts"]}
                    golds = {r["id"]: by_fact[r["id"]]["answer"] for r in section["rows"]}
                tables[key + "/" + section_name] = d8_prediction_table(section["rows"], section["mapping"], membership, golds)
                for role in section["roles"]:
                    lane_tables[key + "/" + section_name + "/" + role] = prediction_tables(
                        policy["exception_lanes"], [r for r in section["rows"] if r["role"] == role],
                        section["mapping"], golds)
            for required in ("transfer/canonical", "transfer/paraphrase", "acquisition_qa", "teaching_qa"):
                if required not in sections:
                    unavailable.append({"arm": arm, "endpoint": label, "section": required,
                                        "reason": "No saved diagnostic predictions for this section."})
    return {"replicate_id": replicate_id, "manifest_sha256": manifest["sha256"], "endpoints": endpoints,
            "exposure_covariates": covariates, "d8_prediction_tables": tables, "unavailable": unavailable,
            "lane_prediction_tables": lane_tables,
            "acquisition_checkpoint_evidence": acquisition_evidence,
            "consumed_file_evidence": consumed, "primary_completion_sha256": completion["sha256"] if completion else None,
            "primary_aggregate_binding_sha256": primary_binding["sha256"] if primary_binding else None,
            "lane_root_balance": balance_disclosures(policy["exception_lanes"], binding, membership),
            "shared_acquisition_duplicated_across_arms_is_not_independent_data": True}


def load_dependency_bundle(bundle, *, original_trace_input_files_by_proof=None):
    """Reload and verify one compact production reference; full fixtures remain supported."""
    if "manifest" in bundle:
        return bundle
    from .prepare import load_prepared
    manifest, schedule, examples = load_prepared(Path(bundle["path"]) / "prepared",
                                                **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))
    require(manifest["sha256"] == bundle.get("manifest_sha256") and
            manifest["confirmation_audit"]["sha256"] == bundle.get("confirmation_audit_sha256"),
            "Reloaded dependency preparation differs from the primary-verified exact reference")
    return dict(bundle, manifest=manifest, schedule=schedule, examples=examples)


def build_saved_dependency_report(bundle_inputs, frozen_policy, prereg, *, expected_evaluator_metadata=None, original_trace_input_files_by_proof=None):
    """Production reader for the primary analyzer's already validated bundles."""
    import time
    import resource
    from .schedule import ARMS
    from .confirmation_lane import complement_contrast
    started = time.monotonic()
    _sealed(frozen_policy)
    require(frozen_policy.get("schema") == FROZEN_POLICY_SCHEMA and frozen_policy.get("final_membership_frozen") is True,
            "Production diagnostics require a frozen final policy")
    policy, membership = frozen_policy["policy"], frozen_policy["policy"]["membership"]
    verify_preregistered_dependencies(prereg, frozen_policy)
    require(bundle_inputs, "Dependency reporting has no audited source context")
    registered = {r["manifest_sha256"]: r["replicate_id"] for r in policy["canonical_bindings"]}
    require(len(bundle_inputs) == prereg["n"] == len(registered) and
            {b["manifest"]["sha256"] if "manifest" in b else b.get("manifest_sha256") for b in bundle_inputs} ==
            set(registered) == set(prereg["replicate_manifest_sha256"]),
            "Dependency report must retain the complete unchanged registered replicate set")
    bundles, first_audit_sha = [], None
    for reference in bundle_inputs:
        bundle = load_dependency_bundle(reference, **sourceqa_validator_keywords(original_trace_input_files_by_proof=original_trace_input_files_by_proof))
        manifest, completion = bundle["manifest"], bundle["completion"]
        audit = manifest["confirmation_audit"]
        if first_audit_sha is None:
            verified = validate_final_policy(policy, audit["source_chunks"], audit["source_catalog"],
                                            frozen_policy["review_receipt"], frozen_policy["review_evidence"],
                                            frozen_policy["source_review_evidence"], **sourceqa_validator_keywords(expected_evaluator_metadata=expected_evaluator_metadata, original_trace_input_files_by_proof=original_trace_input_files_by_proof))
            require(verified == frozen_policy["validation"], "Saved reporting policy validation changed")
            first_audit_sha = audit["sha256"]
        require(audit["sha256"] == first_audit_sha,
                "Saved registered preparations use different exact source/evaluation audits")
        _sealed(completion)
        require(completion["manifest_sha256"] == manifest["sha256"] and completion["preregistration_sha256"] == prereg["sha256"] and
                completion["E"] == prereg["E"] and set(completion["arms"]) == set(ARMS),
                "Dependency reporting lacks existing complete five-arm confirmation proof")
        primary_binding = bundle.get("primary_aggregate_binding")
        require(primary_binding is not None and primary_binding.get("primary_delay") == prereg.get("primary_delay"),
                "Dependency reporting lacks the original primary aggregate/endpoint binding")
        bundles.append(saved_bundle_diagnostics(bundle["path"], manifest, bundle["schedule"], bundle["examples"],
                                                 policy, registered[manifest["sha256"]], acquisition_E=prereg["E"],
                                                 completion=completion, primary_binding=primary_binding))
        # Keep diagnostic rows, never retain another replicate's full audit,
        # packed examples or schedule while loading the next preparation.
        del bundle, manifest, audit, completion, primary_binding
    endpoint_labels = {key.split("/", 1)[1] for b in bundles for key in b["endpoints"]}
    contrasts = (("H2", "EXP", "UNI"), ("H3", "UNI", "MASS"), ("H1", "UNI", "NONE"), ("HG", "UNI", "GEN"))
    reports, lane_contrasts, undefined = [], [], []
    for label in sorted(endpoint_labels):
        for section in ("transfer/canonical", "transfer/paraphrase", "acquisition_qa", "teaching_qa"):
            roles = ("old", "new", "control") if section.startswith("transfer/") else ("qa",) if section == "teaching_qa" else ("old",)
            for role in roles:
                for metric in ("exact_match", "loss"):
                    for contrast, a_arm, b_arm in contrasts:
                        pairs = []
                        for bundle in bundles:
                            sides, mapping = {}, None
                            for side, arm in (("a", a_arm), ("b", b_arm)):
                                saved = bundle["endpoints"].get(arm + "/" + label, {}).get(section)
                                mapping = mapping or (saved["mapping"] if saved else None)
                                rows = [r for r in saved["rows"] if r["role"] == role] if saved else []
                                available = [metric in r for r in rows]
                                require(not any(available) or all(available), "Saved metric is partially missing within a role")
                                if not rows or not all(available):
                                    undefined.append({"replicate_id": bundle["replicate_id"], "arm": arm, "endpoint": label,
                                                      "section": section, "role": role, "metric": metric,
                                                      "reason": "No complete saved predictions for this metric; retained as undefined."})
                                    sides[side] = ([], None)
                                else:
                                    sides[side] = (rows, saved["aggregate"][role][metric + "_event_macro"])
                            mapping = mapping or _global_mapping({}, membership)
                            pairs.append({"replicate_id": bundle["replicate_id"], "mapping": mapping,
                                          "a_rows": sides["a"][0], "b_rows": sides["b"][0],
                                          "primary_a": sides["a"][1], "primary_b": sides["b"][1]})
                        reports.append(complete_contrast_report(pairs, contrast=contrast, endpoint=label, role=role,
                            variant=section, metric=metric, flagged_units=membership["flagged_source_unit_ids"],
                            preserved_units=membership["preserved_source_unit_ids"]))
                        lane_contrasts.append(complement_contrast(pairs, policy["exception_lanes"], contrast=contrast,
                            endpoint=label, role=role, variant=section, metric=metric))
    result = {"schema": "spacing-confirmation-saved-dependency-report-v1", "frozen_policy_sha256": frozen_policy["sha256"],
              "preregistration_sha256": prereg["sha256"], "bundles": [{k: v for k, v in b.items() if k != "endpoints"} for b in bundles],
              "contrasts": reports, "undefined_diagnostics": undefined,
              "lane_descriptive_reporting": {"policy": policy["lane_reporting_policy"],
                  "exception_lanes_sha256": digest(policy["exception_lanes"]),
                  "lane_only_complement_contrasts": lane_contrasts,
                  "balance_and_prediction_tables_location": "bundles[*].lane_root_balance and bundles[*].lane_prediction_tables",
                  "missing_sections_location": "bundles[*].unavailable",
                  "V3R2_algorithm_changed": False, "either_label_success_metric_computed": False},
              "primary_analysis_changed": False, "acquisition_gate_or_E_changed": False,
              "no_retraining_or_new_evaluation": True, "valid_poorly_acquired_replicates_retained": True,
              "runtime_seconds": time.monotonic() - started,
              "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if os.uname().sysname == "Darwin" else 1024)}
    result["sha256"] = digest(result)
    return result
