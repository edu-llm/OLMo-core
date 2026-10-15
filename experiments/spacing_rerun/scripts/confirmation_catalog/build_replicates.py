"""Write the per-replicate confirmation configs and their derived grounding and acquisition bundles.

Run on FarmShare, with HF_HOME set to the pinned cache:
    python -I -B build_replicates.py AUDIT TEMPLATE_CONFIG OUTPUT_ROOT N

For replicate i (1..N), the split seed is 2026100800+i. Roles come from the original
128-candidate metadata rule with the audit's same-role components. Each bundle copies the
audited, approved content verbatim (review_status approved_confirmation) and is checked with
the same validators prepare uses. Nothing semantic is authored here.
"""

import copy
import hashlib
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from spacing_rerun.acquisition import validate_acquisition_bundle  # noqa: E402
from spacing_rerun.common import digest  # noqa: E402
from spacing_rerun.confirmation import PROTOCOL, assign_confirmation_roles, validate_confirmation_audit  # noqa: E402
from spacing_rerun.data import DATASET, MODEL, REVISION, canonicalize, outer_partition  # noqa: E402
from spacing_rerun.grounding import (FACTSHEETS_SHA256, build_grounded_registry, validate_bundle,  # noqa: E402
                                     validate_grounded_registry)
from spacing_rerun.units import GROUNDED_POLICY, build_units  # noqa: E402

AUTHOR_INPUTS = ["source_assertions", "source_answer_labels", "event_factsheets"]
audit_path, template_path, output = (pathlib.Path(p) for p in sys.argv[1:4])
n = int(sys.argv[4])
if output.exists():
    sys.exit("output exists; never overwrite")


def sealed(payload):
    return dict(payload, sha256=digest(payload))


from huggingface_hub import hf_hub_download  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from transformers import PreTrainedTokenizerFast  # noqa: E402

audit = json.load(open(audit_path))
context = validate_confirmation_audit(audit)
template = json.load(open(template_path))
tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL, revision=template["model_revision"])


def token_count(text):
    return len(tokenizer.encode(text, add_special_tokens=False))


rows = pq.read_table(hf_hub_download(DATASET, "fict_qa/train-00000-of-00001.parquet", repo_type="dataset",
                                     revision=REVISION)).to_pylist()
factsheet_path = hf_hub_download(DATASET, "fictsheets/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
assert hashlib.sha256(pathlib.Path(factsheet_path).read_bytes()).hexdigest() == FACTSHEETS_SHA256
pinned_factsheets = {r["event_id"]: r["fictsheet"] for r in pq.read_table(factsheet_path).to_pylist()}
base_facts, source_audit = canonicalize(rows)
partition = outer_partition(base_facts, source_audit["cross_event_links"], 20261007)
assert partition["sha256"] == context["outer_partition_sha256"]
evaluations = {r["id"]: r for r in context["evaluation_records"]}
sources = {r["unit_id"]: r for r in context["sources"]}
groups = {g["id"]: g for g in context["groups"]}
catalog = audit["source_catalog"]
reviewers = sorted({str(r["reviewer"].get("model", r["reviewer"].get("name"))) for r in context["review_identities"]})
provenance = {"audit_path": "../confirmation-audit.json", "confirmation_audit_sha256": audit["sha256"],
              "source_catalog_sha256": catalog["sha256"]}
output.mkdir(parents=True)
shutil.copyfile(audit_path, output / "confirmation-audit.json")
summary = []
for i in range(1, n + 1):
    seed = 2026100800 + i
    split = assign_confirmation_roles(copy.deepcopy(base_facts), partition, seed, context["same_role_components"])
    roles, facts = split["roles"], split["facts"]
    for fact in facts:
        # Same alias/paraphrase override prepare applies before validating bundles.
        fact["aliases"] = [evaluations[fact["id"]]["canonical_answer"]]
        fact["paraphrase"] = evaluations[fact["id"]]["paraphrase"]
    registry = build_units(facts)
    by_probe = {f["id"]: f for f in facts}
    source_rows = []
    for unit in registry["units"]:
        if unit["role"] not in ("old", "new"):
            continue
        audited = sources[unit["id"]]
        row = {"unit_id": unit["id"], "event_id": unit["event"], "role": unit["role"],
               "source_statement": unit["source_statement"],
               "canonical_answers": [by_probe[pid]["answer"] for pid in unit["member_probe_ids"]],
               "probe_ids": unit["member_probe_ids"]}
        row.update(index=len(source_rows), source_statement_sha256=digest(unit["source_statement"]),
                   source_fields_sha256=digest(row), source_unit_sha256=audited["source_unit_sha256"],
                   group_id=audited["group_id"], trained_declaration=groups[audited["group_id"]]["statement"],
                   factsheet_sha256=catalog["factsheets"][unit["event"]]["sha256"], status=audited["status"],
                   conflict_rationale=audited["conflict_rationale"], evidence=copy.deepcopy(audited["evidence"]),
                   nonliteral_answer_labels=audited["nonliteral_answer_labels"], review_status="approved_confirmation")
        source_rows.append(row)
    group_rows = []
    for gid, audited in sorted(groups.items()):
        if roles[audited["event"]] not in ("old", "new"):
            continue
        group = {k: copy.deepcopy(audited[k]) for k in ("id", "event", "source_unit_ids", "statement",
                                                         "grouping_rationale", "evidence")}
        group.update(role=roles[group["event"]], source_indices=[r["index"] for r in source_rows if r["group_id"] == gid],
                     statement_tokens=token_count(group["statement"]), review_status="approved_confirmation")
        group_rows.append(group)
    grounding = sealed({"schema": "spacing-grounded-declarations-v1", "policy": GROUNDED_POLICY, "mode": "confirmation",
                        "source_dataset_revision": REVISION, "source_factsheets_sha256": FACTSHEETS_SHA256,
                        "source_catalog_sha256": catalog["sha256"], "confirmation_audit_sha256": audit["sha256"],
                        "partition_sha256": partition["sha256"], "roles_sha256": digest(roles),
                        "authoring_inputs": ["source_statement", "canonical_answers", "event_factsheet"],
                        "evaluation_question_text_included": False, "model_outcomes_used": False,
                        "independent_agent_review_complete": True, "independent_agent_reviewer": reviewers,
                        "human_review_complete": False, "review_provenance": provenance,
                        "source_unit_count": len(source_rows), "original_evaluation_probe_count": len(facts),
                        "factsheets": {e: catalog["factsheets"][e]["text"] for e in sorted({r["event_id"] for r in source_rows})},
                        "source_units": source_rows, "groups": group_rows,
                        "max_statement_tokens": max(g["statement_tokens"] for g in group_rows)})
    validate_bundle(grounding, facts, registry, mode="confirmation", partition_hash=partition["sha256"], roles=roles,
                    pinned_factsheets=pinned_factsheets, confirmation_audit=audit)
    grounded = build_grounded_registry(facts, registry, grounding)
    validate_grounded_registry(facts, grounded, registry, grounding)
    old_sources = {s["unit_id"]: s for s in source_rows if s["role"] == "old"}
    units = []
    for unit in grounded["units"]:
        if unit["role"] != "old":
            continue
        assertions = [old_sources[sid] for sid in unit["source_unit_ids"]]
        units.append({"unit_id": unit["id"], "event": unit["event"], "statement": unit["statement"],
                      "answer_targets": sorted({a for s in assertions for a in s["canonical_answers"]}),
                      "source_assertions": [{"statement": s["source_statement"],
                                             "answer_targets": sorted(set(s["canonical_answers"])),
                                             "status": s["status"], "evidence": s["evidence"]} for s in assertions]})
    source_input = sealed({"schema": "source-only-acquisition-authoring-input-v1", "dataset_revision": REVISION,
                           "source_catalog_sha256": catalog["sha256"], "confirmation_audit_sha256": audit["sha256"],
                           "grounding_bundle_sha256": grounding["sha256"], "review_provenance": provenance,
                           "human_review_complete": False, "independent_agent_review_complete": True,
                           "authoring_inputs": AUTHOR_INPUTS, "evaluation_question_text_included": False,
                           "model_outcomes_used": False,
                           "input_scope": "old reviewed units; original assertions, answer labels and pinned factsheets",
                           "units": units,
                           "factsheets": {e: grounding["factsheets"][e] for e in sorted({u["event"] for u in units})}})
    records = [dict(copy.deepcopy(r), review_status="approved_confirmation")
               for r in context["acquisition_records"] if roles[r["event"]] == "old"]
    qa = sealed({"schema": "spacing-acquisition-qa-v1", "mode": "confirmation", "source_catalog_sha256": catalog["sha256"],
                 "confirmation_audit_sha256": audit["sha256"], "grounding_bundle_sha256": grounding["sha256"],
                 "source_input_sha256": source_input["sha256"], "authoring_inputs": AUTHOR_INPUTS,
                 "evaluation_question_text_included": False, "model_outcomes_used": False,
                 "independent_agent_review_complete": True, "independent_agent_reviewer": reviewers,
                 "human_review_complete": False, "review_provenance": provenance, "records": records})
    validate_acquisition_bundle(qa, source_input, facts, grounded, grounding, mode="confirmation", confirmation_audit=audit)
    config = {k: v for k, v in template.items() if k not in (
        "development_design_limit", "acquisition_trajectory_policy", "qa_teaching_diagnostics")}
    config.update(mode="confirmation", confirmation_protocol_schema=PROTOCOL, confirmation_audit="../confirmation-audit.json",
                  split_seed=seed, order_seed=2026100900 + i, train_seed=2026101000 + i, eval_seed=2026101100 + i,
                  training_unit_grounding="grounding-declarations.json",
                  acquisition_qa_bundle="acquisition-qa.json", acquisition_qa_source_input="acquisition-source-input.json")
    directory = output / f"replicate-{i:02d}"
    directory.mkdir()
    for name, payload in (("config.json", config), ("grounding-declarations.json", grounding),
                          ("acquisition-source-input.json", source_input), ("acquisition-qa.json", qa)):
        (directory / name).write_text(json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True))
    summary.append({"replicate": i, "split_seed": seed, "roles_sha256": digest(roles),
                    "old_groups": sum(g["role"] == "old" for g in group_rows),
                    "new_groups": sum(g["role"] == "new" for g in group_rows), "old_acquisition_targets": len(records),
                    "grounding_sha256": grounding["sha256"], "acquisition_qa_sha256": qa["sha256"]})
    print(json.dumps(summary[-1]), flush=True)
(output / "replicates.json").write_text(json.dumps({"audit_sha256": audit["sha256"], "replicates": summary}, indent=1))
