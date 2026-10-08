from __future__ import annotations

import collections
import datetime
import hashlib
from pathlib import Path

from . import SCHEMA
from .common import digest, read_json, require, write_json
from .data import (DATASET, MODEL, MODEL_REVISION_PREFIX, REVISION, assign_roles,
                   canonicalize, join_metadata, normalize, outer_partition)
from .encoding import build_examples, generic_slice
from .schedule import compile_schedule, validate_schedule
from .rewrites import apply_development_rewrites
from .units import LEGACY_POLICY, UNIT_POLICY, GROUNDED_POLICY, build_units, schedule_records, validate_units
from .teaching import CANONICAL_QA_POLICY, SOURCE_QA_POLICY, build_teaching_pool, validate_teaching_pool
from .grounding import FACTSHEETS_SHA256, build_grounded_registry, validate_bundle, validate_grounded_registry

MCQ_CONFIG = "gend_mcq_w_grades_03-01-26"
GENERIC_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
SOURCE_SHA256 = {
    "fict_qa": "c178467bfd7db74b17c2d57333bb8908f881c0e1abfbe216c7db75f0934765b7",
    MCQ_CONFIG: "a917d33eba7256467d2a8f554a6ffc48e78bf578c00dcaab8a5752373bff6850",
    "blind_answer_attempts": "f2994c005b88fb48651cc42f2a91bb5ad8e10076465b27d248e41ee23811f20e",
}


def validate_config(config):
    for key in ("batch_size", "microbatch_size", "context_length", "loss_tokens_per_example",
                "eval_context_length", "max_answer_tokens", "eval_batch_size", "generation_batch_size",
                "generic_eval_sequences", "max_attempts", "checkpoint_every_updates", "log_every_updates"):
        require(isinstance(config[key], int) and config[key] > 0, f"Positive integer required: {key}")
    require(0 < config["max_runtime_seconds"] < 24 * 3600, "Per-job runtime must be below24hours")
    require(0 < config["learning_rate"] and config["warmup_steps"] >= 0, "Invalid LR/warmup")
    require(0 < config["acquisition_target_min"] < config["acquisition_target_max"] < 1, "Invalid acquisition targets")
    milestones = config["acquisition_exposures"]
    require(milestones and milestones == sorted(set(milestones)) and all(isinstance(x, int) and x > 0 for x in milestones),
            "Acquisition exposure caps must increase strictly")
    delays = config["eval_delays"]
    require(delays and delays == sorted(set(delays)) and all(isinstance(x, int) and 0 < x <= config["buffer_steps"] for x in delays),
            "Invalid frozen evaluation delays")
    require(config["primary_delay"] in delays, "Primary delay must be a declared evaluation delay")
    require(max(delays) == config["buffer_steps"], "Longest evaluated delay must end the buffer")
    require(all(0 < f < 1 for f in config["stage2_eval_fractions"]), "Stage2 grid lies outside Stage2")
    require(config.get("rehearsal_unit_policy", LEGACY_POLICY) in (LEGACY_POLICY, UNIT_POLICY, GROUNDED_POLICY),
            "Unknown rehearsal unit policy")
    if config.get("rehearsal_unit_policy") == GROUNDED_POLICY:
        require(config["mode"] == "development", "Grounded assertion units are development-only until audited and frozen")
        require(config.get("training_unit_grounding") and not config.get("training_statement_rewrites"),
                "Grounded assertions require their own reviewed bundle without a second rewrite recipe")
    else:
        require(not config.get("training_unit_grounding"), "Grounding bundle requires grounded_assertion_v1")
    require(config.get("qa_teaching_policy", CANONICAL_QA_POLICY) in (CANONICAL_QA_POLICY, SOURCE_QA_POLICY),
            "Unknown QA-teaching policy")
    if config.get("qa_teaching_policy") == SOURCE_QA_POLICY:
        require(config["mode"] == "development", "Expanded source QA teaching is development-only until audited and frozen")
    if config["mode"] == "confirmation":
        require(config.get("rehearsal_unit_policy") == UNIT_POLICY,
                "Confirmation requires distinct supporting-statement rehearsal units")


def prepare(config_path, output, source_directory=None, audit_path=None):
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq
    from transformers import PreTrainedTokenizerFast

    config = read_json(config_path)
    validate_config(config)
    output = Path(output)
    require(not (output / "manifest.json").exists(), "Prepared directory already exists; never overwrite")
    model_revision = HfApi().model_info(MODEL, revision=config["model_revision"]).sha
    require(model_revision.startswith(MODEL_REVISION_PREFIX), "Wrong pretrained checkpoint")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL, revision=model_revision)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = "<|endoftext|>"
    require(tokenizer.eos_token_id is not None, "Tokenizer has no EOS")
    tokenizer.pad_token = tokenizer.eos_token
    tables = {}
    for name in ("fict_qa", MCQ_CONFIG, "blind_answer_attempts"):
        path = Path(source_directory) / (name + ".parquet") if source_directory else hf_hub_download(
            DATASET, f"{name}/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION)
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == SOURCE_SHA256[name],
                f"Source parquet differs from pinned revision: {name}")
        tables[name] = pq.read_table(path).to_pylist()
    require(len(tables["fict_qa"]) == 7500, "Pinned FictionalQA row count changed")
    source_rows = tables.pop("fict_qa")
    facts, source_audit = canonicalize(source_rows)
    source_audit["joins"] = join_metadata(facts, tables)
    partition = outer_partition(facts, source_audit["cross_event_links"], config["outer_seed"])
    audited = read_json(audit_path) if audit_path else {}
    for fact in facts:
        entries = fact["source_metadata"][MCQ_CONFIG]
        require(len(entries) == 1, "MCQ canonical join is not one-to-one")
        row = entries[0]
        choices, correct = list(row["topk_choices"]), int(row["target_idx"])
        require(0 <= correct < len(choices) and normalize(choices[correct]) == normalize(fact["answer"]),
                "MCQ target disagrees with canonical answer")
        require(len(set(map(normalize, choices))) == len(choices), "Duplicate MCQ candidates")
        fact["mcq"] = {"choices": choices, "correct": correct, "source_config": MCQ_CONFIG}
        override = audited.get("facts", {}).get(fact["id"], {})
        fact["aliases"] = override.get("aliases", [fact["answer"]])
        if "paraphrase" in override:
            fact["paraphrase"] = override["paraphrase"]
        # Preserve join evidence in the source audit, without triplicating the raw tables per replicate.
        del fact["source_metadata"]
    split = assign_roles(facts, partition, config["mode"], config["split_seed"])
    facts = split["facts"]
    rewrite_bundle, rewrite_audit = None, None
    if config.get("training_statement_rewrites"):
        rewrite_path = Path(config_path).resolve().parent / config["training_statement_rewrites"]
        rewrite_bundle = read_json(rewrite_path)
        require(rewrite_bundle["source_dataset_revision"] == REVISION, "Rewrite dataset revision differs")
        rewrite_audit = apply_development_rewrites(facts, rewrite_bundle, mode=config["mode"],
                                                   partition_hash=partition["sha256"], roles=split["roles"])
    unit_registry = build_units(facts) if config.get("rehearsal_unit_policy") in (UNIT_POLICY, GROUNDED_POLICY) else None
    source_registry, grounding_bundle = None, None
    if config.get("rehearsal_unit_policy") == GROUNDED_POLICY:
        grounding_path = Path(config_path).resolve().parent / config["training_unit_grounding"]
        grounding_bundle = read_json(grounding_path)
        factsheet_path = (Path(source_directory) / "fictsheets.parquet" if source_directory else hf_hub_download(
            DATASET, "fictsheets/train-00000-of-00001.parquet", repo_type="dataset", revision=REVISION))
        require(hashlib.sha256(Path(factsheet_path).read_bytes()).hexdigest() == FACTSHEETS_SHA256,
                "Factsheet parquet differs from pinned source revision")
        factsheets = {r["event_id"]: r["fictsheet"] for r in pq.read_table(factsheet_path).to_pylist()}
        source_registry = unit_registry
        validate_bundle(grounding_bundle, facts, source_registry, mode=config["mode"],
                        partition_hash=partition["sha256"], roles=split["roles"], pinned_factsheets=factsheets)
        unit_registry = build_grounded_registry(facts, source_registry, grounding_bundle)
        validate_grounded_registry(facts, unit_registry, source_registry, grounding_bundle)
    teaching_pool = (build_teaching_pool(source_rows, facts, split["roles"], config["mode"])
                     if config.get("qa_teaching_policy") == SOURCE_QA_POLICY else None)
    teaching_records = teaching_pool["records"] if teaching_pool else None
    generic_path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet",
                                   repo_type="dataset", revision=GENERIC_REVISION)
    generic_rows = pq.read_table(generic_path).to_pylist()
    generic_text = "\n".join(row["text"] for row in generic_rows if row["text"].strip())
    generic_tokens = tokenizer.encode(generic_text, add_special_tokens=False)
    examples = build_examples(facts, tokenizer, generic_tokens, config, teaching_records)
    if grounding_bundle:
        measured = {g["id"]: len(tokenizer.encode(g["statement"], add_special_tokens=False))
                    for g in grounding_bundle["groups"]}
        require(all(measured[g["id"]] == g["statement_tokens"] for g in grounding_bundle["groups"]) and
                max(measured.values()) == grounding_bundle["max_statement_tokens"], "Grounded token-length audit differs")
    # Encoding adds frozen probes/option order to the same fact objects.
    split["sha256"] = digest({k: v for k, v in split.items() if k != "sha256"})
    training_records = schedule_records(facts, unit_registry, teaching_records)
    schedule = compile_schedule(training_records, config)
    validate_schedule(schedule, training_records, examples)
    # A disjoint generic validation split diagnoses broad language-model degradation.
    validation_path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/validation-00000-of-00001.parquet",
                                      repo_type="dataset", revision=GENERIC_REVISION)
    val_text = "\n".join(r["text"] for r in pq.read_table(validation_path).to_pylist() if r["text"].strip())
    val_tokens = tokenizer.encode(val_text, add_special_tokens=False)
    generic_eval = [generic_slice(val_tokens, config["context_length"], f"generic-eval/{i}")
                    for i in range(config["generic_eval_sequences"])]
    audit_complete = bool(audited.get("cross_event_alias_audit_complete") and
                          audited.get("answer_alias_audit_complete") and
                          all(f.get("paraphrase") for f in facts if f["role"] == "old"))
    if config["mode"] == "confirmation":
        require(audit_complete, "Confirmation requires verified old paraphrases, aliases and entity audit")
    model_revision = str(model_revision)
    manifest = {"schema": SCHEMA, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "mode": config["mode"], "config": config, "model": MODEL,
                "model_revision": model_revision, "tokenizer_sha256": digest(tokenizer.backend_tokenizer.to_str()),
                "dataset": DATASET, "dataset_revision": REVISION, "source_audit": source_audit,
                "partition": partition, "split": split, "schedule_sha256": schedule["sha256"],
                "examples_sha256": digest(examples), "generic_revision": GENERIC_REVISION,
                "generic_source_sha256": digest(generic_text), "audit_complete": audit_complete,
                "audit_sha256": digest(audited), "generic_eval": generic_eval,
                "generic_eval_sha256": digest(generic_eval),
                "fact_counts": dict(collections.Counter(f["role"] for f in facts)),
                "training_budget": {"loss_tokens_per_example": config["loss_tokens_per_example"],
                    "generic_filler_policy": "EOS followed by deterministic pinned WikiText filler; all filler tokens bear loss",
                    "statement_tokens": [len(f["statement_ids"]) for f in facts],
                    "generic_filler_tokens_by_example": {k: v["generic_filler_tokens"] for k, v in examples.items()}}}
    if rewrite_audit is not None:
        manifest["statement_rewrite_audit"] = rewrite_audit
    manifest["rehearsal_unit_policy"] = config.get("rehearsal_unit_policy", LEGACY_POLICY)
    if unit_registry is not None:
        manifest["unit_registry"] = unit_registry
    if teaching_pool is not None:
        manifest["qa_teaching_pool"] = teaching_pool
    if grounding_bundle is not None:
        manifest["source_unit_registry"] = source_registry
        manifest["grounding_audit"] = {"bundle_sha256": grounding_bundle["sha256"],
            "source_factsheets_sha256": FACTSHEETS_SHA256, "source_units": grounding_bundle["source_unit_count"],
            "groups": len(grounding_bundle["groups"]), "independent_agent_review_complete": True,
            "human_review_complete": grounding_bundle["human_review_complete"]}
    manifest["sha256"] = digest(manifest)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output / "tokenizer")
    write_json(output / "examples.json", examples)
    write_json(output / "schedule.json", schedule)
    if rewrite_bundle is not None:
        write_json(output / "statement-rewrites.json", rewrite_bundle)
    if grounding_bundle is not None:
        write_json(output / "grounding-declarations.json", grounding_bundle)
    write_json(output / "manifest.json", manifest)
    return manifest


def load_prepared(path):
    path = Path(path)
    manifest, schedule, examples = [read_json(path / name) for name in
                                     ("manifest.json", "schedule.json", "examples.json")]
    validate_config(manifest["config"])
    for payload in (manifest, schedule, manifest["split"], manifest["partition"]):
        require(digest({k: v for k, v in payload.items() if k != "sha256"}) == payload["sha256"],
                "Manifest integrity check failed")
    require(digest(examples) == manifest["examples_sha256"], "Tokenized examples were modified")
    require(schedule["sha256"] == manifest["schedule_sha256"], "Schedule was modified")
    if manifest.get("statement_rewrite_audit"):
        rewrite_bundle = read_json(path / "statement-rewrites.json")
        require(digest({k: v for k, v in rewrite_bundle.items() if k != "sha256"}) ==
                rewrite_bundle["sha256"] == manifest["statement_rewrite_audit"]["bundle_sha256"],
                "Prepared rewrite audit changed")
    facts = manifest["split"]["facts"]
    registry = manifest.get("unit_registry")
    require(manifest.get("rehearsal_unit_policy", LEGACY_POLICY) ==
            manifest["config"].get("rehearsal_unit_policy", LEGACY_POLICY), "Manifest unit policy differs from config")
    if manifest["config"].get("rehearsal_unit_policy") == GROUNDED_POLICY:
        require(registry is not None and manifest.get("source_unit_registry") and manifest.get("grounding_audit"),
                "Missing original/grounded unit provenance")
        bundle = read_json(path / "grounding-declarations.json")
        require(manifest["grounding_audit"]["bundle_sha256"] == bundle["sha256"] and
                manifest["grounding_audit"]["source_factsheets_sha256"] == FACTSHEETS_SHA256,
                "Prepared grounding evidence differs")
        validate_bundle(bundle, facts, manifest["source_unit_registry"], mode=manifest["mode"],
                        partition_hash=manifest["partition"]["sha256"], roles=manifest["split"]["roles"])
        validate_grounded_registry(facts, registry, manifest["source_unit_registry"], bundle)
    elif manifest["config"].get("rehearsal_unit_policy") == UNIT_POLICY:
        require(registry is not None, "Missing rehearsal unit registry")
        validate_units(facts, registry)
    else:
        require(registry is None, "Legacy schedule unexpectedly contains a unit registry")
    teaching_pool = manifest.get("qa_teaching_pool")
    if manifest["config"].get("qa_teaching_policy") == SOURCE_QA_POLICY:
        require(teaching_pool is not None, "Missing source QA-teaching pool")
        validate_teaching_pool(teaching_pool, facts, manifest["split"]["roles"])
    else:
        require(teaching_pool is None, "Canonical QA-teaching unexpectedly contains a source pool")
    validate_schedule(schedule, schedule_records(facts, registry, teaching_pool["records"] if teaching_pool else None), examples)
    return manifest, schedule, examples
