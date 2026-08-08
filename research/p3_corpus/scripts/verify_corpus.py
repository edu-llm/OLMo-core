"""Deep scan of every example in the corpus.

`sweep_corpus.py` runs the invariant suite, which samples and checks properties.
This reads all 285,908 examples and reconstructs each one from its parts, so a
corruption in a single record cannot hide behind an aggregate.

Checks, in the order a defect would bite:

  STRUCTURE   every record has the schema, with usable types and no empty
              goal, target or fact block
  RENDERING   text is exactly block + separator + goal + target, and
              [mask_start, mask_end) lands precisely on the block — if this
              slips, the split arm masks the wrong tokens and the whole
              experiment is measuring nothing
  REFERENCES  every name in `cited` has a statement, and every fact in the block
              appears in the block text
  IDENTITY    ids are unique corpus-wide, not just per shard
  ACCOUNTING  raw = shards + eval + the examples the splitter dropped, so no
              example vanished silently between stages
  HOLD-OUT    no training example touches a held-out fact; every eval example
              does
  ENCODING    no replacement characters or stray control bytes
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path

from build_atp_shard import (
    ProofStep,
    is_refutation_formula,
    render_target,
    source_dependencies,
)
from split_heldout import (
    canonicalization_metadata,
    exact_atp_signature,
    heldout_exposure,
    normalize_theorem_identity,
    statement_hash,
)

HDR = "I know these mathematical statements:"
LOCAL_HDR = "Local assumptions:"
ATP_LOCAL_HDR = "Local ATP inputs:"
SEP = "---"
REQUIRED = (
    "id",
    "theorem",
    "facts",
    "cited",
    "goal",
    "target",
    "text",
    "mask_start",
    "mask_end",
)
ATP_REQUIRED = ("local_inputs", "goal_name", "proof_steps")
ATP_SHARDS = {"enigma", "prf2"}
SHARD_FAMILY = {
    "enigma": "atp",
    "prf2": "atp",
    "metamath": "metamath",
    "mizar": "mizar",
    "thproofs": "mizar",
    "isabelle": "isabelle",
}
STEP_RE = re.compile(r"^\s*\d+\s+(\S+)\s+(\|-\s.*)$")


def _read_lines(path):
    with open(path, encoding="utf-8") as source_file:
        yield from source_file


def _load_json(path):
    with open(path, encoding="utf-8") as source_file:
        return json.load(source_file)


def _parse_json_line(line):
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return None


def atp_record_errors(record):
    """Return all replay-structure defects in one ATP v2 record."""
    errors = []
    proof_steps = record.get("proof_steps")
    if not isinstance(proof_steps, list) or not proof_steps:
        return [("missing_atp_proof_steps", "")]
    local_inputs = record.get("local_inputs", {})
    if not isinstance(local_inputs, dict):
        local_inputs = {}

    parsed_steps = []
    for index, step in enumerate(proof_steps, 1):
        if not isinstance(step, dict):
            errors.append(("malformed_atp_step", f"step {index} is not an object"))
            continue
        missing = [
            key
            for key in (
                "name",
                "role",
                "formula",
                "rule",
                "parents",
                "parent_sources",
                "source",
            )
            if key not in step
        ]
        if missing:
            errors.append(("malformed_atp_step", f"step {index} missing {missing}"))
            continue
        if (
            not isinstance(step["parents"], list)
            or not all(isinstance(parent, str) for parent in step["parents"])
            or not isinstance(step["parent_sources"], list)
            or not all(isinstance(parent, str) for parent in step["parent_sources"])
        ):
            errors.append(("malformed_atp_step", f"step {index} has bad parents"))
            continue
        parsed = source_dependencies(step["source"])
        if parsed is None:
            errors.append(
                ("atp_step_without_derived_source", f"step {index} {step['name']}")
            )
        else:
            rule, parent_sources, parents = parsed
            if (
                rule != step["rule"]
                or parent_sources != step["parent_sources"]
                or parents != step["parents"]
            ):
                errors.append(
                    (
                        "atp_source_parent_mismatch",
                        f"step {index} {step['name']}",
                    )
                )
        parsed_steps.append(
            ProofStep(
                name=step["name"],
                role=step["role"],
                formula=step["formula"],
                rule=step["rule"],
                parents=step["parents"],
                parent_sources=step["parent_sources"],
                source=step["source"],
            )
        )

    if len(parsed_steps) == len(proof_steps):
        if render_target(parsed_steps) != record.get("target"):
            errors.append(("atp_target_not_reconstructible", ""))

        supplied = set(record.get("facts", {})) | set(local_inputs)
        goal_name = record.get("goal_name")
        if goal_name:
            supplied.add(goal_name)
        all_steps = {step.name for step in parsed_steps}
        seen = set()
        for step in parsed_steps:
            if step.name in seen:
                errors.append(("duplicate_atp_step", step.name))
            for parent in step.parents:
                if parent in supplied or parent in seen:
                    continue
                if parent in all_steps:
                    errors.append(
                        ("atp_parent_not_earlier", f"{step.name} <- {parent}")
                    )
                else:
                    errors.append(("unresolved_atp_parent", f"{step.name} <- {parent}"))
            seen.add(step.name)
        if not is_refutation_formula(parsed_steps[-1].formula):
            errors.append(("atp_target_not_refutation", parsed_steps[-1].name))
    return errors


def resolve_current_index_references(
    shard,
    target,
    index,
    statements,
    *,
    theorem,
):
    """Dispatch current-index replay to each Mizar family's native resolver."""

    if shard == "mizar":
        from build_mizar_human_shard import resolve_global_citations

        resolution = resolve_global_citations(
            target,
            index,
            theorem=theorem,
        )
        return list(resolution.references), list(resolution.unresolved)
    if shard == "thproofs":
        from build_thproofs_shard import resolve_index_references

        return resolve_index_references(
            target,
            index,
            statements,
            theorem=theorem,
        )
    raise ValueError(f"{shard}: current Mizar index resolver is undefined")


def _direct_mizar_source_binding_errors(
    record,
    *,
    declaration,
    declarations,
    index,
):
    """Independently replay primary or conservative secondary source binding."""

    from build_mizar_human_shard import (
        SECONDARY_ALIGNMENT_METHOD,
        SOURCE_INDEX_BINDING_SCHEMA,
        _goals_match,
        _load_anchors,
        _strict_complete_alignment,
    )
    from mizar_current_index import _comparison_key

    source = record["source"]
    cache = getattr(index, "_direct_mizar_binding_cache", None)
    if cache is None:
        cache = {}
        index._direct_mizar_binding_cache = cache
    cache_key = (
        str(source["article"]),
        str(source["file"]),
        str(source["file_sha256"]),
    )
    cached = cache.get(cache_key)
    if cached is None:
        anchors = _load_anchors(index, str(source["article"]))
        if any(left.number >= right.number for left, right in pairwise(anchors)):
            return [("mizar_source_binding_ambiguous", "index order")]
        primary, _ = _strict_complete_alignment(declarations, anchors)
        source_label_counts = Counter(item.label for item in declarations if item.label)
        index_label_counts = Counter(
            anchor.local_label for anchor in anchors if anchor.local_label
        )
        cached = (
            anchors,
            primary,
            source_label_counts,
            index_label_counts,
        )
        cache[cache_key] = cached
    anchors, primary, source_label_counts, index_label_counts = cached
    primary_match = [
        item
        for item in primary
        if item.source.ordinal == declaration.ordinal
        and item.identity == record.get("theorem")
    ]
    binding = record.get("source_index_binding")
    if len(primary_match) == 1:
        if binding is not None:
            return [("mizar_source_binding_mismatch", "primary row")]
        return []

    label = declaration.label
    if binding is None:
        return [("mizar_source_binding_missing", str(label))]
    if (
        declaration.category != "complete_explicit_proof"
        or declaration.target_sha256 is None
        or not label
        or source_label_counts[label] != 1
        or index_label_counts[label] != 1
    ):
        return [("mizar_source_binding_ambiguous", str(label))]

    ordered_primary = sorted(primary, key=lambda item: item.source.ordinal)
    previous = next(
        (
            item
            for item in reversed(ordered_primary)
            if item.source.ordinal < declaration.ordinal
        ),
        None,
    )
    following = next(
        (item for item in ordered_primary if item.source.ordinal > declaration.ordinal),
        None,
    )

    def proof_hash_anchor(item):
        if item is None:
            return None
        if (
            item.anchor.proof_category != "complete_explicit_proof"
            or item.anchor.proof_sha256 is None
            or item.anchor.proof_sha256 != item.source.target_sha256
        ):
            return False
        return {
            "source_ordinal": item.source.ordinal,
            "identity": item.identity,
            "index_number": item.anchor.number,
            "proof_sha256": item.anchor.proof_sha256,
        }

    previous_binding = proof_hash_anchor(previous)
    following_binding = proof_hash_anchor(following)
    if previous_binding is False or following_binding is False:
        return [("mizar_source_binding_ambiguous", "neighbor proof hash")]
    lower = previous.anchor.number if previous is not None else 0
    upper = following.anchor.number if following is not None else sys.maxsize
    used = {item.identity for item in primary}
    candidates = [
        anchor
        for anchor in anchors
        if anchor.identity not in used
        and anchor.article == declaration.article
        and anchor.local_label == label
        and anchor.mml_alignment == "literal_goal_match"
        and anchor.proof_category == "malformed_explicit_proof"
        and anchor.proof_sha256 is None
        and lower < anchor.number < upper
        and _goals_match(declaration, anchor)
    ]
    if len(candidates) != 1 or candidates[0].identity != record.get("theorem"):
        return [("mizar_source_binding_ambiguous", str(label))]
    normalized_source_goal = _comparison_key(declaration.index_source_goal)
    normalized_index_goal = _comparison_key(candidates[0].source_goal or "")
    if normalized_source_goal != normalized_index_goal:
        return [("mizar_source_binding_ambiguous", "normalized goal")]
    expected = {
        "schema_version": SOURCE_INDEX_BINDING_SCHEMA,
        "method": SECONDARY_ALIGNMENT_METHOD,
        "source_label_occurrences": 1,
        "index_label_occurrences": 1,
        "normalized_goal_sha256": hashlib.sha256(
            normalized_source_goal.encode("utf-8")
        ).hexdigest(),
        "previous_proof_hash_anchor": previous_binding,
        "next_proof_hash_anchor": following_binding,
    }
    if binding != expected:
        return [("mizar_source_binding_mismatch", str(label))]
    return []


def direct_mizar_record_errors(
    record,
    *,
    semantic_index,
    mml_root,
    source_manifest,
):
    """Replay one current direct-Mizar row against source and semantic roots."""

    from build_mizar_human_shard import (
        ROW_SCHEMA,
        _read_miz,
        parse_miz_article,
        render_training_text,
    )
    from build_p3_generation import IntegrationError, _validate_source_manifest
    from mizar_current_index import MizarIndex, MizarIndexError

    errors = []

    def bad(kind, detail=""):
        errors.append((kind, detail))

    if record.get("schema_version") != ROW_SCHEMA:
        return [("mizar_schema_mismatch", str(record.get("schema_version")))]
    try:
        manifest = _validate_source_manifest(
            source_manifest,
            family="mizar",
            production=False,
        )
    except IntegrationError as error:
        return [("mizar_source_manifest_mismatch", str(error))]
    if record.get("source_metadata") != manifest["row_source_metadata"]:
        bad("mizar_source_metadata_mismatch")

    index_path = os.fspath(semantic_index)
    actual_index_sha256 = hashlib.sha256(Path(index_path).read_bytes()).hexdigest()
    expected_index_sha256 = manifest["row_source_metadata"]["index_roots"].get(
        "semantic_index_sha256"
    )
    if actual_index_sha256 != expected_index_sha256:
        return [("mizar_index_root_mismatch", actual_index_sha256)]

    source = record.get("source")
    if not isinstance(source, dict):
        return [("mizar_source_provenance_missing", "")]
    source_file = source.get("file")
    if (
        not isinstance(source_file, str)
        or not source_file
        or os.path.basename(source_file) != source_file
    ):
        return [("mizar_source_path_invalid", str(source_file))]
    path = os.path.join(os.fspath(mml_root), source_file)
    try:
        source_text, encoding = _read_miz(Path(path))
        declarations = parse_miz_article(
            source_text,
            article=str(source.get("article", "")),
            source_file=source_file,
        )
        ordinal = int(source["declaration_ordinal"])
        declaration = declarations[ordinal - 1]
    except (OSError, KeyError, TypeError, ValueError, IndexError) as error:
        return [("mizar_source_declaration_missing", str(error))]
    target = record.get("target")
    target_sha256 = hashlib.sha256(str(target).encode("utf-8")).hexdigest()
    if (
        declaration.category != "complete_explicit_proof"
        or declaration.target != target
        or source_text[
            int(source.get("target_start", -1)) : int(source.get("target_end", -1))
        ]
        != target
        or source.get("target_sha256") != target_sha256
        or source.get("file_sha256")
        != hashlib.sha256(Path(path).read_bytes()).hexdigest()
        or source.get("encoding") != encoding
    ):
        bad("mizar_source_target_mismatch")
    if record.get("local_assumptions", {}) != declaration.local_assumptions:
        bad("mizar_local_assumptions_mismatch")

    expected_id = hashlib.sha256(
        "\0".join(
            (
                ROW_SCHEMA,
                str(record.get("theorem", "")),
                source_file,
                str(ordinal),
                target_sha256,
            )
        ).encode("utf-8")
    ).hexdigest()
    if record.get("id") != expected_id:
        bad("mizar_row_id_mismatch")

    try:
        with MizarIndex(index_path) as index:
            statements = index.statement_map()
            if statements.get(record.get("theorem")) != record.get("goal"):
                bad("mizar_goal_index_mismatch")
            facts = record.get("facts", {})
            if not isinstance(facts, dict) or any(
                statements.get(name) != statement for name, statement in facts.items()
            ):
                bad("mizar_fact_index_mismatch")
            references, unresolved = resolve_current_index_references(
                "mizar",
                str(target),
                index,
                statements,
                theorem=str(record.get("theorem", "")),
            )
            if unresolved or references != record.get("cited"):
                bad("mizar_reference_mismatch")
            if "index" in record:
                errors.extend(
                    _direct_mizar_source_binding_errors(
                        record,
                        declaration=declaration,
                        declarations=declarations,
                        index=index,
                    )
                )
    except (
        MizarIndexError,
        OSError,
        KeyError,
        TypeError,
        ValueError,
        sqlite3.Error,
    ) as error:
        bad("mizar_index_replay_failed", str(error))

    try:
        rendered, mask_start, mask_end = render_training_text(
            record["facts"],
            record["goal"],
            record["target"],
        )
        if (
            record.get("text") != rendered
            or record.get("mask_start") != mask_start
            or record.get("mask_end") != mask_end
        ):
            bad("mizar_rendering_mismatch")
    except (KeyError, TypeError, ValueError) as error:
        bad("mizar_rendering_mismatch", str(error))
    return errors


def legacy_audit(argv=None):
    """Run the read-only legacy v2 diagnostic scanner."""

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="corpus")
    ap.add_argument("--max-report", type=int, default=3)
    ap.add_argument(
        "--mizar-html2",
        help="legacy debug-only html2 verification map",
    )
    ap.add_argument(
        "--mizar-semantic-index",
        help="authoritative mizar-semantic-index-v1 for production verification",
    )
    a = ap.parse_args(argv)
    err = Counter()
    samples = defaultdict(list)
    seen_ids = {}
    totals = Counter()

    mizar_statements = {}
    mizar_local = {}
    mizar_index = None
    expected_mizar_source = None
    if a.mizar_semantic_index:
        from mizar_current_index import INDEX_SCHEMA, MizarIndex

        mizar_index = MizarIndex(a.mizar_semantic_index)
        mizar_statements = mizar_index.statement_map()
        metadata = mizar_index.metadata()
        digest = hashlib.sha256()
        with open(a.mizar_semantic_index, "rb") as index_file:
            while chunk := index_file.read(1024 * 1024):
                digest.update(chunk)
        expected_mizar_source = {
            "semantic_index_schema": INDEX_SCHEMA,
            "semantic_index_sha256": digest.hexdigest(),
            "source_manifest_sha256": metadata["source_manifest_sha256"],
            "release": metadata["release"],
            "source_trees": metadata["source_trees"],
        }
    elif a.mizar_html2:
        from build_mizar_shard import parse_article

        sources = sorted(glob.glob(os.path.join(a.mizar_html2, "*")))
        if not sources:
            raise SystemExit(f"no Mizar html2 articles found in {a.mizar_html2}")
        for source in sources:
            if not os.path.isfile(source):
                continue
            statements, local = parse_article(source)
            mizar_statements.update(statements)
            article = os.path.basename(source).split(".", 1)[0].upper()
            mizar_local[article] = local

    def bad(kind, where):
        err[kind] += 1
        if len(samples[kind]) < a.max_report:
            samples[kind].append(where)

    held_all = set()
    held_of = defaultdict(set)
    held_statement_hashes_of = defaultdict(set)
    held_of_family = defaultdict(set)
    held_statement_hashes_of_family = defaultdict(set)
    family_of = dict(SHARD_FAMILY)
    for h in sorted(glob.glob(os.path.join(a.corpus, "heldout", "*.json"))):
        d = _load_json(h)
        stem = os.path.basename(h)[:-5]
        family = d.get("family", SHARD_FAMILY.get(stem, stem))
        shards = d.get("shards", [stem])
        held = set(d["facts"])
        held_all |= held
        held_of_family[family] |= held
        expected_canonicalization = canonicalization_metadata(family)
        manifest_canonicalization = d.get("canonicalization")
        if (
            manifest_canonicalization is not None
            and manifest_canonicalization != expected_canonicalization
        ):
            bad(
                "manifest_canonicalization_mismatch",
                f"heldout/{stem} {manifest_canonicalization}",
            )
        for shard in shards:
            family_of[shard] = family
            held_of[shard] |= held
            if manifest_canonicalization == expected_canonicalization:
                held_statement_hashes_of_family[family] |= set(
                    d.get("statement_hashes", [])
                )

    # Backfill statement identities for legacy manifests so this verifier makes
    # the pre-v2 alias leak visible instead of trusting a missing hash field.
    # Pool across every raw sibling before distributing the result back to all
    # shards, since one ATP renderer can contain the held definition while its
    # sibling contains only an exact statement alias.
    for p in sorted(glob.glob(os.path.join(a.corpus, "raw", "*.jsonl"))):
        shard = os.path.basename(p)[:-6]
        family = family_of.get(shard, shard)
        for line in _read_lines(p):
            record = _parse_json_line(line)
            if record is None:
                continue
            for name, statement in record.get("facts", {}).items():
                if name in held_of_family[family]:
                    held_statement_hashes_of_family[family].add(
                        statement_hash(statement, family=family)
                    )
    for shard, family in family_of.items():
        held_statement_hashes_of[shard] |= held_statement_hashes_of_family[family]

    seen_atp_signatures = {}
    for kind in ("shards", "eval"):
        for p in sorted(glob.glob(os.path.join(a.corpus, kind, "*.jsonl"))):
            sh = os.path.basename(p)[:-6]
            n = 0
            for ln, line in enumerate(_read_lines(p), 1):
                where = f"{kind}/{sh}:{ln}"
                r = _parse_json_line(line)
                if r is None:
                    bad("unparseable_json", where)
                    continue
                n += 1

                missing = [k for k in REQUIRED if k not in r]
                if sh == "metamath" and "local_assumptions" not in r:
                    missing.append("local_assumptions")
                if missing:
                    bad("missing_field", f"{where} {missing}")
                    continue
                if not isinstance(r["facts"], dict) or not r["facts"]:
                    bad("empty_or_bad_facts", where)
                    continue
                if not r["goal"].strip() or not r["target"].strip():
                    bad("empty_goal_or_target", where)

                block = (
                    HDR + "\n" + "\n".join(f"{k} : {v}" for k, v in r["facts"].items())
                )
                local_inputs = r.get("local_inputs", {})
                if sh in ATP_SHARDS:
                    atp_missing = [key for key in ATP_REQUIRED if key not in r]
                    if atp_missing:
                        bad("missing_atp_field", f"{where} {atp_missing}")
                    if not isinstance(local_inputs, dict):
                        bad("bad_local_inputs", where)
                        local_inputs = {}
                    if local_inputs:
                        block += (
                            "\n"
                            + ATP_LOCAL_HDR
                            + "\n"
                            + "\n".join(f"{k} : {v}" for k, v in local_inputs.items())
                        )
                local_assumptions = r.get("local_assumptions", {})
                if not isinstance(local_assumptions, dict):
                    bad("bad_local_assumptions", where)
                    local_assumptions = {}
                if sh == "metamath":
                    block += "\n" + LOCAL_HDR
                    if local_assumptions:
                        block += "\n" + "\n".join(
                            f"{k} : {v}" for k, v in local_assumptions.items()
                        )
                if r["text"][: r["mask_end"]] != block:
                    bad("mask_span_wrong", where)
                if r["mask_start"] != 0:
                    bad("mask_start_nonzero", where)
                if r["text"] != f"{block}\n{SEP}\nGOAL {r['goal']}\n{r['target']}":
                    bad("text_not_reconstructible", where)

                for c in r["cited"]:
                    if c not in r["facts"]:
                        bad("cited_without_statement", f"{where} {c}")
                        break
                if sh in {"mizar", "thproofs"}:
                    from build_mizar_shard import (
                        is_canceled,
                        resolve_references,
                        scan_references,
                        statements_match,
                    )

                    for name, statement in r["facts"].items():
                        if is_canceled(statement):
                            bad("canceled_mizar_fact", f"{where} {name}")

                    # Qualified names are self-describing and can be checked
                    # without source files. Th/Def/Lm labels need the exact
                    # article-local map and are checked below when supplied.
                    qualified, _ = scan_references(r["target"], {})
                    missing_qualified = [
                        name for name in qualified if name not in r["facts"]
                    ]
                    if missing_qualified:
                        bad(
                            "target_reference_without_fact",
                            f"{where} {missing_qualified[:3]}",
                        )

                    if mizar_index is not None:
                        from build_thproofs_shard import BUILD_SOURCE_SCHEMA

                        source_metadata = r.get("source_metadata")
                        if not isinstance(source_metadata, dict):
                            bad("mizar_source_metadata_missing", where)
                        else:
                            if (
                                source_metadata.get("schema_version")
                                != BUILD_SOURCE_SCHEMA
                            ):
                                bad("mizar_source_metadata_schema", where)
                            for key, expected in expected_mizar_source.items():
                                if source_metadata.get(key) != expected:
                                    bad(
                                        "mizar_source_metadata_mismatch",
                                        f"{where} {key}",
                                    )
                            if not isinstance(
                                source_metadata.get("source_roots"), dict
                            ):
                                bad("mizar_source_roots_missing", where)
                            if not isinstance(
                                source_metadata.get("source_archives"), dict
                            ):
                                bad("mizar_source_archives_missing", where)

                        refs, unresolved = resolve_current_index_references(
                            sh,
                            r["target"],
                            mizar_index,
                            mizar_statements,
                            theorem=r["theorem"],
                        )
                        if unresolved:
                            bad(
                                "target_reference_unresolved",
                                f"{where} {unresolved[:3]}",
                            )
                        if set(refs) != set(r["cited"]):
                            bad(
                                "target_citations_disagree",
                                f"{where} target={sorted(set(refs))[:3]} "
                                f"record={sorted(set(r['cited']))[:3]}",
                            )
                        for fact_name, fact_statement in r["facts"].items():
                            source_fact = mizar_statements.get(fact_name)
                            if source_fact is None:
                                bad(
                                    "mizar_fact_source_missing",
                                    f"{where} {fact_name}",
                                )
                            elif fact_statement != source_fact:
                                bad(
                                    "mizar_fact_source_mismatch",
                                    f"{where} {fact_name}",
                                )
                        source_statement = mizar_statements.get(r["theorem"])
                        if source_statement is None:
                            bad(
                                "mizar_gold_source_missing",
                                f"{where} {r['theorem']}",
                            )
                        elif r["goal"] != source_statement:
                            bad(
                                "mizar_gold_source_mismatch",
                                f"{where} {r['theorem']}",
                            )
                    elif a.mizar_html2:
                        article = r["theorem"].split(":", 1)[0]
                        local = mizar_local.get(article)
                        if local is None:
                            bad("mizar_source_article_missing", f"{where} {article}")
                        else:
                            refs, unresolved = resolve_references(
                                r["target"],
                                local,
                                mizar_statements,
                                own_name=r["theorem"],
                                local_by_article=mizar_local,
                            )
                            if unresolved:
                                bad(
                                    "target_reference_unresolved",
                                    f"{where} {unresolved[:3]}",
                                )
                            if set(refs) != set(r["cited"]):
                                bad(
                                    "target_citations_disagree",
                                    f"{where} target={sorted(set(refs))[:3]} "
                                    f"record={sorted(set(r['cited']))[:3]}",
                                )

                        for fact_name, fact_statement in r["facts"].items():
                            source_fact = mizar_statements.get(fact_name)
                            if source_fact is None:
                                bad(
                                    "mizar_fact_source_missing",
                                    f"{where} {fact_name}",
                                )
                            elif fact_statement != source_fact:
                                bad(
                                    "mizar_fact_source_mismatch",
                                    f"{where} {fact_name}",
                                )

                        source_statement = mizar_statements.get(r["theorem"])
                        if source_statement is None:
                            bad(
                                "mizar_gold_source_missing",
                                f"{where} {r['theorem']}",
                            )
                        elif not statements_match(r["goal"], source_statement):
                            bad(
                                "mizar_gold_source_mismatch",
                                f"{where} {r['theorem']}",
                            )

                if sh == "metamath":
                    overlap = set(local_assumptions) & (
                        set(r["facts"]) | set(r["cited"])
                    )
                    if overlap:
                        bad(
                            "local_assumption_is_global_fact",
                            f"{where} {sorted(overlap)[:2]}",
                        )
                    for target_line in r["target"].splitlines():
                        match = STEP_RE.match(target_line)
                        if match is None:
                            bad("malformed_metamath_target", where)
                            break
                        label = match.group(1)
                        if label == "(reuse)":
                            bad("reuse_in_target", where)
                            break
                        if label in local_assumptions:
                            bad("local_assumption_in_target", f"{where} {label}")
                            break
                        if label not in r["facts"]:
                            bad("target_label_without_fact", f"{where} {label}")
                            break
                elif sh in ATP_SHARDS:
                    overlap = set(local_inputs) & (set(r["facts"]) | set(r["cited"]))
                    if overlap:
                        bad(
                            "atp_local_input_is_global_fact",
                            f"{where} {sorted(overlap)[:2]}",
                        )
                    for error, detail in atp_record_errors(r):
                        bad(error, f"{where} {detail}".rstrip())

                key = r["id"]
                if key in seen_ids and seen_ids[key] != (kind, sh):
                    bad("duplicate_id_across_shards", f"{where} {key}")
                seen_ids.setdefault(key, (kind, sh))

                family = family_of.get(sh, SHARD_FAMILY.get(sh, sh))
                if family == "atp":
                    signature = exact_atp_signature(r)
                    previous = seen_atp_signatures.get(signature)
                    if previous is not None:
                        bad(
                            "duplicate_atp_family_proof",
                            f"{where} duplicates {previous}",
                        )
                    else:
                        seen_atp_signatures[signature] = where
                shard_held = held_of[sh]
                exposure = heldout_exposure(
                    r,
                    shard_held,
                    held_statement_hashes_of[sh],
                    family=family,
                )
                if kind == "shards" and exposure.named_fact:
                    touch = (set(r["facts"]) | set(local_inputs)) & shard_held
                    bad("train_touches_heldout", f"{where} {sorted(touch)[:2]}")
                if kind == "shards" and exposure.own_theorem:
                    # Own-proof leakage is supervision even if the fact name is
                    # absent from GOAL: the goal exposes the statement and the
                    # target teaches a complete proof of the held theorem.
                    bad(
                        "train_proves_heldout_theorem",
                        f"{where} "
                        f"{normalize_theorem_identity(r['theorem'], family=family)}",
                    )
                if kind == "shards" and exposure.statement_alias:
                    bad("train_exposes_heldout_statement", where)
                if kind == "eval" and not exposure.should_eval:
                    bad("eval_without_heldout", where)

                t = r["text"]
                if "\ufffd" in t:
                    bad("replacement_char", where)
                if any(ch < " " and ch not in "\n\t" for ch in t):
                    bad("control_char", where)
                if any(unicodedata.category(ch) == "Cs" for ch in t[:2000]):
                    bad("surrogate", where)

            totals[f"{kind}/{sh}"] = n

    print("counts")
    for k in sorted(totals):
        print(f"  {k:<22}{totals[k]:>9,}")

    print("\naccounting: raw == shards + eval + splitter drops")
    ok_acct = True
    for p in sorted(glob.glob(os.path.join(a.corpus, "raw", "*.jsonl"))):
        sh = os.path.basename(p)[:-6]
        raw = sum(1 for _ in _read_lines(p))
        s = totals.get(f"shards/{sh}", 0)
        e = totals.get(f"eval/{sh}", 0)
        d = raw - s - e
        note = "" if d == 0 else f"   {d:,} dropped by the splitter"
        if d < 0:
            ok_acct = False
            note = f"   NEGATIVE ({d})"
        print(f"  {sh:<12} raw {raw:>8,} = train {s:>8,} + eval {e:>6,}{note}")
    ids_dup = sum(1 for k in ()) if False else err["duplicate_id_across_shards"]

    print(
        f"\nunique ids: {len(seen_ids):,}   " f"duplicates across shards: {ids_dup:,}"
    )
    held_sets = {tuple(sorted(values)) for values in held_of.values()}
    print(f"held-out facts: {len(held_all):,} across {len(held_sets)} manifests")

    print("\nfindings")
    if not err:
        print("  none — every example passed every check")
    for k, v in err.most_common():
        print(f"  {k:<28}{v:>8,}   e.g. {samples[k][:2]}")
    print("\nVERIFY CLEAN" if not err and ok_acct else "\nVERIFY FOUND PROBLEMS")
    if mizar_index is not None:
        mizar_index.close()
    return 0 if (not err and ok_acct) else 1


def main(argv=None):
    """Verify only the immutable six-family transaction selected by CURRENT."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="corpus")
    parser.add_argument("--max-report", type=int, default=3)
    parser.add_argument(
        "--mizar-semantic-index",
        help="authoritative current mizar-semantic-index-v1",
    )
    parser.add_argument(
        "--legacy-audit",
        action="store_true",
        help="read-only v2 diagnosis; never returns production-clean status",
    )
    parser.add_argument(
        "--mizar-html2",
        help="legacy html2 map; valid only together with --legacy-audit",
    )
    args = parser.parse_args(argv)

    if args.legacy_audit:
        legacy_args = [
            "--corpus",
            args.corpus,
            "--max-report",
            str(args.max_report),
        ]
        if args.mizar_html2:
            legacy_args.extend(["--mizar-html2", args.mizar_html2])
        if args.mizar_semantic_index:
            legacy_args.extend(["--mizar-semantic-index", args.mizar_semantic_index])
        status = legacy_audit(legacy_args)
        if status == 0:
            print("\nLEGACY AUDIT CLEAN — NONPRODUCTION ONLY")
            return 2
        print("\nLEGACY AUDIT FOUND PROBLEMS — NONPRODUCTION ONLY")
        return 1

    if args.mizar_html2:
        parser.error("--mizar-html2 requires explicit --legacy-audit")

    corpus = os.path.abspath(args.corpus)
    current = os.path.join(corpus, "CURRENT")
    legacy_directories = [
        name
        for name in ("raw", "shards", "eval", "heldout", "sidecars", "artifacts")
        if os.path.lexists(os.path.join(corpus, name))
    ]
    if not os.path.isfile(current):
        detail = (
            f"; legacy directories present: {', '.join(legacy_directories)}"
            if legacy_directories
            else ""
        )
        print(
            f"PRODUCTION VERIFY REFUSED: transaction CURRENT is missing{detail}",
            file=sys.stderr,
        )
        return 1
    if legacy_directories:
        print(
            "PRODUCTION VERIFY REFUSED: legacy directories are forbidden at "
            f"the transaction root: {', '.join(legacy_directories)}",
            file=sys.stderr,
        )
        return 1

    try:
        try:
            from scripts import build_p3_generation
        except ImportError:  # pragma: no cover - direct production CLI.
            import build_p3_generation

        report = build_p3_generation.verify_generation(
            args.corpus,
            production=True,
            mizar_semantic_index=args.mizar_semantic_index,
        )
    except Exception as error:  # noqa: BLE001 - CLI must fail closed on any verifier.
        print(f"PRODUCTION VERIFY REFUSED: {error}", file=sys.stderr)
        return 1
    if report.get("status") != "clean":
        print("PRODUCTION VERIFY FOUND PROBLEMS", file=sys.stderr)
        return 1
    print(
        "PRODUCTION VERIFY CLEAN "
        f"generation={report['generation_id']} "
        f"root={report['logical_root_sha256']} "
        f"families={','.join(report['families'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
