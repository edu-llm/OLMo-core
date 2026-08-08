"""Machine B, second half — the Mizar `thproofs` shard.

`html2` gives us statements for the whole MML but proofs for only part of it;
`thproofs` gives one file per theorem for 76,696 theorems, which is 4.7x what the
html2 shard kept. The two are joined on the fact dictionary: thproofs supplies
goals and proofs, html2 supplies the statements that go in the block.

These files carry no header, so the theorem's own name comes from the filename
(`t36_partpr_1` -> `PARTPR_1:36`). Article-local `Th`, `Lm`, and `Def`
references are resolved through the matching html2 article's actual label map;
their numbers are not assumed to equal global theorem numbers.

Theorems already covered by the html2 shard are skipped, so the two shards do not
restate the same proof.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

from build_mizar_shard import (
    HDR,
    SEP,
    resolve_references,
    shuffled,
    split_outer_proof,
    statements_match,
)
from mizar_current_index import (
    INDEX_SCHEMA,
    SOURCE_MANIFEST_SCHEMA,
    MizarIndex,
    MizarIndexError,
)
from mizar_current_index import (
    verify_source_manifest as verify_current_source_manifest,
)

FNAME = re.compile(r"^t(\d+)_(.+)$")
THEOREM_START = re.compile(r"(?m)^theorem\b")
EXPLICIT_IDENTITY = re.compile(r"::\s*([A-Z][A-Z_0-9]*:\d+)\b")
BUILD_SOURCE_SCHEMA = "mizar-thproof-build-source-v1"
ROW_SCHEMA = "mizar-proof-v2"
LEGACY_SOURCE_MANIFEST_SCHEMA = "mizar-thproof-sources-v1"
# Verified current MML 8.1.15 measurements: 76,696 files and joins, 69,698
# explicit-proof-bearing extracts, and 58,658 complete proofs (84.160%).
HARD_MIN_SOURCE_FILES = 75_000
HARD_MIN_NAME_MATCHES = 75_000
HARD_MIN_NAME_JOIN_RATE = 0.99
HARD_MIN_EXPLICIT_PROOFS = 65_000
HARD_MIN_COMPLETE_PROOFS = 55_000
HARD_MIN_COMPLETION_RATE = 0.80
# Isolated current-index candidate: 50,752/58,658 eligible complete (86.522%).
# These floors retain conservative slack while rejecting degraded builds.
HARD_MIN_ACCEPTED_PROOFS = 45_000
HARD_MIN_ACCEPTED_RATE = 0.80


def split_proof(chunk):
    """The derivation part of a theorem, or None if there is no derivation.

    A theorem discharged by a bare citation — `vars Non a = vars a by Th99;` —
    has no proof to learn. Its target would be the name of the one fact already
    in the block, which is the degenerate case the corpus exists to avoid.
    """
    _, body = split_outer_proof(chunk)
    return body


def _strip_goal_export_syntax(goal):
    """Strip only export labels, source comments, and layout."""
    goal = re.sub(r"^\s*[A-Za-z]\w*\s*:\s*", "", goal)
    goal = re.sub(r"::.*$", "", goal, flags=re.MULTILINE)
    return " ".join(goal.strip().rstrip(";").split())


def goal_diagnostic(export_goal, source_goal):
    """Conservatively classify raw/html2 rendering agreement.

    This is diagnostic only. A verified source manifest and theorem filename
    establish identity; html2 remains the canonical emitted statement.
    """
    export = _strip_goal_export_syntax(export_goal)
    source = _strip_goal_export_syntax(source_goal)
    if statements_match(export, source):
        return "match"

    def official_rendering_shape(statement):
        compact = re.sub(r"\s+", "", statement)
        return re.sub(
            r"\(union([A-Za-z]\w*)\)",
            r"union\1",
            compact,
        )

    if official_rendering_shape(export) == official_rendering_shape(source):
        return "rendering-difference"
    return "different"


def _tree_digest(root):
    digest = hashlib.sha256()
    files = sorted(
        path for path in glob.glob(os.path.join(root, "*")) if os.path.isfile(path)
    )
    for path in files:
        digest.update(os.path.basename(path).encode())
        digest.update(b"\0")
        with open(path, "rb") as source_file:
            digest.update(hashlib.sha256(source_file.read()).digest())
    return len(files), digest.hexdigest()


def verify_legacy_source_manifest(path, html2, thproofs):
    """Verify an exact html2/thproof source pair and return its coverage gates."""
    if not path or not os.path.isfile(path):
        raise ValueError("an exact --source-manifest is required")
    with open(path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("schema_version") != LEGACY_SOURCE_MANIFEST_SCHEMA:
        raise ValueError(
            f"source manifest must use {LEGACY_SOURCE_MANIFEST_SCHEMA}"
        )
    if not manifest.get("mml_version"):
        raise ValueError("source manifest must pin mml_version")

    for key, root in (("html2", html2), ("thproofs", thproofs)):
        pinned = manifest.get(key)
        if not isinstance(pinned, dict) or not pinned.get("version"):
            raise ValueError(f"source manifest must pin {key} version")
        count, digest = _tree_digest(root)
        if (
            pinned.get("file_count") != count
            or pinned.get("tree_sha256") != digest
        ):
            raise ValueError(
                f"{key} source does not match manifest "
                f"(files {count}, sha256 {digest})"
            )

    coverage = manifest.get("coverage")
    required = {
        "minimum_source_files": int,
        "minimum_name_matches": int,
        "minimum_name_join_rate": (int, float),
        "minimum_complete_proofs": int,
        "minimum_completion_rate": (int, float),
        "minimum_accepted_proofs": int,
        "minimum_accepted_rate": (int, float),
    }
    if not isinstance(coverage, dict):
        raise TypeError("source manifest must provide evidence-based coverage gates")
    for key, expected_type in required.items():
        value = coverage.get(key)
        if not isinstance(value, expected_type) or isinstance(value, bool):
            raise TypeError(f"source manifest coverage must pin {key}")
    for key in (
        "minimum_name_join_rate",
        "minimum_completion_rate",
        "minimum_accepted_rate",
    ):
        if not 0 <= coverage[key] <= 1:
            raise ValueError(f"source manifest {key} must be between 0 and 1")
    return manifest


def effective_coverage_floors(manifest_coverage):
    """Combine immutable code floors with stricter manifest requirements."""
    return {
        "minimum_source_files": max(
            HARD_MIN_SOURCE_FILES,
            manifest_coverage.get("minimum_source_files", 0),
        ),
        "minimum_name_matches": max(
            HARD_MIN_NAME_MATCHES,
            manifest_coverage.get("minimum_name_matches", 0),
        ),
        "minimum_name_join_rate": max(
            HARD_MIN_NAME_JOIN_RATE,
            manifest_coverage.get("minimum_name_join_rate", 0.0),
        ),
        "minimum_explicit_proofs": max(
            HARD_MIN_EXPLICIT_PROOFS,
            manifest_coverage.get("minimum_explicit_proofs", 0),
        ),
        "minimum_complete_proofs": max(
            HARD_MIN_COMPLETE_PROOFS,
            manifest_coverage.get("minimum_complete_proofs", 0),
        ),
        "minimum_completion_rate": max(
            HARD_MIN_COMPLETION_RATE,
            manifest_coverage.get("minimum_completion_rate", 0.0),
        ),
        "minimum_accepted_proofs": max(
            HARD_MIN_ACCEPTED_PROOFS,
            manifest_coverage.get("minimum_accepted_proofs", 0),
        ),
        "minimum_accepted_rate": max(
            HARD_MIN_ACCEPTED_RATE,
            manifest_coverage.get("minimum_accepted_rate", 0.0),
        ),
    }


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source_file:
        while chunk := source_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _verified_source_metadata(index, index_path, manifest_path, manifest, roots):
    metadata = index.metadata()
    manifest_sha256 = _sha256_file(manifest_path)
    if metadata.get("schema_version") != INDEX_SCHEMA:
        raise ValueError(f"semantic index must use {INDEX_SCHEMA}")
    if metadata.get("source_manifest_sha256") != manifest_sha256:
        raise ValueError("semantic index was built from a different source manifest")
    expected_trees = {
        name: {
            "file_count": spec["file_count"],
            "tree_sha256": spec["tree_sha256"],
        }
        for name, spec in sorted(manifest["sources"].items())
    }
    if metadata.get("source_trees") != expected_trees:
        raise ValueError("semantic index source-tree metadata disagrees with manifest")
    if metadata.get("release") != manifest["release"]:
        raise ValueError("semantic index release metadata disagrees with manifest")
    quality_policy = {
        "coverage": effective_coverage_floors(
            manifest.get("builder_coverage", {})
        ),
        "requires_complete_references": True,
        "requires_current_index_association": True,
        "requires_source_goal": True,
    }
    return {
        "schema_version": BUILD_SOURCE_SCHEMA,
        "semantic_index_schema": INDEX_SCHEMA,
        "semantic_index_sha256": _sha256_file(index_path),
        "index_roots": {
            "semantic_index_schema": INDEX_SCHEMA,
            "semantic_index_sha256": _sha256_file(index_path),
        },
        "source_manifest_schema": SOURCE_MANIFEST_SCHEMA,
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_root_sha256": manifest_sha256,
        "release": manifest["release"],
        "source_roots": {
            name: str(Path(root).resolve()) for name, root in sorted(roots.items())
        },
        "source_trees": expected_trees,
        "quality_filter_root_sha256": _canonical_sha256(quality_policy),
        "schema_generation_root_sha256": _canonical_sha256(
            {
                "row_schema": ROW_SCHEMA,
                "source_schema": BUILD_SOURCE_SCHEMA,
            }
        ),
        "source_archives": {
            name: {
                "archive_bytes": spec.get("archive_bytes"),
                "archive_sha256": spec["archive_sha256"],
                "archive_url": spec["archive_url"],
            }
            for name, spec in sorted(manifest["sources"].items())
        },
        "licensing": manifest["licensing"],
    }


def _index_thproof(index, identity):
    row = index.connection.execute(
        """
        SELECT file_name, category, source_goal, explicit_identity, proof_sha256
        FROM thproofs
        WHERE identity = ?
        """,
        (identity,),
    ).fetchone()
    if row is None:
        raise KeyError(identity)
    return {
        "file_name": row[0],
        "category": row[1],
        "source_goal": row[2],
        "explicit_identity": row[3],
        "proof_sha256": row[4],
    }


def resolve_index_references(body, index, statements, *, theorem):
    """Resolve proof citations with theorem-contextual local-label semantics."""
    article = theorem.split(":", 1)[0]
    cached = getattr(index, "_thproof_builder_label_cache", None)
    if cached is None:
        all_labels = index.article_local_label_maps()
        unique_by_article = {
            other_article: {
                label: identities[0]
                for label, identities in labels.items()
                if len(identities) == 1
            }
            for other_article, labels in all_labels.items()
        }
        cached = (all_labels, unique_by_article)
        index._thproof_builder_label_cache = cached
    all_labels, unique_by_article = cached
    local = {}
    for label in all_labels.get(article, {}):
        try:
            local[label] = index.resolve_local_label(
                article,
                label,
                at_identity=theorem,
            )
        except KeyError:
            continue
    return resolve_references(
        body,
        local,
        statements,
        own_name=theorem,
        local_by_article=unique_by_article,
    )


def _output_paths(out, name):
    return (
        os.path.join(out, "shards", f"{name}.jsonl"),
        os.path.join(out, "eval", f"{name}.jsonl"),
        os.path.join(out, "heldout", f"{name}.json"),
    )


def _invalidate_outputs(paths):
    """Quarantine prior outputs so a failed rebuild cannot appear fresh."""
    for path in paths:
        if not os.path.exists(path):
            continue
        stale = path + ".stale"
        suffix = 1
        while os.path.exists(stale):
            stale = f"{path}.stale.{suffix}"
            suffix += 1
        os.replace(path, stale)


def _extract_chunk(text):
    starts = list(THEOREM_START.finditer(text))
    if not starts:
        return None
    return text[starts[-1].start():]


def _raw_goal_and_identity(chunk):
    proof_start, _ = split_outer_proof(chunk)
    if proof_start is None:
        return "", None
    declaration = chunk[len("theorem"):proof_start]
    identity_match = EXPLICIT_IDENTITY.search(declaration)
    identity = identity_match.group(1) if identity_match else None
    if identity_match:
        declaration = (
            declaration[:identity_match.start()]
            + declaration[identity_match.end():]
        )
    return _strip_goal_export_syntax(declaration), identity


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/dscount/thproofs/thproofs")
    ap.add_argument("--semantic-index")
    ap.add_argument(
        "--source-manifest",
        help="verified mizar-current-sources-v1 manifest",
    )
    ap.add_argument("--mml-root")
    ap.add_argument("--html-root")
    ap.add_argument("--mizar-archive")
    ap.add_argument("--html-archive")
    ap.add_argument("--thproofs-archive")
    ap.add_argument(
        "--html2",
        help="legacy debug input; cannot authorize production output",
    )
    ap.add_argument("--exclude", default="corpus/shards/mizar.jsonl",
                    help="skip theorems the html2 shard already covers")
    ap.add_argument("--out", default="corpus")
    ap.add_argument("--name", default="thproofs")
    ap.add_argument("--heldout", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260801)
    a = ap.parse_args()
    output_paths = _output_paths(a.out, a.name)
    # Invalidate first. Any later read, parse, source, join, or write failure
    # must be unable to leave an old corpus looking current.
    _invalidate_outputs(output_paths)
    required = {
        "--semantic-index": a.semantic_index,
        "--source-manifest": a.source_manifest,
        "--mml-root": a.mml_root,
        "--html-root": a.html_root,
        "--src": a.src,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        print(
            "production thproof build requires " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    roots = {
        "mml": a.mml_root,
        "html": a.html_root,
        "thproofs": a.src,
    }
    archives = {
        name: path
        for name, path in {
            "mml": a.mizar_archive,
            "html": a.html_archive,
            "thproofs": a.thproofs_archive,
        }.items()
        if path
    }
    index = None
    try:
        source_manifest = verify_current_source_manifest(
            a.source_manifest,
            roots,
            archive_paths=archives,
        )
        index = MizarIndex(a.semantic_index)
        source_metadata = _verified_source_metadata(
            index,
            a.semantic_index,
            a.source_manifest,
            source_manifest,
            roots,
        )
    except (
        MizarIndexError,
        OSError,
        TypeError,
        ValueError,
        sqlite3.Error,
        json.JSONDecodeError,
    ) as error:
        if index is not None:
            index.close()
        print(f"source compatibility failure: {error}", file=sys.stderr)
        return 2
    coverage = effective_coverage_floors(
        source_manifest.get("builder_coverage", {})
    )
    print(
        f"  verified Mizar {source_manifest['release']['mizar_version']} / "
        f"MML {source_manifest['release']['mml_version']} semantic index "
        f"{source_metadata['semantic_index_sha256']}"
    )
    stmt = index.statement_map()
    index_content = index.metadata()["content"]
    print(f"  {len(stmt):,} canonical semantic statements")

    done = set()
    if a.exclude and os.path.exists(a.exclude):
        try:
            with open(a.exclude, encoding="utf-8") as exclude_file:
                for line_number, line in enumerate(exclude_file, 1):
                    record = json.loads(line)
                    theorem = record["theorem"]
                    if not isinstance(theorem, str):
                        raise TypeError(
                            f"exclude theorem on line {line_number} is not text"
                        )
                    done.add(theorem)
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            index.close()
            print(f"invalid --exclude file {a.exclude}: {error}", file=sys.stderr)
            return 2
        print(f"  {len(done):,} theorems already in the html2 shard")

    files = sorted(
        path
        for path in glob.glob(os.path.join(a.src, "*"))
        if os.path.isfile(path)
    )
    if not files:
        index.close()
        print(f"no thproofs files found in {a.src}", file=sys.stderr)
        _invalidate_outputs(output_paths)
        return 2
    print(f"  {len(files):,} thproofs files")

    parsed = []
    counts = {}
    n_skip = n_noproof = n_nogoal = n_nocite = 0
    n_no_source = n_provenance = n_incomplete = n_association = 0
    n_goal_match = n_goal_rendering = n_goal_different = 0
    named_files = joined_names = eligible_complete = accepted_candidates = 0
    proof_categories = Counter()
    for p in files:
        file_name = os.path.basename(p)
        if FNAME.match(file_name) is None:
            continue
        named_files += 1
        try:
            name = index.theorem_identity(file_name)
            indexed_proof = _index_thproof(index, name)
        except KeyError:
            n_no_source += 1
            continue
        if indexed_proof["file_name"] != file_name:
            n_association += 1
            continue
        proof_categories[indexed_proof["category"]] += 1
        source_statement = stmt.get(name)
        if source_statement is None:
            n_no_source += 1
            continue
        joined_names += 1
        if indexed_proof["category"] != "complete_explicit_proof":
            n_noproof += 1
            continue
        with open(p, encoding="utf-8", errors="replace") as proof_file:
            txt = proof_file.read()
        chunk = _extract_chunk(txt)
        if chunk is None:
            n_nogoal += 1
            continue
        body = split_proof(chunk)
        if body is None:
            n_noproof += 1
            continue
        if hashlib.sha256(body.encode("utf-8")).hexdigest() != indexed_proof[
            "proof_sha256"
        ]:
            n_association += 1
            continue
        eligible_complete += 1
        raw_goal, explicit_identity = _raw_goal_and_identity(chunk)
        if not raw_goal:
            n_nogoal += 1
            continue
        source_goal = indexed_proof["source_goal"]
        if not source_goal:
            n_nogoal += 1
            continue
        diagnostic = goal_diagnostic(raw_goal, source_goal)
        if diagnostic == "match":
            n_goal_match += 1
        elif diagnostic == "rendering-difference":
            n_goal_rendering += 1
        else:
            n_goal_different += 1
        provenance = (
            "mismatch"
            if explicit_identity is not None and explicit_identity != name
            else "match"
        )
        if provenance == "mismatch":
            n_provenance += 1
        refs, missing = resolve_index_references(
            body,
            index,
            stmt,
            theorem=name,
        )
        if missing:
            n_incomplete += 1
            continue
        if not refs:
            n_nocite += 1
            continue
        accepted_candidates += 1
        if name in done:
            n_skip += 1
            continue
        diagnostics = {
            "raw_goal": diagnostic,
            "numeric_provenance": provenance,
        }
        parsed.append((name, source_statement, body, refs, diagnostics))
        for r in refs:
            counts[r] = counts.get(r, 0) + 1
    index.close()

    print(f"  usable {len(parsed):,}   skipped(dup) {n_skip:,}   "
          f"no proof {n_noproof:,}   no goal {n_nogoal:,}   "
          f"no resolvable citation {n_nocite:,}")
    print(f"  rejected source-missing {n_no_source:,}   "
          f"numeric provenance mismatch {n_provenance:,}   "
          f"incomplete references {n_incomplete:,}   "
          f"source association {n_association:,}")
    print(
        f"  raw-goal diagnostic match {n_goal_match:,}   "
        f"harmless rendering {n_goal_rendering:,}   "
        f"different {n_goal_different:,}"
    )
    join_rate = joined_names / named_files if named_files else 0.0
    complete_proofs = proof_categories["complete_explicit_proof"]
    explicit_proofs = (
        complete_proofs + proof_categories["malformed_explicit_proof"]
    )
    completion_rate = (
        complete_proofs / explicit_proofs if explicit_proofs else 0.0
    )
    accepted_rate = (
        accepted_candidates / eligible_complete if eligible_complete else 0.0
    )
    print(
        f"  source-name join {joined_names:,}/{named_files:,} "
        f"({join_rate:.3%}); complete explicit proofs {complete_proofs:,}/"
        f"{explicit_proofs:,} ({completion_rate:.3%}); accepted "
        f"{accepted_candidates:,}/{eligible_complete:,} eligible complete "
        f"({accepted_rate:.3%})"
    )
    print(
        "  enforced floors: "
        f"source files {coverage['minimum_source_files']:,}; "
        f"join {coverage['minimum_name_matches']:,} and "
        f"{coverage['minimum_name_join_rate']:.3%}; explicit "
        f"{coverage['minimum_explicit_proofs']:,}; completion "
        f"{coverage['minimum_complete_proofs']:,} and "
        f"{coverage['minimum_completion_rate']:.3%}; accepted "
        f"{coverage['minimum_accepted_proofs']:,} and "
        f"{coverage['minimum_accepted_rate']:.3%}"
    )
    index_counts_match = (
        named_files == index_content["thproof_files"]
        and joined_names == index_content["thproof_join_count"]
        and explicit_proofs
        == index_content["explicit_proof_bearing_extracts"]
        and complete_proofs == index_content["complete_explicit_proofs"]
        and n_association == 0
    )
    gates_pass = (
        index_counts_match
        and len(files) >= coverage["minimum_source_files"]
        and joined_names >= coverage["minimum_name_matches"]
        and join_rate >= coverage["minimum_name_join_rate"]
        and explicit_proofs >= coverage["minimum_explicit_proofs"]
        and complete_proofs >= coverage["minimum_complete_proofs"]
        and completion_rate >= coverage["minimum_completion_rate"]
        and accepted_candidates >= coverage["minimum_accepted_proofs"]
        and accepted_rate >= coverage["minimum_accepted_rate"]
    )
    if not gates_pass:
        print(
            "source coverage/completion gate failed; outputs invalidated",
            file=sys.stderr,
        )
        _invalidate_outputs(output_paths)
        return 2
    if not parsed:
        print("no accepted thproof rows; outputs invalidated", file=sys.stderr)
        _invalidate_outputs(output_paths)
        return 1

    tail = sorted(n for n, c in counts.items() if c in (1, 2))
    held = set(random.Random(a.seed).sample(tail, min(a.heldout, len(tail))))
    kept = ev = dup = 0
    tb = 0
    seen = set()
    sp = output_paths[0]
    for directory in ("shards", "eval", "heldout"):
        os.makedirs(os.path.join(a.out, directory), exist_ok=True)
    temp_paths = tuple(f"{path}.tmp.{os.getpid()}" for path in output_paths)
    train_tmp, eval_tmp, heldout_tmp = temp_paths
    try:
        with open(train_tmp, "w", encoding="utf-8") as fh, open(
            eval_tmp, "w", encoding="utf-8"
        ) as evf:
            for name, goal, body, refs, diagnostics in parsed:
                eid = hashlib.md5(f"thproofs/{name}".encode()).hexdigest()[:12]
                blk = {r: stmt[r] for r in shuffled(refs, eid)}
                block = HDR + "\n" + "\n".join(
                    f"{n} : {s}" for n, s in blk.items()
                )
                text = f"{block}\n{SEP}\nGOAL {goal}\n{body}"
                if text in seen:
                    dup += 1
                    continue
                seen.add(text)
                rec = {
                    "schema_version": ROW_SCHEMA,
                    "id": eid,
                    "theorem": name,
                    "facts": blk,
                    "cited": refs,
                    "goal": goal,
                    "target": body,
                    "text": text,
                    "mask_start": 0,
                    "mask_end": len(block),
                    "source_diagnostics": diagnostics,
                    "source_metadata": source_metadata,
                }
                if name in held or set(refs) & held:
                    evf.write(json.dumps(rec) + "\n")
                    ev += 1
                else:
                    fh.write(json.dumps(rec) + "\n")
                    kept += 1
                    tb += len(text.encode())
        with open(heldout_tmp, "w", encoding="utf-8") as heldout_file:
            json.dump(
                {
                    "facts": sorted(held),
                    "seed": a.seed,
                    "corpus": a.name,
                    "source_metadata": source_metadata,
                    "build_coverage": {
                        "source_files": len(files),
                        "name_matches": joined_names,
                        "name_join_rate": join_rate,
                        "explicit_proof_bearing_extracts": explicit_proofs,
                        "complete_explicit_proofs": complete_proofs,
                        "explicit_completion_rate": completion_rate,
                        "eligible_complete_proofs": eligible_complete,
                        "accepted_proofs": accepted_candidates,
                        "accepted_rate": accepted_rate,
                    },
                    "policy": "facts cited 1-2x; citing proofs and own proof removed",
                },
                heldout_file,
                indent=1,
            )
        for temp, final in zip(temp_paths, output_paths):
            os.replace(temp, final)
    except Exception:
        _invalidate_outputs(output_paths)
        raise
    finally:
        for temp in temp_paths:
            if os.path.exists(temp):
                os.remove(temp)
    print(f"  held out {len(held):,} of {len(tail):,} facts cited 1-2x")

    print(f"\n  train {kept:,}   eval {ev:,}   duplicate {dup:,}")
    print(f"  {tb/1e6:.1f} MB text  ~{tb/2.2/1e6:.0f}M GPT-2 tokens")
    print(f"  wrote {sp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
