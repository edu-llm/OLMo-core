"""Invariants every corpus shard must satisfy before it enters training.

These are the acceptance criteria for the four extraction jobs. Each machine runs
this file against its own shard; a shard that fails any test does not ship.

The invariants exist because each one corresponds to a mistake already made during
the survey:

  I1 oracle completeness  — a fact block missing a cited statement silently turns a
                            perfect-retriever example into an imperfect one.
  I2 held-out isolation   — a held-out fact leaks if any training example cites it
                            OR if its own proof survives (96.4% of facts are proved
                            in-corpus, so the goal line is a second leak path).
  I3 name stability       — one name must denote one statement, or the split arm has
                            nothing stable to key on.
  I4 no degenerate targets— empty or unchanged targets teach nothing.
  I5 mask well-formedness — the masked span must be exactly the fact block.
"""

import hashlib
import json
import os
import re

import pytest

HDR = "I know these mathematical statements:"
SEP = "---"
REAL_ARTIFACT_CONFIG_HELP = (
    "real-artifact invariants are disabled: set both SHARD_PATH and HELDOUT_PATH "
    "explicitly; artifact-backed tests skip when neither is set, while inline unit "
    "regressions still run"
)
FLEXARY1_20_NAME = "FLEXARY1:20"
FLEXARY1_20_STATEMENT = (
    "for n being Nat for f being finite complex-valued Function holds "
    "(f . n) + ((f,(n + 1)) +...) = (f,n) +..."
)
FLEXARY1_20_STATEMENT_SHA256 = (
    "dd40c1aefebfb49e8242b487cd687d7410dc2cb3968f77c244798dedeab416f4"
)
FLEXARY1_20_SOURCE_BINDING = (
    "mizar-build-source-v2",
    "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762",
    "mizar-semantic-index-v1",
    "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835",
    "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d",
    "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242",
)
FLEXARY1_20_OCCURRENCE_BINDINGS = frozenset(
    {
        (
            "2af2fb7c7d4ddb154ab9a4a125473ce1155f926a2863ccdac1c490cf48b461a6",
            "EULRPART:12",
            "EULRPART",
            "eulrpart.miz",
            "0342a4e3663241ddb34e0d960922656ce6fae2b5a8475a4876c137c0286fddbc",
            12,
            "eb7c44c55cb09849cf2fb9f7bb36b18c9c789500f89fa7e5386c6c08a7c9acf4",
            "fc4c1fda590c637e7e3f5c01c1ec7da98121f708e90f0a7988e15ba341dcae83",
            "EULRPART:12",
            "02acb73e4542619066cca952ac920c32fea9977abf59191790950df876fe3f8b",
            "fc4c1fda590c637e7e3f5c01c1ec7da98121f708e90f0a7988e15ba341dcae83",
        ),
        (
            "c704b4918600038142be1197df61a8ebe65beeabf16331a6ea96abd3defb186a",
            "FLEXARY1:22",
            "FLEXARY1",
            "flexary1.miz",
            "ecfe91138077414400a3a14e78e2a20ed3cbb6a2adfb1d8953128a6bc0a82836",
            22,
            "dcc34870af9e15ee07f86acbfb72830fac42745e35133407d5fcee161e47aa02",
            "03de27c6a3f4d0ba6ad74203f4d8b003a84d269e41e4c920c8684b4be471f52f",
            "FLEXARY1:22",
            "3b8a6e9757783f9529b4199c35f0004532112c47d9a91dad74e92b41fc09cc5b",
            "03de27c6a3f4d0ba6ad74203f4d8b003a84d269e41e4c920c8684b4be471f52f",
        ),
    }
)


def _artifact_paths_from_environment(environment):
    shard = environment.get("SHARD_PATH")
    heldout = environment.get("HELDOUT_PATH")
    if not shard and not heldout:
        return None
    if not shard or not heldout:
        missing = "HELDOUT_PATH" if shard else "SHARD_PATH"
        raise pytest.UsageError(
            "real-artifact configuration error: SHARD_PATH and HELDOUT_PATH must be "
            f"set together (missing {missing})"
        )
    return shard, heldout


def load(path, limit=None):
    if not os.path.exists(path):
        pytest.fail(
            f"real-artifact configuration points to a missing file: {path}",
            pytrace=False,
        )
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                out.append(json.loads(line))
    return out


@pytest.fixture(scope="module")
def artifact_paths():
    paths = _artifact_paths_from_environment(os.environ)
    if paths is None:
        pytest.skip(REAL_ARTIFACT_CONFIG_HELP)
    missing = [
        f"{variable}={path!r}"
        for variable, path in zip(("SHARD_PATH", "HELDOUT_PATH"), paths)
        if not os.path.isfile(path)
    ]
    if missing:
        pytest.fail(
            "real-artifact configuration points to missing files: " + ", ".join(missing),
            pytrace=False,
        )
    return paths


@pytest.fixture(scope="module")
def rows(artifact_paths):
    shard, _ = artifact_paths
    return load(shard, limit=200_000)


@pytest.fixture(scope="module")
def heldout_manifest(artifact_paths):
    _, heldout_path = artifact_paths
    with open(heldout_path, encoding="utf-8") as heldout_file:
        return json.load(heldout_file)


@pytest.fixture(scope="module")
def heldout(heldout_manifest):
    return set(heldout_manifest["facts"])


def _normalized_layout(text):
    return " ".join(str(text).split())


def _without_isabelle_declaration(text):
    normalized = _normalized_layout(text)
    return re.sub(
        r"^(?:lemma|theorem|corollary|proposition)\s+",
        "",
        normalized,
        count=1,
    )


def _boundary_contains(text, needle):
    text = _normalized_layout(text)
    needle = _normalized_layout(needle)
    start = 0
    while needle and (index := text.find(needle, start)) >= 0:
        before = text[index - 1] if index else ""
        after_index = index + len(needle)
        after = text[after_index] if after_index < len(text) else ""
        if not re.match(r"[A-Za-z0-9_'.]", before) and not re.match(
            r"[A-Za-z0-9_'.]",
            after,
        ):
            return True
        start = index + 1
    return False


def _flexary1_source_binding(row):
    source_metadata = row.get("source_metadata")
    if not isinstance(source_metadata, dict):
        return None
    index_roots = source_metadata.get("index_roots")
    if not isinstance(index_roots, dict):
        return None
    return (
        source_metadata.get("schema_version"),
        source_metadata.get("source_manifest_root_sha256"),
        index_roots.get("semantic_index_schema"),
        index_roots.get("semantic_index_sha256"),
        source_metadata.get("quality_filter_root_sha256"),
        source_metadata.get("schema_generation_root_sha256"),
    )


def _flexary1_occurrence_binding(row):
    source = row.get("source")
    index = row.get("index")
    if not isinstance(source, dict) or not isinstance(index, dict):
        return None
    return (
        row.get("id"),
        row.get("theorem"),
        source.get("article"),
        source.get("file"),
        source.get("file_sha256"),
        source.get("declaration_ordinal"),
        source.get("declaration_sha256"),
        source.get("target_sha256"),
        index.get("identity"),
        index.get("statement_sha256"),
        index.get("proof_sha256"),
    )


def _is_authoritative_flexary1_20_fact(row, fact_name, statement):
    if row.get("schema_version") != "mizar-proof-v2":
        return False
    if fact_name != FLEXARY1_20_NAME or statement != FLEXARY1_20_STATEMENT:
        return False
    if (
        hashlib.sha256(statement.encode()).hexdigest()
        != FLEXARY1_20_STATEMENT_SHA256
    ):
        return False
    if _flexary1_source_binding(row) != FLEXARY1_20_SOURCE_BINDING:
        return False
    return _flexary1_occurrence_binding(row) in FLEXARY1_20_OCCURRENCE_BINDINGS


def _statement_looks_truncated(row, fact_name, statement):
    if not isinstance(statement, str):
        return True
    stripped = statement.strip()
    if len(stripped) < 3:
        return True
    if not stripped.endswith(("…", "...")):
        return False
    return not _is_authoritative_flexary1_20_fact(row, fact_name, statement)


def structured_heldout_exposures(row, manifest):
    """Independently scan every structured train field for heldout exposure."""

    held_names = set(manifest.get("facts", []))
    statements = {
        name: _normalized_layout(statement)
        for name, statement in manifest.get("statements", {}).items()
        if name in held_names
    }
    exposures = []

    for field in ("facts", "premise_aliases", "local_names"):
        values = row.get(field, {})
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            for name in held_names:
                if _boundary_contains(key, name) or _boundary_contains(value, name):
                    exposures.append((f"{field}.{key}", name, "name"))

    exact_statement_fields = {
        "facts": row.get("facts", {}),
        "local_assumptions": row.get("local_assumptions", {}),
    }
    for field, values in exact_statement_fields.items():
        if not isinstance(values, dict):
            continue
        for key, value in values.items():
            normalized = _normalized_layout(value)
            for name, statement in statements.items():
                if normalized == statement:
                    exposures.append((f"{field}.{key}", name, "statement"))

    theorem_statement = row.get("theorem_statement", "")
    for name, statement in statements.items():
        if _without_isabelle_declaration(theorem_statement) == statement:
            exposures.append(("theorem_statement", name, "declaration"))

    for field in ("goal", "state_before", "state_after", "target"):
        value = row.get(field, "")
        for name, statement in statements.items():
            if _boundary_contains(value, statement):
                exposures.append((field, name, "statement"))

    for field in ("cited",):
        values = row.get(field, [])
        if isinstance(values, list):
            for value in values:
                if value in held_names:
                    exposures.append((field, value, "name"))
    for field in ("theorem",):
        value = row.get(field)
        if value in held_names:
            exposures.append((field, value, "name"))

    return exposures


def test_real_artifact_configuration_has_no_implicit_defaults():
    assert _artifact_paths_from_environment({}) is None
    assert _artifact_paths_from_environment(
        {
            "SHARD_PATH": "/fixtures/mizar.jsonl",
            "HELDOUT_PATH": "/fixtures/mizar.json",
        }
    ) == ("/fixtures/mizar.jsonl", "/fixtures/mizar.json")


@pytest.mark.parametrize(
    "environment",
    [
        {"SHARD_PATH": "/fixtures/mizar.jsonl"},
        {"HELDOUT_PATH": "/fixtures/mizar.json"},
    ],
)
def test_real_artifact_configuration_rejects_a_partial_path_pair(environment):
    with pytest.raises(pytest.UsageError, match="SHARD_PATH and HELDOUT_PATH must be set together"):
        _artifact_paths_from_environment(environment)


def _accepted_flexary1_20_occurrences():
    source_metadata = {
        "schema_version": "mizar-build-source-v2",
        "source_manifest_root_sha256": (
            "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"
        ),
        "index_roots": {
            "semantic_index_schema": "mizar-semantic-index-v1",
            "semantic_index_sha256": (
                "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835"
            ),
        },
        "quality_filter_root_sha256": (
            "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d"
        ),
        "schema_generation_root_sha256": (
            "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242"
        ),
    }
    return [
        {
            "schema_version": "mizar-proof-v2",
            "id": (
                "2af2fb7c7d4ddb154ab9a4a125473ce1155f926a2863ccdac1c490cf48b461a6"
            ),
            "theorem": "EULRPART:12",
            "source_metadata": json.loads(json.dumps(source_metadata)),
            "source": {
                "article": "EULRPART",
                "file": "eulrpart.miz",
                "file_sha256": (
                    "0342a4e3663241ddb34e0d960922656ce6fae2b5a8475a4876c137c0286fddbc"
                ),
                "declaration_ordinal": 12,
                "declaration_sha256": (
                    "eb7c44c55cb09849cf2fb9f7bb36b18c9c789500f89fa7e5386c6c08a7c9acf4"
                ),
                "target_sha256": (
                    "fc4c1fda590c637e7e3f5c01c1ec7da98121f708e90f0a7988e15ba341dcae83"
                ),
            },
            "index": {
                "identity": "EULRPART:12",
                "statement_sha256": (
                    "02acb73e4542619066cca952ac920c32fea9977abf59191790950df876fe3f8b"
                ),
                "proof_sha256": (
                    "fc4c1fda590c637e7e3f5c01c1ec7da98121f708e90f0a7988e15ba341dcae83"
                ),
            },
        },
        {
            "schema_version": "mizar-proof-v2",
            "id": (
                "c704b4918600038142be1197df61a8ebe65beeabf16331a6ea96abd3defb186a"
            ),
            "theorem": "FLEXARY1:22",
            "source_metadata": json.loads(json.dumps(source_metadata)),
            "source": {
                "article": "FLEXARY1",
                "file": "flexary1.miz",
                "file_sha256": (
                    "ecfe91138077414400a3a14e78e2a20ed3cbb6a2adfb1d8953128a6bc0a82836"
                ),
                "declaration_ordinal": 22,
                "declaration_sha256": (
                    "dcc34870af9e15ee07f86acbfb72830fac42745e35133407d5fcee161e47aa02"
                ),
                "target_sha256": (
                    "03de27c6a3f4d0ba6ad74203f4d8b003a84d269e41e4c920c8684b4be471f52f"
                ),
            },
            "index": {
                "identity": "FLEXARY1:22",
                "statement_sha256": (
                    "3b8a6e9757783f9529b4199c35f0004532112c47d9a91dad74e92b41fc09cc5b"
                ),
                "proof_sha256": (
                    "03de27c6a3f4d0ba6ad74203f4d8b003a84d269e41e4c920c8684b4be471f52f"
                ),
            },
        },
    ]


def _canonical_flexary1_20_statement():
    return (
        "for n being Nat for f being finite complex-valued Function holds "
        "(f . n) + ((f,(n + 1)) +...) = (f,n) +..."
    )


def test_truncation_check_accepts_both_authoritative_flexary1_occurrences():
    flexary1_20 = _canonical_flexary1_20_statement()
    occurrences = _accepted_flexary1_20_occurrences()

    assert len(occurrences) == 2
    assert {
        row["source_metadata"]["source_manifest_root_sha256"] for row in occurrences
    } == {"fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"}
    for row in occurrences:
        assert not _statement_looks_truncated(row, "FLEXARY1:20", flexary1_20)


@pytest.mark.parametrize("occurrence_index", [0, 1])
def test_truncation_check_rejects_binding_copied_to_abcmiz_row(occurrence_index):
    row = _accepted_flexary1_20_occurrences()[occurrence_index]
    row["id"] = "f971bf1e3a8a6695a63f016ce6bd5b76825d46b22e0c73b02f1b02463e4a34dd"

    assert _statement_looks_truncated(
        row,
        "FLEXARY1:20",
        _canonical_flexary1_20_statement(),
    )


@pytest.mark.parametrize(("source_index", "destination_index"), [(0, 1), (1, 0)])
def test_truncation_check_rejects_swapped_canonical_row_identity(
    source_index,
    destination_index,
):
    occurrences = _accepted_flexary1_20_occurrences()
    row = occurrences[source_index]
    row["id"] = occurrences[destination_index]["id"]

    assert _statement_looks_truncated(
        row,
        "FLEXARY1:20",
        _canonical_flexary1_20_statement(),
    )


@pytest.mark.parametrize(
    ("fact_name", "statement"),
    [
        ("FLEXARY1:21", _canonical_flexary1_20_statement()),
        ("FLEXARY1:20", "not a complete mathematical statement +..."),
        ("XBOOLE_0:1", "for X being set holds X = X +..."),
    ],
)
def test_truncation_check_rejects_adversarial_mizar_ellipsis(fact_name, statement):
    for row in _accepted_flexary1_20_occurrences():
        assert _statement_looks_truncated(row, fact_name, statement)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "wrong-schema",
        "wrong-index-root",
        "wrong-manifest-root",
        "wrong-quality-root",
        "wrong-generation-root",
    ],
)
def test_truncation_check_rejects_untrusted_flexary1_source_metadata(mutation):
    row = _accepted_flexary1_20_occurrences()[0]
    metadata = row["source_metadata"]
    if mutation == "missing":
        row.pop("source_metadata")
    elif mutation == "wrong-schema":
        metadata["schema_version"] = "mizar-build-source-v1"
    elif mutation == "wrong-index-root":
        metadata["index_roots"]["semantic_index_sha256"] = "0" * 64
    elif mutation == "wrong-manifest-root":
        metadata["source_manifest_root_sha256"] = "0" * 64
    elif mutation == "wrong-quality-root":
        metadata["quality_filter_root_sha256"] = "0" * 64
    else:
        metadata["schema_generation_root_sha256"] = "0" * 64

    assert _statement_looks_truncated(
        row,
        "FLEXARY1:20",
        _canonical_flexary1_20_statement(),
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-source",
        "wrong-source-file",
        "missing-index",
        "wrong-index-statement",
    ],
)
def test_truncation_check_rejects_mutated_occurrence_replay_binding(mutation):
    row = _accepted_flexary1_20_occurrences()[0]
    if mutation == "missing-source":
        row.pop("source")
    elif mutation == "wrong-source-file":
        row["source"]["file_sha256"] = "0" * 64
    elif mutation == "missing-index":
        row.pop("index")
    else:
        row["index"]["statement_sha256"] = "0" * 64

    assert _statement_looks_truncated(
        row,
        "FLEXARY1:20",
        _canonical_flexary1_20_statement(),
    )


def test_truncation_check_keeps_ordinary_empty_and_clipped_detection():
    source_backed_mizar = _accepted_flexary1_20_occurrences()[0]
    flexary1_20 = _canonical_flexary1_20_statement()

    assert _statement_looks_truncated({}, "FLEXARY1:20", flexary1_20)
    assert _statement_looks_truncated(
        source_backed_mizar,
        "FLEXARY1:21",
        "for x being set holds x = ...",
    )
    assert _statement_looks_truncated(source_backed_mizar, "FLEXARY1:21", "")
    assert _statement_looks_truncated(source_backed_mizar, "FLEXARY1:21", "x")


# ----------------------------------------------------------------- I1
def test_every_example_has_a_nonempty_fact_block(rows):
    bad = [r["id"] for r in rows if not r.get("facts")]
    assert not bad, f"{len(bad)} examples have an empty fact block, e.g. {bad[:3]}"


def test_every_fact_carries_a_statement(rows):
    bad = []
    for r in rows:
        for name, stmt in r["facts"].items():
            if not stmt or not stmt.strip():
                bad.append((r["id"], name))
    assert not bad, (f"{len(bad)} facts have a name but no statement — the block is "
                     f"not an oracle, e.g. {bad[:3]}")


def test_every_cited_name_appears_in_the_block(rows):
    """The derivation may only cite facts the block supplies."""
    bad = []
    for r in rows:
        missing = set(r.get("cited", [])) - set(r["facts"])
        if missing:
            bad.append((r["id"], sorted(missing)[:3]))
    assert not bad, (f"{len(bad)} examples cite a fact absent from their block, "
                     f"e.g. {bad[:3]}")


# ----------------------------------------------------------------- I2
def test_no_training_example_cites_a_heldout_fact(rows, heldout):
    bad = [(r["id"], sorted(set(r.get("cited", [])) & heldout)[:3])
           for r in rows if set(r.get("cited", [])) & heldout]
    assert not bad, f"{len(bad)} training examples cite a held-out fact: {bad[:3]}"


def test_no_training_example_proves_a_heldout_fact(rows, heldout):
    """The goal-line leak: a fact's own proof exposes its statement."""
    bad = [r["id"] for r in rows if r.get("theorem") in heldout]
    assert not bad, (f"{len(bad)} training examples ARE the proof of a held-out "
                     f"fact, leaking its statement as the goal: {bad[:3]}")


def test_no_heldout_statement_appears_in_structured_training_fields(
    rows,
    heldout_manifest,
):
    """Scan locals, declarations, goals, states, and targets independently."""

    if not rows:
        pytest.skip("empty shard")
    if not heldout_manifest.get("statements"):
        pytest.skip("heldout manifest does not enumerate statements")
    for r in rows:
        exposures = structured_heldout_exposures(r, heldout_manifest)
        assert not exposures, (
            f"{r['id']} exposes heldout data in structured fields: {exposures[:3]}"
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        (
            "local_assumptions",
            {"iso_assoc": "iso_assoc: fixes a shows P"},
        ),
        (
            "theorem_statement",
            "lemma iso_assoc: fixes a shows P",
        ),
        (
            "goal",
            "THEOREM\niso_assoc: fixes a shows P\nSTATE_BEFORE\nsafe",
        ),
        ("state_before", "proof goal: 1. iso_assoc: fixes a shows P"),
        ("state_after", "proof goal: 1. iso_assoc: fixes a shows P"),
        (
            "target",
            "TACTIC\napply\nSTATE_AFTER\niso_assoc: fixes a shows P",
        ),
        (
            "local_names",
            {"iso_assoc": "MonoidalCategory.iso_assoc"},
        ),
    ],
)
def test_structured_exposure_helper_covers_every_isabelle_leak_field(field, value):
    row = {
        "id": field,
        "facts": {"Safe.fact": "safe statement"},
        "cited": ["Safe.fact"],
        "premise_aliases": {"safe": "Safe.fact"},
        "local_assumptions": {},
        "local_names": {},
        "theorem": "Safe/theorem",
        "theorem_statement": "safe theorem",
        "goal": "safe goal",
        "state_before": "safe before",
        "state_after": "safe after",
        "target": "safe target",
    }
    row[field] = value
    manifest = {
        "facts": ["MonoidalCategory.iso_assoc"],
        "statements": {
            "MonoidalCategory.iso_assoc": "iso_assoc: fixes a shows P",
        },
    }

    assert structured_heldout_exposures(row, manifest)


# ----------------------------------------------------------------- I3
def test_one_name_denotes_one_statement(rows):
    seen = {}
    clashes = []
    for r in rows:
        for n, s in r["facts"].items():
            k = " ".join(s.split())
            if n in seen and seen[n] != k:
                clashes.append(n)
            seen.setdefault(n, k)
    uniq = sorted(set(clashes))
    assert not uniq, (f"{len(uniq)} names denote more than one statement — the store "
                      f"is not persistent: {uniq[:5]}")


# ----------------------------------------------------------------- I4
def test_target_is_nonempty_and_differs_from_the_goal(rows):
    bad = [r["id"] for r in rows
           if not r.get("target", "").strip()
           or " ".join(r["target"].split()) == " ".join(r.get("goal", "").split())]
    assert not bad, f"{len(bad)} examples have an empty or unchanged target: {bad[:3]}"


def test_no_constant_target_dominates(rows):
    """41% of LeanDojo's targets were the literal string 'no goals'."""
    from collections import Counter
    c = Counter(" ".join(r.get("target", "").split()) for r in rows)
    if not c:
        pytest.skip("empty shard")
    top, n = c.most_common(1)[0]
    share = n / len(rows)
    assert share < 0.05, (f"target {top[:40]!r} accounts for {share:.1%} of the "
                          f"shard — degenerate")


# ----------------------------------------------------------------- I5
def test_rendered_text_has_exactly_one_mask_span(rows):
    for r in rows[:5000]:
        t = r["text"]
        assert t.count(HDR) == 1, (
            f"{r['id']}: fact-block header appears {t.count(HDR)}x"
        )
        assert t.count(f"\n{SEP}\n") == 1, f"{r['id']}: separator is not unique"
        assert t.index(HDR) < t.index(f"\n{SEP}\n"), f"{r['id']}: block after separator"


def test_mask_span_covers_the_block_and_nothing_else(rows):
    for r in rows[:5000]:
        t, a, b = r["text"], r["mask_start"], r["mask_end"]
        span = t[a:b]
        assert span.startswith(HDR), f"{r['id']}: mask does not start at the header"
        assert SEP not in span, f"{r['id']}: mask swallows the separator"
        for name in r["facts"]:
            assert name in span, f"{r['id']}: fact {name} sits outside the mask"


def test_masked_fraction_is_in_the_design_band(rows):
    if not rows:
        pytest.skip("empty shard")
    fr = [(r["mask_end"] - r["mask_start"]) / max(len(r["text"]), 1) for r in rows]
    mean = sum(fr) / len(fr)
    assert 0.05 < mean < 0.60, (f"masked fraction {mean:.1%} is outside the 5–60% "
                                f"band; ~17–30% is the design target")


# ----------------------------------------------------------------- I6
def test_no_duplicate_examples(rows):
    ids = [r["id"] for r in rows]
    txt = [r["text"] for r in rows]
    dup_id = len(ids) - len(set(ids))
    dup_tx = len(txt) - len(set(txt))
    assert dup_id == 0, f"{dup_id} duplicate ids"
    assert dup_tx == 0, (f"{dup_tx} examples are byte-identical — the model would "
                         f"see them twice per epoch")


# ----------------------------------------------------------------- I7
def test_train_and_eval_do_not_overlap(rows, artifact_paths):
    """Example-level leak, distinct from the held-out fact check in I2."""
    shard, _ = artifact_paths
    ev_path = shard.replace(".jsonl", "_eval.jsonl")
    if not os.path.exists(ev_path):
        pytest.skip("no eval file beside this shard")
    ev = load(ev_path)
    tr_txt = {r["text"] for r in rows}
    tr_thm = {r.get("theorem") for r in rows}
    same_txt = [r["id"] for r in ev if r["text"] in tr_txt]
    same_thm = [r["id"] for r in ev if r.get("theorem") in tr_thm]
    assert not same_txt, f"{len(same_txt)} eval examples appear verbatim in train"
    assert not same_thm, (f"{len(same_thm)} eval theorems are also proved in train — "
                          f"the same result reached another way still leaks")


# ----------------------------------------------------------------- I8
def test_text_is_clean(rows):
    bad_ctrl, bad_repl = [], []
    for r in rows[:50_000]:
        t = r["text"]
        if "\ufffd" in t:
            bad_repl.append(r["id"])
        if any(ord(c) < 9 or 13 < ord(c) < 32 for c in t):
            bad_ctrl.append(r["id"])
    assert not bad_repl, f"{len(bad_repl)} examples contain U+FFFD (bad decode)"
    assert not bad_ctrl, f"{len(bad_ctrl)} examples contain control characters"


def test_statements_are_not_truncated(rows):
    bad = [
        (r["id"], name)
        for r in rows
        for name, statement in r["facts"].items()
        if _statement_looks_truncated(r, name, statement)
    ]
    assert not bad, (f"{len(bad)} statements look truncated — a clipped fact makes "
                     f"the block a bad oracle: {bad[:3]}")


# ----------------------------------------------------------------- I9
def test_goals_are_nondegenerate(rows):
    bad = [r["id"] for r in rows if len(r.get("goal", "").strip()) < 3]
    assert not bad, f"{len(bad)} examples have an empty or trivial goal: {bad[:3]}"


# ----------------------------------------------------------------- I10
def test_fact_block_order_does_not_leak_the_proof(rows):
    """If the block is listed in citation order, the model reads the step
    sequence straight off the prompt without deriving it."""
    multi = [r for r in rows if len(r["facts"]) >= 3]
    if len(multi) < 50:
        pytest.skip("too few multi-fact examples to judge ordering")
    same = sum(1 for r in multi if list(r["facts"]) == r["cited"])
    share = same / len(multi)
    assert share < 0.20, (f"{share:.0%} of multi-fact blocks are in citation order — "
                          f"the block leaks the derivation sequence. Shuffle it with "
                          f"a per-example deterministic seed.")


# ----------------------------------------------------------------- I11
def test_heldout_manifest_is_the_shared_one(heldout):
    """Every machine must mask the same 500 facts or the eval is contaminated."""
    expected = os.environ.get("HELDOUT_SHA256")
    if not expected:
        pytest.skip("set HELDOUT_SHA256 to pin the shared manifest")
    import hashlib
    got = hashlib.sha256(
        json.dumps(sorted(heldout)).encode()).hexdigest()
    assert got == expected, (f"held-out set does not match the shared manifest\n"
                             f"  expected {expected}\n  got      {got}")
