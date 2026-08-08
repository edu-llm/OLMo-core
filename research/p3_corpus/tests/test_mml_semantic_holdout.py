"""Contract tests for the pooled MML semantic holdout planner."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

holdout = importlib.import_module("split_mml_semantic_holdout")
build_atp_shard = importlib.import_module("build_atp_shard")


SHARDS = ("mizar", "thproofs", "prf2", "enigma")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _common_row(row_id, theorem, facts, *, goal=None, target="proof"):
    goal = goal or f"goal {row_id}"
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    text = f"{block}\n---\nGOAL {goal}\n{target}"
    return {
        "id": row_id,
        "theorem": theorem,
        "facts": facts,
        "cited": list(facts),
        "goal": goal,
        "target": target,
        "text": text,
        "mask_start": 0,
        "mask_end": len(block),
    }


def _mizar_row(row_id, theorem, facts, **overrides):
    row = _common_row(row_id, theorem, facts)
    row.update(overrides)
    return row


def _atp_row(
    row_id,
    theorem,
    facts,
    *,
    local_inputs=None,
    goal="goal(a)",
    formula="$false",
    rule="resolution",
):
    local_inputs = local_inputs or {}
    fact_names = list(facts)
    step = {
        "name": "step_x",
        "role": "plain",
        "formula": formula,
        "rule": rule,
        "parents": [*fact_names, "theorem_x"],
        "parent_sources": [*fact_names, "theorem_x"],
        "source": (
            f"inference({rule},[status(thm)],[{','.join([*fact_names, 'theorem_x'])}])"
        ),
    }
    target = build_atp_shard.render_target([build_atp_shard.ProofStep(**step)])
    row = _common_row(row_id, theorem, facts, goal=goal, target=target)
    if local_inputs:
        block = row["text"].split("\n---\n", 1)[0]
        block += "\nLocal ATP inputs:\n" + "\n".join(
            f"{name} : {statement}" for name, statement in local_inputs.items()
        )
        row["text"] = f"{block}\n---\nGOAL {goal}\n{target}"
        row["mask_end"] = len(block)
    row.update(
        {
            "schema_version": "atp-v2",
            "local_inputs": local_inputs,
            "goal_name": "theorem_x",
            "proof_steps": [step],
        }
    )
    return row


def _line(row):
    return (
        json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        + b"\n"
    )


def _sources(rows_by_shard):
    sources = {}
    for shard in SHARDS:
        lines = tuple(_line(row) for row in rows_by_shard.get(shard, ()))
        sources[shard] = holdout.MemoryShardSource(
            name=shard,
            logical_path=f"raw/{shard}.jsonl",
            lines=lines,
            expected_input_sha256=holdout.digest_lines(lines),
            source_snapshots=(
                holdout.SourceSnapshot(
                    reference=f"{shard}-source@test",
                    sha256=SHA_A,
                ),
            ),
            source_manifest_root_sha256=SHA_B,
            quality_filter_root_sha256=SHA_C,
            schema_generation_root_sha256=SHA_D,
        )
    return sources


def _source_policy(sources):
    return holdout.SourceIdentityPolicy(
        policy_id="synthetic-mml-source-policy-v1",
        test_only=True,
        shards={
            shard: holdout.ApprovedShardSource(
                input_sha256=source.expected_input_sha256,
                source_snapshots=source.source_snapshots,
                source_manifest_root_sha256=source.source_manifest_root_sha256,
                quality_filter_root_sha256=source.quality_filter_root_sha256,
                schema_generation_root_sha256=source.schema_generation_root_sha256,
            )
            for shard, source in sources.items()
        },
    )


def _sha(label):
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _source_manifest_root(manifest):
    def without_recursive_roots(value):
        if isinstance(value, dict):
            return {
                key: without_recursive_roots(item)
                for key, item in value.items()
                if key
                not in {
                    "manifest_root_sha256",
                    "source_manifest_root_sha256",
                }
            }
        if isinstance(value, list):
            return [without_recursive_roots(item) for item in value]
        return value

    payload = json.dumps(
        without_recursive_roots(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload + b"\n").hexdigest()


def _finalized_policy_records(tmp_path):
    records = {}
    for shard in SHARDS:
        raw_path = tmp_path / shard / "raw" / f"{shard}.jsonl"
        raw_path.parent.mkdir(parents=True)
        raw_path.write_bytes(_line(_mizar_row(f"{shard}-row", "SAFE:1", {"SAFE:2": "safe"})))
        metadata = {
            "schema_version": f"{shard}-build-source-v2",
            "source_roots": {
                "fixture": {
                    "reference": f"fixture://{shard}",
                    "sha256": _sha(f"{shard}:snapshot"),
                }
            },
            "index_roots": {},
            "quality_filter_root_sha256": _sha(f"{shard}:quality"),
            "schema_generation_root_sha256": _sha(f"{shard}:schema"),
        }
        manifest = {
            "schema_version": "p3-family-source-manifest/v2",
            "family": shard,
            "row_schema_version": (
                "mizar-proof-v2" if shard in {"mizar", "thproofs"} else "atp-v2"
            ),
            "row_source_metadata": metadata,
            "source_snapshots": [
                {
                    "reference": f"fixture://{shard}",
                    "sha256": _sha(f"{shard}:snapshot"),
                }
            ],
            "builder": {
                "driver": "fixture",
                "partition_mode": "pooled-mml-1000-v1",
            },
            "license": {
                "approved": False,
                "identifier": "unresolved-fixture",
                "status": "not a publication assertion",
            },
            "source_verifier_acceptance": {
                "accepted": True,
                "status": "fixture source replay clean",
            },
            "test_only": False,
        }
        root = _source_manifest_root(manifest)
        manifest["manifest_root_sha256"] = root
        metadata["source_manifest_root_sha256"] = root
        records[shard] = holdout.finalize_production_source_record(
            shard,
            raw_path=raw_path,
            source_manifest=manifest,
        )
    return records


def _tokenizer(token_counts=None):
    token_counts = token_counts or {}
    return holdout.TokenizerSeam(
        seal=holdout.approved_tokenizer_seal(),
        count_text_plus_eos=lambda text: token_counts.get(text, 32),
    )


def _plan(
    rows_by_shard,
    *,
    token_counts=None,
    sources=None,
    policy_pins=None,
    source_policy=None,
):
    sources = sources or _sources(rows_by_shard)
    return holdout.plan_semantic_holdout(
        sources,
        tokenizer=_tokenizer(token_counts),
        policy_pins=policy_pins or holdout.current_policy_pins(),
        source_policy=source_policy or _source_policy(sources),
    )


def _mizar_fillers(count=999):
    return {f"FILLER_{index}:1": f"filler statement {index}" for index in range(count)}


def _atp_fillers(count=999):
    return {f"t1_filler_{index}": f"filler({index})" for index in range(count)}


def _selected_index():
    class_id = "mml:v1:theorem:ARTICLE:7"
    mizar_digest = holdout.statement_digest("mizar", "selected statement")
    atp_digest = holdout.statement_digest("atp", "selected(statement)")
    return holdout.ExposureIndex(
        selected_class_ids=frozenset({class_id}),
        members_by_class={
            class_id: {
                "mizar": frozenset({"ARTICLE:7"}),
                "atp": frozenset({"t7_article"}),
            }
        },
        statement_classes_by_representation={
            "mizar": {mizar_digest: frozenset({class_id})},
            "atp": {atp_digest: frozenset({class_id})},
        },
        canonical_statement_classes_by_representation={
            "mizar": {"selected statement": frozenset({class_id})},
            "atp": {"selected(statement)": frozenset({class_id})},
        },
    )


@pytest.mark.parametrize(
    ("representation", "name", "theorem_identity", "class_id", "kind"),
    [
        (
            "mizar",
            "ARTICLE:12",
            False,
            "mml:v1:theorem:ARTICLE:12",
            "theorem",
        ),
        (
            "mizar",
            "ARTICLE:def_9",
            False,
            "mml:v1:definition:ARTICLE:9",
            "definition",
        ),
        (
            "atp",
            "t12_article",
            False,
            "mml:v1:theorem:ARTICLE:12",
            "theorem",
        ),
        (
            "atp",
            "d9_article",
            False,
            "mml:v1:definition:ARTICLE:9",
            "definition",
        ),
        (
            "atp",
            "enigma:t12_article#2",
            True,
            "mml:v1:theorem:ARTICLE:12",
            "theorem",
        ),
        (
            "atp",
            "prf2:d9_article#10",
            True,
            "mml:v1:definition:ARTICLE:9",
            "definition",
        ),
    ],
)
def test_mapping_grammar_joins_only_approved_mml_names(
    representation, name, theorem_identity, class_id, kind
):
    identity = holdout.semantic_identity(
        name,
        representation=representation,
        theorem_identity=theorem_identity,
    )

    assert identity.class_id == class_id
    assert identity.kind == kind
    assert identity.mapped


@pytest.mark.parametrize(
    "name",
    [
        "l100_article",
        "t1_article__scheme_instance",
        "e1_article",
        "c1_article",
        "de1_article",
        "ie1_article",
        "rd1_article",
        "r1_generated_registration",
        "t_article",
        "t01_article",
        "t1_",
        "t7_ARTICLE",
        "prf2:t1_article",
        "t1_article#2",
        "not a tptp atom",
    ],
)
def test_rejected_atp_names_get_representation_singletons(name):
    identity = holdout.semantic_identity(name, representation="atp")

    assert not identity.mapped
    assert identity.kind == "representation_singleton"
    assert identity.class_id.startswith("mml:v1:singleton:atp:")
    assert ":theorem:" not in identity.class_id


def test_quoted_atoms_are_decoded_for_mapping_and_preserved_for_audit():
    mapped = holdout.semantic_identity("'t7_art\\icle'", representation="atp")
    wrong_case = holdout.semantic_identity("'t7_\\ARTICLE'", representation="atp")
    singleton = holdout.semantic_identity("'odd atom'", representation="atp")

    assert mapped.class_id == "mml:v1:theorem:ARTICLE:7"
    assert mapped.native_name == "t7_article"
    assert mapped.raw_name == "'t7_art\\icle'"
    assert not wrong_case.mapped
    assert singleton.native_name == "odd atom"
    assert singleton.raw_name == "'odd atom'"


def test_terminal_alternate_suffix_is_only_normalized_for_theorem_identity():
    theorem = holdout.semantic_identity(
        "enigma:t7_article#10",
        representation="atp",
        theorem_identity=True,
    )
    fact = holdout.semantic_identity(
        "enigma:t7_article#10",
        representation="atp",
        theorem_identity=False,
    )

    assert theorem.class_id == "mml:v1:theorem:ARTICLE:7"
    assert not fact.mapped


def test_enigma_base_and_numbered_variants_route_together():
    pool = _atp_row(
        "pool",
        "prf2:t100_source",
        _atp_fillers(1000),
    )
    variants = []
    for suffix, formula in (
        ("", "safe_base(a)"),
        ("#1", "safe_one(a)"),
        ("#2", "safe_two(a)"),
        ("#3", "filler(7)"),
        ("#4", "safe_four(a)"),
    ):
        row = _atp_row(
            f"variant-{suffix or 'base'}",
            f"enigma:t99_variant{suffix}",
            {"safe_unmapped": "safe(a)"},
        )
        visible = {
            **row["proof_steps"][0],
            "name": "visible_step",
            "formula": formula,
        }
        refutation = {
            "name": "refutation_step",
            "role": "plain",
            "formula": "$false",
            "rule": "resolution",
            "parents": ["visible_step"],
            "parent_sources": ["visible_step"],
            "source": "inference(resolution,[status(thm)],[visible_step])",
        }
        row["proof_steps"] = [visible, refutation]
        row["target"] = build_atp_shard.render_target(
            [build_atp_shard.ProofStep(**step) for step in row["proof_steps"]]
        )
        variants.append(row)

    plan = _plan({"prf2": [pool], "enigma": variants})
    routes = plan.rows["enigma"]

    assert [route.disposition for route in routes] == ["eval"] * 5
    assert all(route.exposure and route.exposure.should_eval for route in routes)
    grouping = plan.manifest["enigma_variant_grouping"]
    assert grouping["groups_with_multiple_variants"] == 1
    assert grouping["rows_promoted_to_eval"] == 4
    assert grouping["route_together"] is True


def test_unmapped_facts_join_only_within_the_same_representation():
    mizar = holdout.semantic_identity("stable_odd_name", representation="mizar")
    atp = holdout.semantic_identity("stable_odd_name", representation="atp")
    atp_again = holdout.semantic_identity("stable_odd_name", representation="atp")

    assert atp.class_id == atp_again.class_id
    assert mizar.class_id != atp.class_id


def test_statement_hashes_are_representation_scoped_and_quote_aware():
    assert holdout.statement_digest("atp", " p ( 'a b' ) ") == (
        holdout.statement_digest("atp", "p('a b')")
    )
    assert holdout.statement_digest("atp", "p('a b')") != (
        holdout.statement_digest("atp", "p('ab')")
    )
    assert holdout.statement_digest("mizar", 'x = "a  b"') != (
        holdout.statement_digest("mizar", 'x = "a b"')
    )
    assert holdout.statement_digest("mizar", "p(x)") != (
        holdout.statement_digest("atp", "p(x)")
    )


@pytest.mark.parametrize(
    ("wrapped", "canonical"),
    [
        ("(selected(statement))", "selected(statement)"),
        ("((selected(statement)))", "selected(statement)"),
        (" ( ( selected ( statement ) ) ) ", "selected(statement)"),
        ("(($false))", "$false"),
        ("((f(a)=b))", "f(a)=b"),
        ("((p(a)&q(a)))", "p(a)&q(a)"),
        ("((p(a)|q(a)))", "p(a)|q(a)"),
        ("((p(a)=>q(a)))", "p(a)=>q(a)"),
        ("((p(a)<=>q(a)))", "p(a)<=>q(a)"),
        ("((![X]:p(X)))", "![X]:p(X)"),
        ("((?[X]:p(X)))", "?[X]:p(X)"),
        ("((p@a))", "p@a"),
    ],
)
def test_tptp_canonicalization_strips_complete_redundant_formula_parentheses(
    wrapped,
    canonical,
):
    assert holdout.canonical_statement(wrapped, representation="atp") == canonical
    assert holdout.statement_digest("atp", wrapped) == holdout.statement_digest(
        "atp",
        canonical,
    )


@pytest.mark.parametrize(
    ("structured", "near_structure"),
    [
        ("selected((statement))", "selected(statement)"),
        ("(selected(statement)", "selected(statement)"),
        ("selected(statement))", "selected(statement)"),
        ("(selected(statement)),other", "selected(statement),other"),
        ("((p(a)&q(a))", "p(a)&q(a)"),
        ("(p(a)|q(a)))", "p(a)|q(a)"),
        ("((![X]:p(X))", "![X]:p(X)"),
        ("(p@a),q", "p@a,q"),
    ],
)
def test_tptp_canonicalization_preserves_required_or_unbalanced_parentheses(
    structured,
    near_structure,
):
    assert holdout.canonical_statement(structured, representation="atp") != (
        holdout.canonical_statement(near_structure, representation="atp")
    )
    assert holdout.statement_digest("atp", structured) != holdout.statement_digest(
        "atp",
        near_structure,
    )


def test_tptp_outer_parentheses_scan_is_quote_and_escape_safe():
    assert (
        holdout.canonical_statement(
            "('(selected(statement))')",
            representation="atp",
        )
        == "'(selected(statement))'"
    )
    assert (
        holdout.canonical_statement(
            "('quoted ) atom')",
            representation="atp",
        )
        == "'quoted ) atom'"
    )
    assert (
        holdout.canonical_statement(
            "('quoted \\' ) atom')",
            representation="atp",
        )
        == "'quoted \\' ) atom'"
    )
    assert (
        holdout.canonical_statement(
            "('unterminated)",
            representation="atp",
        )
        == "('unterminated)"
    )


@pytest.mark.parametrize(
    "malformed",
    [
        "([{]})",
        "({[}])",
        r"(\))",
        "p(a))",
        "(p(a)",
        "([)]",
        r"p(a)\q",
    ],
)
def test_tptp_canonicalization_preserves_malformed_delimiter_text(malformed):
    assert holdout.canonical_statement(malformed, representation="atp") == malformed


def test_tptp_malformed_layout_is_not_normalized_for_digest_or_dedup():
    compact = "([{]})"
    spaced = "([ { ] })"
    assert holdout.statement_digest("atp", compact) != holdout.statement_digest(
        "atp", spaced
    )

    compact_row = _atp_row(
        "compact",
        "prf2:t7_article",
        {"t1_article": compact},
    )
    spaced_row = _atp_row(
        "spaced",
        "enigma:t7_article",
        {"t1_article": spaced},
    )
    assert holdout.exact_atp_signature(compact_row) != holdout.exact_atp_signature(
        spaced_row
    )


@pytest.mark.parametrize(
    ("wrapped", "canonical"),
    [
        ("((f([g({a})])))", "f([g({a})])"),
        ("('quoted ([{]}) delimiters')", "'quoted ([{]}) delimiters'"),
        ('("quoted ({[}]) delimiters")', '"quoted ({[}]) delimiters"'),
        (r"('quoted \')([{}]) delimiters')", r"'quoted \')([{}]) delimiters'"),
    ],
)
def test_tptp_ordered_delimiter_stack_accepts_valid_nesting_and_quoted_text(
    wrapped,
    canonical,
):
    assert holdout.canonical_statement(wrapped, representation="atp") == canonical


def test_malformed_tptp_layout_cannot_alias_into_exposure():
    class_id = "mml:v1:theorem:ARTICLE:7"
    compact = "([{]})"
    spaced = "([ { ] })"
    index = holdout.ExposureIndex(
        selected_class_ids=frozenset({class_id}),
        members_by_class={class_id: {"atp": frozenset({"t7_article"})}},
        statement_classes_by_representation={
            "mizar": {},
            "atp": {
                holdout.statement_digest("atp", compact): frozenset({class_id}),
            },
        },
        canonical_statement_classes_by_representation={
            "mizar": {},
            "atp": {compact: frozenset({class_id})},
        },
    )
    row = _atp_row(
        "malformed-visible",
        "enigma:t3_other",
        {"other": "q(a)"},
        formula=spaced,
    )

    exposure = holdout.classify_exposure(row, shard="enigma", index=index)

    assert not exposure.visible_target
    assert not exposure.should_eval


def test_outer_parenthesized_atp_visible_target_cannot_leak_via_stale_projection():
    row = _atp_row(
        "outer-target",
        "enigma:t3_other",
        {"other": "q(a)"},
        formula="(selected(statement))",
    )

    exposure = holdout.classify_exposure(
        row,
        shard="enigma",
        index=_selected_index(),
    )

    assert exposure.visible_target
    assert exposure.should_eval


@pytest.mark.parametrize(
    "formula",
    [
        "p(a)&q(a)",
        "p(a)|q(a)",
        "p(a)=>q(a)",
        "p(a)<=>q(a)",
        "![X]:p(X)",
        "?[X]:p(X)",
        "f(a)=b",
        "selected(statement)",
        "p@a",
    ],
)
def test_complete_outer_wrappers_cannot_hide_any_tptp_formula_exposure(formula):
    class_id = "mml:v1:theorem:ARTICLE:7"
    digest = holdout.statement_digest("atp", formula)
    index = holdout.ExposureIndex(
        selected_class_ids=frozenset({class_id}),
        members_by_class={class_id: {"atp": frozenset({"t7_article"})}},
        statement_classes_by_representation={
            "mizar": {},
            "atp": {digest: frozenset({class_id})},
        },
        canonical_statement_classes_by_representation={
            "mizar": {},
            "atp": {formula: frozenset({class_id})},
        },
    )
    row = _atp_row(
        f"wrapped-{formula}",
        "enigma:t3_other",
        {"other": "q(a)"},
        formula=f"(({formula}))",
    )

    exposure = holdout.classify_exposure(row, shard="enigma", index=index)

    assert exposure.visible_target
    assert exposure.should_eval


def test_exposure_classifier_distinguishes_fact_own_and_combined_paths():
    index = _selected_index()
    fact_only = _mizar_row(
        "fact",
        "OTHER:1",
        {"ARTICLE:7": "selected statement"},
    )
    own_only = _mizar_row(
        "own",
        "ARTICLE:7",
        {"OTHER:2": "other statement"},
        goal="selected statement",
    )
    combined = _mizar_row(
        "combined",
        "ARTICLE:7",
        {"ARTICLE:7": "selected statement"},
        goal="selected statement",
    )

    fact = holdout.classify_exposure(fact_only, shard="mizar", index=index)
    own = holdout.classify_exposure(own_only, shard="mizar", index=index)
    both = holdout.classify_exposure(combined, shard="mizar", index=index)

    assert fact.direct_native_citation and not fact.own_theorem
    assert own.own_theorem and not own.direct_native_citation
    assert both.direct_native_citation and both.own_theorem
    assert fact.should_eval and own.should_eval and both.should_eval


def test_exposure_classifier_covers_alias_local_input_proof_and_visible_target():
    index = _selected_index()
    alias = _atp_row(
        "alias",
        "enigma:t1_other",
        {"unmapped_alias": "selected(statement)"},
    )
    local = _atp_row(
        "local",
        "enigma:t2_other",
        {"other": "q(a)"},
        local_inputs={"local_alias": "selected(statement)"},
    )
    proof = _atp_row(
        "proof",
        "enigma:t3_other",
        {"other": "q(a)"},
        formula="selected(statement)",
    )
    visible_target = _mizar_row(
        "target",
        "OTHER:4",
        {"OTHER:5": "other"},
        target="selected statement",
    )

    assert holdout.classify_exposure(alias, shard="enigma", index=index).statement_alias
    assert holdout.classify_exposure(local, shard="enigma", index=index).statement_alias
    assert holdout.classify_exposure(proof, shard="enigma", index=index).visible_target
    assert holdout.classify_exposure(
        visible_target, shard="mizar", index=index
    ).visible_target


def test_exposure_aliases_never_compare_mizar_text_to_tptp_text():
    index = _selected_index()
    row = _atp_row(
        "cross-language-text",
        "enigma:t1_other",
        {"other": "q(a)"},
        local_inputs={"same_text": "selected statement"},
    )

    exposure = holdout.classify_exposure(row, shard="enigma", index=index)

    assert not exposure.statement_alias
    assert not exposure.visible_target
    assert not exposure.should_eval


def test_mapped_cross_representation_path_is_reported():
    exposure = holdout.classify_exposure(
        _atp_row(
            "cross",
            "enigma:t1_other",
            {"t7_article": "selected(statement)"},
        ),
        shard="enigma",
        index=_selected_index(),
    )

    assert exposure.direct_native_citation
    assert exposure.mapped_cross_representation


def test_draw_is_exactly_1000_and_manifest_is_authoritative():
    facts = {"ARTICLE:7": "selected statement", **_mizar_fillers()}
    plan = _plan({"mizar": [_mizar_row("pool", "OTHER:1", facts)]})
    manifest = plan.manifest

    assert manifest["schema_version"] == holdout.MANIFEST_SCHEMA_VERSION
    assert manifest["seed"] == 20260801
    assert manifest["requested_classes"] == 1000
    assert manifest["actual_classes"] == 1000
    assert len(manifest["class_records"]) == 1000
    assert [item["shard"] for item in manifest["ordered_inputs"]] == list(SHARDS)
    assert holdout.verify_manifest_root(manifest)
    assert set(plan.compatibility_projections) == {"mizar", "atp"}
    for projection in plan.compatibility_projections.values():
        assert (
            projection["authoritative_manifest_root_sha256"]
            == (manifest["manifest_root_sha256"])
        )


def test_insufficient_tail_is_refused_instead_of_downsampling():
    facts = _mizar_fillers(999)

    with pytest.raises(holdout.HoldoutError, match="insufficient tail"):
        _plan({"mizar": [_mizar_row("short", "OTHER:1", facts)]})


def test_exact_atp_duplicates_are_removed_before_pooled_counts():
    facts = {"t7_article": "selected(statement)", **_atp_fillers()}
    first = _atp_row("first", "prf2:t100_source", facts)
    duplicate = _atp_row(
        "duplicate", "enigma:t100_source#2", dict(reversed(facts.items()))
    )
    duplicate["proof_steps"] = json.loads(json.dumps(first["proof_steps"]))
    duplicate["target"] = first["target"]
    second_citation = _atp_row(
        "second",
        "prf2:t101_source",
        {"t7_article": "selected(statement)"},
        goal="different(goal)",
    )

    plan = _plan({"prf2": [first, second_citation], "enigma": [duplicate]})
    selected = {record["class_id"]: record for record in plan.manifest["class_records"]}
    target = selected["mml:v1:theorem:ARTICLE:7"]

    assert target["selected_tail_row_citations"] == 2
    assert plan.rows["enigma"][0].disposition == "drop"
    assert plan.rows["enigma"][0].drop_reason == "exact_atp_duplicate"
    assert plan.manifest["atp_deduplication"]["duplicates_by_shard"] == {
        "prf2": 0,
        "enigma": 1,
    }


def test_direct_mizar_covered_thproof_trajectory_is_dropped_before_counting_and_written_natively(
    tmp_path,
):
    pool = _mizar_row("pool", "POOL:1", _mizar_fillers(1000))
    direct = _mizar_row(
        "direct",
        "OVERLAP:1",
        {"DIRECT_FACT:1": "direct fact"},
        goal="shared goal",
        target="shared proof",
    )
    duplicate = _mizar_row(
        "duplicate",
        "OVERLAP:1",
        {"DUPLICATE_ONLY:1": "must not be counted"},
        goal="shared goal",
        target="shared proof",
    )
    thproof_only = _mizar_row(
        "thproof-only",
        "THPROOF_ONLY:1",
        {"THPROOF_FACT:1": "legitimate thproof-only fact"},
        goal="thproof-only goal",
        target="thproof-only proof",
    )
    rows = {
        "mizar": [pool, direct],
        "thproofs": [duplicate, thproof_only],
    }
    sources = _sources(rows)

    plan = _plan(rows, sources=sources)

    duplicate_route, thproof_only_route = plan.rows["thproofs"]
    assert duplicate_route.disposition == "drop"
    assert duplicate_route.drop_reason == "direct_mizar_trajectory_duplicate"
    assert thproof_only_route.drop_reason is None
    assert thproof_only_route.disposition in {"train", "eval"}
    assert plan.manifest["tail_classes_available"] == 1002
    assert plan.manifest["drop_reason_counts"][
        "direct_mizar_trajectory_duplicate"
    ] == 1
    dedup = plan.manifest["mizar_thproofs_deduplication"]
    assert dedup["priority"] == ["mizar", "thproofs"]
    assert dedup["direct_mizar_trajectories"] == 2
    assert dedup["thproofs_trajectories"] == 2
    assert dedup["thproofs_only_trajectories"] == 1
    assert dedup["duplicates_by_shard"] == {"mizar": 0, "thproofs": 1}
    assert dedup["duplicates_total"] == 1
    assert dedup["duplicate_route_root_sha256"] == (
        holdout.mizar_thproofs_duplicate_route_root(plan.manifest["row_routes"])
    )
    assert (
        sum(
            plan.manifest["partition_projections"]["by_shard"]["thproofs"][
                disposition
            ]["rows"]
            for disposition in ("train", "eval", "drop")
        )
        == 2
    )

    output = tmp_path / "partition"
    holdout.write_partition_atomically(plan, sources=sources, output=output)
    assert (output / "dropped" / "thproofs.jsonl").read_bytes() == _line(duplicate)
    surviving_path = (
        output
        / ("eval" if thproof_only_route.disposition == "eval" else "shards")
        / "thproofs.jsonl"
    )
    assert _line(thproof_only) in surviving_path.read_bytes()
    drop_records = [
        json.loads(line)
        for line in (output / "sidecars" / "drop_reasons.jsonl")
        .read_text()
        .splitlines()
    ]
    assert next(record for record in drop_records if record["row_id"] == "duplicate")[
        "reason"
    ] == "direct_mizar_trajectory_duplicate"


def test_direct_mizar_thproof_theorem_conflict_is_refused():
    pool = _mizar_row("pool", "POOL:1", _mizar_fillers(1000))
    direct = _mizar_row(
        "direct",
        "OVERLAP:1",
        {"DIRECT_FACT:1": "direct fact"},
        goal="shared goal",
        target="direct proof",
    )
    conflicting = _mizar_row(
        "conflicting",
        "OVERLAP:1",
        {"THPROOF_FACT:1": "thproof fact"},
        goal="shared goal",
        target="different proof",
    )

    with pytest.raises(holdout.HoldoutError, match="direct Mizar trajectory"):
        _plan({"mizar": [pool, direct], "thproofs": [conflicting]})


def test_overlength_rows_are_dropped_before_class_counting():
    valid = _mizar_row(
        "valid",
        "OTHER:1",
        {"ARTICLE:7": "selected statement", **_mizar_fillers()},
    )
    long_one = _mizar_row(
        "long-1",
        "OTHER:2",
        {"ARTICLE:7": "selected statement"},
    )
    long_two = _mizar_row(
        "long-2",
        "OTHER:3",
        {"ARTICLE:7": "selected statement"},
    )
    token_counts = {
        valid["text"]: 16_384,
        long_one["text"]: 16_385,
        long_two["text"]: 16_385,
    }

    plan = _plan(
        {"mizar": [valid, long_one, long_two]},
        token_counts=token_counts,
    )
    target = next(
        record
        for record in plan.manifest["class_records"]
        if record["class_id"] == "mml:v1:theorem:ARTICLE:7"
    )

    assert target["selected_tail_row_citations"] == 1
    assert [row.drop_reason for row in plan.rows["mizar"]] == [
        None,
        "overlength",
        "overlength",
    ]


def test_each_class_is_counted_at_most_once_per_row():
    row = _atp_row(
        "aliases",
        "prf2:t1_source",
        {
            "t7_article": "selected(statement)",
            **_atp_fillers(),
        },
    )
    row["cited"].append("t7_article")

    plan = _plan({"prf2": [row]})
    target = next(
        record
        for record in plan.manifest["class_records"]
        if record["class_id"] == "mml:v1:theorem:ARTICLE:7"
    )

    assert target["selected_tail_row_citations"] == 1


@pytest.mark.parametrize("direction", ["mizar-to-atp", "atp-to-mizar"])
def test_cross_family_train_leak_is_routed_to_eval_in_both_directions(direction):
    if direction == "mizar-to-atp":
        pool = _mizar_row(
            "mizar-pool",
            "OTHER:1",
            {"ARTICLE:7": "selected statement", **_mizar_fillers()},
        )
        own = _atp_row(
            "atp-own",
            "enigma:t7_article#10",
            {"safe_unmapped": "safe"},
        )
        safe_2 = _atp_row(
            "safe-2",
            "enigma:t200_safe",
            {"safe_unmapped": "safe"},
        )
        safe_3 = _atp_row(
            "safe-3",
            "enigma:t201_safe",
            {"safe_unmapped": "safe"},
        )
        plan = _plan({"mizar": [pool], "enigma": [own, safe_2, safe_3]})
        leaked = plan.rows["enigma"][0]
    else:
        pool = _atp_row(
            "atp-pool",
            "prf2:t100_source",
            {"t7_article": "selected(statement)", **_atp_fillers()},
        )
        own = _mizar_row(
            "mizar-own",
            "ARTICLE:7",
            {"SAFE:1": "safe"},
            goal="selected statement",
        )
        safe_2 = _mizar_row("safe-2", "SAFE:2", {"SAFE:1": "safe"})
        safe_3 = _mizar_row("safe-3", "SAFE:3", {"SAFE:1": "safe"})
        plan = _plan({"prf2": [pool], "mizar": [own, safe_2, safe_3]})
        leaked = plan.rows["mizar"][0]

    assert leaked.disposition == "eval"
    assert leaked.exposure.own_theorem
    assert leaked.exposure.mapped_cross_representation


def test_statement_disagreements_are_explicit_drops():
    filler = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    first = _atp_row(
        "first",
        "prf2:t1_source",
        {"stable_unmapped": "p(a)"},
    )
    second = _atp_row(
        "second",
        "enigma:t2_source",
        {"stable_unmapped": "q(a)"},
    )

    plan = _plan({"mizar": [filler], "prf2": [first], "enigma": [second]})

    assert plan.rows["prf2"][0].drop_reason == "statement_disagreement"
    assert plan.rows["enigma"][0].drop_reason == "statement_disagreement"


def test_mapped_native_disagreements_are_explicit_drops():
    filler = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    first = _atp_row("first", "prf2:t1_source", {"t7_article": "p(a)"})
    second = _atp_row("second", "enigma:t2_source", {"t7_article": "q(a)"})

    plan = _plan({"mizar": [filler], "prf2": [first], "enigma": [second]})

    assert plan.rows["prf2"][0].drop_reason == "statement_disagreement"
    assert plan.rows["enigma"][0].drop_reason == "statement_disagreement"


def test_bookkeeping_names_in_global_facts_are_refused():
    row = _atp_row("bad", "prf2:t1_source", {"cc4_generated": "p(a)"})

    with pytest.raises(holdout.HoldoutError, match="bookkeeping"):
        _plan({"prf2": [row]})


def test_atp_schema_drift_is_refused():
    row = _atp_row("bad", "prf2:t1_source", {"t1_article": "p(a)"})
    row["schema_version"] = "atp-v1"

    with pytest.raises(holdout.HoldoutError, match="atp-v2"):
        _plan({"prf2": [row]})


def test_mapping_policy_drift_is_refused():
    pins = holdout.current_policy_pins()
    pins = replace(pins, mapping_sha256="0" * 64)

    with pytest.raises(holdout.HoldoutError, match="mapping policy"):
        _plan({}, policy_pins=pins)


def test_source_hash_drift_is_refused():
    sources = _sources({})
    sources["mizar"] = replace(
        sources["mizar"],
        expected_input_sha256="0" * 64,
    )

    with pytest.raises(holdout.HoldoutError, match="input SHA-256"):
        _plan({}, sources=sources)


def test_source_provenance_gap_is_refused():
    sources = _sources({})
    sources["mizar"] = replace(sources["mizar"], source_snapshots=())

    with pytest.raises(holdout.HoldoutError, match="source snapshot"):
        _plan({}, sources=sources)


def test_tokenizer_hash_drift_is_refused():
    seal = holdout.approved_tokenizer_seal()
    seal["behavior_digest"] = "0" * 64
    tokenizer = holdout.TokenizerSeam(
        seal=seal,
        count_text_plus_eos=lambda text: 32,
    )

    with pytest.raises(holdout.HoldoutError, match="tokenizer"):
        sources = _sources({})
        holdout.plan_semantic_holdout(
            sources,
            tokenizer=tokenizer,
            policy_pins=holdout.current_policy_pins(),
            source_policy=_source_policy(sources),
        )


def test_source_iteration_order_cannot_change_the_result():
    facts = {"ARTICLE:7": "selected statement", **_mizar_fillers()}
    rows = {"mizar": [_mizar_row("pool", "OTHER:1", facts)]}
    sources = _sources(rows)
    reversed_sources = dict(reversed(list(sources.items())))

    first = _plan(rows, sources=sources)
    second = _plan(rows, sources=reversed_sources)

    assert first.manifest == second.manifest
    assert first.rows == second.rows


def test_atomic_writer_preserves_native_bytes_and_uses_sidecars(tmp_path):
    row = _mizar_row(
        "pool",
        "OTHER:1",
        {"ARTICLE:7": "selected statement", **_mizar_fillers()},
    )
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    output = tmp_path / "partition"

    holdout.write_partition_atomically(plan, sources=sources, output=output)

    assert (output / "eval" / "mizar.jsonl").read_bytes() == _line(row)
    assert (output / "shards" / "mizar.jsonl").read_bytes() == b""
    assert (output / "dropped" / "mizar.jsonl").read_bytes() == b""
    assert json.loads((output / "heldout" / "mml.json").read_text()) == plan.manifest
    exposure = [
        json.loads(line)
        for line in (output / "sidecars" / "eval_exposure.jsonl")
        .read_text()
        .splitlines()
    ]
    assert exposure[0]["row_id"] == "pool"
    assert exposure[0]["paths"]["direct_native_citation"]
    assert "holdout_exposure" not in json.loads(
        (output / "eval" / "mizar.jsonl").read_text()
    )
    for projection in ("mizar", "atp"):
        payload = json.loads((output / "heldout" / f"{projection}.json").read_text())
        assert payload == plan.compatibility_projections[projection]


def test_writer_refuses_changed_sources_before_publication(tmp_path):
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    changed = dict(sources)
    changed_line = _line(_mizar_row("changed", "OTHER:2", _mizar_fillers(1000)))
    changed["mizar"] = replace(changed["mizar"], lines=(changed_line,))

    with pytest.raises(holdout.HoldoutError, match="changed after planning"):
        holdout.write_partition_atomically(
            plan,
            sources=changed,
            output=tmp_path / "partition",
        )

    assert not (tmp_path / "partition").exists()


def test_writer_refuses_manifest_or_projection_hash_drift(tmp_path):
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    sources = _sources({"mizar": [row]})
    manifest_drift = _plan({"mizar": [row]}, sources=sources)
    manifest_drift.manifest["seed"] = 0

    with pytest.raises(holdout.HoldoutError, match="manifest root"):
        holdout.write_partition_atomically(
            manifest_drift,
            sources=sources,
            output=tmp_path / "manifest-drift",
        )

    projection_drift = _plan({"mizar": [row]}, sources=sources)
    projection_drift.compatibility_projections["mizar"]["facts"].append("DRIFT:1")

    with pytest.raises(holdout.HoldoutError, match="compatibility projection"):
        holdout.write_partition_atomically(
            projection_drift,
            sources=sources,
            output=tmp_path / "projection-drift",
        )


def test_manifest_binds_every_row_route_and_rejects_eval_to_train_mutation(tmp_path):
    row = _mizar_row(
        "pool",
        "OTHER:1",
        {"ARTICLE:7": "selected statement", **_mizar_fillers()},
    )
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    route = plan.manifest["row_routes"]["mizar"][0]

    assert route == {
        "disposition": "eval",
        "drop_reason": None,
        "exposure_sidecar_sha256": plan.rows["mizar"][0].exposure_sidecar_sha256,
        "line_number": 1,
        "native_row_sha256": holdout.digest_lines((_line(row),)),
        "row_id": "pool",
        "text_plus_eos_tokens": 32,
    }
    assert plan.manifest["route_plan_root_sha256"] == holdout.route_plan_root(
        plan.manifest["row_routes"]
    )

    plan.rows["mizar"] = (
        replace(
            plan.rows["mizar"][0],
            disposition="train",
            exposure=None,
            exposure_sidecar_sha256=None,
        ),
    )
    with pytest.raises(holdout.HoldoutError, match="route plan"):
        holdout.write_partition_atomically(
            plan,
            sources=sources,
            output=tmp_path / "hostile-route",
        )
    assert not (tmp_path / "hostile-route").exists()


def test_manifest_root_covers_route_records():
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    plan = _plan({"mizar": [row]})

    plan.manifest["row_routes"]["mizar"][0]["disposition"] = "train"

    assert not holdout.verify_manifest_root(plan.manifest)


def test_atp_v2_validation_rejects_formula_only_step():
    row = _atp_row("bad", "prf2:t1_source", {"t1_article": "p(a)"})
    row["proof_steps"] = [{"formula": "$false"}]

    with pytest.raises(holdout.HoldoutError, match="missing ATP step fields"):
        holdout.validate_atp_v2_record(row, where="prf2:1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source-parent-mismatch", "source-parent mismatch"),
        ("duplicate-parent", "source-parent mismatch"),
        ("unresolved-parent", "unresolved parent"),
        ("target-mismatch", "target reconstruction"),
        ("non-refutation", "final semantic"),
        ("supply-overlap", "global/local supply overlap"),
    ],
)
def test_atp_v2_validation_is_fail_closed(mutation, message):
    row = _atp_row(
        "atp",
        "prf2:t1_source",
        {"t1_article": "p(a)"},
        local_inputs={"input_a": "q(a)"},
    )
    step = row["proof_steps"][0]
    if mutation == "source-parent-mismatch":
        step["parents"] = ["input_a", "theorem_x"]
    elif mutation == "duplicate-parent":
        step["parents"].insert(0, "t1_article")
        step["parent_sources"].insert(0, "t1_article")
    elif mutation == "unresolved-parent":
        step["parents"].append("missing")
        step["parent_sources"].append("missing")
        step["source"] = (
            "inference(resolution,[status(thm)],[t1_article,theorem_x,missing])"
        )
    elif mutation == "target-mismatch":
        row["target"] += "\ncorruption"
    elif mutation == "non-refutation":
        step["formula"] = "p(a)"
        row["target"] = build_atp_shard.render_target(
            [build_atp_shard.ProofStep(**step)]
        )
    elif mutation == "supply-overlap":
        row["local_inputs"]["t1_article"] = "p(a)"

    with pytest.raises(holdout.HoldoutError, match=message):
        holdout.validate_atp_v2_record(row, where="prf2:1")


def test_atp_v2_accepts_real_source_derived_nested_csr_parent_occurrences():
    nested_parent_source = (
        "inference ( csr , [ status ( thm ) ] , [ inference ( csr , "
        "[ status ( thm ) ] , [ inference ( csr , [ status ( thm ) ] , "
        "[ c_0_23 , c_0_24 ] ) , c_0_24 ] ) , c_0_24 ] )"
    )
    source = (
        "inference ( csr , [ status ( thm ) ] , [ inference ( csr , "
        "[ status ( thm ) ] , [ inference ( csr , [ status ( thm ) ] , "
        "[ inference ( csr , [ status ( thm ) ] , [ c_0_23 , c_0_24 ] ) , "
        "c_0_24 ] ) , c_0_24 ] ) , c_0_24 ] )"
    )
    parents = ["c_0_23", "c_0_24", "c_0_24", "c_0_24", "c_0_24"]
    parent_sources = [nested_parent_source, "c_0_24"]
    row = _atp_row(
        "beb090448a52",
        "prf2:l100_geomtrap",
        {"c_0_23": "p(a)", "c_0_24": "q(a)"},
    )
    step = {
        "name": "c_0_35",
        "role": "plain",
        "formula": "$false",
        "rule": "csr",
        "parents": list(parents),
        "parent_sources": list(parent_sources),
        "source": source,
    }
    row["proof_steps"] = [step]
    row["target"] = build_atp_shard.render_target(
        [build_atp_shard.ProofStep(**step)]
    )

    assert build_atp_shard.source_dependencies(source) == (
        "csr",
        parent_sources,
        parents,
    )
    holdout.validate_atp_v2_record(row, where="prf2:3")
    assert step["parents"] == parents
    assert step["parent_sources"] == parent_sources
    assert step["source"] == source


def test_atp_v2_accepts_source_derived_repeated_spm_dependency_operand():
    row = _atp_row("spm-repeat", "prf2:t1_source", {"t1_article": "p(a)"})
    first = {
        "name": "step_1",
        "role": "plain",
        "formula": "q(a)",
        "rule": "resolution",
        "parents": ["t1_article", "theorem_x"],
        "parent_sources": ["t1_article", "theorem_x"],
        "source": "inference(resolution,[status(thm)],[t1_article,theorem_x])",
    }
    second = {
        "name": "step_2",
        "role": "plain",
        "formula": "$false",
        "rule": "spm",
        "parents": ["step_1", "step_1"],
        "parent_sources": ["step_1", "step_1"],
        "source": "inference(spm,[status(thm)],[step_1,step_1])",
    }
    row["proof_steps"] = [first, second]
    row["target"] = build_atp_shard.render_target(
        [build_atp_shard.ProofStep(**step) for step in row["proof_steps"]]
    )

    holdout.validate_atp_v2_record(row, where="prf2:spm")
    assert second["parents"] == ["step_1", "step_1"]


def test_atp_v2_validation_rejects_duplicate_and_late_steps():
    row = _atp_row("atp", "prf2:t1_source", {"t1_article": "p(a)"})
    first = {
        "name": "step_1",
        "role": "plain",
        "formula": "q(a)",
        "rule": "resolution",
        "parents": ["t1_article", "theorem_x"],
        "parent_sources": ["t1_article", "theorem_x"],
        "source": "inference(resolution,[status(thm)],[t1_article,theorem_x])",
    }
    second = {
        "name": "step_2",
        "role": "plain",
        "formula": "$false",
        "rule": "resolution",
        "parents": ["step_1"],
        "parent_sources": ["step_1"],
        "source": "inference(resolution,[status(thm)],[step_1])",
    }
    row["proof_steps"] = [second, first]
    row["target"] = build_atp_shard.render_target(
        [build_atp_shard.ProofStep(**step) for step in row["proof_steps"]]
    )

    with pytest.raises(holdout.HoldoutError, match="late parent"):
        holdout.validate_atp_v2_record(row, where="prf2:1")

    row["proof_steps"] = [first, {**second, "name": "step_1"}]
    row["target"] = build_atp_shard.render_target(
        [build_atp_shard.ProofStep(**step) for step in row["proof_steps"]]
    )
    with pytest.raises(holdout.HoldoutError, match="duplicate ATP step"):
        holdout.validate_atp_v2_record(row, where="prf2:1")


@pytest.mark.parametrize(
    "location",
    [
        "facts",
        "local_inputs",
        "goal",
        "proof_formula",
        "proof_source",
        "parent_source",
        "target",
    ],
)
def test_atp_v2_validation_rejects_malformed_tptp_delimiters_everywhere(location):
    row = _atp_row(
        "bad-delimiters",
        "prf2:t1_source",
        {"t1_article": "p(a)"},
        local_inputs={"local_1": "q(a)"},
    )
    malformed = "([{]})"
    if location == "facts":
        row["facts"]["t1_article"] = malformed
    elif location == "local_inputs":
        row["local_inputs"]["local_1"] = malformed
    elif location == "goal":
        row["goal"] = malformed
    elif location == "proof_formula":
        row["proof_steps"][0]["formula"] = malformed
        row["target"] = build_atp_shard.render_target(
            [build_atp_shard.ProofStep(**row["proof_steps"][0])]
        )
    elif location == "proof_source":
        row["proof_steps"][0]["source"] = malformed
    elif location == "parent_source":
        row["proof_steps"][0]["parent_sources"][0] = malformed
    else:
        row["target"] += malformed

    with pytest.raises(holdout.HoldoutError, match="malformed TPTP delimiters"):
        holdout.validate_atp_v2_record(row, where="prf2:1")


def test_planner_rejects_malformed_atp_before_tail_counting():
    malformed = _atp_row("bad", "prf2:t1_source", {"t1_article": "p(a)"})
    malformed["proof_steps"] = [{"formula": "$false"}]

    with pytest.raises(holdout.HoldoutError, match="missing ATP step fields"):
        _plan({"prf2": [malformed]})


def test_mizar_target_detects_embedded_statement_without_substring_false_positive():
    embedded = _mizar_row(
        "embedded",
        "OTHER:1",
        {"OTHER:2": "other"},
        target="thus selected statement by OTHER:2;",
    )
    near_substring = _mizar_row(
        "near",
        "OTHER:3",
        {"OTHER:4": "other"},
        target="thus selected statements by OTHER:4;",
    )
    identifier_substring = _mizar_row(
        "identifier",
        "OTHER:5",
        {"OTHER:6": "other"},
        target="thus unselected statement by OTHER:6;",
    )

    assert holdout.classify_exposure(
        embedded,
        shard="mizar",
        index=_selected_index(),
    ).visible_target
    assert not holdout.classify_exposure(
        near_substring,
        shard="mizar",
        index=_selected_index(),
    ).visible_target
    assert not holdout.classify_exposure(
        identifier_substring,
        shard="mizar",
        index=_selected_index(),
    ).visible_target


def test_duplicate_ids_are_rejected_globally_before_selection():
    first = _mizar_row("duplicate-id", "OTHER:1", {"A:1": "a"})
    second = _mizar_row("duplicate-id", "OTHER:2", {"B:1": "b"})

    with pytest.raises(holdout.HoldoutError, match="duplicate raw row id"):
        _plan({"mizar": [first], "thproofs": [second]})


def test_source_contract_binds_manifest_quality_and_schema_roots():
    facts = _mizar_fillers(1000)
    sources = _sources({"mizar": [_mizar_row("pool", "OTHER:1", facts)]})
    plan = _plan({"mizar": []}, sources=sources)
    first_input = plan.manifest["ordered_inputs"][0]

    assert first_input["source_manifest_root_sha256"] == SHA_B
    assert first_input["quality_filter_root_sha256"] == SHA_C
    assert first_input["schema_generation_root_sha256"] == SHA_D
    assert plan.manifest["source_identity_policy"]["policy_id"] == (
        "synthetic-mml-source-policy-v1"
    )
    assert plan.manifest["source_root_sha256"] == holdout.source_root(
        plan.manifest["ordered_inputs"]
    )

    policy = _source_policy(sources)
    drifted = dict(sources)
    drifted["mizar"] = replace(
        drifted["mizar"],
        quality_filter_root_sha256="0" * 64,
    )
    with pytest.raises(holdout.HoldoutError, match="quality-filter root"):
        _plan({}, sources=drifted, source_policy=policy)


def test_direct_mizar_production_source_policy_is_final_and_drift_closed():
    table = holdout.PRODUCTION_SOURCE_IDENTITY_TABLE

    assert table["mizar"]["status"] == "finalized"
    assert table["mizar"]["source_manifest_schema"] == "p3-family-source-manifest/v2"
    assert table["mizar"]["input_rows"] == 55_353
    assert table["mizar"]["input_sha256"] == (
        "54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f"
    )
    assert table["mizar"]["source_manifest_root_sha256"] == (
        "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"
    )
    assert table["mizar"]["quality_filter_root_sha256"] == (
        "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d"
    )
    assert table["mizar"]["schema_generation_root_sha256"] == (
        "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242"
    )
    assert table["mizar"]["semantic_index_sha256"] == (
        "8deb18e7ab38d7d42d852828667a7f0b8000f3141b5bad7cbd940b617f9bd835"
    )
    assert table["mizar"]["acceptance_roots"] == {
        "recovery_identity_set_sha256": (
            "048f47cf87e6eaeccf87f3aafb202236373dea000719ae221c5ee33896dad8cd"
        ),
        "recovery_identity_source_order_sha256": (
            "6d113a43ff0b0af8aae13325908d2507b9b63aadcc01d50c37d73e29549396fa"
        ),
        "recovery_source_binding_sha256": (
            "790c86db30604c5836be70e28df527bb6c1a41b30620cfaf327122db047be65c"
        ),
        "recovery_text_hash_sequence_sha256": (
            "e116af514ee5cc7fc3415d01a68ec42037206849f3b62e6bdddbfefe4637659f"
        ),
        "recovery_token_sequence_sha256": (
            "3391cb491f1e7e8ec23b7725d27ceb95b4d5d51bd5856a8e46372507102d5ca4"
        ),
    }
    assert (
        "3d1af5b3e840aca5631541b42510b35c1b15dfa988af70ce463f58c899e88714"
        in table["mizar"]["approved_tree_sha256"]
    )
    assert table["mizar"]["deduplication_root_sha256"] == (
        holdout.production_deduplication_root("mizar")
    )
    assert holdout.verify_finalized_production_source_record(
        "mizar",
        table["mizar"],
    )
    approved = holdout.production_approved_shard_source("mizar")
    source = holdout.MemoryShardSource(
        name="mizar",
        logical_path="raw/mizar.jsonl",
        lines=(),
        expected_input_sha256=approved.input_sha256,
        source_snapshots=approved.source_snapshots,
        source_manifest_root_sha256=approved.source_manifest_root_sha256,
        quality_filter_root_sha256=approved.quality_filter_root_sha256,
        schema_generation_root_sha256=approved.schema_generation_root_sha256,
    )
    holdout.validate_production_shard_source("mizar", source, test_only=False)

    with pytest.raises(holdout.HoldoutError, match="test-only"):
        holdout.validate_production_shard_source("mizar", source, test_only=True)
    with pytest.raises(holdout.HoldoutError, match="raw input"):
        holdout.validate_production_shard_source(
            "mizar",
            replace(source, expected_input_sha256="0" * 64),
            test_only=False,
        )


def test_production_source_policy_materializes_all_accepted_real_roots():
    table = holdout.PRODUCTION_SOURCE_IDENTITY_TABLE
    expected = {
        "mizar": {
            "input_rows": 55_353,
            "input_sha256": (
                "54206c1fe89d09dec7ec36c927612439b687814ba95e1086e4b09db036ad486f"
            ),
            "source_manifest_root_sha256": (
                "fa21f98fa551ae3e54b17e4e31aacebfde48c0be3ea8b99f5ff85f4ee08fb762"
            ),
            "quality_filter_root_sha256": (
                "9fb4b02b9c632d0dfdf5f8730798b25a981a7da46bc0c06f770ee3df14ee7d7d"
            ),
            "schema_generation_root_sha256": (
                "ea8deb4c5912f9b10f5da674fcd86c9f8c8b5cf521522ad70b6168a5bf554242"
            ),
        },
        "thproofs": {
            "input_rows": 50_743,
            "input_sha256": (
                "8bdb66128d6385f03d1b70d064bbc089f28c2f4f529f405d761e728080657854"
            ),
            "source_manifest_root_sha256": (
                "17d2aa537ef9cf05e9acb26573143cec2de9ec8d7ae272b4324a09becd25ff63"
            ),
            "quality_filter_root_sha256": (
                "895dc288d07504d0f6106a8641aeb4a268a86caa84dcdf1a55078939da60eb74"
            ),
            "schema_generation_root_sha256": (
                "52429daebe81fa70cde000ec1fffe1fb8af0ee9a2421af3d089ce0b14e75a62b"
            ),
        },
        "prf2": {
            "input_rows": 24_797,
            "input_sha256": (
                "fdde1aececef6de1c88cac8e17945c7a55491bd7bb527784c26099f67f63ab3d"
            ),
            "source_manifest_root_sha256": (
                "029f23d490c96521bf90e29c12581459b831d991fef597c58f81ce39abcd4fdf"
            ),
            "quality_filter_root_sha256": (
                "1d4ba91cfc3152cade7a3743d34801f654bdce1a682872d6ba35c403cdab6ab6"
            ),
            "schema_generation_root_sha256": (
                "bd0caede34f0dea92401bc9a306ae1ef82f81a26ed3bedeecd6e6a4a653a1a60"
            ),
        },
        "enigma": {
            "input_rows": 29_166,
            "input_sha256": (
                "7fddf832938404f6e76f33fae06a6e8731b923cde65d9c32795288ac4250a3f7"
            ),
            "source_manifest_root_sha256": (
                "c33d5b87696e56276f8c2eb81fd1acb274ab766308e60a96e6ee999d6d76fd2e"
            ),
            "quality_filter_root_sha256": (
                "cdd95f8ab0314ec2b11b44273fd412c7445e2ab61e73d43a33e84c158b7c4ce8"
            ),
            "schema_generation_root_sha256": (
                "bd0caede34f0dea92401bc9a306ae1ef82f81a26ed3bedeecd6e6a4a653a1a60"
            ),
        },
    }

    policy = holdout.production_source_policy()

    assert isinstance(policy, holdout.SourceIdentityPolicy)
    assert not policy.test_only
    assert set(policy.shards) == set(SHARDS)
    for shard, roots in expected.items():
        assert table[shard]["status"] == "finalized"
        assert holdout.verify_finalized_production_source_record(shard, table[shard])
        assert {key: table[shard][key] for key in roots} == roots
        approved = policy.shards[shard]
        assert approved.input_sha256 == roots["input_sha256"]
        assert (
            approved.source_manifest_root_sha256
            == roots["source_manifest_root_sha256"]
        )
        assert approved.quality_filter_root_sha256 == roots[
            "quality_filter_root_sha256"
        ]
        assert approved.schema_generation_root_sha256 == roots[
            "schema_generation_root_sha256"
        ]
    assert table["enigma"]["acceptance_roots"] == {
        "acceptance_audit_sha256": (
            "b3189c4a8589beb7d066c27246adf3257adadccb0729fecd2732a4c10165c5c8"
        ),
        "alternative_proof_policy_root_sha256": (
            "646a531f28aeb7c5a8ef78de97e8000e866816508a6e8413b51474d6c5cf4669"
        ),
        "selected_occurrence_root_sha256": (
            "587b81fa01ba84b4245745f10e83c89d9fa46771558c907148a00864afdee7e0"
        ),
    }


@pytest.mark.parametrize(
    ("shard", "root_name"),
    [
        ("mizar", "recovery_identity_set_sha256"),
        ("mizar", "recovery_source_binding_sha256"),
        ("enigma", "acceptance_audit_sha256"),
        ("enigma", "alternative_proof_policy_root_sha256"),
        ("enigma", "selected_occurrence_root_sha256"),
    ],
)
def test_production_source_policy_rejects_acceptance_root_mutation(
    monkeypatch,
    shard,
    root_name,
):
    table = {
        name: {
            **record,
            "acceptance_roots": dict(record.get("acceptance_roots", {})),
        }
        for name, record in holdout.PRODUCTION_SOURCE_IDENTITY_TABLE.items()
    }
    table[shard]["acceptance_roots"][root_name] = _sha(
        f"drift:{shard}:{root_name}"
    )
    monkeypatch.setattr(holdout, "PRODUCTION_SOURCE_IDENTITY_TABLE", table)

    with pytest.raises(holdout.HoldoutError, match="finalization root drift"):
        holdout.production_source_policy()


def test_production_source_policy_rejects_finalized_root_drift_before_pending_siblings(
    monkeypatch,
):
    table = {
        shard: dict(record)
        for shard, record in holdout.PRODUCTION_SOURCE_IDENTITY_TABLE.items()
    }
    table["mizar"]["quality_filter_root_sha256"] = _sha("drifted-direct-mizar")
    monkeypatch.setattr(holdout, "PRODUCTION_SOURCE_IDENTITY_TABLE", table)

    with pytest.raises(holdout.HoldoutError, match="finalization root drift"):
        holdout.production_source_policy()


def test_record_finalization_seam_is_deterministic_and_returns_typed_policy(
    tmp_path,
    monkeypatch,
):
    first = _finalized_policy_records(tmp_path / "first")
    second = _finalized_policy_records(tmp_path / "second")
    assert first == second

    monkeypatch.setattr(holdout, "PRODUCTION_SOURCE_IDENTITY_TABLE", first)
    policy = holdout.production_source_policy()

    assert isinstance(policy, holdout.SourceIdentityPolicy)
    assert not policy.test_only
    assert set(policy.shards) == set(SHARDS)
    assert policy.deduplication_roots == {
        shard: holdout.production_deduplication_root(shard) for shard in SHARDS
    }
    assert all(
        isinstance(source, holdout.ApprovedShardSource)
        for source in policy.shards.values()
    )


@pytest.mark.parametrize(
    "root_field",
    [
        "input_sha256",
        "source_manifest_root_sha256",
        "quality_filter_root_sha256",
        "schema_generation_root_sha256",
        "deduplication_root_sha256",
        "finalization_root_sha256",
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "placeholder", "drift"])
@pytest.mark.parametrize("shard", SHARDS)
def test_production_source_policy_rejects_each_unfinished_or_drifted_root(
    tmp_path,
    monkeypatch,
    shard,
    root_field,
    mutation,
):
    records = _finalized_policy_records(tmp_path)
    damaged = dict(records[shard])
    if mutation == "missing":
        damaged.pop(root_field)
    elif mutation == "placeholder":
        damaged[root_field] = "0" * 64
    else:
        damaged[root_field] = _sha(f"drift:{root_field}")
    records[shard] = damaged
    monkeypatch.setattr(holdout, "PRODUCTION_SOURCE_IDENTITY_TABLE", records)

    with pytest.raises(
        holdout.HoldoutError,
        match="incomplete|missing|placeholder|drift|SHA-256",
    ):
        holdout.production_source_policy()


@pytest.mark.parametrize("shard", SHARDS)
@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("input_rows", 2),
        ("source_manifest_schema", "drifted-source-manifest-v1"),
        (
            "source_snapshots",
            [{"reference": "fixture://drifted", "sha256": SHA_A}],
        ),
    ],
)
def test_production_source_policy_rejects_each_nonroot_input_mutation(
    tmp_path,
    monkeypatch,
    shard,
    field,
    replacement,
):
    records = _finalized_policy_records(tmp_path)
    records[shard] = {**records[shard], field: replacement}
    monkeypatch.setattr(holdout, "PRODUCTION_SOURCE_IDENTITY_TABLE", records)

    with pytest.raises(holdout.HoldoutError, match="drift"):
        holdout.production_source_policy()


@pytest.mark.parametrize(
    ("observed_shard", "projection", "expected_alias"),
    [
        ("mizar", "atp", "t7_article"),
        ("prf2", "mizar", "ARTICLE:7"),
    ],
)
def test_reverse_canonical_aliases_exist_without_observed_sibling(
    observed_shard,
    projection,
    expected_alias,
):
    if observed_shard == "mizar":
        row = _mizar_row(
            "pool",
            "OTHER:1",
            {"ARTICLE:7": "selected statement", **_mizar_fillers()},
        )
    else:
        row = _atp_row(
            "pool",
            "prf2:t100_source",
            {"t7_article": "selected(statement)", **_atp_fillers()},
        )

    plan = _plan({observed_shard: [row]})
    projected = plan.compatibility_projections[projection]

    assert expected_alias in projected["facts"]
    assert projected["shards"]
    assert projected["mapping"]["version"] == holdout.MAPPING_VERSION
    assert projected["canonicalization"]
    assert projected["classes"]
    assert projected["source_root_sha256"] == plan.manifest["source_root_sha256"]
    assert (
        projected["tokenizer_root_sha256"] == (plan.manifest["tokenizer_root_sha256"])
    )


def test_class_and_projection_records_bind_route_totals_and_hashes():
    plan = _plan(
        {
            "mizar": [
                _mizar_row(
                    "pool",
                    "OTHER:1",
                    {"ARTICLE:7": "selected statement", **_mizar_fillers()},
                )
            ]
        }
    )
    class_record = next(
        record
        for record in plan.manifest["class_records"]
        if record["class_id"] == "mml:v1:theorem:ARTICLE:7"
    )

    assert class_record["route_totals"]["eval"] == 1
    assert class_record["route_root_sha256"]
    for projection in plan.compatibility_projections.values():
        assert (
            projection["route_totals"]
            == (plan.manifest["partition_projections"]["totals"])
        )
        assert (
            projection["route_plan_root_sha256"]
            == (plan.manifest["route_plan_root_sha256"])
        )


def test_synthetic_loader_validates_authoritative_and_projection_contract(tmp_path):
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    output = tmp_path / "partition"
    holdout.write_partition_atomically(plan, sources=sources, output=output)

    contract = holdout.load_holdout_contract(output, production=False)

    assert (
        contract.manifest["manifest_root_sha256"]
        == (plan.manifest["manifest_root_sha256"])
    )
    assert contract.selected_class_ids == frozenset(
        record["class_id"] for record in plan.manifest["class_records"]
    )
    assert contract.projection("mizar")["family"] == "mizar"
    assert contract.projection("atp")["family"] == "atp"


def test_stale_v2_projection_can_be_refreshed_from_authoritative_manifest(tmp_path):
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    output = tmp_path / "partition"
    holdout.write_partition_atomically(plan, sources=sources, output=output)
    stale = {"schema_version": "legacy-v2", "facts": ["wrong"]}
    (output / "heldout" / "mizar.json").write_text(json.dumps(stale))

    holdout.refresh_compatibility_projections(output)
    contract = holdout.load_holdout_contract(output, production=False)

    assert contract.projection("mizar") == plan.compatibility_projections["mizar"]


def _published_mixed_partition(tmp_path):
    pool = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    train_rows = [
        _mizar_row(f"train-{index}", f"SAFE:{index + 2}", {"SAFE:1": "safe"})
        for index in range(3)
    ]
    dropped = _mizar_row("dropped", "DROP:2", {"DROP:1": "drop"})
    rows = {"mizar": [pool, *train_rows, dropped]}
    sources = _sources(rows)
    plan = _plan(
        rows,
        sources=sources,
        token_counts={dropped["text"]: 16_385},
    )
    output = tmp_path / "partition"
    holdout.write_partition_atomically(plan, sources=sources, output=output)
    return output, plan


def _rewrite_authoritative(output, mutate):
    manifest_path = output / "heldout" / "mml.json"
    manifest = json.loads(manifest_path.read_text())
    mutate(manifest)
    body = dict(manifest)
    body.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = holdout._manifest_root(body)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    projections = holdout.derive_compatibility_projections(manifest)
    for family, projection in projections.items():
        (output / "heldout" / f"{family}.json").write_text(
            json.dumps(projection, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        )


def test_manifest_binds_complete_publication_inventory_and_loader_schema(tmp_path):
    output, plan = _published_mixed_partition(tmp_path)
    inventory = {
        record["path"]: record for record in plan.manifest["artifact_inventory"]
    }
    expected = {
        *(f"shards/{shard}.jsonl" for shard in SHARDS),
        *(f"eval/{shard}.jsonl" for shard in SHARDS),
        *(f"dropped/{shard}.jsonl" for shard in SHARDS),
        "sidecars/eval_exposure.jsonl",
        "sidecars/drop_reasons.jsonl",
        "heldout/mml.json",
        "heldout/mizar.json",
        "heldout/atp.json",
    }

    assert set(inventory) == expected
    assert plan.manifest["loader_contract"]["schema_version"] == (
        holdout.LOADER_CONTRACT_SCHEMA_VERSION
    )
    assert plan.manifest["artifact_inventory_root_sha256"] == (
        holdout.artifact_inventory_root(plan.manifest["artifact_inventory"])
    )
    for path, record in inventory.items():
        assert record["schema"]
        if not path.startswith("heldout/"):
            assert len(record["sha256"]) == 64
            assert record["bytes"] >= 0
            assert record["rows"] >= 0

    contract = holdout.load_holdout_contract(output, production=False)
    assert set(contract.artifacts) == expected


def test_manifest_and_projections_bind_exact_complete_contract_tuple():
    plan = _plan({"mizar": [_mizar_row("pool", "OTHER:1", _mizar_fillers(1000))]})
    manifest = plan.manifest
    contract = manifest["contract_tuple"]

    assert contract == holdout.canonical_contract_tuple(
        manifest["source_identity_policy"]
    )
    assert contract["schema_version"] == holdout.CONTRACT_TUPLE_SCHEMA_VERSION
    assert contract["edullm_data_commit"] == (
        "38bf831a6c3f445e394784018441fd59288b876c"
    )
    assert (
        holdout.MANIFEST_SCHEMA_VERSION,
        holdout.POLICY_VERSION,
        holdout.MAPPING_VERSION,
        holdout.STATEMENT_HASH_VERSION,
        holdout.ATP_DEDUPLICATION_POLICY,
        holdout.COMPATIBILITY_SCHEMA_VERSION,
        holdout.SOURCE_IDENTITY_POLICY_VERSION,
        holdout.LOADER_CONTRACT_SCHEMA_VERSION,
        holdout.CANONICALIZATION_CONTRACT_VERSION,
        holdout.CONTRACT_TUPLE_SCHEMA_VERSION,
    ) == (
        "mml-semantic-holdout-manifest-v7",
        "mml-semantic-holdout-policy-v8",
        "mml-semantic-name-map-v2",
        "mml-semantic-statement-v4",
        "mml-atp-exact-structured-v5",
        "mml-semantic-holdout-compat-v7",
        "mml-source-identity-policy-v3",
        "mml-semantic-holdout-loader-v7",
        "mml-semantic-canonicalization-v4",
        "mml-semantic-holdout-contract-tuple-v5",
    )
    assert set(contract["components"]) == {
        "manifest",
        "loader",
        "compatibility",
        "policy",
        "mapping",
        "statement_hash",
        "atp_deduplication",
        "mizar_thproofs_deduplication",
        "enigma_variant_grouping",
        "source_policy",
        "canonicalization",
    }
    assert all(
        len(component["sha256"]) == 64 for component in contract["components"].values()
    )
    assert manifest["contract_tuple_sha256"] == holdout._json_sha256(contract)
    for projection in plan.compatibility_projections.values():
        assert (
            projection["contract_tuple_sha256"] == (manifest["contract_tuple_sha256"])
        )


@pytest.mark.parametrize(
    "component",
    [
        "manifest",
        "loader",
        "compatibility",
        "policy",
        "mapping",
        "statement_hash",
        "atp_deduplication",
        "mizar_thproofs_deduplication",
        "enigma_variant_grouping",
        "source_policy",
        "canonicalization",
        "edullm_data_commit",
        "contract_tuple_schema",
    ],
)
def test_loader_rejects_resigned_contract_downgrade_or_arbitrary_mutation(
    tmp_path,
    component,
):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        contract = manifest["contract_tuple"]
        if component == "edullm_data_commit":
            contract["edullm_data_commit"] = "e0984c88b7c5-unrelated-0.8.0"
        elif component == "contract_tuple_schema":
            contract["schema_version"] = "downgraded-contract-tuple"
        else:
            contract["components"][component] = {
                "version": "downgraded-or-arbitrary",
                "sha256": "0" * 64,
            }
        manifest["contract_tuple_sha256"] = holdout._json_sha256(contract)

    _rewrite_authoritative(output, mutate)

    with pytest.raises(holdout.HoldoutError, match="contract tuple"):
        holdout.load_holdout_contract(output, production=False)


@pytest.mark.parametrize(
    ("component", "old_version"),
    [
        ("policy", "mml-semantic-holdout-policy-v6"),
        ("compatibility", "mml-semantic-holdout-compat-v5"),
        ("loader", "mml-semantic-holdout-loader-v5"),
        ("contract_tuple_schema", "mml-semantic-holdout-contract-tuple-v3"),
    ],
)
def test_loader_rejects_resigned_repeated_parent_rule_downgrade(
    tmp_path,
    component,
    old_version,
):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        contract = manifest["contract_tuple"]
        if component == "contract_tuple_schema":
            contract["schema_version"] = old_version
        else:
            contract["components"][component]["version"] = old_version
        manifest["contract_tuple_sha256"] = holdout._json_sha256(contract)

    _rewrite_authoritative(output, mutate)

    with pytest.raises(holdout.HoldoutError, match="contract tuple"):
        holdout.load_holdout_contract(output, production=False)


def test_writer_and_loader_reject_reordered_inventory_with_recomputed_roots(tmp_path):
    row = _mizar_row("pool", "OTHER:1", _mizar_fillers(1000))
    sources = _sources({"mizar": [row]})
    plan = _plan({"mizar": [row]}, sources=sources)
    manifest = json.loads(json.dumps(plan.manifest))
    manifest["artifact_inventory"][0], manifest["artifact_inventory"][1] = (
        manifest["artifact_inventory"][1],
        manifest["artifact_inventory"][0],
    )
    manifest["artifact_inventory_root_sha256"] = holdout._json_sha256(
        manifest["artifact_inventory"]
    )
    body = dict(manifest)
    body.pop("manifest_root_sha256", None)
    manifest["manifest_root_sha256"] = holdout._manifest_root(body)
    hostile_plan = replace(
        plan,
        manifest=manifest,
        compatibility_projections=holdout.derive_compatibility_projections(manifest),
        sealed_manifest_root_sha256=manifest["manifest_root_sha256"],
    )

    with pytest.raises(holdout.HoldoutError, match="canonical path order|sorted"):
        holdout.write_partition_atomically(
            hostile_plan,
            sources=sources,
            output=tmp_path / "writer-reordered",
        )
    assert not (tmp_path / "writer-reordered").exists()

    output, _ = _published_mixed_partition(tmp_path)

    def reorder(published_manifest):
        inventory = published_manifest["artifact_inventory"]
        inventory[0], inventory[1] = inventory[1], inventory[0]
        published_manifest["artifact_inventory_root_sha256"] = holdout._json_sha256(
            inventory
        )

    _rewrite_authoritative(output, reorder)
    with pytest.raises(holdout.HoldoutError, match="canonical path order|sorted"):
        holdout.load_holdout_contract(output, production=False)


@pytest.mark.parametrize("mutation", ["duplicate", "non-normalized"])
def test_loader_rejects_duplicate_or_non_normalized_inventory_paths(
    tmp_path,
    mutation,
):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        inventory = manifest["artifact_inventory"]
        if mutation == "duplicate":
            inventory.append(dict(inventory[0]))
        else:
            inventory[0]["path"] = f"./{inventory[0]['path']}"
        manifest["artifact_inventory_root_sha256"] = holdout._json_sha256(inventory)

    _rewrite_authoritative(output, mutate)
    with pytest.raises(
        holdout.HoldoutError,
        match="duplicate|normalized|canonical",
    ):
        holdout.load_holdout_contract(output, production=False)


@pytest.mark.parametrize(
    "relative_path",
    [
        "shards/mizar.jsonl",
        "eval/mizar.jsonl",
        "dropped/mizar.jsonl",
        "sidecars/eval_exposure.jsonl",
        "sidecars/drop_reasons.jsonl",
    ],
)
def test_loader_rejects_same_size_artifact_corruption(tmp_path, relative_path):
    output, _ = _published_mixed_partition(tmp_path)
    path = output / relative_path
    corrupted = bytearray(path.read_bytes())
    corrupted[len(corrupted) // 2] ^= 1
    path.write_bytes(corrupted)

    with pytest.raises(holdout.HoldoutError, match="SHA-256|artifact"):
        holdout.load_holdout_contract(output, production=False)


def test_loader_rejects_swapped_native_rows(tmp_path):
    output, _ = _published_mixed_partition(tmp_path)
    path = output / "shards" / "mizar.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    lines[0], lines[1] = lines[1], lines[0]
    path.write_bytes(b"".join(lines))

    with pytest.raises(holdout.HoldoutError, match="SHA-256|route"):
        holdout.load_holdout_contract(output, production=False)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_loader_rejects_missing_or_extra_eval_sidecars(tmp_path, mutation):
    output, _ = _published_mixed_partition(tmp_path)
    path = output / "sidecars" / "eval_exposure.jsonl"
    lines = path.read_bytes().splitlines(keepends=True)
    if mutation == "missing":
        path.write_bytes(b"")
    else:
        path.write_bytes(b"".join([*lines, lines[0]]))

    with pytest.raises(holdout.HoldoutError, match="sidecar|artifact"):
        holdout.load_holdout_contract(output, production=False)


@pytest.mark.parametrize("mutation", ["extra-file", "missing-file"])
def test_loader_requires_exact_artifact_inventory(tmp_path, mutation):
    output, _ = _published_mixed_partition(tmp_path)
    if mutation == "extra-file":
        (output / "shards" / "extra.jsonl").write_bytes(b"")
    else:
        (output / "eval" / "enigma.jsonl").unlink()

    with pytest.raises(holdout.HoldoutError, match="inventory"):
        holdout.load_holdout_contract(output, production=False)


def test_loader_reconciles_routes_sidecars_and_totals(tmp_path):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        manifest["partition_projections"]["totals"]["train"]["rows"] += 1

    _rewrite_authoritative(output, mutate)

    with pytest.raises(holdout.HoldoutError, match="totals"):
        holdout.load_holdout_contract(output, production=False)


def test_loader_rejects_resigned_mizar_thproof_dedup_accounting(tmp_path):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        manifest["mizar_thproofs_deduplication"][
            "thproofs_only_trajectories"
        ] += 1

    _rewrite_authoritative(output, mutate)

    with pytest.raises(holdout.HoldoutError, match="manifest root"):
        holdout.load_holdout_contract(output, production=False)


def test_loader_checks_route_id_against_native_bytes(tmp_path):
    output, _ = _published_mixed_partition(tmp_path)

    def mutate(manifest):
        manifest["row_routes"]["mizar"][1]["row_id"] = "wrong-id"
        manifest["route_plan_root_sha256"] = holdout.route_plan_root(
            manifest["row_routes"]
        )
        manifest["partition_projections"]["route_plan_root_sha256"] = manifest[
            "route_plan_root_sha256"
        ]
        manifest["partition_projections"]["by_shard"]["mizar"]["route_root_sha256"] = (
            holdout._json_sha256(manifest["row_routes"]["mizar"])
        )

    _rewrite_authoritative(output, mutate)

    with pytest.raises(holdout.HoldoutError, match="row id|route"):
        holdout.load_holdout_contract(output, production=False)


def test_loader_separates_test_and_production_source_policies(tmp_path):
    output, _ = _published_mixed_partition(tmp_path)

    with pytest.raises(holdout.HoldoutError, match="test-only"):
        holdout.load_holdout_contract(output)

    test_contract = holdout.load_holdout_contract(output, production=False)
    assert test_contract.test_only
    assert not test_contract.production

    def mutate(manifest):
        manifest["source_identity_policy"]["test_only"] = False
        manifest["source_identity_policy"]["injected_test_seams"] = False
        manifest["source_identity_policy"]["policy_id"] = (
            "production-mml-source-policy-v1"
        )
        manifest["loader_contract"]["publication_mode"] = "production"
        manifest["contract_tuple"] = holdout.canonical_contract_tuple(
            manifest["source_identity_policy"]
        )
        manifest["contract_tuple_sha256"] = holdout._json_sha256(
            manifest["contract_tuple"]
        )

    _rewrite_authoritative(output, mutate)
    with pytest.raises(holdout.HoldoutError, match="not approved"):
        holdout.load_holdout_contract(output, production=True)


def test_validated_contract_exposes_shared_integration_api(tmp_path):
    output, plan = _published_mixed_partition(tmp_path)

    contract = holdout.load_holdout_contract(output, production=False)

    assert isinstance(contract, holdout.ValidatedHoldoutContract)
    assert contract.authoritative_root == plan.manifest["manifest_root_sha256"]
    assert contract.family_paths["mizar"].train == output / "shards" / "mizar.jsonl"
    assert contract.family_paths["mizar"].eval == output / "eval" / "mizar.jsonl"
    assert contract.family_paths["mizar"].dropped == (
        output / "dropped" / "mizar.jsonl"
    )
    assert ("mizar", "pool") in contract.exposure_index
    assert contract.tokenizer_root_sha256 == plan.manifest["tokenizer_root_sha256"]
    assert contract.source_root_sha256 == plan.manifest["source_root_sha256"]
    assert contract.quality_filter_roots_by_shard == {shard: SHA_C for shard in SHARDS}
    assert contract.schema_generation_roots_by_shard == {
        shard: SHA_D for shard in SHARDS
    }
    assert contract.projection("atp")["facts"]
