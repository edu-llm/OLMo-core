"""Regression tests for lossless ATP proofs and family-wide holdout isolation."""

import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

build_atp_shard = importlib.import_module("build_atp_shard")
split_heldout = importlib.import_module("split_heldout")
verify_corpus = importlib.import_module("verify_corpus")


def _proof_with_parents(parent_count=7):
    inputs = "\n".join(
        f"fof(parent_{i}, axiom, p{i}(a), file('problem.p', parent_{i}))."
        for i in range(parent_count)
    )
    parents = ",".join(f"parent_{i}" for i in range(parent_count))
    return f"""
{inputs}
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)], [{parents}])).
"""


def test_atp_parser_preserves_more_than_six_parents():
    proof = build_atp_shard.parse(_proof_with_parents(), fenced=False)

    assert proof is not None
    assert proof.steps[-1].parents == [f"parent_{i}" for i in range(7)]
    assert proof.steps[-1].source == (
        "inference(resolution, [status(thm)], "
        "[parent_0,parent_1,parent_2,parent_3,parent_4,parent_5,parent_6])"
    )


def test_deep_verifier_compares_all_parents_to_tstp_source():
    proof = build_atp_shard.parse(_proof_with_parents(), fenced=False)
    assert proof is not None
    step = proof.steps[-1]
    serialized = {
        "name": step.name,
        "role": step.role,
        "formula": step.formula,
        "rule": step.rule,
        "parents": list(step.parents),
        "parent_sources": list(step.parent_sources),
        "source": step.source,
    }
    record = {
        "facts": proof.facts,
        "local_inputs": proof.local_inputs,
        "goal_name": proof.goal_name,
        "proof_steps": [serialized],
        "target": build_atp_shard.render_target(proof.steps),
    }
    assert verify_corpus.atp_record_errors(record) == []

    serialized["parents"] = serialized["parents"][:6]
    serialized["parent_sources"] = serialized["parent_sources"][:6]
    assert (
        "atp_source_parent_mismatch",
        "step 1 step_x",
    ) in verify_corpus.atp_record_errors(record)


def test_atp_builder_emits_replayable_structured_trace(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "proof_1").write_text(_proof_with_parents())
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_atp_shard.py",
            "--src",
            str(source),
            "--name",
            "prf2",
            "--out",
            str(output),
            "--heldout",
            "0",
            "--min-steps",
            "1",
        ],
    )

    assert build_atp_shard.main() == 0
    records = [
        json.loads(line)
        for line in (output / "shards" / "prf2.jsonl").read_text().splitlines()
    ]

    assert len(records) == 1
    assert records[0]["schema_version"] == "atp-v2"
    assert records[0]["source_metadata"]["schema_version"] == "atp-build-source-v2"
    assert records[0]["source_metadata"]["source_roots"]
    assert records[0]["source_metadata"]["index_roots"] == {}
    assert len(
        records[0]["source_metadata"]["source_manifest_root_sha256"]
    ) == 64
    assert len(
        records[0]["source_metadata"]["quality_filter_root_sha256"]
    ) == 64
    assert len(
        records[0]["source_metadata"]["schema_generation_root_sha256"]
    ) == 64
    assert records[0]["proof_steps"][0]["parents"] == [
        f"parent_{i}" for i in range(7)
    ]
    assert verify_corpus.atp_record_errors(records[0]) == []


def test_atp_parser_supplies_external_and_bookkeeping_inputs_locally():
    text = """
fof(global_decl, axiom, p(a), file('problem.p', global_fact)).
fof('input-1', plain, q(a), file('problem.p', 'input-1')).
fof(bookkeeping_decl, axiom, b(a), file('problem.p', dt_generated)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)],
              [global_fact, 'input-1', dt_generated, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.facts == {"global_fact": "p(a)"}
    assert proof.local_inputs == {"input-1": "q(a)", "dt_generated": "b(a)"}
    assert [step.name for step in proof.steps] == ["step_x"]
    assert proof.steps[0].parents == [
        "global_fact",
        "input-1",
        "dt_generated",
        "theorem_x",
    ]
    assert build_atp_shard.dependency_errors(proof) == []


def test_atp_dependency_closure_reports_unresolved_target_parent():
    text = """
fof(premise, axiom, p(a), file('problem.p', premise)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)], [premise, 'missing-parent'])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert build_atp_shard.dependency_errors(proof) == [
        "step_x: unresolved parent missing-parent"
    ]


def test_theorem_identity_normalizes_family_and_alternate_suffix():
    assert (
        split_heldout.normalize_theorem_identity(
            "enigma:t42_example#2", family="atp"
        )
        == "t42_example"
    )
    assert (
        split_heldout.normalize_theorem_identity(
            "prf2:t42_example#10", family="atp"
        )
        == "t42_example"
    )
    assert (
        split_heldout.normalize_theorem_identity("AFF_1:50", family="mizar")
        == "AFF_1:50"
    )
    assert (
        split_heldout.normalize_theorem_identity("set:mp2", family="metamath")
        == "mp2"
    )
    assert (
        split_heldout.normalize_theorem_identity(
            "Session:qualified", family="isabelle"
        )
        == "Session:qualified"
    )


def test_heldout_exposure_matches_exact_statement_aliases():
    held_hashes = {split_heldout.statement_hash("![X] : p(X)", family="atp")}
    alias_row = {
        "theorem": "enigma:other_theorem",
        "facts": {"different_name": " ! [ X ] : p ( X ) "},
        "local_inputs": {},
        "goal": "q",
        "proof_steps": [],
    }

    exposure = split_heldout.heldout_exposure(
        alias_row,
        held_facts={"held_name"},
        held_statement_hashes=held_hashes,
        family="atp",
    )

    assert exposure.statement_alias
    assert exposure.should_eval


def test_heldout_exposure_matches_alternate_proofs_of_held_theorem():
    for suffix in ("#2", "#10"):
        alternate = {
            "theorem": f"enigma:held_name{suffix}",
            "facts": {"other": "q"},
            "local_inputs": {},
            "goal": "q",
            "proof_steps": [],
        }

        exposure = split_heldout.heldout_exposure(
            alternate,
            held_facts={"held_name"},
            held_statement_hashes=set(),
            family="atp",
        )

        assert exposure.own_theorem
        assert exposure.should_eval


def test_family_split_removes_alternate_proofs_and_statement_aliases(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)

    def row(row_id, theorem, facts, goal):
        return {
            "id": row_id,
            "theorem": theorem,
            "facts": facts,
            "local_inputs": {},
            "goal": goal,
            "proof_steps": [],
            "target": "",
        }

    rows = [
        row("held", "enigma:source", {"held_name": "p(a)"}, "source_goal"),
        row("alternate", "enigma:held_name#2", {"alias": "p(a)"}, "p(a)"),
        *[
            row(f"alias-{i}", f"enigma:other_{i}", {"alias": "p(a)"}, "other")
            for i in range(3)
        ],
        *[
            row(f"safe-{i}", f"enigma:safe_{i}", {"safe": "q(a)"}, "safe_goal")
            for i in range(3)
        ],
    ]
    (corpus / "raw" / "enigma.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_heldout.py",
            "--corpus",
            str(corpus),
            "--family",
            "atp=enigma",
            "--heldout",
            "1",
        ],
    )

    assert split_heldout.main() == 0
    train = [
        json.loads(line)
        for line in (corpus / "shards" / "enigma.jsonl").read_text().splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (corpus / "eval" / "enigma.jsonl").read_text().splitlines()
    ]
    manifest = json.loads((corpus / "heldout" / "atp.json").read_text())

    assert {record["id"] for record in train} == {
        "safe-0",
        "safe-1",
        "safe-2",
    }
    assert {record["id"] for record in eval_rows} == {
        "held",
        "alternate",
        "alias-0",
        "alias-1",
        "alias-2",
    }
    assert manifest["facts"] == ["held_name"]
    assert manifest["statement_hashes"] == [
        split_heldout.statement_hash("p(a)", family="atp")
    ]
    assert manifest["canonicalization"] == {
        "family": "atp",
        "scheme": "tptp-layout-v2",
        "version": 2,
    }


def _render_target(steps):
    lines = []
    for i, step in enumerate(steps):
        name = build_atp_shard.render_tptp_atom(step["name"])
        rule = build_atp_shard.render_tptp_atom(step["rule"])
        parents = " ".join(
            build_atp_shard.render_tptp_atom(parent)
            for parent in step["parents"]
        )
        line = f"{i + 1:>3}  {name:<10} {step['formula']}   [{rule}"
        lines.append(line + (f" {parents}]" if parents else "]"))
    return "\n".join(lines)


def _atp_record(
    *,
    theorem="enigma:theorem_x",
    facts=None,
    local_inputs=None,
    proof_steps=None,
):
    facts = facts or {"premise": "p(a)"}
    local_inputs = local_inputs or {}
    proof_steps = proof_steps or [
        {
            "name": "step_x",
            "role": "plain",
            "formula": "$false",
            "rule": "resolution",
            "parents": ["premise", "theorem_x"],
            "parent_sources": ["premise", "theorem_x"],
            "source": "inference(resolution,[status(thm)],[premise,theorem_x])",
        }
    ]
    target = _render_target(proof_steps)
    block = "I know these mathematical statements:\n" + "\n".join(
        f"{name} : {statement}" for name, statement in facts.items()
    )
    if local_inputs:
        block += "\nLocal ATP inputs:\n" + "\n".join(
            f"{name} : {statement}" for name, statement in local_inputs.items()
        )
    return {
        "id": "atp-example",
        "theorem": theorem,
        "facts": facts,
        "cited": list(facts),
        "local_inputs": local_inputs,
        "goal_name": "theorem_x",
        "goal": "goal(a)",
        "proof_steps": proof_steps,
        "target": target,
        "text": f"{block}\n---\nGOAL goal(a)\n{target}",
        "mask_start": 0,
        "mask_end": len(block),
    }


def _write_corpus(tmp_path, record, held_manifest):
    corpus = tmp_path / "corpus"
    for name in ("raw", "shards", "eval", "heldout"):
        (corpus / name).mkdir(parents=True, exist_ok=True)
    line = json.dumps(record) + "\n"
    (corpus / "raw" / "enigma.jsonl").write_text(line)
    (corpus / "shards" / "enigma.jsonl").write_text(line)
    (corpus / "eval" / "enigma.jsonl").write_text("")
    (corpus / "heldout" / "atp.json").write_text(json.dumps(held_manifest))
    return corpus


def _verify(corpus, monkeypatch):
    del monkeypatch
    return verify_corpus.legacy_audit(
        ["--corpus", str(corpus), "--max-report", "10"],
    )


def test_deep_verifier_accepts_complete_atp_dependency_closure(tmp_path, monkeypatch):
    record = _atp_record(local_inputs={"input-1": "q(a)"})
    record["proof_steps"][0]["parents"].insert(1, "input-1")
    record["proof_steps"][0]["parent_sources"].insert(1, "'input-1'")
    record["proof_steps"][0]["source"] = (
        "inference(resolution,[status(thm)],[premise,'input-1',theorem_x])"
    )
    record["target"] = _render_target(record["proof_steps"])
    block = (
        "I know these mathematical statements:\n"
        "premise : p(a)\n"
        "Local ATP inputs:\n"
        "input-1 : q(a)"
    )
    record["text"] = f"{block}\n---\nGOAL goal(a)\n{record['target']}"
    record["mask_end"] = len(block)
    corpus = _write_corpus(
        tmp_path,
        record,
        {"facts": [], "shards": ["enigma"], "statement_hashes": []},
    )

    assert _verify(corpus, monkeypatch) == 0


def test_deep_verifier_rejects_unresolved_atp_parent(tmp_path, monkeypatch):
    record = _atp_record()
    record["proof_steps"][0]["parents"].append("missing-parent")
    record["proof_steps"][0]["parent_sources"].append("'missing-parent'")
    record["target"] = _render_target(record["proof_steps"])
    block = "I know these mathematical statements:\npremise : p(a)"
    record["text"] = f"{block}\n---\nGOAL goal(a)\n{record['target']}"
    corpus = _write_corpus(
        tmp_path,
        record,
        {"facts": [], "shards": ["enigma"], "statement_hashes": []},
    )

    assert _verify(corpus, monkeypatch) == 1


def test_deep_verifier_rejects_alternate_proof_of_held_theorem(
    tmp_path, monkeypatch
):
    record = _atp_record(theorem="enigma:held_name#2")
    corpus = _write_corpus(
        tmp_path,
        record,
        {"facts": ["held_name"], "shards": ["enigma"], "statement_hashes": []},
    )

    assert _verify(corpus, monkeypatch) == 1


def test_deep_verifier_rejects_exact_held_statement_alias(tmp_path, monkeypatch):
    held_statement = "![X]:p(X)"
    record = _atp_record(facts={"alias_name": " ! [ X ] : p ( X ) "})
    corpus = _write_corpus(
        tmp_path,
        record,
        {
            "facts": ["held_name"],
            "shards": ["enigma"],
            "statement_hashes": [
                split_heldout.statement_hash(held_statement, family="atp")
            ],
        },
    )

    assert _verify(corpus, monkeypatch) == 1


def test_family_split_preserves_mizar_colon_identity(tmp_path, monkeypatch):
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)

    def row(row_id, theorem, facts, goal):
        return {
            "id": row_id,
            "theorem": theorem,
            "facts": facts,
            "cited": list(facts),
            "goal": goal,
            "target": "thus thesis;",
        }

    rows = [
        row("citation", "OTHER:1", {"AFF_1:50": "held statement"}, "other goal"),
        row("own-proof", "AFF_1:50", {"common": "safe statement"}, "own goal"),
        row("safe-1", "OTHER:2", {"common": "safe statement"}, "safe goal 1"),
        row("safe-2", "OTHER:3", {"common": "safe statement"}, "safe goal 2"),
    ]
    (corpus / "raw" / "mizar.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in rows)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_heldout.py",
            "--corpus",
            str(corpus),
            "--family",
            "mizar=mizar",
            "--heldout",
            "1",
        ],
    )

    assert split_heldout.main() == 0
    train = [
        json.loads(line)
        for line in (corpus / "shards" / "mizar.jsonl").read_text().splitlines()
    ]
    eval_rows = [
        json.loads(line)
        for line in (corpus / "eval" / "mizar.jsonl").read_text().splitlines()
    ]

    assert {record["id"] for record in train} == {"safe-1", "safe-2"}
    assert {record["id"] for record in eval_rows} == {"citation", "own-proof"}


def test_numbered_mptp_bookkeeping_stays_local_and_distinct():
    text = """
fof(global_decl, axiom, p(a), file('problem.p', global_fact)).
fof(cc_decl, axiom, cc(a), file('problem.p', cc4_generated)).
fof(fc_decl, axiom, fc(a), file('problem.p', fc3_generated)).
fof(rc_decl, axiom, rc(a), file('problem.p', rc1_generated)).
fof(old_decl, axiom, old(a), file('problem.p', cc_generated)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)],
              [global_fact, cc4_generated, fc3_generated,
               rc1_generated, cc_generated, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.facts == {"global_fact": "p(a)"}
    assert proof.local_inputs == {
        "cc4_generated": "cc(a)",
        "fc3_generated": "fc(a)",
        "rc1_generated": "rc(a)",
        "cc_generated": "old(a)",
    }
    assert build_atp_shard.dependency_errors(proof) == []


def test_unknown_file_provenance_uses_declaration_names_without_collapsing():
    text = """
fof(first_decl, axiom, first(a), file('problem.p', unknown)).
fof(second_decl, axiom, second(a), file('problem.p', unknown)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)],
              [first_decl, second_decl, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.facts == {}
    assert proof.local_inputs == {
        "first_decl": "first(a)",
        "second_decl": "second(a)",
    }
    assert build_atp_shard.dependency_errors(proof) == []


def test_unknown_file_provenance_does_not_create_unknown_parent_alias():
    text = """
fof(first_decl, axiom, first(a), file('problem.p', unknown)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)], [unknown, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.local_inputs == {"first_decl": "first(a)"}
    assert build_atp_shard.dependency_errors(proof) == [
        "step_x: unresolved parent unknown"
    ]


def test_known_parent_alias_remains_resolvable():
    text = """
fof(declaration_name, plain, first(a), file('problem.p', 'parent alias')).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(resolution, [status(thm)], ['parent alias', theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.local_inputs == {"parent alias": "first(a)"}
    assert build_atp_shard.dependency_errors(proof) == []


def test_statement_canonicalization_is_syntax_aware_and_family_scoped():
    assert split_heldout.canonical_statement(
        " p ( X ) ", family="atp"
    ) == split_heldout.canonical_statement("p(X)", family="atp")
    assert split_heldout.statement_hash(
        "p('a b')", family="atp"
    ) != split_heldout.statement_hash("p('ab')", family="atp")
    assert split_heldout.statement_hash(
        "|- ph   ps", family="metamath"
    ) == split_heldout.statement_hash("|- ph ps", family="metamath")
    assert split_heldout.statement_hash(
        "|- ph ps", family="metamath"
    ) != split_heldout.statement_hash("|- phps", family="metamath")
    assert split_heldout.statement_hash(
        "p(X)", family="atp"
    ) != split_heldout.statement_hash("p(X)", family="mizar")
    assert split_heldout.canonical_statement(
        'for x holds x = "a  b"', family="mizar"
    ).endswith('"a  b"')


def test_metamath_holdout_exposure_includes_local_and_target_expressions():
    held_hashes = {
        split_heldout.statement_hash("|- ph", family="metamath")
    }
    local_only = {
        "theorem": "set:safe",
        "facts": {"ext": "|- ps"},
        "local_assumptions": {"safe.1": "|- ph"},
        "goal": "|- ch",
        "target": "  1  ext            |- ps",
    }
    target_only = {
        "theorem": "set:safe",
        "facts": {"ext": "|- ps => |- ph"},
        "local_assumptions": {},
        "goal": "|- ch",
        "target": "  1  ext            |- ph",
    }

    assert split_heldout.heldout_exposure(
        local_only, {"held"}, held_hashes, family="metamath"
    ).statement_alias
    assert split_heldout.heldout_exposure(
        target_only, {"held"}, held_hashes, family="metamath"
    ).statement_alias


def test_atp_target_quotes_special_step_and_parent_names_losslessly():
    text = r"""
fof('parent with space', axiom, p(a),
    file('problem.p', 'parent with space')).
fof('parent\\slash\'quote,[]', axiom, q(a),
    file('problem.p', 'parent\\slash\'quote,[]')).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof('step with space', plain, ($false),
    inference(resolution, [status(thm)],
              ['parent with space', 'parent\\slash\'quote,[]', theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.steps[0].name == "step with space"
    assert proof.steps[0].parents == [
        "parent with space",
        "parent\\slash'quote,[]",
        "theorem_x",
    ]
    target = build_atp_shard.render_target(proof.steps)
    assert "'step with space'" in target
    assert "'parent with space'" in target
    assert "'parent\\\\slash\\'quote,[]'" in target
    record = {
        "facts": proof.facts,
        "local_inputs": proof.local_inputs,
        "goal_name": proof.goal_name,
        "proof_steps": [
            {
                "name": step.name,
                "role": step.role,
                "formula": step.formula,
                "rule": step.rule,
                "parents": step.parents,
                "parent_sources": step.parent_sources,
                "source": step.source,
            }
            for step in proof.steps
        ],
        "target": target,
    }
    assert verify_corpus.atp_record_errors(record) == []


def test_refutation_predicate_accepts_only_balanced_outer_false():
    assert build_atp_shard.is_refutation_formula("$false")
    assert build_atp_shard.is_refutation_formula(" ( ( $false ) ) ")
    assert not build_atp_shard.is_refutation_formula("($false")
    assert not build_atp_shard.is_refutation_formula("($false | p(a))")
    assert not build_atp_shard.is_refutation_formula("p(a)")


def _write_stale_atp_outputs(output):
    stale = {}
    for directory in ("shards", "eval", "heldout"):
        path = output / directory / "prf2.jsonl"
        if directory == "heldout":
            path = path.with_suffix(".json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"stale-{directory}")
        stale[directory] = path
    return stale


def _assert_outputs_invalidated(stale):
    for path in stale.values():
        assert not path.exists()
        quarantined = list(path.parent.glob(path.name + ".stale*"))
        assert len(quarantined) == 1
        assert quarantined[0].read_text().startswith("stale-")


def test_atp_builder_missing_source_invalidates_stale_outputs(
    tmp_path, monkeypatch
):
    output = tmp_path / "output"
    stale = _write_stale_atp_outputs(output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_atp_shard.py",
            "--src",
            str(tmp_path / "missing"),
            "--name",
            "prf2",
            "--out",
            str(output),
            "--min-steps",
            "1",
        ],
    )

    assert build_atp_shard.main() != 0
    _assert_outputs_invalidated(stale)


def test_atp_builder_rejects_non_refutation_and_invalidates_stale_outputs(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "closed_but_not_false").write_text(
        """
fof(premise, axiom, p(a), file('problem.p', premise)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, p(a),
    inference(resolution, [status(thm)], [premise, theorem_x])).
"""
    )
    output = tmp_path / "output"
    stale = _write_stale_atp_outputs(output)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_atp_shard.py",
            "--src",
            str(source),
            "--name",
            "prf2",
            "--out",
            str(output),
            "--heldout",
            "0",
            "--min-steps",
            "1",
        ],
    )

    assert build_atp_shard.main() != 0
    _assert_outputs_invalidated(stale)


def _split_atp_row(row_id, theorem, facts):
    fact_names = sorted(facts)
    steps = [
        {
            "name": "step_x",
            "role": "plain",
            "formula": "$false",
            "rule": "resolution",
            "parents": [*fact_names, "theorem_x"],
            "parent_sources": [*fact_names, "theorem_x"],
            "source": "inference(resolution,[status(thm)],["
            + ",".join([*fact_names, "theorem_x"])
            + "])",
        }
    ]
    return {
        "id": row_id,
        "schema_version": "atp-v2",
        "theorem": theorem,
        "facts": facts,
        "cited": list(facts),
        "local_inputs": {},
        "goal_name": "theorem_x",
        "goal": "goal(a)",
        "proof_steps": steps,
        "target": _render_target(steps),
    }


def test_exact_atp_signature_ignores_wrapper_id_and_fact_order_only():
    first = _split_atp_row(
        "prf2-id",
        "prf2:shared#2",
        {"fact_a": "p(a)", "fact_b": "q(a)"},
    )
    sibling = _split_atp_row(
        "enigma-id",
        "enigma:shared#10",
        {"fact_b": "q(a)", "fact_a": "p(a)"},
    )

    assert split_heldout.exact_atp_signature(first) == (
        split_heldout.exact_atp_signature(sibling)
    )

    sibling["proof_steps"][0]["rule"] = "different_rule"
    assert split_heldout.exact_atp_signature(first) != (
        split_heldout.exact_atp_signature(sibling)
    )

    legacy_first = {
        "id": "legacy-prf2",
        "theorem": "prf2:legacy_shared",
        "facts": {"fact_a": "p(a)", "fact_b": "q(a)"},
        "local_inputs": {"input_a": "r(a)"},
        "goal": "goal(a)",
        "target": "  1  step_x $false   [resolution fact_a fact_b]",
    }
    legacy_sibling = {
        "id": "legacy-enigma",
        "theorem": "enigma:legacy_shared",
        "facts": {"fact_b": "q(a)", "fact_a": "p(a)"},
        "local_inputs": {"input_a": "r(a)"},
        "goal": " goal ( a ) ",
        "target": "  1  step_x $false   [resolution fact_a fact_b]",
    }
    assert split_heldout.exact_atp_signature(legacy_first) == (
        split_heldout.exact_atp_signature(legacy_sibling)
    )
    legacy_sibling["target"] = (
        "  1  step_x $false   [different_rule fact_a fact_b]"
    )
    assert split_heldout.exact_atp_signature(legacy_first) != (
        split_heldout.exact_atp_signature(legacy_sibling)
    )


def test_family_split_deduplicates_before_counts_and_prefers_first_shard(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    (corpus / "raw").mkdir(parents=True)
    prf2_rows = [
        _split_atp_row(
            "first-shard-copy", "prf2:shared_theorem", {"held_name": "p(a)"}
        ),
        _split_atp_row(
            "other-proof", "prf2:other_theorem", {"held_name": "p(a)"}
        ),
    ]
    enigma_rows = [
        _split_atp_row(
            "later-sibling-copy",
            "enigma:shared_theorem",
            {"held_name": "p(a)"},
        )
    ]
    for shard, rows in (("prf2", prf2_rows), ("enigma", enigma_rows)):
        (corpus / "raw" / f"{shard}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "split_heldout.py",
            "--corpus",
            str(corpus),
            "--family",
            "atp=prf2,enigma",
            "--heldout",
            "1",
        ],
    )

    assert split_heldout.main() == 0
    prf2_eval = [
        json.loads(line)
        for line in (corpus / "eval" / "prf2.jsonl").read_text().splitlines()
    ]
    enigma_train = (corpus / "shards" / "enigma.jsonl").read_text()
    enigma_eval = (corpus / "eval" / "enigma.jsonl").read_text()
    manifest = json.loads((corpus / "heldout" / "atp.json").read_text())

    assert {row["id"] for row in prf2_eval} == {
        "first-shard-copy",
        "other-proof",
    }
    assert enigma_train == ""
    assert enigma_eval == ""
    assert manifest["facts"] == ["held_name"]
    assert manifest["deduplication"] == {
        "policy": "atp-exact-structured",
        "version": 1,
        "priority": "ordered family shards then source line",
        "duplicates_total": 1,
        "duplicates_by_shard": {"prf2": 0, "enigma": 1},
    }


def _write_two_shard_atp_corpus(tmp_path, prf2, enigma, manifest):
    corpus = tmp_path / "corpus"
    for directory in ("raw", "shards", "eval", "heldout"):
        (corpus / directory).mkdir(parents=True, exist_ok=True)
    for shard, record in (("prf2", prf2), ("enigma", enigma)):
        line = json.dumps(record) + "\n"
        (corpus / "raw" / f"{shard}.jsonl").write_text(line)
        (corpus / "shards" / f"{shard}.jsonl").write_text(line)
        (corpus / "eval" / f"{shard}.jsonl").write_text("")
    (corpus / "heldout" / "atp.json").write_text(json.dumps(manifest))
    return corpus


def test_deep_verifier_rejects_exact_atp_sibling_across_shards(
    tmp_path, monkeypatch
):
    prf2 = _atp_record(theorem="prf2:shared_theorem")
    prf2["id"] = "prf2-copy"
    enigma = _atp_record(theorem="enigma:shared_theorem")
    enigma["id"] = "enigma-copy"
    corpus = _write_two_shard_atp_corpus(
        tmp_path,
        prf2,
        enigma,
        {
            "facts": [],
            "family": "atp",
            "shards": ["prf2", "enigma"],
            "statement_hashes": [],
            "canonicalization": split_heldout.canonicalization_metadata("atp"),
        },
    )

    assert _verify(corpus, monkeypatch) == 1


def test_legacy_held_statement_backfill_is_pooled_across_atp_family(
    tmp_path, monkeypatch
):
    corpus = tmp_path / "corpus"
    for directory in ("raw", "shards", "eval", "heldout"):
        (corpus / directory).mkdir(parents=True, exist_ok=True)

    held_definition = {
        "id": "held-definition",
        "theorem": "prf2:source",
        "facts": {"held_name": "p(a)"},
    }
    alias = _atp_record(
        theorem="enigma:other_theorem",
        facts={"premise": "p(a)"},
    )
    alias["id"] = "cross-shard-alias"
    (corpus / "raw" / "prf2.jsonl").write_text(
        json.dumps(held_definition) + "\n"
    )
    (corpus / "raw" / "enigma.jsonl").write_text(json.dumps(alias) + "\n")
    (corpus / "shards" / "prf2.jsonl").write_text("")
    (corpus / "shards" / "enigma.jsonl").write_text(json.dumps(alias) + "\n")
    (corpus / "eval" / "prf2.jsonl").write_text("")
    (corpus / "eval" / "enigma.jsonl").write_text("")
    (corpus / "heldout" / "atp.json").write_text(
        json.dumps(
            {
                "facts": ["held_name"],
                "family": "atp",
                "shards": ["prf2", "enigma"],
            }
        )
    )

    assert _verify(corpus, monkeypatch) == 1


def test_atp_target_quotes_special_inference_rule_losslessly():
    text = r"""
fof(premise, axiom, p(a), file('problem.p', premise)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference('rule\\slash\'quote,[]', [status(thm)],
              [premise, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.steps[0].rule == "rule\\slash'quote,[]"
    target = build_atp_shard.render_target(proof.steps)
    assert "['rule\\\\slash\\'quote,[]' premise theorem_x]" in target
    record = {
        "facts": proof.facts,
        "local_inputs": proof.local_inputs,
        "goal_name": proof.goal_name,
        "proof_steps": [
            {
                "name": step.name,
                "role": step.role,
                "formula": step.formula,
                "rule": step.rule,
                "parents": step.parents,
                "parent_sources": step.parent_sources,
                "source": step.source,
            }
            for step in proof.steps
        ],
        "target": target,
    }
    assert verify_corpus.atp_record_errors(record) == []


def _nested_source_proof(parent_source):
    return f"""
fof(parent_decl, axiom, p(a), file('problem.p', parent_fact)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_nested, plain, q(a),
    inference(variable_rename, [status(thm)], [{parent_source}])).
fof(step_false, plain, $false,
    inference(resolution, [status(thm)], [step_nested, theorem_x])).
"""


def test_real_prf2_nested_source_recovers_true_parent_and_provenance():
    # From extracted prf2/t7_matrix17, reduced to one preprocessing chain.
    nested = (
        "inference ( fof_nnf , [ status ( thm ) ] , "
        "[ inference ( fof_simplification , [ status ( thm ) ] , "
        "[ parent_fact ] ) ] )"
    )
    proof = build_atp_shard.parse(_nested_source_proof(nested), fenced=False)

    assert proof is not None
    assert proof.steps[0].parents == ["parent_fact"]
    assert proof.steps[0].parent_sources == [nested]
    assert proof.steps[0].source == (
        "inference(variable_rename, [status(thm)], [" + nested + "])"
    )
    assert build_atp_shard.dependency_errors(proof) == []
    assert "parent_fact" in build_atp_shard.render_target(proof.steps)
    assert "fof_nnf" not in build_atp_shard.render_target(proof.steps)


def test_real_enigma_nested_source_recovers_true_parent_and_provenance():
    # From extracted ENIGMA mzr01/t7_matrix17, c_0_11.
    nested = "inference(fof_nnf,[status(thm)],[parent_fact])"
    proof = build_atp_shard.parse(_nested_source_proof(nested), fenced=False)

    assert proof is not None
    assert proof.steps[0].parents == ["parent_fact"]
    assert proof.steps[0].parent_sources == [nested]
    assert proof.steps[0].source == (
        "inference(variable_rename, [status(thm)], [" + nested + "])"
    )
    assert build_atp_shard.dependency_errors(proof) == []


def test_nested_source_parser_distinguishes_provenance_from_true_parents():
    source = (
        r"inference('outer,rule[]',[status(thm),"
        r"info('quoted,info[not a parent]')],["
        r"(inference(inner_rule,[status(thm)],['quoted,parent[]'])"
        r":[details('comma,bracket[]')]),"
        r"file('problem,[]','file,parent[]'),"
        r"introduced(definition,[info('not,parent')]),"
        r"theory(equality,[info('not,parent')]),"
        r"unknown,"
        r"step_plain])"
    )

    parsed = build_atp_shard.source_dependencies(source)

    assert parsed is not None
    rule, parent_sources, parents = parsed
    assert rule == "outer,rule[]"
    assert parent_sources == [
        (
            r"(inference(inner_rule,[status(thm)],['quoted,parent[]'])"
            r":[details('comma,bracket[]')])"
        ),
        r"file('problem,[]','file,parent[]')",
        r"introduced(definition,[info('not,parent')])",
        r"theory(equality,[info('not,parent')])",
        "unknown",
        "step_plain",
    ]
    assert parents == [
        "quoted,parent[]",
        "file,parent[]",
        "unknown",
        "step_plain",
    ]


def test_nested_source_parser_flattens_nested_parent_lists_without_fake_terms():
    source = (
        "inference(resolution,[status(thm)],"
        "[[parent_a,inference(rewrite,[status(thm)],[parent_b])],"
        "introduced(tautology),theory(equality),unknown])"
    )

    assert build_atp_shard.source_dependencies(source) == (
        "resolution",
        [
            "[parent_a,inference(rewrite,[status(thm)],[parent_b])]",
            "introduced(tautology)",
            "theory(equality)",
            "unknown",
        ],
        ["parent_a", "parent_b", "unknown"],
    )


def test_malformed_nested_source_is_not_reinterpreted_as_parent_name():
    malformed = (
        "inference(resolution,[status(thm)],"
        "[parent_a,inference(rewrite,[status(thm)],[parent_b])"
    )

    assert build_atp_shard.source_dependencies(malformed) is None


def test_nested_true_parent_still_rejects_when_late():
    text = """
fof(parent_fact, axiom, p(a), file('problem.p', parent_fact)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_early, plain, q(a),
    inference(rewrite, [status(thm)],
              [inference(fof_nnf, [status(thm)], [step_late])])).
fof(step_late, plain, r(a),
    inference(rewrite, [status(thm)], [parent_fact])).
fof(step_false, plain, $false,
    inference(resolution, [status(thm)], [step_early, theorem_x])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.steps[0].parents == ["step_late"]
    assert build_atp_shard.dependency_errors(proof) == [
        "step_early: parent step_late is not earlier"
    ]


def test_real_e_reserved_clause_id_is_a_true_unresolved_parent():
    # From prf2/t25_zf_fund1 and solved ENIGMA copies. E prints this
    # reserved signed clause identifier as an sr argument but emits no node.
    reserved_parent = "c_0_-9223372036854775806"
    text = f"""
fof(parent_fact, axiom, p(a), file('problem.p', parent_fact)).
fof(theorem_x, conjecture, goal(a), file('problem.p', theorem_x)).
fof(step_x, plain, $false,
    inference(sr, [status(thm)], [parent_fact, {reserved_parent}])).
"""
    proof = build_atp_shard.parse(text, fenced=False)

    assert proof is not None
    assert proof.source_errors == []
    assert proof.steps[0].parents == ["parent_fact", reserved_parent]
    assert build_atp_shard.dependency_errors(proof) == [
        f"step_x: unresolved parent {reserved_parent}"
    ]


def test_rejected_real_inventory_dispositions_are_fatal_without_threshold():
    rejected_builds = {
        "prf2": (
            25_060,
            {
                "accepted": 15,
                "too_thin": 259,
                "incomplete_trace": 24_786,
            },
        ),
        "enigma": (
            231_520,
            {
                "accepted": 19,
                "unsolved_or_unfenced": 174_036,
                "too_thin": 964,
                "incomplete_trace": 56_465,
                "redundant_reproof": 36,
            },
        ),
    }

    for files, dispositions in rejected_builds.values():
        errors = build_atp_shard.source_inventory_errors(files, dispositions)
        assert not any("accounting" in error for error in errors)
        assert any("incomplete_trace" in error for error in errors)


def test_source_inventory_accounting_requires_exact_closure():
    assert build_atp_shard.source_inventory_errors(
        4,
        {
            "accepted": 2,
            "too_thin": 1,
            "unsolved_or_unfenced": 1,
        },
    ) == []
    assert any(
        "accounting" in error
        for error in build_atp_shard.source_inventory_errors(
            4,
            {"accepted": 2, "too_thin": 1},
        )
    )
    assert build_atp_shard.source_inventory_errors(
        4,
        {
            "accepted": 2,
            "unresolved_parent": 1,
            "late_or_cyclic_parent": 1,
        },
    ) == []


def test_builder_types_and_drops_true_unresolved_parent_with_exact_accounting(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "accepted").write_text(_proof_with_parents())
    (source / "unresolved").write_text(
        _proof_with_parents().replace(
            "[parent_0,parent_1,parent_2,parent_3,parent_4,parent_5,parent_6]",
            "[parent_0,parent_1,parent_2,parent_3,parent_4,parent_5,"
            "parent_6,missing_parent]",
        )
    )
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_atp_shard.py",
            "--src",
            str(source),
            "--name",
            "prf2",
            "--out",
            str(output),
            "--heldout",
            "0",
            "--min-steps",
            "1",
        ],
    )

    assert build_atp_shard.main() == 0
    rows = [
        json.loads(line)
        for line in (output / "shards" / "prf2.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 1
    assert rows[0]["theorem"] == "prf2:accepted"


def test_repaired_real_inventory_dispositions_close_without_quality_floor():
    assert build_atp_shard.source_inventory_errors(
        25_060,
        {
            "accepted": 24_797,
            "too_thin": 259,
            "unresolved_parent": 4,
        },
    ) == []
    assert build_atp_shard.source_inventory_errors(
        231_520,
        {
            "accepted": 27_079,
            "unsolved_or_unfenced": 174_036,
            "too_thin": 964,
            "unresolved_parent": 4,
            "redundant_reproof": 29_437,
        },
    ) == []


def _low_tier_feature(
    *,
    base="t1_example",
    run="mzr03",
    raw_sha256="1" * 64,
    text_sha256="2" * 64,
    exact_signature_sha256="3" * 64,
    text_plus_eos_tokens=8_192,
    existing_variants=1,
    dead_steps=0,
    material=True,
    alpha_formulas=frozenset({"a", "x", "y", "z", "w"}),
    backward_edges=frozenset({"e1", "x1", "x2", "x3"}),
    rule_bigrams=frozenset({("r1", "r2"), ("r2", "r3")}),
    premises=frozenset({("different", "p")}),
    core_rules=frozenset({"resolution", "paramodulation"}),
    record=None,
):
    return build_atp_shard.AlternativeProofFeatures(
        base=base,
        run=run,
        raw_sha256=raw_sha256,
        text_sha256=text_sha256,
        exact_signature_sha256=exact_signature_sha256,
        text_plus_eos_tokens=text_plus_eos_tokens,
        existing_variants=existing_variants,
        dead_steps=dead_steps,
        material=material,
        alpha_formulas=alpha_formulas,
        backward_edges=backward_edges,
        rule_bigrams=rule_bigrams,
        premises=premises,
        core_rules=core_rules,
        paste_steps=0,
        record=record,
    )


def _accepted_low_tier_feature(*, base="t1_example"):
    return _low_tier_feature(
        base=base,
        run="mzr01",
        raw_sha256="a" * 64,
        text_sha256="b" * 64,
        exact_signature_sha256="c" * 64,
        alpha_formulas=frozenset({"a", "b", "c", "d"}),
        backward_edges=frozenset({"e1", "e2", "e3"}),
        rule_bigrams=frozenset({("r1", "r2"), ("r2", "r4")}),
        premises=frozenset({("accepted", "p")}),
        core_rules=frozenset({"resolution"}),
    )


def test_conservative_enigma_policy_contains_only_approved_low_tier():
    policy = build_atp_shard.ENIGMA_LOW_TIER_POLICY

    assert policy["schema_version"] == "enigma-alternative-proof-low-tier-v1"
    assert policy["expected_base_rows"] == 27_079
    assert policy["expected_redundant_dispositions"] == 29_437
    assert policy["expected_selected_rows"] == 2_087
    assert policy["expected_text_plus_eos_tokens"] == 9_655_618
    assert policy["expected_packed_16384_tokens"] == 9_666_560
    assert policy["max_total_variants"] == 2
    assert policy["max_text_plus_eos_tokens"] == 8_192
    assert policy["max_alpha_formula_jaccard"] == 0.67
    assert policy["max_backward_dag_edge_jaccard"] == 0.70
    assert policy["min_new_alpha_formulas"] == 3
    assert policy["min_new_alpha_formula_fraction"] == 0.20
    assert policy["run_priority"] == ["mzr01", "mzr03", "mzr02", "mzr08"]
    assert "central" not in policy
    assert "high" not in policy


def test_conservative_enigma_contract_rejects_any_source_or_tokenizer_drift():
    contract = json.loads(
        json.dumps(build_atp_shard.ENIGMA_LOW_TIER_SOURCE_CONTRACT)
    )
    build_atp_shard.validate_enigma_low_tier_contract(contract)

    contract["source_dispositions"]["redundant_reproof"] -= 1
    with pytest.raises(ValueError, match="source/audit contract mismatch"):
        build_atp_shard.validate_enigma_low_tier_contract(contract)


@pytest.mark.parametrize(
    "candidate",
    [
        _low_tier_feature(text_plus_eos_tokens=8_193),
        _low_tier_feature(dead_steps=1),
        _low_tier_feature(existing_variants=2),
        _low_tier_feature(material=False),
        _low_tier_feature(alpha_formulas=frozenset({"a", "x", "y"})),
        _low_tier_feature(
            alpha_formulas=frozenset({"a", "b", "c", "d", "e", "x"})
        ),
        _low_tier_feature(
            backward_edges=frozenset({"e1", "e2", "e3", "x"})
        ),
        _low_tier_feature(
            premises=frozenset({("accepted", "p")}),
            core_rules=frozenset({"resolution"}),
        ),
    ],
)
def test_conservative_enigma_policy_rejects_nonapproved_candidates(candidate):
    accepted = _accepted_low_tier_feature()

    assert not build_atp_shard.conservative_alternative_is_eligible(
        candidate,
        accepted,
    )


def test_conservative_enigma_selector_is_deterministic_and_uses_audited_rank():
    accepted = _accepted_low_tier_feature()
    lower_run_priority = _low_tier_feature(
        run="mzr02",
        raw_sha256="f" * 64,
        text_sha256="4" * 64,
        exact_signature_sha256="5" * 64,
        text_plus_eos_tokens=100,
    )
    audited_winner = replace(
        lower_run_priority,
        run="mzr03",
        raw_sha256="0" * 64,
        text_sha256="6" * 64,
        exact_signature_sha256="7" * 64,
    )
    longer = replace(
        audited_winner,
        raw_sha256="9" * 64,
        text_sha256="8" * 64,
        exact_signature_sha256="9" * 64,
        text_plus_eos_tokens=101,
    )

    forward = build_atp_shard.select_conservative_alternatives(
        [lower_run_priority, audited_winner, longer],
        {"t1_example": [accepted]},
        existing_text_sha256s={accepted.text_sha256},
        existing_signature_sha256s={accepted.exact_signature_sha256},
    )
    reverse = build_atp_shard.select_conservative_alternatives(
        [longer, audited_winner, lower_run_priority],
        {"t1_example": [accepted]},
        existing_text_sha256s={accepted.text_sha256},
        existing_signature_sha256s={accepted.exact_signature_sha256},
    )

    assert [candidate.raw_sha256 for candidate in forward] == ["0" * 64]
    assert [candidate.raw_sha256 for candidate in reverse] == ["0" * 64]


def test_conservative_enigma_selector_globally_deduplicates_text_and_signature():
    first_accepted = _accepted_low_tier_feature(base="t1_example")
    second_accepted = replace(first_accepted, base="t2_example")
    first = _low_tier_feature(base="t1_example", raw_sha256="1" * 64)
    second = replace(
        first,
        base="t2_example",
        run="mzr02",
        raw_sha256="2" * 64,
    )

    selected = build_atp_shard.select_conservative_alternatives(
        [second, first],
        {
            "t1_example": [first_accepted],
            "t2_example": [second_accepted],
        },
        existing_text_sha256s=set(),
        existing_signature_sha256s=set(),
    )

    assert [(candidate.base, candidate.run) for candidate in selected] == [
        ("t1_example", "mzr03")
    ]


def test_low_tier_writer_preserves_every_existing_byte_and_variant_id(tmp_path):
    existing = [
        {"id": "base", "theorem": "enigma:t1_example"},
        {"id": "two", "theorem": "enigma:t2_example#2"},
        {"id": "three", "theorem": "enigma:t3_example#3"},
        {"id": "four", "theorem": "enigma:t4_example#4"},
    ]
    base_bytes = b"".join(
        json.dumps(record).encode("utf-8") + b"\n" for record in existing
    )
    base = tmp_path / "base.jsonl"
    base.write_bytes(base_bytes)
    output = tmp_path / "output.jsonl"
    addition = {
        "id": "new",
        "theorem": "enigma:t1_example#1",
        "text": "new trace",
    }

    stats = build_atp_shard.write_preserved_base_with_alternatives(
        base,
        output,
        [addition],
    )
    output_bytes = output.read_bytes()

    assert output_bytes[: len(base_bytes)] == base_bytes
    assert output_bytes.splitlines(keepends=True)[:4] == base_bytes.splitlines(
        keepends=True
    )
    assert [json.loads(line)["id"] for line in output_bytes.splitlines()] == [
        "base",
        "two",
        "three",
        "four",
        "new",
    ]
    assert stats["base_sha256"] == hashlib.sha256(base_bytes).hexdigest()
    assert stats["base_bytes"] == len(base_bytes)
    assert stats["added_rows"] == 1


def test_all_enigma_variant_suffixes_share_current_theorem_identity():
    identities = {
        split_heldout.normalize_theorem_identity(
            f"enigma:t42_example{suffix}",
            family="atp",
        )
        for suffix in ("", "#1", "#2", "#3", "#4")
    }

    assert identities == {"t42_example"}
