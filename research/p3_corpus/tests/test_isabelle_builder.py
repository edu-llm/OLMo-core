"""Synthetic regression tests for the Isabelle/Magnushammer corpus builder."""

import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import build_isabelle_shard as builder

GLOBAL_PREMISE = {"g": ["Global.fact", "global statement"]}


class FakeQwenTokenizer:
    """Small tokenizer-shaped test double with an explicit EOS contract."""

    identity = "Qwen/Qwen2.5-0.5B (synthetic test double)"
    sha256 = "f" * 64
    tokenizer_json_sha256 = "f" * 64
    tokenizer_config_sha256 = "e" * 64
    behavior_digest = "d" * 64
    tokenizers_version = "synthetic-test-double"
    eos_token_id = 151643

    def __init__(self, length=None):
        self._length = length or (lambda text: max(1, len(text.split())))

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=[7] * self._length(text))


class TwoPassFactory:
    """Re-iterable source fixture that records streaming pass count."""

    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.calls = 0

    def __call__(self):
        self.calls += 1
        yield from self.trajectories


def transition(state, step="apply rule", premises=None):
    return {
        "state": state,
        "step": step,
        "premises": GLOBAL_PREMISE if premises is None else premises,
    }


def trajectory(statement, transitions):
    return {"statement": statement, "transitions": transitions}


def source_item(theory, index, proof):
    return theory, index, proof


def pinned_test_gate(_path):
    """Explicit function-level bypass used only for synthetic source fixtures."""

    return builder.pinned_source_metadata()


def run_build(
    tmp_path,
    trajectories,
    *,
    heldout=0,
    seed=23,
    tokenizer=None,
    name="isabelle",
    out_name="corpus",
):
    source = tmp_path / f"{out_name}-synthetic-source.json"
    source.write_text("synthetic fixture", encoding="utf-8")
    output = tmp_path / out_name
    factory = TwoPassFactory(trajectories)
    stats = builder.build_corpus(
        source=source,
        out=output,
        name=name,
        heldout=heldout,
        seed=seed,
        tokenizer=tokenizer or FakeQwenTokenizer(),
        source_gate=pinned_test_gate,
        trajectory_iter_factory=factory,
    )
    return output, stats, factory


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def raw_rows(output, name="isabelle"):
    return read_jsonl(output / "raw" / f"{name}.jsonl")


def heldout_manifest(output, name="isabelle"):
    return json.loads(
        (output / "heldout" / f"{name}.json").read_text(encoding="utf-8")
    )


def test_source_reader_streams_theory_arrays_across_tiny_chunks(
    tmp_path,
    monkeypatch,
):
    proofs = {
        "First.Theory": [
            trajectory(
                "first",
                [
                    transition("first before alpha", "apply"),
                    transition("first after omega"),
                ],
            ),
            trajectory(
                "second",
                [
                    transition("second before alpha", "apply"),
                    transition("second after omega"),
                ],
            ),
        ],
        "Second.Theory": [
            trajectory(
                "third",
                [
                    transition("third before alpha", "apply"),
                    transition("third after omega"),
                ],
            )
        ],
    }
    source = tmp_path / "streamed.json"
    source.write_text(json.dumps(proofs, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(builder, "STREAM_CHUNK_BYTES", 7, raising=False)

    streamed = list(builder.iter_source_trajectories(source))

    identities = [
        (theory, index, proof["statement"])
        for theory, index, proof in streamed
    ]
    assert identities == [
        ("First.Theory", 0, "first"),
        ("First.Theory", 1, "second"),
        ("Second.Theory", 0, "third"),
    ]


def test_adjacent_states_final_rejection_and_nested_qed_are_literal(tmp_path):
    long_statement = "theorem " + "very_long_body " * 45
    proof = trajectory(
        long_statement,
        [
            transition("alpha beta gamma delta", "proof"),
            transition("one two three four", "next"),
            transition("red blue green yellow", "qed"),
            transition("finished terminal state", "unused final step"),
        ],
    )
    invalid = trajectory(
        "invalid fields",
        [
            transition("", "has tactic"),
            transition("nonempty state", ""),
            transition("same state words", "apply"),
            transition("same state words", "apply"),
            transition("", "final"),
        ],
    )

    output, _, factory = run_build(
        tmp_path,
        [
            source_item("Nested.Theory", 7, proof),
            source_item("Nested.Theory", 8, invalid),
        ],
    )
    rows = raw_rows(output)

    assert factory.calls == 2, "the 2.3 GB source must be streamed in two passes"
    assert [row["transition_index"] for row in rows] == [0, 1, 2]
    assert [row["tactic"] for row in rows] == ["proof", "next", "qed"]
    assert [(row["state_before"], row["state_after"]) for row in rows] == [
        ("alpha beta gamma delta", "one two three four"),
        ("one two three four", "red blue green yellow"),
        ("red blue green yellow", "finished terminal state"),
    ]
    assert all(row["theorem_statement"] == long_statement.strip() for row in rows)
    assert len(rows[0]["theorem_statement"]) > 400
    assert all(row["transition_index"] != 3 for row in rows)


def test_goal_target_schema_and_rendered_output_reconstruct_exactly(tmp_path):
    theorem = "shows " + ("FULL_UNTRUNCATED_STATEMENT " * 25)
    proof = trajectory(
        theorem,
        [
            transition("before alpha beta gamma", "by exact_tactic"),
            transition("after delta epsilon zeta"),
        ],
    )

    output, _, _ = run_build(tmp_path, [source_item("Theory", 0, proof)])
    row = raw_rows(output)[0]

    expected_goal = (
        f"THEOREM\n{theorem.strip()}\n"
        "STATE_BEFORE\nbefore alpha beta gamma"
    )
    expected_target = (
        "TACTIC\nby exact_tactic\n"
        "STATE_AFTER\nafter delta epsilon zeta"
    )
    required = {
        "schema_version",
        "id",
        "trajectory_id",
        "transition_index",
        "theorem_statement",
        "facts",
        "cited",
        "premise_aliases",
        "local_assumptions",
        "local_names",
        "state_before",
        "tactic",
        "state_after",
        "goal",
        "target",
        "text",
        "mask_start",
        "mask_end",
        "source_metadata",
    }
    assert required <= set(row)
    assert row["schema_version"] == "isabelle-transition-v2"
    assert row["source_metadata"]["schema_version"] == "isabelle-build-source-v2"
    assert row["source_metadata"]["source_roots"]
    assert row["source_metadata"]["index_roots"] == {}
    assert len(row["source_metadata"]["source_manifest_root_sha256"]) == 64
    assert len(row["source_metadata"]["quality_filter_root_sha256"]) == 64
    assert len(row["source_metadata"]["schema_generation_root_sha256"]) == 64
    assert row["goal"] == expected_goal
    assert row["target"] == expected_target
    assert len(row["id"]) == len(row["trajectory_id"]) == 64
    assert re.fullmatch(r"[0-9a-f]{64}", row["id"])
    assert re.fullmatch(r"[0-9a-f]{64}", row["trajectory_id"])

    block, continuation = row["text"].split("\n---\n", 1)
    assert row["mask_start"] == 0
    assert row["mask_end"] == len(block)
    assert row["text"][: row["mask_end"]] == block
    assert continuation == f"GOAL\n{row['goal']}\n{row['target']}"
    assert row["text"] == f"{block}\n---\nGOAL\n{expected_goal}\n{expected_target}"


def test_state_carryover_paste_boundary_is_strictly_below_half(tmp_path):
    cases = [
        ("below", "a b c d", "a e f g", 0.25),
        ("equal", "h i j k", "h i m n", 0.5),
        ("above", "p q r s", "p q r t", 0.75),
        ("repeated_equal", "a", "a b b b", 0.5),
    ]
    proofs = [
        source_item(
            "Paste",
            i,
            trajectory(
                label,
                [transition(before, "apply"), transition(after)],
            ),
        )
        for i, (label, before, after, _) in enumerate(cases)
    ]

    for _, before, after, expected in cases:
        assert builder.paste_share(after, before) == expected
    assert builder.paste_share("a , ,", ",") == 0.5
    assert builder.paste_share("a, b!", "a ?") == 0.25
    assert builder.paste_share("", "anything") == 1.0

    output, _, _ = run_build(tmp_path, proofs)
    assert [row["theorem_statement"] for row in raw_rows(output)] == ["below"]


def test_aliases_locals_and_qualified_global_facts_render_losslessly(tmp_path):
    premises = {
        "z": ["Global.Z", "zeta statement"],
        "a": ["Global.A", "alpha   statement"],
        "also_a": ["Global.A", "alpha statement"],
        "h": ["local.assms(1)", "x > 0"],
    }
    proof = trajectory(
        "alias theorem",
        [
            transition("before unique tokens", "apply alias", premises),
            transition("after different symbols"),
        ],
    )

    output, _, _ = run_build(tmp_path, [source_item("Aliases", 0, proof)])
    row = raw_rows(output)[0]
    block = row["text"][: row["mask_end"]]

    assert row["facts"] == {
        "Global.A": "alpha statement",
        "Global.Z": "zeta statement",
    }
    assert row["cited"] == ["Global.A", "Global.Z"]
    assert row["premise_aliases"] == {
        "a": "Global.A",
        "also_a": "Global.A",
        "z": "Global.Z",
    }
    assert row["local_assumptions"] == {"h": "x > 0"}
    assert row["local_names"] == {"h": "local.assms(1)"}
    assert block.splitlines() == [
        "I know these mathematical statements:",
        "a [Global.A] : alpha statement",
        "also_a [Global.A] : alpha statement",
        "z [Global.Z] : zeta statement",
        "Local assumptions:",
        "h [local.assms(1)] : x > 0",
    ]
    assert block in row["text"].split("\n---\n", 1)[0]


@pytest.mark.parametrize(
    "premises",
    [
        {"ok": ["Global.ok", "ok"], "bad": "not a pair"},
        {"ok": ["Global.ok", "ok"], "bad": ["Global.bad"]},
        {"ok": ["Global.ok", "ok"], "bad": ["Global.bad", "bad", "extra"]},
        {"ok": ["Global.ok", "ok"], "bad": ["", "bad"]},
        {"ok": ["Global.ok", "ok"], "bad": ["Global.bad", ""]},
        ["not", "a", "mapping"],
    ],
)
def test_malformed_premise_rejects_whole_transition(tmp_path, premises):
    malformed = trajectory(
        "malformed",
        [
            transition("malformed before state", "apply", premises),
            transition("malformed after state"),
        ],
    )
    valid = trajectory(
        "valid",
        [
            transition("valid before alpha", "apply"),
            transition("valid after omega"),
        ],
    )

    output, _, _ = run_build(
        tmp_path,
        [
            source_item("Premises", 0, malformed),
            source_item("Premises", 1, valid),
        ],
    )
    assert [row["theorem_statement"] for row in raw_rows(output)] == ["valid"]


def test_local_only_transition_is_rejected_because_a_global_fact_is_required(tmp_path):
    local_only = trajectory(
        "local only",
        [
            transition(
                "local before alpha",
                "apply",
                {"h": ["local.assms", "assumption"]},
            ),
            transition("local after omega"),
        ],
    )
    valid = trajectory(
        "valid",
        [
            transition("valid before alpha", "apply"),
            transition("valid after omega"),
        ],
    )

    output, _, _ = run_build(
        tmp_path,
        [
            source_item("Premises", 0, local_only),
            source_item("Premises", 1, valid),
        ],
    )
    assert [row["theorem_statement"] for row in raw_rows(output)] == ["valid"]


def test_ambiguous_global_rejects_all_users_without_alpha_renaming(tmp_path):
    proofs = [
        source_item(
            "Facts",
            0,
            trajectory(
                "alpha x",
                [
                    transition(
                        "x before alpha",
                        "apply",
                        {"a": ["Global.alpha", "P ?x"]},
                    ),
                    transition("x after omega"),
                ],
            ),
        ),
        source_item(
            "Facts",
            1,
            trajectory(
                "alpha y",
                [
                    transition(
                        "y before alpha",
                        "apply",
                        {"a": ["Global.alpha", "P ?y"]},
                    ),
                    transition("y after omega"),
                ],
            ),
        ),
        source_item(
            "Facts",
            2,
            trajectory(
                "stable one",
                [
                    transition(
                        "stable one before",
                        "apply",
                        {"s": ["Stable.fact", "stable   statement"]},
                    ),
                    transition("omega ending symbols"),
                ],
            ),
        ),
        source_item(
            "Facts",
            3,
            trajectory(
                "stable two",
                [
                    transition(
                        "stable two before",
                        "apply",
                        {"s": ["Stable.fact", "stable statement"]},
                    ),
                    transition("zeta terminal symbols"),
                ],
            ),
        ),
    ]

    output, stats, _ = run_build(tmp_path, proofs)
    rows = raw_rows(output)

    assert [row["theorem_statement"] for row in rows] == [
        "stable one",
        "stable two",
    ]
    assert all(row["facts"] == {"Stable.fact": "stable statement"} for row in rows)
    assert stats["ambiguous_global_names"] == 1
    assert stats["dropped_ambiguous_fact"] == 2


@pytest.mark.parametrize(
    "abort_step",
    ["oops", "sorry", "abort", "(OoPs)", "by; ABORT"],
)
def test_abort_word_rejects_the_entire_trajectory(tmp_path, abort_step):
    aborted = trajectory(
        "must all disappear",
        [
            transition("good before alpha", "apply"),
            transition("good middle omega", abort_step),
            transition("terminal final state"),
        ],
    )
    adversarial = trajectory(
        "ordinary substrings",
        [
            transition(
                "substring before alpha",
                "simp add: oopsie sorryful abortive",
            ),
            transition("substring after omega"),
        ],
    )

    output, stats, _ = run_build(
        tmp_path,
        [
            source_item("Abort", 0, aborted),
            source_item("Abort", 1, adversarial),
        ],
    )

    assert [row["theorem_statement"] for row in raw_rows(output)] == [
        "ordinary substrings"
    ]
    assert stats["dropped_aborted_trajectories"] == 1


@pytest.mark.parametrize(
    "safe_step",
    [
        "simp add: Foo.sorry Bar.oops Baz.abort",
        "simp add: oopsie sorryful abortive",
        'simp "sorry oops abort"',
        "simp (* sorry (* oops *) abort *)",
        r"simp \<open>sorry (* oops *) abort\<close>",
        'simp ‹sorry "oops" abort›',
        'simp "quoted \\"sorry\\"" (* abort *) Foo.oops',
    ],
)
def test_abort_scanner_ignores_isabelle_noncode_and_qualified_names(safe_step):
    transitions = [transition("before state", safe_step)]
    assert not builder._is_aborted(transitions)


@pytest.mark.parametrize(
    "actual_command",
    [
        "by sorry",
        "oops",
        "abort",
        "simp (* hidden sorry *) ; sorry",
        r"simp \<open>hidden abort\<close>; oops",
    ],
)
def test_abort_scanner_rejects_standalone_commands(actual_command):
    transitions = [transition("before state", actual_command)]
    assert builder._is_aborted(transitions)


def test_exact_rendered_dedup_is_deterministic(tmp_path):
    duplicate = trajectory(
        "same theorem",
        [
            transition("same before alpha", "same tactic"),
            transition("same after omega"),
        ],
    )
    items = [
        source_item("Dedup", 0, duplicate),
        source_item("Dedup", 1, duplicate),
    ]

    output_a, stats_a, _ = run_build(tmp_path, items, out_name="first")
    output_b, stats_b, _ = run_build(tmp_path, items, out_name="second")
    bytes_a = (output_a / "raw" / "isabelle.jsonl").read_bytes()
    bytes_b = (output_b / "raw" / "isabelle.jsonl").read_bytes()

    assert len(raw_rows(output_a)) == 1
    assert stats_a["dropped_duplicate"] == stats_b["dropped_duplicate"] == 1
    assert bytes_a == bytes_b


def test_qwen_text_plus_eos_must_fit_16384_before_holdout_counting(tmp_path):
    def encoded_length(text):
        if "accepted_16383" in text:
            return 16_383
        if "rejected_16384" in text:
            return 16_384
        return 10

    proofs = [
        source_item(
            "Length",
            0,
            trajectory(
                "accepted_16383",
                [
                    transition(
                        "accepted before alpha",
                        "apply",
                        {"a": ["Accepted.fact", "accepted statement"]},
                    ),
                    transition("accepted after omega"),
                ],
            ),
        ),
        source_item(
            "Length",
            1,
            trajectory(
                "rejected_16384",
                [
                    transition(
                        "rejected before alpha",
                        "apply",
                        {"r": ["Rejected.fact", "rejected statement"]},
                    ),
                    transition("rejected after omega"),
                ],
            ),
        ),
    ]

    output, stats, _ = run_build(
        tmp_path,
        proofs,
        tokenizer=FakeQwenTokenizer(encoded_length),
    )
    manifest = heldout_manifest(output)

    assert [row["theorem_statement"] for row in raw_rows(output)] == [
        "accepted_16383"
    ]
    assert stats["dropped_overlength"] == 1
    assert manifest["eligible_fact_names"] == ["Accepted.fact"]
    assert manifest["tokenizer"]["eos_token_id"] == 151643
    assert manifest["tokenizer"]["identity"].startswith("Qwen/Qwen2.5")
    assert manifest["tokenizer"]["tokenizer_json_sha256"] == "f" * 64
    assert manifest["tokenizer"]["tokenizer_config_sha256"] == "e" * 64
    assert manifest["tokenizer"]["behavior_digest"] == "d" * 64
    assert (
        manifest["tokenizer"]["tokenizers_version"]
        == "synthetic-test-double"
    )


def test_family_local_holdout_isolates_trajectories_and_own_proof_leaks(tmp_path):
    citing = trajectory(
        "citing theorem",
        [
            transition(
                "cite before alpha",
                "cite held",
                {"held": ["Held.fact", "held statement"]},
            ),
            transition(
                "sibling middle omega",
                "safe sibling",
                {"common": ["Common.fact", "common statement"]},
            ),
            transition("citing terminal zeta"),
        ],
    )
    own_proof = trajectory(
        "held statement",
        [
            transition(
                "own before alpha",
                "prove own",
                {"common": ["Common.fact", "common statement"]},
            ),
            transition("own after omega"),
        ],
    )
    safe = [
        source_item(
            "Heldout",
            index,
            trajectory(
                f"safe theorem {index}",
                [
                    transition(
                        f"safe{index} before alpha",
                        "safe",
                        {"common": ["Common.fact", "common statement"]},
                    ),
                    transition(f"safe{index} after omega"),
                ],
            ),
        )
        for index in range(2, 5)
    ]
    statement_aliases = [
        source_item(
            "Heldout",
            5,
            trajectory(
                "alias one theorem",
                [
                    transition(
                        "alias one before",
                        "alias",
                        {"one": ["Alias.one", "duplicate statement"]},
                    ),
                    transition("first terminal omega"),
                ],
            ),
        ),
        source_item(
            "Heldout",
            6,
            trajectory(
                "alias two theorem",
                [
                    transition(
                        "alias two before",
                        "alias",
                        {"two": ["Alias.two", "duplicate statement"]},
                    ),
                    transition("second terminal zeta"),
                ],
            ),
        ),
    ]

    output, _, _ = run_build(
        tmp_path,
        [
            source_item("Heldout", 0, citing),
            source_item("Heldout", 1, own_proof),
            *safe,
            *statement_aliases,
        ],
        heldout=1,
        seed=9,
    )
    train = read_jsonl(output / "shards" / "isabelle.jsonl")
    eval_rows = read_jsonl(output / "eval" / "isabelle.jsonl")
    manifest = heldout_manifest(output)

    assert manifest["facts"] == ["Held.fact"]
    assert manifest["statement_hashes"] == [
        builder.canonical_statement_hash("held statement")
    ]
    assert {row["theorem_statement"] for row in eval_rows} == {"citing theorem"}
    assert [row["transition_index"] for row in eval_rows] == [0]
    assert "held statement" not in {
        row["theorem_statement"] for row in train
    }
    assert not any(
        row["theorem_statement"] == "citing theorem"
        and row["transition_index"] == 1
        for row in train + eval_rows
    )
    assert {row["theorem_statement"] for row in train} == {
        "safe theorem 2",
        "safe theorem 3",
        "safe theorem 4",
        "alias one theorem",
        "alias two theorem",
    }
    assert manifest["trajectory_counts"]["direct_eval"] == 1
    assert manifest["trajectory_counts"]["own_proof"] == 1
    assert manifest["trajectory_counts"]["excluded_from_train"] == 2
    assert manifest["row_counts"]["eval"] == 1
    assert manifest["row_counts"]["dropped_siblings"] == 1
    assert manifest["row_counts"]["dropped_own_proof"] == 1
    assert manifest["canonicalization"] == {
        "family": "isabelle",
        "scheme": "quoted-layout-v2",
        "version": 2,
    }


ISO_ASSOC_STATEMENT = (
    "iso_assoc: fixes a :: \"'a\" and b :: \"'a\" and c :: \"'a\" "
    'assumes "ide a" and "ide b" and "ide c" '
    'shows "local.iso \\<a>[a, b, c]"'
)
SUBST_CLOSED_SUBSET_STATEMENT = (
    'substClosedSubset: fixes Rel :: "(pi \\<times> pi) set" '
    'shows "substClosed Rel \\<subseteq> Rel"'
)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (f"lemma {SUBST_CLOSED_SUBSET_STATEMENT}", SUBST_CLOSED_SUBSET_STATEMENT),
        (f"theorem {SUBST_CLOSED_SUBSET_STATEMENT}", SUBST_CLOSED_SUBSET_STATEMENT),
        (f"corollary {SUBST_CLOSED_SUBSET_STATEMENT}", SUBST_CLOSED_SUBSET_STATEMENT),
        (f"proposition {SUBST_CLOSED_SUBSET_STATEMENT}", SUBST_CLOSED_SUBSET_STATEMENT),
        (
            f"lemmas {SUBST_CLOSED_SUBSET_STATEMENT}",
            f"lemmas {SUBST_CLOSED_SUBSET_STATEMENT}",
        ),
        (
            f"context lemma {SUBST_CLOSED_SUBSET_STATEMENT}",
            f"context lemma {SUBST_CLOSED_SUBSET_STATEMENT}",
        ),
        (
            f"lemma_aux {SUBST_CLOSED_SUBSET_STATEMENT}",
            f"lemma_aux {SUBST_CLOSED_SUBSET_STATEMENT}",
        ),
    ],
)
def test_own_proof_declaration_normalization_is_conservative(source, expected):
    assert builder.normalize_declaration_statement(source) == expected


@pytest.mark.parametrize(
    ("held_name", "prefix_only_name"),
    [
        (
            "Isometries.geodesic_segment_dist",
            "Isometries.geodesic_segment_dist_le",
        ),
        (
            "Caratheodory.extend_measure_caratheodory",
            "Caratheodory.extend_measure_caratheodory_pair",
        ),
        ("CR.one_subst", "CR.one_subst_aux"),
    ],
)
def test_qualified_name_prefix_alerts_are_not_exposures(
    held_name,
    prefix_only_name,
):
    assert builder.contains_qualified_name(held_name, held_name)
    assert not builder.contains_qualified_name(prefix_only_name, held_name)
    assert not builder.contains_qualified_name(f"x{held_name}", held_name)


@pytest.mark.parametrize(
    "candidate",
    [
        f"x{SUBST_CLOSED_SUBSET_STATEMENT}",
        f"{SUBST_CLOSED_SUBSET_STATEMENT}_aux",
        SUBST_CLOSED_SUBSET_STATEMENT.replace(
            "substClosedSubset",
            "substClosedSubset_pair",
            1,
        ),
        SUBST_CLOSED_SUBSET_STATEMENT.replace(
            "\\<subseteq> Rel",
            "\\<subseteq> RelExtra",
            1,
        ),
    ],
)
def test_embedded_statement_matching_rejects_near_substrings(candidate):
    assert not builder.contains_normalized_statement(
        candidate,
        SUBST_CLOSED_SUBSET_STATEMENT,
    )


def test_embedded_statement_matching_accepts_layout_and_field_boundaries():
    wrapped = (
        "THEOREM\nlemma "
        + SUBST_CLOSED_SUBSET_STATEMENT.replace(" fixes ", "\n  fixes ")
        + "\nSTATE_BEFORE\nproof state"
    )
    assert builder.contains_normalized_statement(
        wrapped,
        SUBST_CLOSED_SUBSET_STATEMENT,
    )


def test_real_held_statement_leaks_become_typed_trajectory_drops(tmp_path):
    common = {"common": ["Common.fact", "common statement"]}
    direct_iso = trajectory(
        "direct iso citation",
        [
            transition(
                "direct iso before alpha",
                "use iso",
                {
                    "iso_assoc": [
                        "MonoidalCategory.elementary_monoidal_category.iso_assoc",
                        ISO_ASSOC_STATEMENT,
                    ]
                },
            ),
            transition("iso terminal omega zeta"),
        ],
    )
    direct_subst = trajectory(
        "direct subst citation",
        [
            transition(
                "direct subst before alpha",
                "use subst",
                {
                    "substClosedSubset": [
                        "Rel.substClosedSubset",
                        SUBST_CLOSED_SUBSET_STATEMENT,
                    ]
                },
            ),
            transition("subst terminal omega zeta"),
        ],
    )
    local_leak = trajectory(
        "MonoidalCategory local proof",
        [
            transition(
                "local leak before alpha",
                "local first",
                {
                    **common,
                    "iso_assoc": ["local.iso_assoc", ISO_ASSOC_STATEMENT],
                },
            ),
            transition(
                "local sibling middle omega",
                "local sibling",
                common,
            ),
            transition("local terminal zeta"),
        ],
    )
    declaration_leak = trajectory(
        f"lemma {SUBST_CLOSED_SUBSET_STATEMENT}",
        [
            transition("declaration before alpha", "prove", common),
            transition("declaration after omega"),
        ],
    )
    state_target_leak = trajectory(
        "state target leak",
        [
            transition("state leak before alpha", "advance", common),
            transition(f"proof goal (1 subgoal): 1. {ISO_ASSOC_STATEMENT}"),
        ],
    )
    state_before_leak = trajectory(
        "state before leak",
        [
            transition(
                f"proof goal (1 subgoal): 1. {ISO_ASSOC_STATEMENT}",
                "advance",
                common,
            ),
            transition("state before terminal omega zeta"),
        ],
    )
    safe = [
        source_item(
            "Leaks",
            index,
            trajectory(
                f"safe theorem {index}",
                [
                    transition(f"safe {index} before alpha", "safe", common),
                        transition(f"terminal {index} omega zeta"),
                ],
            ),
        )
        for index in range(6, 9)
    ]

    output, stats, _ = run_build(
        tmp_path,
        [
            source_item("Leaks", 0, direct_iso),
            source_item("Leaks", 1, direct_subst),
            source_item("Leaks", 2, local_leak),
            source_item("Leaks", 3, declaration_leak),
            source_item("Leaks", 4, state_target_leak),
            source_item("Leaks", 5, state_before_leak),
            *safe,
        ],
        heldout=2,
        seed=20260801,
    )
    train = read_jsonl(output / "shards" / "isabelle.jsonl")
    eval_rows = read_jsonl(output / "eval" / "isabelle.jsonl")
    manifest = heldout_manifest(output)

    assert manifest["facts"] == [
        "MonoidalCategory.elementary_monoidal_category.iso_assoc",
        "Rel.substClosedSubset",
    ]
    assert manifest["statements"] == {
        "MonoidalCategory.elementary_monoidal_category.iso_assoc": (
            ISO_ASSOC_STATEMENT
        ),
        "Rel.substClosedSubset": SUBST_CLOSED_SUBSET_STATEMENT,
    }
    assert {row["theorem_statement"] for row in eval_rows} == {
        "direct iso citation",
        "direct subst citation",
    }
    assert {row["theorem_statement"] for row in train} == {
        "safe theorem 6",
        "safe theorem 7",
        "safe theorem 8",
    }
    assert manifest["trajectory_counts"]["local_statement_exposure"] == 1
    assert manifest["trajectory_counts"]["own_proof_declaration_exposure"] == 1
    assert manifest["trajectory_counts"]["target_state_exposure"] == 2
    assert manifest["row_counts"]["dropped_local_statement_exposure"] == 2
    assert manifest["row_counts"]["dropped_own_proof_declaration_exposure"] == 1
    assert manifest["row_counts"]["dropped_target_state_exposure"] == 2
    assert manifest["row_counts"]["dropped_statement_exposure"] == 5
    assert manifest["row_counts"]["eval"] == 2
    assert stats["train_rows"] == 3
    assert stats["eval_rows"] == 2
    assert manifest["source"]["sha256"] == builder.SOURCE_SHA256
    assert manifest["tokenizer"]["tokenizer_json_sha256"] == "f" * 64
    assert manifest["tokenizer"]["tokenizer_config_sha256"] == "e" * 64
    assert manifest["tokenizer"]["behavior_digest"] == "d" * 64


def test_positive_heldout_refuses_when_safe_tail_is_too_small(tmp_path):
    proof = trajectory(
        "only tail theorem",
        [
            transition(
                "before alpha beta",
                "apply",
                {"only": ["Only.fact", "only statement"]},
            ),
            transition("after omega zeta"),
        ],
    )

    with pytest.raises(
        builder.BuildError,
        match=r"requested 2 heldout facts.*only 1 safe tail",
    ):
        run_build(
            tmp_path,
            [source_item("Heldout", 0, proof)],
            heldout=2,
        )


def test_heldout_zero_emits_raw_staging_with_pinned_manifest(tmp_path):
    proof = trajectory(
        "staging theorem",
        [
            transition("staging before alpha", "apply"),
            transition("staging after omega"),
        ],
    )

    output, _, _ = run_build(tmp_path, [source_item("Raw", 0, proof)])
    manifest = heldout_manifest(output)

    assert (output / "raw" / "isabelle.jsonl").is_file()
    assert not (output / "shards" / "isabelle.jsonl").exists()
    assert not (output / "eval" / "isabelle.jsonl").exists()
    assert manifest["mode"] == "raw_staging"
    assert "debug/staging-only" in manifest["policy"]
    assert "not scientifically complete" in manifest["policy"]
    assert manifest["source"] == {
        "dataset": "Simontwice/premise_selection_in_isabelle",
        "revision": "f947ccc827ccd236464e19cd4cc23dfda7fc5575",
        "file": "raw_data/human_data/all_data.json",
        "size_bytes": 2_327_313_460,
        "sha256": "aa71609de90fee138835cfdf9e954becb1b231a293ac19bd98951e6d8bec8e7d",
        "license": "Apache-2.0",
    }
    assert manifest["seed"] == 23


def test_statement_hash_and_manifest_use_shared_quoted_layout_v2():
    statement = 'for x holds x = "a  b"'
    payload = "\0".join(
        (
            "statement",
            "2",
            "isabelle",
            "quoted-layout-v2",
            builder.normalize_layout(statement),
        )
    )
    assert builder.canonical_statement_hash(statement) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def test_finalized_vendored_tokenizer_has_the_approved_four_part_seal():
    tokenizer_root = (
        Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
    )
    tokenizer = builder.load_vendored_tokenizer(tokenizer_root)

    assert (
        builder.APPROVED_TOKENIZER_JSON_SHA256
        == "3fd169731d2cbde95e10bf356d66d5997fd885dd8dbb6fb4684da3f23b2585d8"
    )
    assert (
        builder.APPROVED_TOKENIZER_CONFIG_SHA256
        == "ddb9f850ca6559a928bb25d511f72e3c6eff81395334a4e0eeec670448333d09"
    )
    assert (
        builder.APPROVED_TOKENIZER_BEHAVIOR_SHA256
        == "aa90434a251a434bbc938ddb3be6683a73fa94150377b5ccd2cbd7880358661a"
    )
    assert builder.APPROVED_TOKENIZERS_VERSION == "0.22.2"
    assert (
        tokenizer.tokenizer_json_sha256
        == builder.APPROVED_TOKENIZER_JSON_SHA256
    )
    assert (
        tokenizer.tokenizer_config_sha256
        == builder.APPROVED_TOKENIZER_CONFIG_SHA256
    )
    assert (
        tokenizer.behavior_digest
        == builder.APPROVED_TOKENIZER_BEHAVIOR_SHA256
    )
    assert tokenizer.tokenizers_version == builder.APPROVED_TOKENIZERS_VERSION


def test_tokenizer_loader_refuses_same_family_but_unapproved_bytes(tmp_path):
    tokenizer_root = (
        Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
    )
    config = json.loads(
        (tokenizer_root / "tokenizer_config.json").read_text(encoding="utf-8")
    )

    bad_tokenizer = tmp_path / "bad-tokenizer"
    bad_tokenizer.mkdir()
    (bad_tokenizer / "tokenizer.json").write_text("{}")
    (bad_tokenizer / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(builder.BuildError, match="tokenizer.json SHA-256"):
        builder.load_vendored_tokenizer(bad_tokenizer)

    bad_config = tmp_path / "bad-config"
    bad_config.mkdir()
    (bad_config / "tokenizer.json").symlink_to(
        tokenizer_root / "tokenizer.json"
    )
    config["model_max_length"] += 1
    (bad_config / "tokenizer_config.json").write_text(json.dumps(config))
    with pytest.raises(builder.BuildError, match="tokenizer_config.json SHA-256"):
        builder.load_vendored_tokenizer(bad_config)


def test_tokenizer_loader_enforces_version_and_behavior_seals(
    monkeypatch,
):
    tokenizer_root = (
        Path(__file__).resolve().parents[1] / "tokenizers" / "qwen25-vendored"
    )
    monkeypatch.setattr(
        builder,
        "APPROVED_TOKENIZERS_VERSION",
        "0.0.0-rejected",
        raising=False,
    )
    with pytest.raises(builder.BuildError, match="tokenizers implementation"):
        builder.load_vendored_tokenizer(tokenizer_root)

    monkeypatch.setattr(
        builder,
        "APPROVED_TOKENIZERS_VERSION",
        "0.22.2",
        raising=False,
    )
    monkeypatch.setattr(
        builder,
        "APPROVED_TOKENIZER_BEHAVIOR_SHA256",
        "0" * 64,
        raising=False,
    )
    with pytest.raises(builder.BuildError, match="behavior digest"):
        builder.load_vendored_tokenizer(tokenizer_root)


def _production_documentation_blocks():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "corpus" / "README.md").read_text(encoding="utf-8")
    plan = (root / "CORPUS_BUILD_PLAN.md").read_text(encoding="utf-8")
    return root, {
        "readme": readme.split("## Rebuild", 1)[1].split(
            "## Quality flags",
            1,
        )[0],
        "plan": plan.split("### Orchestration", 1)[1].split(
            "### Intra-job parallelism",
            1,
        )[0],
    }


def _production_plan_commands(plan):
    commands = []
    in_fence = False
    inspect_fence = False
    previous_nonempty = ""
    historical_marker = "<!-- NON-EXECUTABLE / HISTORICAL -->"

    for line_number, line in enumerate(plan.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            if in_fence:
                in_fence = False
                inspect_fence = False
            else:
                language = stripped[3:].strip().lower()
                in_fence = True
                inspect_fence = (
                    language in {"", "bash", "sh", "shell"}
                    and previous_nonempty != historical_marker
                )
            previous_nonempty = stripped
            continue

        inspect_line = not in_fence or inspect_fence
        script_command = re.search(
            r"\bpython3?\s+(scripts/(?:build|split|merge)_[A-Za-z0-9_./-]+\.py)\b",
            line,
        )
        shell_command = inspect_fence or bool(
            re.match(r"\s*(?:cat|python3?|SHARD_PATH=)", line)
        )
        flat_output = shell_command and re.search(
            r"\$(?:OUT|SHARED)/(?:shards/)?(?:"
            r"\{[^}\n]*(?:metamath|mizar|atp|isabelle)[^}\n]*\}|"
            r"(?:metamath|mizar|atp|isabelle)(?:_eval)?)\.jsonl\b",
            line,
        )
        if inspect_line and (script_command or flat_output):
            commands.append(
                {
                    "line_number": line_number,
                    "line": line,
                    "script": script_command.group(1) if script_command else None,
                    "flat_output": bool(flat_output),
                }
            )

        if stripped:
            previous_nonempty = stripped

    return commands


def test_entire_plan_has_only_current_production_commands():
    root = Path(__file__).resolve().parents[1]
    plan = (root / "CORPUS_BUILD_PLAN.md").read_text(encoding="utf-8")
    commands = _production_plan_commands(plan)
    assert commands

    pending_semantic_builders = {
        "scripts/build_atp_shard.py",
        "scripts/build_mizar_shard.py",
        "scripts/build_thproofs_shard.py",
    }
    failures = []
    for command in commands:
        location = f"line {command['line_number']}: {command['line'].strip()}"
        if command["script"] and not (root / command["script"]).is_file():
            failures.append(f"missing script at {location}")
        if (
            command["script"] in pending_semantic_builders
            and "--heldout" in command["line"]
        ):
            failures.append(
                f"pending semantic builder performs its own split at {location}"
            )
        if command["script"] and command["script"].startswith("scripts/split_"):
            failures.append(f"pending semantic splitter is executable at {location}")
        if command["script"] and command["script"].startswith("scripts/merge_"):
            failures.append(f"obsolete merge is executable at {location}")
        if command["flat_output"]:
            failures.append(f"flat family output is executable at {location}")

    assert not failures, "\n".join(failures)
    assert all(
        command["script"] != "scripts/build_mizar_shard.py" for command in commands
    )


def test_heldout_section_is_historical_without_choosing_pending_policy():
    root = Path(__file__).resolve().parents[1]
    plan = (root / "CORPUS_BUILD_PLAN.md").read_text(encoding="utf-8")
    section = plan.split("## 4.", 1)[1].split("## 5.", 1)[0]
    normalized = " ".join(section.split())

    assert "NON-EXECUTABLE / HISTORICAL" in section
    assert "python scripts/build_mizar_shard.py" not in section
    assert "Emits `heldout.json`" not in section
    assert "Merge into one `heldout.json`" not in section
    for family in ("Mizar", "thproofs", "prf2", "ENIGMA"):
        assert family in section
    assert "pending authoritative semantic-class splitter" in normalized
    assert "pooled-versus-strata policy remains unresolved" in normalized
    assert "Metamath" in section
    assert "positive-heldout isabelle" in normalized.lower()


def test_production_documentation_references_only_existing_scripts():
    root, blocks = _production_documentation_blocks()
    for block in blocks.values():
        references = re.findall(
            r"(?m)^\s*python\s+(scripts/[A-Za-z0-9_./-]+\.py)\b",
            block,
        )
        assert references
        assert all((root / reference).is_file() for reference in references)
        command_lines = [
            line for line in block.splitlines() if line.lstrip().startswith("python ")
        ]
        assert all(not line.rstrip().endswith("\\") for line in command_lines)


def test_documented_isabelle_paths_and_split_are_builder_local():
    _, blocks = _production_documentation_blocks()
    combined = "\n".join(blocks.values())

    assert "merge_heldout.py" not in combined
    assert "$OUT/isabelle.jsonl" not in combined
    assert not re.search(r"\$OUT/\{[^}\n]*isabelle[^}\n]*\}\.jsonl", combined)
    assert "--family isabelle=isabelle" not in combined

    assert "`corpus/shards/isabelle.jsonl`" in blocks["readme"]
    assert "`corpus/eval/isabelle.jsonl`" in blocks["readme"]
    assert "`corpus/heldout/isabelle.json`" in blocks["readme"]
    assert "`$OUT/shards/isabelle.jsonl`" in blocks["plan"]
    assert "`$OUT/eval/isabelle.jsonl`" in blocks["plan"]
    assert "`$OUT/heldout/isabelle.json`" in blocks["plan"]


def test_documented_workflow_is_gated_and_uses_positive_isabelle_heldout():
    _, blocks = _production_documentation_blocks()
    for block in blocks.values():
        commands = [
            line.strip()
            for line in block.splitlines()
            if "python scripts/build_isabelle_shard.py" in line
        ]
        assert commands
        assert all("--heldout 500" in command for command in commands)
        assert all(
            "--tokenizer-path tokenizers/qwen25-vendored" in command
            for command in commands
        )

    assert "NON-EXECUTABLE / HISTORICAL" in blocks["plan"]
    assert "checklist.md" in blocks["plan"]
    for block in blocks.values():
        assert "scripts/split_heldout.py" in block
        assert "raw" in block
        assert "pending" in block.lower()


def test_source_gate_refuses_missing_empty_wrong_size_and_wrong_digest(tmp_path):
    missing = tmp_path / "missing.json"
    with pytest.raises(builder.BuildError, match="missing"):
        builder.verify_source_file(missing, expected_size=3, expected_sha256="0" * 64)

    empty = tmp_path / "empty.json"
    empty.write_bytes(b"")
    with pytest.raises(builder.BuildError, match="empty"):
        builder.verify_source_file(empty, expected_size=0, expected_sha256="0" * 64)

    source = tmp_path / "source.json"
    source.write_bytes(b"abc")
    digest = hashlib.sha256(b"abc").hexdigest()
    with pytest.raises(builder.BuildError, match="byte size"):
        builder.verify_source_file(source, expected_size=4, expected_sha256=digest)
    with pytest.raises(builder.BuildError, match="SHA-256"):
        builder.verify_source_file(source, expected_size=3, expected_sha256="0" * 64)

    metadata = builder.verify_source_file(
        source,
        expected_size=3,
        expected_sha256=digest,
    )
    assert metadata["size_bytes"] == 3
    assert metadata["sha256"] == digest


def test_real_pinned_source_gate_skips_honestly_when_gated_file_is_absent():
    source = Path(
        getattr(
            builder,
            "DEFAULT_SOURCE",
            "/tmp/dscount/magnushammer/raw_data/human_data/all_data.json",
        )
    )
    if not source.exists():
        pytest.skip("pinned gated Magnushammer source is not available locally")
    builder.verify_source_file(source)


def test_production_tokenizer_loader_refuses_a_missing_vendored_path(tmp_path):
    with pytest.raises(builder.BuildError, match="tokenizer.*missing"):
        builder.load_vendored_tokenizer(tmp_path / "missing-tokenizer")


def test_empty_rebuild_fails_and_invalidates_every_active_output(tmp_path):
    source = tmp_path / "source.json"
    source.write_text("synthetic fixture", encoding="utf-8")
    output = tmp_path / "corpus"
    active = [
        output / "raw" / "isabelle.jsonl",
        output / "shards" / "isabelle.jsonl",
        output / "eval" / "isabelle.jsonl",
        output / "heldout" / "isabelle.json",
    ]
    for path in active:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("STALE", encoding="utf-8")
    invalid = trajectory(
        "no eligible rows",
        [
            transition("same state", "apply"),
            transition("same state"),
        ],
    )

    with pytest.raises(builder.BuildError, match="no accepted output"):
        builder.build_corpus(
            source=source,
            out=output,
            name="isabelle",
            heldout=0,
            seed=1,
            tokenizer=FakeQwenTokenizer(),
            source_gate=pinned_test_gate,
            trajectory_iter_factory=TwoPassFactory(
                [source_item("Empty", 0, invalid)]
            ),
        )

    assert not any(path.exists() for path in active)
    assert all(list(path.parent.glob(path.name + ".stale*")) for path in active)
