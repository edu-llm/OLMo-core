"""
Tests for the tool_call_holdout.py script.

The carve is what makes the test set mean anything, and every way it can go wrong is silent: a
held-out tool leaking into training, a held-out orphan nobody trained against, a phrasing bank that
re-splits when it grows. Each of those is a test here.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


holdout = _load("tool_call_holdout", "src/scripts/data/tool_call_holdout.py")
ser = _load("tool_call_serializer", "src/scripts/data/tool_call_serializer.py")


@pytest.fixture(scope="module")
def registry():
    return holdout.load_registry()


# ==================== the shipped registry ====================


def test_shipped_registry_loads_and_matches_the_design(registry):
    assert len(registry.tools) == 57
    assert len(registry.heldout_names()) == 9


def test_every_held_out_tool_has_a_trained_sibling(registry):
    """Hold out the sibling, not the orphan — an orphan measures nothing."""
    for name in registry.heldout_names():
        tool = registry.by_name(name)
        assert tool.sibling_of, f"{name} is held out with no sibling"
        sibling = registry.by_name(tool.sibling_of)
        assert sibling is not None and not sibling.held_out
        assert sibling.domain == tool.domain


def test_registry_agrees_with_the_tools_olmo_core_actually_ships(registry):
    """If someone adds or renames a runtime tool, this fails before data is generated."""
    holdout.validate_against_runtime(registry)


def test_no_implemented_tool_is_held_out(registry):
    """Holding out a tool the product ships trades real capability for a measurement."""
    for name in holdout.IMPLEMENTED_TOOLS:
        assert not registry.by_name(name).held_out


def test_every_registry_name_is_flat(registry):
    """A dotted name raises in the runtime parser, so it must never reach the data."""
    assert [t.name for t in registry.tools if "." in t.name] == []


def test_training_pool_never_contains_a_held_out_tool(registry):
    assert not (registry.pool(split="train") & registry.heldout_names())


def test_dominant_tools_are_flagged_as_unholdable(registry):
    for name in ("calculator", "symbolic_math", "web_search"):
        tool = registry.by_name(name)
        assert tool.cannot_hold_out
        assert not tool.held_out
        # ...and the domain must then say what it substitutes, or the claim would be unfounded.
        assert registry.domains[tool.domain]["substitute_carve_axis"]


# ==================== registry validation ====================


def _write(tmp_path: Path, tools: list[dict]) -> Path:
    p = tmp_path / "reg.json"
    p.write_text(json.dumps({"version": 1, "domains": {"general": {}}, "tools": tools}))
    return p


def test_rejects_a_duplicate_tool_name(tmp_path):
    path = _write(
        tmp_path, [{"name": "f", "domain": "general"}, {"name": "f", "domain": "general"}]
    )
    with pytest.raises(ValueError, match="duplicate tool name"):
        holdout.load_registry(path)


def test_rejects_a_held_out_orphan(tmp_path):
    path = _write(tmp_path, [{"name": "f", "domain": "general", "held_out": True}])
    with pytest.raises(ValueError, match="Holding out an orphan"):
        holdout.load_registry(path)


def test_rejects_holding_out_both_halves_of_a_pair(tmp_path):
    path = _write(
        tmp_path,
        [
            {"name": "a", "domain": "general", "held_out": True, "sibling_of": "b"},
            {"name": "b", "domain": "general", "held_out": True, "sibling_of": "a"},
        ],
    )
    with pytest.raises(ValueError, match="BOTH held out"):
        holdout.load_registry(path)


def test_rejects_a_dangling_sibling_reference(tmp_path):
    path = _write(
        tmp_path, [{"name": "a", "domain": "general", "held_out": True, "sibling_of": "ghost"}]
    )
    with pytest.raises(ValueError, match="not in the registry"):
        holdout.load_registry(path)


# ==================== phrasing bank ====================


def test_template_split_is_deterministic():
    bank = [f"template {i}" for i in range(500)]
    assert holdout.split_templates(bank) == holdout.split_templates(bank)


def test_template_split_is_disjoint_and_complete():
    bank = [f"template {i}" for i in range(500)]
    train, held = holdout.split_templates(bank)
    assert not set(train) & set(held)
    assert len(train) + len(held) == len(bank)


def test_template_split_is_roughly_the_requested_fraction():
    bank = [f"template {i}" for i in range(2000)]
    _, held = holdout.split_templates(bank, fraction=0.15)
    assert 0.12 < len(held) / len(bank) < 0.18


def test_growing_the_bank_never_moves_an_existing_template():
    """Hashing rather than shuffling: adding phrasings must not leak a previously held-out one."""
    small = [f"template {i}" for i in range(100)]
    big = small + [f"extra {i}" for i in range(100)]
    _, held_small = holdout.split_templates(small)
    _, held_big = holdout.split_templates(big)
    assert set(held_small) <= set(held_big)


# ==================== corpus checking ====================


TOOL = ser.ToolSchema("compound_interest", "Future value.", {"type": "object", "properties": {}})
HELD = ser.ToolSchema("percent_change", "Percent change.", {"type": "object", "properties": {}})


def _row(schemas, call_name):
    return ser.build_row(schemas=schemas, user="q", calls=[ser.Call(call_name, {})])


def test_clean_corpus_reports_no_violations(registry):
    rows = [
        (
            "conversations/arithmetic/single-call/train-00000.jsonl",
            _row([TOOL], "compound_interest"),
        ),
        (
            "conversations/arithmetic/single-call/heldout-00000.jsonl",
            _row([HELD, TOOL], "percent_change"),
        ),
    ]
    report = holdout.check_corpus(rows, registry, parse_row=ser.parse_row)
    # Only the arithmetic held-out tools are exercised here, so ignore the "unused" note.
    real = [v for v in report.violations if "never used as a gold call" not in v]
    assert real == []


def test_catches_a_held_out_tool_offered_in_training(registry):
    rows = [
        (
            "conversations/arithmetic/single-call/train-00000.jsonl",
            _row([HELD], "percent_change"),
        )
    ]
    report = holdout.check_corpus(rows, registry, parse_row=ser.parse_row)
    assert any("offers held-out tool" in v for v in report.violations)


def test_catches_a_test_row_whose_gold_tool_was_trained(registry):
    """A test row calling a trained tool measures recall, not generalisation."""
    rows = [
        (
            "conversations/pedagogy/single-call/heldout-00000.jsonl",
            _row(
                [ser.ToolSchema("post_score", "d", {"type": "object", "properties": {}})],
                "post_score",
            ),
        )
    ]
    report = holdout.check_corpus(rows, registry, parse_row=ser.parse_row)
    assert any("measures recall rather than generalisation" in v for v in report.violations)


def test_allows_a_trained_gold_tool_where_the_domain_has_a_substitute_axis(registry):
    """arithmetic/web-search cannot hold out their dominant tool, so this is expected there."""
    calc = ser.ToolSchema("calculator", "Evaluate.", {"type": "object", "properties": {}})
    rows = [
        ("conversations/arithmetic/single-call/heldout-00000.jsonl", _row([calc], "calculator"))
    ]
    report = holdout.check_corpus(rows, registry, parse_row=ser.parse_row)
    assert not any("measures recall" in v for v in report.violations)


def test_catches_a_row_filed_under_the_wrong_domain(registry):
    """Gate 34: the domain is the gold tool's domain, not the topic of the question."""
    rows = [
        (
            "conversations/pedagogy/single-call/train-00000.jsonl",
            _row([TOOL], "compound_interest"),
        )
    ]
    report = holdout.check_corpus(rows, registry, parse_row=ser.parse_row)
    assert any("but sits under" in v for v in report.violations)
