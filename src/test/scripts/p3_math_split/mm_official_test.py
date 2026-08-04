"""Cross-check the internal verifier against the pinned official executable."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from . import load_project_module

mm_expand = load_project_module("mm_expand")
mm_official = load_project_module("mm_official")
build_tool = load_project_module("build_pinned_metamath")

PROJECT_DIR = Path("src/scripts/train/p3_math_split")
FIXTURE = (
    Path(__file__).parent / "fixtures" / "mm_verify_soundness.mm"
)
MANIFEST = PROJECT_DIR / "metamath_verifier_manifest.json"


def proof(label: str, expression: str) -> str:
    return f"  1  {label:<12} {expression}"


def test_official_verifier_manifest_is_immutable_and_complete() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["source"]["repository"] == (
        "https://github.com/metamath/metamath-exe.git"
    )
    assert manifest["source"]["commit"] == (
        "69a5d47fc755c21c125453407270cc26857d51b5"
    )
    assert manifest["build"]["compiler"] == (
        "gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0"
    )
    assert manifest["binary"]["sha256"] == (
        "b43a56d75e1489dc5e568c283a7684f0e7da7a2016e90e6ec020b01852b4eeed"
    )
    assert manifest["binary"]["version"] == "0.199.pre 29-Jan-2022"


def test_build_tool_reproduces_manifest_binary_hash(tmp_path: Path) -> None:
    checkout = Path("/tmp/metamath-exe-69a5d47")
    if not (checkout / ".git").is_dir():
        pytest.skip("pinned official Metamath source checkout is unavailable")

    output = build_tool.build(tmp_path / "metamath", checkout)
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["binary"][
        "sha256"
    ]

    assert hashlib.sha256(output.read_bytes()).hexdigest() == expected


@pytest.fixture(scope="module")
def official_binary() -> Path:
    candidate = Path(
        os.environ.get(
            "METAMATH_BIN",
            "/tmp/metamath-exe-69a5d47/metamath-pinned",
        )
    )
    if not candidate.is_file():
        pytest.skip("pinned official Metamath executable is not built")
    expected = json.loads(MANIFEST.read_text(encoding="utf-8"))["binary"][
        "sha256"
    ]
    actual = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert actual == expected
    return candidate


@pytest.mark.parametrize(
    (
        "target_label",
        "generated",
        "goal",
        "facts",
        "local_assumptions",
        "expected",
    ),
    [
        (
            "class-target",
            proof("emit", "|- A"),
            "|- A",
            {"emit": "|- ph"},
            {},
            "invalid",
        ),
        (
            "bad-rebind",
            proof("syl", "|- ( ps -> Z )"),
            "|- ( ps -> Z )",
            {
                "syl": (
                    "|- ( ph -> ps ) & |- ( ps -> ch ) "
                    "=> |- ( ph -> ch )"
                )
            },
            {
                "bad-rebind.1": "|- ( X -> X )",
                "bad-rebind.2": "|- ( X -> Z )",
            },
            "invalid",
        ),
        (
            "good-rebind",
            proof("syl", "|- ( ps -> Z )"),
            "|- ( ps -> Z )",
            {
                "syl": (
                    "|- ( ph -> ps ) & |- ( ps -> ch ) "
                    "=> |- ( ph -> ch )"
                )
            },
            {
                "good-rebind.1": "|- ( ps -> X )",
                "good-rebind.2": "|- ( X -> Z )",
            },
            "valid",
        ),
        (
            "bad-d-target",
            proof("pair", "|- P z z"),
            "|- P z z",
            {"pair": "|- P x y"},
            {},
            "invalid",
        ),
        (
            "good-d-target",
            proof("pair", "|- P z w"),
            "|- P z w",
            {"pair": "|- P x y"},
            {},
            "valid",
        ),
        (
            "dup-target",
            "\n".join(
                [
                    "  1  ax-1        |- ( a -> ( b -> a ) )",
                    "  2  dup         |- ( a -> ( b -> a ) )",
                ]
            ),
            "|- ( a -> ( b -> a ) )",
            {
                "ax-1": "|- ( ph -> ( ps -> ph ) )",
                "dup": "|- ph & |- ph => |- ph",
            },
            {},
            "valid",
        ),
    ],
)
def test_minimal_type_essential_disjoint_and_rebinding_cases_match_official(
    official_binary: Path,
    target_label: str,
    generated: str,
    goal: str,
    facts: dict[str, str],
    local_assumptions: dict[str, str],
    expected: str,
) -> None:
    result = mm_official.verify_expression_trace(
        source_path=FIXTURE,
        binary_path=official_binary,
        target_label=target_label,
        generated=generated,
        goal=goal,
        fact_block=facts,
        local_assumptions=local_assumptions,
    )

    assert result.status == expected, result.reason


def test_pinned_real_source_proofs_are_officially_valid(
    official_binary: Path,
) -> None:
    source = Path("/tmp/p3-audit-sources/set.mm")
    if not source.is_file():
        pytest.skip("pinned set.mm audit source is unavailable")

    for label in ("pm4.39", "sb9", "sbcop1", "jm2.27"):
        result = mm_official.verify_source_proof(
            source_path=source,
            binary_path=official_binary,
            target_label=label,
        )
        assert result.status == "valid", (label, result.reason)


@pytest.mark.parametrize(
    "fixture_name",
    ["invalid_essential.mm", "invalid_disjoint.mm"],
)
def test_corpus_red_source_proofs_are_officially_invalid(
    official_binary: Path,
    fixture_name: str,
) -> None:
    source = (
        Path("/home/vs/AlphaAI/memorysplit-requery-exact")
        / "tests/fixtures/metamath"
        / fixture_name
    )
    if not source.is_file():
        pytest.skip("corpus Metamath red fixtures are unavailable")

    result = mm_official.verify_source_proof(
        source_path=source,
        binary_path=official_binary,
        target_label="bad",
    )

    assert result.status == "invalid", result.output


def test_supported_real_expression_traces_crosscheck_official_oracle(
    official_binary: Path,
) -> None:
    source = Path("/tmp/p3-audit-sources/set.mm")
    rows_path = Path(
        "/home/vs/AlphaAI/memorysplit-requery-exact/corpus/eval/metamath.jsonl"
    )
    if not source.is_file() or not rows_path.is_file():
        pytest.skip("pinned real expression-trace fixtures are unavailable")

    wanted = {"pm4.39", "sb9", "sbcop1", "jm2.27"}
    rows = {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        database, label = row["theorem"].split(":", 1)
        if database == "set" and label in wanted:
            rows[label] = row

    mm = mm_expand.MM().parse(source)
    expected = {
        "pm4.39": "valid",
        "sb9": "valid",
        "sbcop1": "valid",
        "jm2.27": "unknown",
    }
    for label, expected_status in expected.items():
        expression, mandatory, _, trace = mm_expand.expand(mm, label)
        mandatory_e = {
            hyp_label
            for kind, hyp_label, _ in mandatory
            if kind == "$e"
        }
        local_assumptions = {}
        steps = []
        for step_label, formula, _ in trace:
            if step_label in mandatory_e:
                local_assumptions.setdefault(step_label, " ".join(formula))
            elif (
                step_label != "(reuse)"
                and formula
                and formula[0] == "|-"
            ):
                steps.append((step_label, " ".join(formula)))
        generated = "\n".join(
            f"{index:>3}  {step_label:<14} {formula}"
            for index, (step_label, formula) in enumerate(steps, 1)
        )
        result = mm_official.verify_expression_trace(
            source_path=source,
            binary_path=official_binary,
            target_label=label,
            generated=generated,
            goal=" ".join(expression),
            fact_block=rows[label]["facts"],
            local_assumptions=local_assumptions,
        )

        assert result.status == expected_status, (
            label,
            result.reason_code,
            result.reason,
        )
        if expected_status == "unknown":
            assert result.reason_code == "trace_conversion_unsupported"
