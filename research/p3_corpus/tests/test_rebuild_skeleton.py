"""CI-safe tests for the P3 rebuild skeleton (no multi-GB payloads)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARCHIVE_ROOT = Path(__file__).resolve().parents[1]


def test_source_lock_has_required_upstream_pins() -> None:
    lock = json.loads((ARCHIVE_ROOT / "source-lock.json").read_text(encoding="utf-8"))
    assert lock["schema"] == "p3-source-lock/v1"
    sources = lock["sources"]
    for key in ("metamath", "mizar_current", "prf2", "enigma", "isabelle"):
        assert key in sources
    mm_files = {item["name"] for item in sources["metamath"]["files"]}
    assert mm_files == {"set.mm", "iset.mm", "nf.mm"}
    mizar_archives = {item["name"] for item in sources["mizar_current"]["archives"]}
    assert mizar_archives == {"mml", "html", "thproofs"}
    required_enigma = {
        item["name"]
        for item in sources["enigma"]["archives"]
        if not item.get("optional")
    }
    assert required_enigma == {"mzr01", "mzr02", "mzr03", "mzr08"}


def test_expected_release_matches_sealed_provenance() -> None:
    expected = json.loads(
        (ARCHIVE_ROOT / "expected-release-v3.json").read_text(encoding="utf-8")
    )
    sealed = json.loads(
        (ARCHIVE_ROOT / "provenance/sealed-corpus-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert expected["sealed_jsonl"]["total_train_rows"] == sealed["total_train_rows"]
    assert expected["sealed_jsonl"]["total_eval_rows"] == sealed["total_eval_rows"]
    assert (
        expected["sealed_jsonl"]["manifest_root_sha256"]
        == sealed["manifest_root_sha256"]
    )
    for family, fam_exp in expected["sealed_jsonl"]["families"].items():
        fam_sealed = sealed["families"][family]
        assert fam_exp["train_rows"] == fam_sealed["train"]["rows"]
        assert fam_exp["eval_rows"] == fam_sealed["eval"]["rows"]
        assert fam_exp["train_sha256"] == fam_sealed["train"]["sha256"]


def test_generation_templates_and_scripts_exist() -> None:
    templates = ARCHIVE_ROOT / "templates/generation-inputs"
    for name in (
        "policies.json",
        "tokenizer-seal.json",
        "SUMMARY.json",
        "metamath.json",
        "mizar.json",
        "thproofs.json",
        "prf2.json",
        "enigma.json",
        "isabelle.json",
    ):
        payload = json.loads((templates / name).read_text(encoding="utf-8"))
        assert payload
    for name in (
        "bootstrap_sources.py",
        "materialize_generation_inputs.py",
        "build_accepted_bases.py",
        "orchestrate_rebuild.py",
        "verify_rebuild.py",
        "build_p3_generation.py",
        "verify_corpus.py",
    ):
        assert (ARCHIVE_ROOT / "scripts" / name).is_file()


def test_generation_templates_use_portable_placeholders_only() -> None:
    templates = ARCHIVE_ROOT / "templates/generation-inputs"
    private = "/home/vs/AlphaAI/" + "memorysplit-requery-exact"
    for path in templates.glob("*.json"):
        text = path.read_text(encoding="utf-8")
        assert private not in text
        assert "/tmp/p3-" not in text


def test_materialize_generation_inputs_roundtrip(tmp_path: Path) -> None:
    import subprocess
    import sys

    out = tmp_path / "generation-inputs"
    subprocess.run(
        [
            sys.executable,
            str(ARCHIVE_ROOT / "scripts/materialize_generation_inputs.py"),
            "--out",
            str(out),
            "--corpus-root",
            str(ARCHIVE_ROOT),
            "--sources-root",
            str(tmp_path / "sources"),
            "--work-root",
            str(tmp_path / "work"),
        ],
        check=True,
        cwd=ARCHIVE_ROOT,
    )
    summary = json.loads((out / "SUMMARY.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "p3-generation-input-summary-v1"
    assert set(summary["source_manifest_roots"]) == {
        "enigma",
        "isabelle",
        "metamath",
        "mizar",
        "prf2",
        "thproofs",
    }
    metamath = json.loads((out / "metamath.json").read_text(encoding="utf-8"))
    argv = metamath["builder"]["raw"]["argv"]
    assert str(ARCHIVE_ROOT / "scripts/build_metamath_shard.py") in argv
    assert str(tmp_path / "sources/metamath") in argv


def test_generation_template_digests_are_well_formed() -> None:
    import re

    hex64 = re.compile(r"^[0-9a-f]{64}$")
    offenders = []

    def scan(node, where: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if (
                    isinstance(value, str)
                    and "sha256" in key
                    and not hex64.match(value)
                ):
                    offenders.append(f"{where}.{key} = {value!r} ({len(value)} chars)")
                scan(value, f"{where}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                scan(value, f"{where}[{index}]")

    for path in sorted((ARCHIVE_ROOT / "templates/generation-inputs").glob("*.json")):
        scan(json.loads(path.read_text(encoding="utf-8")), path.name)

    assert offenders == []


def test_every_contract_enigma_run_is_a_required_lock_entry() -> None:
    import sys

    sys.path.insert(0, str(ARCHIVE_ROOT / "scripts"))
    from build_atp_shard import ENIGMA_LOW_TIER_SOURCE_CONTRACT

    lock = json.loads((ARCHIVE_ROOT / "source-lock.json").read_text(encoding="utf-8"))
    required = {
        item["name"]
        for item in lock["sources"]["enigma"]["archives"]
        if not item.get("optional")
    }
    # bootstrap_sources.py never downloads an optional archive, so any run the
    # accepted base is built from has to be required or the rebuild silently
    # produces a shard that cannot match its pin.
    assert set(ENIGMA_LOW_TIER_SOURCE_CONTRACT["source_order"]) <= required


def test_accepted_base_stage_is_wired_into_the_orchestrator() -> None:
    import sys

    sys.path.insert(0, str(ARCHIVE_ROOT / "scripts"))
    from build_atp_shard import ENIGMA_LOW_TIER_SOURCE_CONTRACT

    template = json.loads(
        (ARCHIVE_ROOT / "templates/generation-inputs/enigma.json").read_text(
            encoding="utf-8"
        )
    )
    low_tier_argv = template["builder"]["raw"]["argv"]
    runs = ENIGMA_LOW_TIER_SOURCE_CONTRACT["source_order"]
    assert [f"{{{{P3_SOURCES_ROOT}}}}/extracted/{run}" for run in runs] == [
        item for item in low_tier_argv if "/extracted/" in item
    ]

    from orchestrate_rebuild import STAGES

    # The base must exist before preflight validates that its path is readable,
    # and cannot be built before its sources are on disk.
    assert STAGES.index("bootstrap_sources") < STAGES.index("build_accepted_bases")
    assert STAGES.index("build_accepted_bases") < STAGES.index("generation_preflight")


def test_accepted_base_command_drops_low_tier_flags(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(ARCHIVE_ROOT / "scripts"))
    import build_accepted_bases as bases

    sources = tmp_path / "sources"
    for run in ("mzr01", "mzr02", "mzr03", "mzr08"):
        (sources / "extracted" / run).mkdir(parents=True)

    cmd = bases.enigma_command(
        sources_root=sources,
        out_dir=tmp_path / "atp/enigma-accepted-base-v1",
        python="python3",
    )
    assert "--enigma-low-tier-base" not in cmd
    assert "--tokenizer-json" not in cmd
    assert cmd[cmd.index("--seed") + 1] == "20260801"
    assert cmd[cmd.index("--min-steps") + 1] == "4"
    assert cmd[cmd.index("--jaccard") + 1] == "0.5"
    assert "--fenced" in cmd and "--dedup" in cmd
    src_index = cmd.index("--src")
    assert cmd[src_index + 1 : src_index + 5] == [
        str(sources / "extracted" / run) for run in ("mzr01", "mzr03", "mzr02", "mzr08")
    ]


def test_accepted_base_rejects_incomplete_enigma_sources(tmp_path: Path) -> None:
    import sys

    import pytest

    sys.path.insert(0, str(ARCHIVE_ROOT / "scripts"))
    import build_accepted_bases as bases

    sources = tmp_path / "sources"
    (sources / "extracted" / "mzr01").mkdir(parents=True)
    with pytest.raises(SystemExit):
        bases.enigma_source_dirs(sources)


def test_tokenizer_seal_matches_expected_release() -> None:
    expected = json.loads(
        (ARCHIVE_ROOT / "expected-release-v3.json").read_text(encoding="utf-8")
    )
    seal = json.loads(
        (ARCHIVE_ROOT / "templates/generation-inputs/tokenizer-seal.json").read_text(
            encoding="utf-8"
        )
    )
    tok = expected["tokenizer"]
    assert seal["behavior_digest"] == tok["composite_sha256"]
    assert seal["eos_token_id"] == tok["eos_token_id"]
    assert seal["max_text_plus_eos_tokens"] == expected["training_constants"]["sequence_length"]


def test_bootstrap_script_is_executable_python() -> None:
    script = ARCHIVE_ROOT / "scripts/bootstrap_sources.py"
    text = script.read_text(encoding="utf-8")
    assert "source-lock.json" in text
    assert "sha256" in text
    compile(text, str(script), "exec")
