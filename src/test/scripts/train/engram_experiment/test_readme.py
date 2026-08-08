import re
from pathlib import Path

from scripts.train.engram_experiment import base_moe, common, engram_moe, lngram_moe

README = (
    Path(__file__).resolve().parents[5]
    / "src"
    / "scripts"
    / "train"
    / "engram_experiment"
    / "README.md"
)

ARMS = {
    "base": "base-experiment-slug",
    "engram": "engram-experiment-slug",
    "lngram": "lngram-experiment-slug",
}
ARM_MODULES = (base_moe, engram_moe, lngram_moe)


def readme_text() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_names_designs_papers_and_exact_accounting():
    text = readme_text()

    for name in ("Base MoE", "Engram MoE", "Lngram MoE"):
        assert name in text
    assert "arXiv:2601.07372" in text
    assert "arXiv:2605.24869" in text

    for arm in ARM_MODULES:
        total, active = common.parameter_counts(arm.build_model_config())
        assert f"{total:,}" in text
        assert f"{active:,}" in text
    assert "<= 1%" in text
    assert "projection-driven active-parameter bump" in text


def test_readme_seals_schedule_data_and_registry_contracts():
    text = readme_text()

    required = (
        "10,000,000,000",
        "19,074",
        "100,352",
        "2,048",
        "uint32",
        "headerless",
        "native little-endian",
        "s3://edullm-data/pretrain/regmix-10b/v1/tokens/<source>/train-*.u32le.bin",
        "DOCUMENTATION ONLY",
        "pretrain/regmix-10b",
        "v1",
        "tokenizer/dolma2-bpe",
        "62,347",
        "62,421",
        "5292e5d6c0f40b67cc765fe41bec991cf4345b5c",
        "memory-paper-fidelity-v1",
        "fresh run IDs/checkpoint prefixes",
        "sealed registry identity",
        "no hardcoded run paths",
        "no downloads",
        "no evals",
    )
    for phrase in required:
        assert phrase in text


def test_readme_has_exact_paired_commands_for_three_independent_nodes():
    text = readme_text()

    assert "three separate single-node submissions" in text
    assert "platform runs a commit, never the working tree" in text
    assert "edullm/engram-lngram-moe-400m" in text
    for arm, experiment in ARMS.items():
        shared = (
            f"--spec .edullm/run-{arm}.yaml "
            f"--experiment <{experiment}> "
            "--dataset <registered-regmix-release> "
            "--compute gpu-8xa100"
        )
        assert f"edullm check --json {shared}" in text
        assert f"edullm submit {shared}" in text

    assert text.count("edullm check --json --spec") == 3
    assert text.count("edullm submit --spec") == 3
    assert text.count("--dataset <registered-regmix-release>") == 6
    assert text.count("--compute gpu-8xa100") == 6


def test_readme_documents_pipeline_recovery_observability_and_deferred_work():
    text = readme_text()

    required = (
        "FSDP2",
        "bfloat16",
        "ConfigSaver",
        "checkpoint prefix",
        "resume",
        "torn",
        "W&B",
        "EP/TP table sharding",
        "inference host-offload prefetch",
        "CP/PP routing and sequence-boundary halos",
    )
    for phrase in required:
        assert phrase in text


def test_readme_does_not_publish_operational_guesses_or_side_channels():
    text = readme_text()

    forbidden = (
        r"\$\s*\d",
        r"\b\d+(?:\.\d+)?\s*(?:hours?|hrs?|minutes?|mins?)\b",
        r"\bapprovers?\b",
        r"\bapproval count\b",
        r"\bcost ceiling\b",
        r"\b(?:curl|wget|boto3|aws cli)\b",
        r"s3://[^\s`]*checkpoints?",
    )
    for pattern in forbidden:
        assert re.search(pattern, text, flags=re.IGNORECASE) is None
