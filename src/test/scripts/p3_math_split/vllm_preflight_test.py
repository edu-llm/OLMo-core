"""Environment gates for the P3 vLLM generation backend."""

from . import EVALS_DIR, load_project_module

preflight = load_project_module("preflight_vllm")


def test_preflight_accepts_the_pinned_cuda12_l4_stack():
    errors = preflight.validate_runtime_facts(
        python_version=(3, 12, 13),
        package_versions={
            "torch": "2.10.0",
            "transformers": "5.7.0",
            "vllm": "0.19.1",
        },
        torch_cuda_version="12.8",
        cuda_available=True,
        compute_capability=(8, 9),
    )

    assert errors == []


def test_preflight_rejects_the_cuda13_python313_stack_that_failed_the_fleet():
    errors = preflight.validate_runtime_facts(
        python_version=(3, 13, 5),
        package_versions={
            "torch": "2.10.0",
            "transformers": "5.14.1",
            "vllm": "0.19.1",
        },
        torch_cuda_version="13.0",
        cuda_available=True,
        compute_capability=(8, 9),
    )

    assert any("Python 3.12.13" in error for error in errors)
    assert any("transformers==5.7.0" in error for error in errors)
    assert any("CUDA 12" in error for error in errors)


def test_preflight_rejects_cpu_or_pre_ampere_gpu():
    no_cuda = preflight.validate_runtime_facts(
        python_version=(3, 12, 13),
        package_versions=preflight.PINNED_PACKAGE_VERSIONS,
        torch_cuda_version="12.8",
        cuda_available=False,
        compute_capability=None,
    )
    t4 = preflight.validate_runtime_facts(
        python_version=(3, 12, 13),
        package_versions=preflight.PINNED_PACKAGE_VERSIONS,
        torch_cuda_version="12.8",
        cuda_available=True,
        compute_capability=(7, 5),
    )

    assert any("CUDA GPU" in error for error in no_cuda)
    assert any("compute capability" in error for error in t4)


def test_bootstrap_builds_an_isolated_pinned_environment():
    script = (EVALS_DIR / "bootstrap_vllm_env.sh").read_text(encoding="utf-8")

    assert "3.12.13" in script
    for requirement in (
        "torch==2.10.0",
        "transformers==5.7.0",
        "vllm==0.19.1",
    ):
        assert requirement in script
    assert "uv venv" in script
    assert "preflight_vllm.py" in script


def test_bootstrap_installs_the_pinned_tokenizer_reader():
    script = (EVALS_DIR / "bootstrap_vllm_env.sh").read_text(encoding="utf-8")

    # The exporter imports edullm_data to resolve the vendored tokenizer, so the
    # isolated environment must install it or every arm fails during export.
    assert "edullm-data" in script
    assert "38bf831a6c3f445e394784018441fd59288b876c" in script


def test_prepare_base_model_script_pins_control_and_vendored_tokenizer():
    script = (EVALS_DIR / "prepare_base_model.sh").read_text(encoding="utf-8")

    assert script.startswith("#!/usr/bin/env bash")
    assert "set -euo pipefail" in script
    # The untrained control must be the exact snapshot the arms initialized from.
    assert "Qwen/Qwen2.5-0.5B" in script
    assert "060db6499f32faf8b98477b0a26969ef7d8b9987" in script
    # Pull only weights + config from the Hub; the stock tokenizer is excluded.
    assert "config.json" in script
    assert "model.safetensors" in script
    assert "allow_patterns" in script
    # Overlay the vendored tokenizer via the same artifact the exporter uses so
    # tokenizer_sha256 matches dense/split.
    assert "fetch_tokenizer_artifact" in script
    assert "TOKENIZER_ARTIFACT" in script
    # The control arm carries no export provenance.
    assert "model_provenance.json" in script
