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
