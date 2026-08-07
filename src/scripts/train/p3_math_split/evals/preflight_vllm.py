"""Fail fast when the P3 vLLM process is using an incompatible CUDA stack."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys
from collections.abc import Mapping, Sequence

PINNED_PYTHON_VERSION = (3, 12, 13)
PINNED_PACKAGE_VERSIONS = {
    "torch": "2.10.0",
    "transformers": "5.7.0",
    "vllm": "0.19.1",
}
REQUIRED_CUDA_MAJOR = 12
MINIMUM_COMPUTE_CAPABILITY = (8, 0)


def validate_runtime_facts(
    *,
    python_version: Sequence[int],
    package_versions: Mapping[str, str],
    torch_cuda_version: str | None,
    cuda_available: bool,
    compute_capability: tuple[int, int] | None,
) -> list[str]:
    """Return every incompatibility instead of hiding the first one."""

    errors = []
    actual_python = tuple(python_version[:3])
    if actual_python != PINNED_PYTHON_VERSION:
        errors.append(
            "expected Python "
            f"{'.'.join(map(str, PINNED_PYTHON_VERSION))}, got "
            f"{'.'.join(map(str, actual_python))}"
        )

    for package, expected in PINNED_PACKAGE_VERSIONS.items():
        actual = package_versions.get(package)
        if actual != expected:
            errors.append(f"expected {package}=={expected}, got {actual or 'not installed'}")

    if not cuda_available:
        errors.append("a CUDA GPU is required for the vLLM backend")

    if not torch_cuda_version or torch_cuda_version.split(".", 1)[0] != str(REQUIRED_CUDA_MAJOR):
        errors.append(
            f"expected the PyTorch CUDA 12 runtime used by the vLLM wheel, "
            f"got {torch_cuda_version or 'none'}"
        )

    if cuda_available and (
        compute_capability is None or compute_capability < MINIMUM_COMPUTE_CAPABILITY
    ):
        errors.append(
            "BF16 vLLM requires compute capability >= "
            f"{MINIMUM_COMPUTE_CAPABILITY[0]}.{MINIMUM_COMPUTE_CAPABILITY[1]}, "
            f"got {compute_capability!r}"
        )
    return errors


def main() -> int:
    package_versions = {}
    for package in PINNED_PACKAGE_VERSIONS:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = ""

    try:
        import torch
    except Exception as error:  # noqa: BLE001 - native loader failures are the gate
        print(f"P3 vLLM preflight FAILED: torch import failed: {error}", file=sys.stderr)
        return 1

    cuda_available = torch.cuda.is_available()
    compute_capability = torch.cuda.get_device_capability() if cuda_available else None
    gpu_name = torch.cuda.get_device_name() if cuda_available else None
    report = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "packages": package_versions,
        "torch_runtime_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "compute_capability": compute_capability,
    }
    print(json.dumps(report, indent=2))

    errors = validate_runtime_facts(
        python_version=sys.version_info,
        package_versions=package_versions,
        torch_cuda_version=torch.version.cuda,
        cuda_available=cuda_available,
        compute_capability=compute_capability,
    )
    try:
        importlib.import_module("vllm._C")
    except Exception as error:  # noqa: BLE001 - catches CUDA and native ABI failures
        errors.append(f"vllm._C import failed: {type(error).__name__}: {error}")

    # The checkpoint exporter resolves the vendored tokenizer through these; a
    # missing reader fails every arm during export, not at eval time.
    for module in ("edullm_data.read", "edullm_data.s3"):
        try:
            importlib.import_module(module)
        except Exception as error:  # noqa: BLE001 - missing reader dependency
            errors.append(f"{module} import failed: {type(error).__name__}: {error}")

    if errors:
        print("P3 vLLM preflight FAILED:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("P3 vLLM preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
