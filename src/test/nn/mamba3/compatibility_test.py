import importlib.util
import re
from pathlib import Path

import pytest


def _load_compatibility_module():
    path = Path("src/olmo_core/nn/mamba3/compatibility.py")
    spec = importlib.util.spec_from_file_location("mamba3_compatibility", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_official_mamba_kernel_accepts_only_torch_2_10_and_triton_3_6():
    compatibility = _load_compatibility_module()

    compatibility.assert_official_mamba3_runtime_compatible("2.10.0+cu128", "3.6.0")

    with pytest.raises(RuntimeError, match=r"requires PyTorch 2\.10\.x"):
        compatibility.assert_official_mamba3_runtime_compatible("2.11.0", "3.6.0")
    with pytest.raises(RuntimeError, match=r"requires Triton 3\.6\.x"):
        compatibility.assert_official_mamba3_runtime_compatible("2.10.0", "3.7.0")


def test_official_mamba_kernel_contract_matches_the_pinned_image():
    compatibility = _load_compatibility_module()
    makefile = Path("Makefile").read_text()
    dockerfile = Path("src/Dockerfile").read_text()

    makefile_torch = re.search(r"^TORCH_VERSION = (\S+)$", makefile, re.MULTILINE)
    docker_torch = re.search(r"^ARG TORCH_VERSION=(\S+)$", dockerfile, re.MULTILINE)
    assert makefile_torch is not None
    assert docker_torch is not None
    assert makefile_torch.group(1) == docker_torch.group(1) == "2.10.0"
    assert compatibility.OFFICIAL_MAMBA3_TORCH_MAJOR_MINOR == (2, 10)
    assert compatibility.OFFICIAL_MAMBA3_TRITON_MAJOR_MINOR == (3, 6)


def test_official_kernel_availability_checks_the_runtime_contract():
    api_source = Path("src/olmo_core/nn/mamba3/mamba3_ssd_api.py").read_text()

    assert (
        "assert_official_mamba3_runtime_compatible(torch.__version__, triton.__version__)"
        in api_source
    )
