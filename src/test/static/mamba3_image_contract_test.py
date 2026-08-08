import re
from pathlib import Path


MAMBA3_FUSED_IMPORT = (
    "from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined " "import mamba3_siso_combined"
)


def test_pyproject_and_docker_pin_the_same_exact_mamba_revision():
    pyproject = Path("pyproject.toml").read_text()
    dockerfile = Path("src/Dockerfile").read_text()

    pyproject_pins = re.findall(r"state-spaces/mamba\.git@([0-9a-f]{40})", pyproject)
    docker_pins = re.findall(r"^ARG MAMBA3_COMMIT=([0-9a-f]{40})$", dockerfile, re.MULTILINE)

    assert len(pyproject_pins) == 1
    assert len(docker_pins) == 1
    assert pyproject_pins == docker_pins


def test_mamba_source_build_explicitly_includes_a100_sm80():
    dockerfile = Path("src/Dockerfile").read_text()

    assert 'ARG MAMBA3_CUDA_ARCHS="8.0;9.0;10.0"' in dockerfile
    assert 'TORCH_CUDA_ARCH_LIST="${MAMBA3_CUDA_ARCHS}" MAMBA_FORCE_BUILD=TRUE' in dockerfile


def test_docker_build_and_release_validation_import_mamba_triton_liger_and_fused_symbol():
    dockerfile = Path("src/Dockerfile").read_text()
    makefile = Path("Makefile").read_text()
    required_imports = (
        "import mamba_ssm",
        "import triton",
        "import liger_kernel",
        MAMBA3_FUSED_IMPORT,
        "assert callable(mamba3_siso_combined)",
    )

    for source in (dockerfile, makefile):
        for required in required_imports:
            assert required in source
