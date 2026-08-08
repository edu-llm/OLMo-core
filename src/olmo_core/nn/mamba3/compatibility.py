"""Runtime compatibility contract for the pinned official Mamba-3 kernel."""

import re


OFFICIAL_MAMBA3_TORCH_MAJOR_MINOR = (2, 10)
"""PyTorch release series validated for the pinned official Mamba-3 kernel."""

OFFICIAL_MAMBA3_TRITON_MAJOR_MINOR = (3, 6)
"""Triton release series validated for the pinned official Mamba-3 kernel."""


def _major_minor(version: str, dependency: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise RuntimeError(f"could not parse {dependency} version {version!r}")
    return int(match.group(1)), int(match.group(2))


def assert_official_mamba3_runtime_compatible(
    torch_version: str,
    triton_version: str,
) -> None:
    """
    Validate the runtime used by the pinned official Mamba-3 Triton kernel.

    PyTorch 2.10 ships Triton 3.6, the combination validated by this branch.
    Later Triton releases changed TMA descriptor lowering and are not accepted
    until the pinned upstream kernel has explicit compatibility coverage.

    :param torch_version: The value of :data:`torch.__version__`.
    :param triton_version: The value of :data:`triton.__version__`.

    :raises RuntimeError: If either dependency is outside the validated release series.
    """
    torch_major_minor = _major_minor(torch_version, "PyTorch")
    triton_major_minor = _major_minor(triton_version, "Triton")
    if torch_major_minor != OFFICIAL_MAMBA3_TORCH_MAJOR_MINOR:
        raise RuntimeError(
            "the pinned official Mamba-3 kernel requires PyTorch 2.10.x; "
            f"found {torch_version}. Use the repository's pinned training image."
        )
    if triton_major_minor != OFFICIAL_MAMBA3_TRITON_MAJOR_MINOR:
        raise RuntimeError(
            "the pinned official Mamba-3 kernel requires Triton 3.6.x; "
            f"found {triton_version}. Use the repository's pinned training image."
        )
