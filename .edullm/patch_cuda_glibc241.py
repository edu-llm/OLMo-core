"""Patch CUDA 12.8 math declarations for glibc 2.41 compatibility."""

import re
from pathlib import Path


CUDA_CRT = Path("/usr/local/cuda-12.8/targets/x86_64-linux/include/crt")


def replace_once(text: str, pattern: str, replacement: str, *, label: str) -> str:
    """Apply exactly one compatibility replacement."""
    updated, count = re.subn(pattern, replacement, text)
    if count != 1:
        raise RuntimeError(f"expected one {label} declaration, found {count}")
    return updated


def main() -> None:
    """Align CUDA's pi-trig exception specifications with glibc 2.41."""
    declarations = CUDA_CRT / "math_functions.h"
    text = declarations.read_text()
    for name, scalar in (
        ("sinpi", "double"),
        ("sinpif", "float"),
        ("cospi", "double"),
        ("cospif", "float"),
    ):
        pattern = (
            rf"(extern\s+__DEVICE_FUNCTIONS_DECL__\s+__device_builtin__\s+"
            rf"{scalar}\s+{name}\({scalar}\s+x\))\s*;"
        )
        text = replace_once(
            text,
            pattern,
            r"\1 noexcept (true);",
            label=name,
        )
    declarations.write_text(text)

    definitions = CUDA_CRT / "math_functions.hpp"
    text = definitions.read_text()
    for name, scalar in (
        ("sinpi", "double"),
        ("cospi", "double"),
        ("sinpif", "float"),
        ("cospif", "float"),
    ):
        pattern = rf"(__func__\({scalar}\s+{name}\((?:const\s+)?{scalar}\s+a\)\))"
        text = replace_once(
            text,
            pattern,
            r"\1 throw()",
            label=f"{name} definition",
        )
    definitions.write_text(text)


if __name__ == "__main__":
    main()
