"""Build the ahead-of-time native Flash PD-SSM CUDA extension."""

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = Path(__file__).parent
SOURCE = ROOT / "src/olmo_core/nn/flash_pd_native/csrc"
SUPPORTED_ARCHITECTURES = ("8.0", "12.0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", ";".join(SUPPORTED_ARCHITECTURES))

cuda_home = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda"))
cuda_library_dir = cuda_home / "lib64"
if not cuda_library_dir.exists():
    cuda_library_dir = cuda_home / "lib"
local_library_dirs: list[str] = []
if not (cuda_library_dir / "libcudart.so").exists():
    versioned_cudart = cuda_library_dir / "libcudart.so.13"
    if versioned_cudart.exists():
        # NVIDIA's pip toolkit omits the unversioned development symlink expected by
        # torch's CUDAExtension. Keep the compatibility shim inside the build tree.
        linker_shim = ROOT / "build/flash_pd_native_cuda_lib"
        linker_shim.mkdir(parents=True, exist_ok=True)
        shim = linker_shim / "libcudart.so"
        if not shim.exists():
            shim.symlink_to(versioned_cudart)
        local_library_dirs.append(str(linker_shim))

setup(
    name="flash-pd-native-kernels",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="_flash_pd_native_cuda",
            sources=[
                str(SOURCE / "flash_pd_native.cpp"),
                str(SOURCE / "flash_pd_native_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math", "-lineinfo", "-Xptxas=-v"],
            },
            library_dirs=local_library_dirs,
            extra_link_args=[f"-Wl,-rpath,{cuda_library_dir}"],
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)
