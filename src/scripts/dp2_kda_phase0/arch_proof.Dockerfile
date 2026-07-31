# Targeted proof for the sm_89 arch-list edit in src/Dockerfile:55,59.
#
# This reproduces the two edited layers of the real image -- grouped_gemm (TORCH_CUDA_ARCH_LIST)
# and flash-attn 2 (FLASH_ATTN_CUDA_ARCHS) -- against the same bases, pins and SHAs the Makefile
# passes, and then inspects the produced binaries with `cuobjdump` to report which SASS/PTX
# architectures each one actually contains. The question is not "does it build" but "does the
# resulting object carry sm_89 code", which only the cuobjdump output can answer.
#
# Everything up to the two RUN layers under test is byte-identical to src/Dockerfile.
ARG UBUNTU_VERSION=22.04
ARG CUDA_VERSION=12.8.1
ARG DEVEL_BASE_IMAGE=docker.io/nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu${UBUNTU_VERSION}

FROM ${DEVEL_BASE_IMAGE} AS build

WORKDIR /app/build

ARG TARGET_PLATFORM=x86_64
ARG PYTHON_VERSION=3.12
ARG CUDA_VERSION_PATH=cu128
ARG TORCH_VERSION=2.10.0
ARG INSTALL_CHANNEL=whl

# DEVIATION FROM src/Dockerfile (build-host workaround, not a change to what is compiled).
# FarmShare has no /etc/subuid entry for the user, so rootless podman maps exactly one UID into
# the container. apt's unprivileged download sandbox then cannot seteuid to _apt and dies with
# "setgroups (1: Operation not permitted)". Telling apt to keep running as root restores it. This
# affects only how packages are fetched on this host; it does not touch any compiler flag, arch
# list, pin, or source. The apt-get layer below is byte-identical to src/Dockerfile.
RUN echo 'APT::Sandbox::User "root";' > /etc/apt/apt.conf.d/00-no-sandbox

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential ca-certificates cmake curl wget libxml2-dev libjpeg-dev libpng-dev \
        gcc git && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/conda/bin:$PATH
RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${TARGET_PLATFORM}.sh"
RUN chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} cmake conda-build pyyaml numpy ipython && \
    /opt/conda/bin/python -m pip install --upgrade --no-cache-dir pip wheel packaging "setuptools<70.0.0" ninja && \
    /opt/conda/bin/conda clean -ya

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/${INSTALL_CHANNEL}/${CUDA_VERSION_PATH}/ \
    torch==${TORCH_VERSION}

# ---------------------------------------------------------------------------------------------
# LAYER UNDER TEST 1 -- src/Dockerfile:55, TORCH_CUDA_ARCH_LIST="8.9 9.0 10.0"
# ---------------------------------------------------------------------------------------------
ARG GROUPED_GEMM_SHA="f1429a3c44c98f7912aa4b00125144cdf4e7fdb2"
RUN TORCH_CUDA_ARCH_LIST="8.9 9.0 10.0" GROUPED_GEMM_CUTLASS="1" pip install --no-build-isolation --no-cache-dir "grouped_gemm @ git+https://git@github.com/tgale96/grouped_gemm.git@${GROUPED_GEMM_SHA}"

# ---------------------------------------------------------------------------------------------
# LAYER UNDER TEST 2 -- src/Dockerfile:59, FLASH_ATTN_CUDA_ARCHS="89;90;100"
# ---------------------------------------------------------------------------------------------
ARG FLASH_ATTN_VERSION=2.8.2
RUN FLASH_ATTN_CUDA_ARCHS="89;90;100" pip install --no-build-isolation --no-cache-dir flash-attn==${FLASH_ATTN_VERSION}

# ---------------------------------------------------------------------------------------------
# Verdict: report the architectures actually present in each built object.
# ---------------------------------------------------------------------------------------------
RUN set -x && \
    GG=$(python -c "import grouped_gemm, pathlib; print(pathlib.Path(grouped_gemm.__file__).parent)") && \
    FA=$(python -c "import flash_attn, pathlib; print(pathlib.Path(flash_attn.__file__).parent.parent)") && \
    echo "=== grouped_gemm objects ===" && find "$GG" -name '*.so' && \
    echo "=== flash-attn objects ===" && find "$FA" -maxdepth 1 -name 'flash_attn_2_cuda*.so' && \
    for so in $(find "$GG" -name '*.so') $(find "$FA" -maxdepth 1 -name 'flash_attn_2_cuda*.so'); do \
        echo "########## $so ##########" ; \
        echo "-- ELF SASS arches --" ; cuobjdump "$so" -lelf 2>/dev/null | sed 's/.*\.\(sm_[0-9]*\)\..*/\1/' | sort -u ; \
        echo "-- PTX arches --" ; cuobjdump "$so" -lptx 2>/dev/null | sed 's/.*\.\(compute_[0-9]*\)\..*/\1/' | sort -u ; \
    done
