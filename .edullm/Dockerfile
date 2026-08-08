ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# Research image for the eduLLM platform build pipeline.
#
# The platform supplies the registered base image as --build-arg BASE_IMAGE,
# already pinned to a digest. Keep the ARG global and without a default so the
# platform's base-image provenance gate remains authoritative.
ENV PYTHONPATH=/opt/olmo-core/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TRITON_F32_DEFAULT=ieee \
    CC=gcc-14 \
    CXX=g++-14 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    PATH=/usr/local/cuda-12.8/bin:${PATH} \
    TORCH_CUDA_ARCH_LIST="8.0"

LABEL ai.edullm.mamba-comparison="true" \
      ai.edullm.xlstm.license="NXAI Community License; organizational approval confirmed" \
      ai.edullm.cuda-architecture="sm_80"

WORKDIR /opt/olmo-core

# Build tools are fixed to the compiler major used for the CUDA 12.8 extensions.
# NVIDIA's CUDA 12 wheel named ``nvidia-cuda-nvcc-cu12`` contains ptxas but not
# the nvcc driver. NVIDIA's Debian 13 repository carries only CUDA 13, so use
# its signed Debian 12 repository, which still publishes the 12.8 compiler.
COPY .edullm/patch_cuda_glibc241.py /tmp/patch_cuda_glibc241.py
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates wget gpgv gcc-14 g++-14; \
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb; \
    dpkg -i cuda-keyring_1.1-1_all.deb; \
    rm cuda-keyring_1.1-1_all.deb; \
    apt-get -o APT::Key::GPGVCommand=/usr/bin/gpgv update; \
    apt-get install -y --no-install-recommends cuda-nvcc-12-8=12.8.93-1; \
    python /tmp/patch_cuda_glibc241.py; \
    rm /tmp/patch_cuda_glibc241.py; \
    rm -rf /var/lib/apt/lists/*; \
    gcc-14 --version; \
    nvcc --version

# The xLSTM kernels are distributed under the NXAI Community License. Keep the
# organizational research approval and upstream attribution in the image.
RUN mkdir -p /usr/local/share/licenses/nxai && \
    printf '%s\n' \
      'Built with technology from NXAI.' \
      'This image contains mlstm-kernels and FlashRNN under the NXAI Community License.' \
      'Copyright © NXAI GmbH. All Rights Reserved.' \
      'Organizational approval for this research image is confirmed.' \
      > /usr/local/share/licenses/nxai/NOTICE

# Runtime and build dependencies precede source copies so ordinary research edits
# retain the expensive Torch, FlashAttention, Mamba-3, and CUDA-extension layers.
RUN set -eux; \
    python -m pip install --no-cache-dir \
      "pip==25.2" \
      "setuptools==69.5.1" \
      "wheel==0.45.1" \
      "ninja==1.13.0" \
      "packaging==25.0"; \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu128 \
      "torch==2.10.0"; \
    python -m pip install --no-cache-dir \
      "numpy==2.3.5" \
      "cached-path==1.8.0" \
      "filelock==3.20.0" \
      "requests==2.32.5" \
      "dataclass-extensions==0.3.0" \
      "PyYAML==6.0.3" \
      "rich==13.9.4" \
      "safetensors==0.7.0" \
      "importlib-resources==6.5.2" \
      "bettermap==1.3.1" \
      "pandas==2.3.3" \
      "boto3==1.40.70" \
      "botocore[crt]==1.40.70" \
      "wandb==0.23.0" \
      "flash-linear-attention==0.4.1"; \
    python -m pip install --no-cache-dir \
      "edullm-data @ https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"

# The official Mamba-3 source revision and xLSTM kernels build only for A100 sm80.
# The four shared attention layers use PyTorch fused SDPA, avoiding a separate
# FlashAttention source build on the research-image runner.
RUN set -eux; \
    TORCH_CUDA_ARCH_LIST="8.0" MAMBA_FORCE_BUILD=TRUE \
      python -m pip install --no-cache-dir --no-build-isolation \
      "mamba-ssm @ https://github.com/state-spaces/mamba/archive/e9594ce1c732d97440f0332fdc43170a2294dbfa.tar.gz"; \
    python -m pip install --no-cache-dir --no-build-isolation \
      "xlstm==2.0.5" "mlstm-kernels==2.0.4" "flashrnn==1.0.6"

# Build the repository's current native PD extension without invalidating the
# dependency layers when unrelated Python source changes.
COPY flash_pd_native_setup.py ./
COPY src/olmo_core/nn/flash_pd_native/csrc ./src/olmo_core/nn/flash_pd_native/csrc
RUN set -eux; \
    TORCH_CUDA_ARCH_LIST="8.0" python flash_pd_native_setup.py bdist_wheel; \
    python -m pip install --no-cache-dir --no-deps dist/flash_pd_native_kernels-*.whl; \
    rm -rf build dist

COPY . .
RUN set -eux; \
    python -m pip install --no-cache-dir --no-deps .; \
    python -c "import torch; assert torch.__version__.startswith('2.10.0'); assert torch.version.cuda == '12.8', torch.version.cuda; assert 'sm_80' in torch.cuda.get_arch_list()"; \
    python -c "from mamba_ssm.ops.triton.mamba3.mamba3_siso_combined import mamba3_siso_combined; assert callable(mamba3_siso_combined)"; \
    python -c "import _flash_pd_native_cuda; assert callable(_flash_pd_native_cuda.forward); assert callable(_flash_pd_native_cuda.backward); assert callable(_flash_pd_native_cuda.mamba3_forward); assert callable(_flash_pd_native_cuda.paper_backward)"; \
    python -c "import importlib.metadata as m; expected={'xlstm':'2.0.5','mlstm-kernels':'2.0.4','flashrnn':'1.0.6'}; assert {k:m.version(k) for k in expected} == expected"; \
    python -c "from olmo_core.nn.xlstm.olmo_xlstm import XLSTMMixerConfig; from olmo_core.nn.xlstm.olmo_slstm import SLSTMMixerConfig; assert XLSTMMixerConfig().chunkwise_kernel == 'chunkwise--triton_xl_chunk'; assert SLSTMMixerConfig(backend='cuda_fused').backend == 'cuda_fused'"; \
    python -c "from edullm_data.profiles.registry import available; assert sorted(available()) == ['eval-results/v1', 'pretrain-tokens/v1', 'sft-conversations/v1', 'token-order/v1', 'tokenizer/v1']"; \
    python -c "from edullm_data.read import dataset_paths, resolve_latest; print(dataset_paths.__module__, resolve_latest.__module__)"; \
    test -s /usr/local/share/licenses/nxai/NOTICE

# The platform supplies the command; this image deliberately defines no CMD.
