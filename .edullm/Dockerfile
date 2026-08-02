# Research image for the eduLLM platform build pipeline.
#
# The platform supplies the registered base image as --build-arg BASE_IMAGE, already
# pinned to a digest. Its verify_dockerfile_base gate runs before the build and refuses
# this file unless BASE_IMAGE is declared in the global scope with no default and every
# stage roots at ${BASE_IMAGE}. The ARG is bare on purpose: a default would be a base
# image nobody registered, sitting here waiting for the day the build-arg is dropped,
# and the provenance record would still name the registered digest.
ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# THIS FILE USED TO INSTALL NOTHING, AND THE REASON IT STOPPED IS WORTH READING.
#
# The claim was that installing was impossible here: ai2-olmo-core declares a torch
# dependency, so `pip install .` would fetch a multi-gigabyte wheel and take the build past what a
# standard runner can do, and `pip install --no-deps .` would leave an installed
# distribution whose imports fail anyway. Putting the source on PYTHONPATH was the answer.
#
# Both halves of that were measured in Phase 4 and the first half is false. A build of this
# file against the registered python base produced a working 2.94 GB image on an ordinary
# runner, and the torch wheel PyPI serves for linux x86_64 is CUDA-enabled, so it runs on a
# GPU. The install the comment was avoiding is affordable and always was.
#
# WHAT THE PLATFORM REGISTERS IS STILL A BARE PYTHON BASE, AND THAT IS A KNOWN LIMITATION
# RATHER THAN AN OVERSIGHT. The obvious alternative was to register a base that already
# carries torch, so the platform's pinned base digest would cover it. That was tried and
# withdrawn: the platform's Phase 1 rebuild study is asserted to have been made from the
# currently registered base, and re-recording it needs the new image published, which needs
# the registry change merged -- the change that invalidates the study is a prerequisite for
# the evidence that repairs it. The ways through were a knowingly broken main or relaxing a
# control and restoring it afterwards, and neither is worth a tidier narrative.
#
# So the honest statement is: the registered base digest describes a small fraction of this
# image, and the rest is pinned here instead. Reproducibility is preserved -- an exact
# version against an index that never republishes a version is as deterministic as a digest
# -- but a reader of the provenance record who takes base_image_digest as a description of
# the image will be wrong, and this paragraph is where they find out.
ENV PYTHONPATH=/opt/olmo-core/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# The base sets no PYTHONPATH of its own. If a future base does set one, this line silently
# discards it. Prepending is not the safe-looking fix it appears to be:
# ENV PYTHONPATH=/opt/olmo-core/src:${PYTHONPATH} leaves a trailing colon when the base
# sets nothing, and Python reads a trailing colon as the current working directory.

WORKDIR /opt/olmo-core

# A C COMPILER, BECAUSE torch.compile IS UNUSABLE WITHOUT ONE AND THE EXAMPLE TURNS IT ON.
#
# The registered base is a bare python image and carries no toolchain. That is invisible
# until something compiles, and the first thing that does is Inductor: a training run built
# from src/examples/llm/train.py sets compile_model=True, and the run dies with
# `InductorError: RuntimeError: Failed to find C compiler. Please specify via CC
# environment variable or set triton.knobs.build.impl.`
#
# Found the way these things are: a twelve-hour submission on 2026-08-01 reached the GPU,
# built the model, wrote its first checkpoint object to S3, and died on the first compiled
# region. Not at build time, not in a smoke test, and not until an A10G had been paid for.
#
# gcc rather than build-essential. Inductor and Triton shell out to a C compiler to build
# the kernels they generate; they do not need make, dpkg-dev, or the rest of the
# metapackage, and this image is what an untrusted training command runs inside -- so the
# smaller toolchain is both lighter and less to reason about. g++ comes with it because
# Inductor's CPU path emits C++.
#
# Pinned to no version deliberately, unlike torch above. The pin on torch exists because a
# resolved wheel changes what the model computes; a compiler that builds a kernel and then
# is never shipped anywhere does not, and pinning to a Debian point release would make this
# line expire on the base's next security update.
#
# apt-get is the right tool here rather than an assumption: the registered base digest
# sha256:57cd7c3a…10de resolves to Debian trixie carrying python 3.12.13, read out of the
# image config on Docker Hub rather than inferred from the repository name. An Alpine base
# would need apk and would not have manylinux wheels for torch either, so a base swap that
# broke this line would already have broken the install above it.
#
# `cc --version` at the end of the RUN is the assertion, in the same spirit as the torch
# checks below: it fails the build if the compiler is somehow not on PATH afterwards, which
# is the whole condition this layer exists to guarantee.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends gcc g++; \
    rm -rf /var/lib/apt/lists/*; \
    cc --version

# The build context is the repository root, so the root .dockerignore is what keeps .git
# out of this layer, and out of the context entirely.
COPY . .

# Keep the established OLMo-core image contract: pinned torch 2.9, this source checkout,
# W&B, boto3 for OLMo-core's S3 I/O, and the platform-pinned edullm-data reader. Weighted
# training uses OLMo-core's built-in SourceMixtureDatasetConfig and needs no ai2-olmo,
# FlashAttention, evaluator, or vendored experiment stack.
RUN set -eux; \
    python -m pip install --no-cache-dir "torch==2.9.0"; \
    python -m pip install --no-cache-dir ".[wandb]" boto3; \
    python -m pip install --no-cache-dir "edullm-data @ https://github.com/edu-llm/edullm-data/archive/38bf831a6c3f445e394784018441fd59288b876c.tar.gz"; \
    python -c "import torch; assert torch.__version__.startswith('2.9.0'), f'torch was moved off the pin: {torch.__version__}'"; \
    python -c "import torch; assert torch.version.cuda, 'the CUDA build of torch was replaced by a CPU wheel'"; \
    python -c "import boto3, wandb; from olmo_core.data import NumpyFSLDatasetConfig; from olmo_core.data.source_mixture import SourceMixtureDatasetConfig; from olmo_core.nn.transformer import TransformerConfig; print(TransformerConfig.olmo2_370M(vocab_size=100352).__class__.__name__)"; \
    python -c "from edullm_data.read import dataset_paths; print(dataset_paths.__module__)"

# No CMD. The entrypoint of the registered base is inherited deliberately, so swapping the
# base does not leave a stale command behind.
#
# No USER either, so this image runs as root, inherited from the base. That was recorded
# here as a decision for the phase that introduces the workload role and can settle it
# against a real runtime. Phase 4 is that phase and the answer is: still root, for now. The
# container's blast radius is bounded by the workload role rather than by its uid -- that
# role may write one team's output prefix and read it back, and holds nothing else -- and
# the training base expects to own /opt/conda, which a non-root uid cannot write to without
# a chown layer that would roughly double the image. Revisit when the image is built for a
# fixed uid rather than adapted to one.
