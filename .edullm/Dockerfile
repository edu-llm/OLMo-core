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

# Nothing is installed. ai2-olmo-core declares torch>=2.6.0 as a hard runtime
# dependency, so `pip install .` would fetch a multi-gigabyte wheel and take the build
# past what a standard runner can do; `pip install --no-deps .` would avoid that but
# leave an installed distribution whose imports fail anyway, having bought nothing.
# Putting the source on PYTHONPATH instead keeps this file correct across the Phase 4
# base change: on a training base that already carries torch the code becomes importable
# with no edit here, and because PYTHONPATH precedes site-packages, the commit that was
# built stays the copy that runs even if the base ships its own ai2-olmo-core.
#
# Phase 4 landmine: this ENV overwrites PYTHONPATH, it does not prepend to it. Check the
# incoming training base for a PYTHONPATH of its own before changing the registry, or
# whatever the base put there silently disappears. Prepending is not the safe fix it
# looks like, because ENV PYTHONPATH=/opt/olmo-core/src:${PYTHONPATH} leaves a trailing
# colon when the base sets nothing, and Python reads a trailing colon as the current
# working directory, which is a worse failure than the one it avoids.
ENV PYTHONPATH=/opt/olmo-core/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/olmo-core

# The build context is the repository root, so the root .dockerignore is what keeps .git
# out of this layer, and out of the context entirely.
COPY . .

# No CMD. The entrypoint of the registered base is inherited deliberately, so swapping
# the base in Phase 4 does not leave a stale command behind.
#
# Phase 4 landmine: no USER either, so this image runs as root, inherited from the base.
# That is a deliberate omission rather than an oversight. A USER here would have to guess
# the uid the training base expects and could break it outright, and choosing who this
# workload runs as is a policy decision for the phase that introduces the workload role
# and can settle it against a real runtime. It is not a decision this file should make.
