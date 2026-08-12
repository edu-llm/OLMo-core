# Base is injected by the platform build as ${BASE_IMAGE} and verified pre-build by
# tools/verify_dockerfile_base.py -- naming an unregistered image fails closed, so
# do not write a literal FROM here.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /app

# Kept deliberately small. Every layer rebuilds on every push to an edullm/**
# branch, and build cache is one of the two levers on the bill.
RUN pip install --no-cache-dir \
      "torch>=2.6" \
      "numpy>=2.0" \
      "PyYAML>=6.0" \
      "tiktoken>=0.7" \
      "boto3>=1.34"

COPY memsplit/ /app/memsplit/
COPY scripts/  /app/scripts/
COPY configs/  /app/configs/
COPY .edullm/  /app/.edullm/

ENV PYTHONPATH=/app PYTHONUNBUFFERED=1
