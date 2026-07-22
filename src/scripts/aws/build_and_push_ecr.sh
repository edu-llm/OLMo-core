#!/bin/bash
#
# Build the OLMo-core training image and push it to AWS ECR.
#
# Run this on an x86_64 Linux Docker host (NOT macOS/Apple Silicon): building flash-attn and
# TransformerEngine from source needs CUDA toolchain, lots of CPU/RAM, and time (tens of
# minutes). Once pushed, point the scheduler at the image via OLMO_ECR_IMAGE (see env.example.sh).
#
# Usage: ./build_and_push_ecr.sh [region] [repo] [tag]
#   region  AWS region for the ECR repo   (default: us-east-2)
#   repo    ECR repository name           (default: olmo-core)
#   tag     image tag                     (default: today's date)
set -euo pipefail

REGION="${1:-us-east-2}"
REPO="${2:-olmo-core}"
TAG="${3:-$(date +%Y-%m-%d)}"

# Build args mirror the repo Makefile's `docker-image` target; the Dockerfile carries defaults
# for the rest (flash-attn, TE, etc.).
CUDA_VERSION=12.8.1
CUDA_VERSION_PATH=cu128
PYTHON_VERSION=3.12
TORCH_VERSION=2.10.0

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE="${REGISTRY}/${REPO}:${TAG}"

# cd to repo root (this script lives at src/scripts/aws/).
cd "$(cd "$(dirname "$0")/../../.." && pwd)"

echo "==> Ensuring ECR repository '${REPO}' exists in ${REGION}..."
aws ecr describe-repositories --region "$REGION" --repository-names "$REPO" >/dev/null 2>&1 \
    || aws ecr create-repository --region "$REGION" --repository-name "$REPO" >/dev/null

echo "==> Logging Docker into ECR (${REGISTRY})..."
aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$REGISTRY"

echo "==> Building ${IMAGE} (this takes a while)..."
docker build -f src/Dockerfile \
    --build-arg CUDA_VERSION="$CUDA_VERSION" \
    --build-arg CUDA_VERSION_PATH="$CUDA_VERSION_PATH" \
    --build-arg PYTHON_VERSION="$PYTHON_VERSION" \
    --build-arg TORCH_VERSION="$TORCH_VERSION" \
    --target release \
    -t "$IMAGE" .

echo "==> Pushing ${IMAGE}..."
docker push "$IMAGE"

echo ""
echo "✓ Pushed ${IMAGE}"
echo "  Add to your env.sh:  export OLMO_ECR_IMAGE=${IMAGE}"
