#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [env-file]" >&2
  exit 1
fi

env_file=""
if [ "$#" -eq 1 ]; then
  env_file="$1"
elif [ -f "$repo_root/scripts/gke.env" ]; then
  env_file="$repo_root/scripts/gke.env"
fi

if [ -n "$env_file" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$env_file"
  set +a
fi

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project ID}"

REGION="${REGION:-europe-west6}"
REPOSITORY="${REPOSITORY:-colorflow}"
TAG="${TAG:-${IMAGE_TAG:-latest}}"
PLATFORM="${PLATFORM:-linux/amd64}"
REGISTRY_HOST="${REGION}-docker.pkg.dev"
IMAGE_PREFIX="${REGISTRY_HOST}/${PROJECT_ID}/${REPOSITORY}"
MLFLOW_IMAGE="${IMAGE_PREFIX}/colorflow-mlflow:${TAG}"

echo "Configuring Docker auth for ${REGISTRY_HOST}"
gcloud auth configure-docker "$REGISTRY_HOST" --quiet

echo "Building and pushing linux/amd64 images for GKE"
echo "Using platform: ${PLATFORM}"

docker buildx build \
  --platform "${PLATFORM}" \
  --tag "${MLFLOW_IMAGE}" \
  --push \
  services/mlflow

docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg BASE_IMAGE="${MLFLOW_IMAGE}" \
  --tag "${IMAGE_PREFIX}/colorflow-registry:${TAG}" \
  --push \
  services/registry

docker buildx build \
  --platform "${PLATFORM}" \
  --file services/trainer/Dockerfile \
  --tag "${IMAGE_PREFIX}/colorflow-trainer:${TAG}" \
  --push \
  .

docker buildx build \
  --platform "${PLATFORM}" \
  --tag "${IMAGE_PREFIX}/colorflow-ui:${TAG}" \
  --push \
  services/ui

docker buildx build \
  --platform "${PLATFORM}" \
  --build-arg BASE_IMAGE="${MLFLOW_IMAGE}" \
  --tag "${IMAGE_PREFIX}/colorflow-mlserver:${TAG}" \
  --push \
  services/mlserver

cat <<EOF
Published images:
  ${MLFLOW_IMAGE}
  ${IMAGE_PREFIX}/colorflow-registry:${TAG}
  ${IMAGE_PREFIX}/colorflow-trainer:${TAG}
  ${IMAGE_PREFIX}/colorflow-ui:${TAG}
  ${IMAGE_PREFIX}/colorflow-mlserver:${TAG}
EOF
