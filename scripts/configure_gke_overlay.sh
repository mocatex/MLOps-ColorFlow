#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

if [ "$#" -gt 1 ]; then
  echo "Usage: $0 [env-file]" >&2
  exit 1
fi

if [ "$#" -eq 1 ]; then
  env_file="$1"
  # shellcheck disable=SC1090
  . "$env_file"
fi

: "${PROJECT_ID:?Set PROJECT_ID to your Google Cloud project ID}"

REGION="${REGION:-europe-west6}"
REPOSITORY="${REPOSITORY:-colorflow}"
IMAGE_TAG="${IMAGE_TAG:-${TAG:-latest}}"
APP_HOST="${APP_HOST:-}"
GCP_SERVICE_ACCOUNT_EMAIL="${GCP_SERVICE_ACCOUNT_EMAIL:?Set GCP_SERVICE_ACCOUNT_EMAIL to your Workload Identity service account email}"

if [ -n "${MLFLOW_ARTIFACT_ROOT:-}" ]; then
  artifact_root="$MLFLOW_ARTIFACT_ROOT"
else
  : "${MLFLOW_ARTIFACT_BUCKET:?Set MLFLOW_ARTIFACT_BUCKET or MLFLOW_ARTIFACT_ROOT}"
  artifact_root="gs://${MLFLOW_ARTIFACT_BUCKET}/artifacts"
fi

registry_host="${REGION}-docker.pkg.dev"
image_prefix="${registry_host}/${PROJECT_ID}/${REPOSITORY}"

cat > k8s/overlays/gke/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
images:
  - name: colorflow-mlflow
    newName: ${image_prefix}/colorflow-mlflow
    newTag: ${IMAGE_TAG}
  - name: colorflow-model-registry
    newName: ${image_prefix}/colorflow-model-registry
    newTag: ${IMAGE_TAG}
  - name: colorflow-trainer
    newName: ${image_prefix}/colorflow-trainer
    newTag: ${IMAGE_TAG}
  - name: colorflow-ui
    newName: ${image_prefix}/colorflow-ui
    newTag: ${IMAGE_TAG}
  - name: colorflow-mlserver
    newName: ${image_prefix}/colorflow-mlserver
    newTag: ${IMAGE_TAG}
patches:
  - path: namespace-labels-patch.yaml
  - path: checkpoints-pvc-patch.yaml
  - path: ingress-host-patch.yaml
  - path: runtime-serviceaccount-patch.yaml
  - path: mlflow-artifact-root-patch.yaml
  - path: mlflow-no-artifacts-volume-patch.yaml
  - path: mlserver-no-artifacts-volume-patch.yaml
EOF

mkdir -p k8s/jobs/gke/trainer k8s/jobs/gke/model-registry

cat > k8s/jobs/gke/trainer/trainer-dvc-pull-patch.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: trainer
  namespace: colorflow
spec:
  template:
    spec:
      containers:
        - name: trainer
          env:
            - name: DVC_PULL_DATA
              value: "true"
            - name: DVC_PULL_TARGET
              value: data/images.dvc
EOF

cat > k8s/jobs/gke/trainer/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base/trainer
images:
  - name: colorflow-trainer
    newName: ${image_prefix}/colorflow-trainer
    newTag: ${IMAGE_TAG}
patches:
  - path: trainer-dvc-pull-patch.yaml
EOF

cat > k8s/jobs/gke/model-registry/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base/model-registry
images:
  - name: colorflow-model-registry
    newName: ${image_prefix}/colorflow-model-registry
    newTag: ${IMAGE_TAG}
EOF

if [ -n "$APP_HOST" ]; then
cat > k8s/overlays/gke/ingress-host-patch.yaml <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: colorflow
  namespace: colorflow
  annotations:
    kubernetes.io/ingress.class: gce
spec:
  defaultBackend:
    service:
      name: ui
      port:
        number: 80
  rules:
    - host: ${APP_HOST}
      http:
        paths:
          - path: /v2
            pathType: Prefix
            backend:
              service:
                name: mlserver
                port:
                  number: 8080
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ui
                port:
                  number: 80
EOF
else
cat > k8s/overlays/gke/ingress-host-patch.yaml <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: colorflow
  namespace: colorflow
  annotations:
    kubernetes.io/ingress.class: gce
spec:
  defaultBackend:
    service:
      name: ui
      port:
        number: 80
  rules:
    - http:
        paths:
          - path: /v2
            pathType: Prefix
            backend:
              service:
                name: mlserver
                port:
                  number: 8080
          - path: /
            pathType: Prefix
            backend:
              service:
                name: ui
                port:
                  number: 80
EOF
fi

cat > k8s/overlays/gke/mlflow-artifact-root-patch.yaml <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: colorflow
spec:
  template:
    spec:
      containers:
        - name: mlflow
          env:
            - name: MLFLOW_ARTIFACT_ROOT
              value: ${artifact_root}
EOF

cat > k8s/overlays/gke/runtime-serviceaccount-patch.yaml <<EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: colorflow-runtime
  namespace: colorflow
  annotations:
    iam.gke.io/gcp-service-account: ${GCP_SERVICE_ACCOUNT_EMAIL}
EOF

cat <<EOF
Configured GKE overlays with:
  PROJECT_ID=${PROJECT_ID}
  REGION=${REGION}
  REPOSITORY=${REPOSITORY}
  IMAGE_TAG=${IMAGE_TAG}
  APP_HOST=${APP_HOST:-<none>}
  MLFLOW_ARTIFACT_ROOT=${artifact_root}
  GCP_SERVICE_ACCOUNT_EMAIL=${GCP_SERVICE_ACCOUNT_EMAIL}

Next:
  1. Push images with scripts/build_and_push_gke_images.sh using the same PROJECT_ID/REGION/REPOSITORY/IMAGE_TAG values.
  2. Apply k8s/overlays/gke.
  3. Trigger k8s/jobs/gke/trainer and k8s/jobs/gke/model-registry only when you want them.
EOF
