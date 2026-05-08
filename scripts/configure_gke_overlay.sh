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
IMAGE_TAG="${TAG:-${IMAGE_TAG:-latest}}"
APP_HOST="${APP_HOST:-}"
GCP_SERVICE_ACCOUNT_EMAIL="${GCP_SERVICE_ACCOUNT_EMAIL:?Set GCP_SERVICE_ACCOUNT_EMAIL to your Workload Identity service account email}"
MLFLOW_ARTIFACT_STORAGE_MODE="${MLFLOW_ARTIFACT_STORAGE_MODE:-bucket}"

artifact_resources=""
artifact_overlay_patches=""
mlflow_stage_artifact_delete=""
mlserver_stage_artifact_delete=""
ui_stage_artifact_delete=""

case "$MLFLOW_ARTIFACT_STORAGE_MODE" in
  bucket)
    if [ -n "${MLFLOW_ARTIFACT_ROOT:-}" ]; then
      artifact_root="$MLFLOW_ARTIFACT_ROOT"
    else
      : "${MLFLOW_ARTIFACT_BUCKET:?Set MLFLOW_ARTIFACT_BUCKET or MLFLOW_ARTIFACT_ROOT}"
      artifact_root="gs://${MLFLOW_ARTIFACT_BUCKET}/artifacts"
    fi

    artifact_overlay_patches=$(cat <<'EOF'
  - path: mlflow-no-artifacts-volume-patch.yaml
  - path: mlserver-no-artifacts-volume-patch.yaml
EOF
)

    mlflow_stage_artifact_delete=$(cat <<'EOF'
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-artifacts
  namespace: colorflow

$patch: delete
EOF
)

    mlserver_stage_artifact_delete="$mlflow_stage_artifact_delete"
    ui_stage_artifact_delete="$mlflow_stage_artifact_delete"
    ;;
  filesystem)
    : "${MLFLOW_ARTIFACT_NFS_SERVER:?Set MLFLOW_ARTIFACT_NFS_SERVER when MLFLOW_ARTIFACT_STORAGE_MODE=filesystem}"
    MLFLOW_ARTIFACT_NFS_PATH="${MLFLOW_ARTIFACT_NFS_PATH:-/colorflow}"
    MLFLOW_ARTIFACT_PVC_SIZE="${MLFLOW_ARTIFACT_PVC_SIZE:-10Gi}"
    artifact_root="${MLFLOW_ARTIFACT_ROOT:-/outputs/mlruns}"

    artifact_resources=$(cat <<'EOF'
  - mlflow-artifacts-pv.yaml
EOF
)

    artifact_overlay_patches=$(cat <<'EOF'
  - path: mlflow-artifacts-pvc-patch.yaml
EOF
)
    ;;
  *)
    echo "Unsupported MLFLOW_ARTIFACT_STORAGE_MODE: $MLFLOW_ARTIFACT_STORAGE_MODE" >&2
    echo "Expected one of: bucket, filesystem" >&2
    exit 1
    ;;
esac

registry_host="${REGION}-docker.pkg.dev"
image_prefix="${registry_host}/${PROJECT_ID}/${REPOSITORY}"

{
cat <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base
EOF

if [ -n "$artifact_resources" ]; then
  printf '%s\n' "$artifact_resources"
fi

cat <<EOF
  - checkpoints-bucket-pv.yaml
  - checkpoints-bucket-pvc.yaml
images:
  - name: colorflow-mlflow
    newName: ${image_prefix}/colorflow-mlflow
    newTag: ${IMAGE_TAG}
  - name: colorflow-registry
    newName: ${image_prefix}/colorflow-registry
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
  - path: ingress-host-patch.yaml
  - path: runtime-serviceaccount-patch.yaml
  - path: mlflow-artifact-root-patch.yaml
EOF

if [ -n "$artifact_overlay_patches" ]; then
  printf '%s\n' "$artifact_overlay_patches"
fi

cat <<'EOF'
  - path: mlserver-resources-patch.yaml
EOF
} > k8s/overlays/gke/kustomization.yaml

mkdir -p k8s/stages/gke/mlflow k8s/stages/gke/mlserver k8s/stages/gke/ui

cat > k8s/stages/gke/mlflow/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

{
if [ -n "$mlflow_stage_artifact_delete" ]; then
  printf '%s\n' "$mlflow_stage_artifact_delete"
  printf '%s\n' '---'
fi

cat <<'EOF'
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-checkpoints-bucket

$patch: delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlserver
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: mlserver
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ui
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: ui
  namespace: colorflow

$patch: delete
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: colorflow
  namespace: colorflow

$patch: delete
EOF
} > k8s/stages/gke/mlflow/exclude-resources-patch.yaml

cat > k8s/stages/gke/mlserver/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

{
cat <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-secrets
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data
  namespace: colorflow

$patch: delete
EOF

if [ -n "$mlserver_stage_artifact_delete" ]; then
  printf '%s\n' "---"
  printf '%s\n' "$mlserver_stage_artifact_delete"
fi

cat <<'EOF'
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-checkpoints-bucket

$patch: delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ui
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: ui
  namespace: colorflow

$patch: delete
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: colorflow
  namespace: colorflow

$patch: delete
EOF
} > k8s/stages/gke/mlserver/exclude-resources-patch.yaml

cat > k8s/stages/gke/ui/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

{
cat <<'EOF'
apiVersion: v1
kind: Secret
metadata:
  name: mlflow-secrets
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-data
  namespace: colorflow

$patch: delete
EOF

if [ -n "$ui_stage_artifact_delete" ]; then
  printf '%s\n' "---"
  printf '%s\n' "$ui_stage_artifact_delete"
fi

cat <<'EOF'
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-checkpoints-bucket

$patch: delete
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: mlflow
  namespace: colorflow

$patch: delete
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlserver
  namespace: colorflow

$patch: delete
---
apiVersion: v1
kind: Service
metadata:
  name: mlserver
  namespace: colorflow

$patch: delete
EOF
} > k8s/stages/gke/ui/exclude-resources-patch.yaml

mkdir -p k8s/jobs/gke/trainer k8s/jobs/gke/registry

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
              value: images.dvc
EOF

cat > k8s/jobs/gke/trainer-checkpoints-bucket-patch.yaml <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: trainer
  namespace: colorflow
spec:
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
    spec:
      containers:
        - name: trainer
          env:
            - name: COLORFLOW_CHECKPOINT_DIR
              value: /checkpoints
            - name: COLORFLOW_CHECKPOINT_URI_PREFIX
              value: gs://mlops-checkpoints
          volumeMounts:
            - name: checkpoints
              mountPath: /checkpoints
      volumes:
        - name: checkpoints
          persistentVolumeClaim:
            claimName: model-checkpoints
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
  - path: trainer-checkpoints-bucket-patch.yaml
EOF

cat > k8s/jobs/gke/registry/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../base/registry
images:
  - name: colorflow-registry
    newName: ${image_prefix}/colorflow-registry
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

if [ "$MLFLOW_ARTIFACT_STORAGE_MODE" = "filesystem" ]; then
cat > k8s/overlays/gke/mlflow-artifacts-pv.yaml <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mlflow-artifacts-filestore
spec:
  capacity:
    storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  mountOptions:
    - nfsvers=3
  nfs:
    server: ${MLFLOW_ARTIFACT_NFS_SERVER}
    path: ${MLFLOW_ARTIFACT_NFS_PATH}
EOF

cat > k8s/overlays/gke/mlflow-artifacts-pvc-patch.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-artifacts
  namespace: colorflow
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  storageClassName: ""
  volumeName: mlflow-artifacts-filestore
EOF
fi

if [ "$MLFLOW_ARTIFACT_STORAGE_MODE" != "filesystem" ]; then
  rm -f k8s/overlays/gke/mlflow-artifacts-pv.yaml k8s/overlays/gke/mlflow-artifacts-pvc-patch.yaml
fi

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
  MLFLOW_ARTIFACT_STORAGE_MODE=${MLFLOW_ARTIFACT_STORAGE_MODE}
  MLFLOW_ARTIFACT_ROOT=${artifact_root}
  GCP_SERVICE_ACCOUNT_EMAIL=${GCP_SERVICE_ACCOUNT_EMAIL}
EOF

if [ "$MLFLOW_ARTIFACT_STORAGE_MODE" = "filesystem" ]; then
cat <<EOF
  MLFLOW_ARTIFACT_NFS_SERVER=${MLFLOW_ARTIFACT_NFS_SERVER}
  MLFLOW_ARTIFACT_NFS_PATH=${MLFLOW_ARTIFACT_NFS_PATH}
  MLFLOW_ARTIFACT_PVC_SIZE=${MLFLOW_ARTIFACT_PVC_SIZE}
EOF
fi

cat <<EOF
Next:
  1. Push images with scripts/build_and_push_gke_images.sh using the same PROJECT_ID/REGION/REPOSITORY/IMAGE_TAG values.
  2. For ordered startup, apply k8s/stages/gke/mlflow, then k8s/stages/gke/mlserver, then k8s/stages/gke/ui.
  3. Or apply k8s/overlays/gke if you still want the whole platform at once.
  4. Trigger k8s/jobs/gke/trainer and k8s/jobs/gke/registry only when you want them.
EOF
