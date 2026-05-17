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
MLFLOW_ARTIFACT_PVC_SIZE="${MLFLOW_ARTIFACT_PVC_SIZE:-10Gi}"

artifact_resources=$(cat <<'EOF'
  - mlflow-artifacts-pv.yaml
  - checkpoints-pv.yaml
EOF
)

artifact_overlay_patches=$(cat <<'EOF'
  - path: mlflow-artifacts-pvc-patch.yaml
  - path: checkpoints-pvc-patch.yaml
EOF
)

bucket_overlay_patches=""
trainer_gcsfuse_patch=""
uploader_gcsfuse_patch=""

case "$MLFLOW_ARTIFACT_STORAGE_MODE" in
  bucket)
    : "${MLFLOW_ARTIFACT_BUCKET:?Set MLFLOW_ARTIFACT_BUCKET when MLFLOW_ARTIFACT_STORAGE_MODE=bucket}"
    : "${COLORFLOW_CHECKPOINT_BUCKET:?Set COLORFLOW_CHECKPOINT_BUCKET when MLFLOW_ARTIFACT_STORAGE_MODE=bucket}"
    artifact_root="${MLFLOW_ARTIFACT_ROOT:-/outputs/mlruns}"

    bucket_overlay_patches=$(cat <<'EOF'
  - path: mlflow-gcsfuse-annotation-patch.yaml
  - path: mlserver-gcsfuse-annotation-patch.yaml
EOF
)

    trainer_gcsfuse_patch=$(cat <<'EOF'
  - path: gcsfuse-annotation-patch.yaml
EOF
)

    uploader_gcsfuse_patch=$(cat <<'EOF'
patches:
  - path: gcsfuse-annotation-patch.yaml
EOF
)
    ;;
  filesystem)
    : "${MLFLOW_ARTIFACT_NFS_SERVER:?Set MLFLOW_ARTIFACT_NFS_SERVER when MLFLOW_ARTIFACT_STORAGE_MODE=filesystem}"
    MLFLOW_ARTIFACT_NFS_PATH="${MLFLOW_ARTIFACT_NFS_PATH:-/colorflow}"
    artifact_root="${MLFLOW_ARTIFACT_ROOT:-/outputs/mlruns}"
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

printf '%s\n' "$artifact_resources"

cat <<EOF
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

printf '%s\n' "$artifact_overlay_patches"

if [ -n "$bucket_overlay_patches" ]; then
  printf '%s\n' "$bucket_overlay_patches"
fi

cat <<'EOF'
  - path: mlserver-resources-patch.yaml
EOF
} > k8s/overlays/gke/kustomization.yaml

mkdir -p \
  k8s/stages/gke/mlflow \
  k8s/stages/gke/mlserver \
  k8s/stages/gke/ui \
  k8s/jobs/gke/trainer \
  k8s/jobs/gke/registry \
  k8s/tools/gke/uploader

cat > k8s/stages/gke/mlflow/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

cat > k8s/stages/gke/mlflow/exclude-resources-patch.yaml <<'EOF'
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

cat > k8s/stages/gke/mlserver/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

cat > k8s/stages/gke/mlserver/exclude-resources-patch.yaml <<'EOF'
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

cat > k8s/stages/gke/ui/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - ../../../overlays/gke
patches:
  - path: exclude-resources-patch.yaml
EOF

cat > k8s/stages/gke/ui/exclude-resources-patch.yaml <<'EOF'
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

if [ -n "$trainer_gcsfuse_patch" ]; then
  printf '%s\n' "$trainer_gcsfuse_patch" >> k8s/jobs/gke/trainer/kustomization.yaml
fi

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

cat > k8s/tools/gke/uploader/kustomization.yaml <<EOF
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - uploader-pod.yaml
EOF

if [ -n "$uploader_gcsfuse_patch" ]; then
  printf '%s\n' "$uploader_gcsfuse_patch" >> k8s/tools/gke/uploader/kustomization.yaml
fi

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

if [ "$MLFLOW_ARTIFACT_STORAGE_MODE" = "bucket" ]; then
cat > k8s/overlays/gke/mlflow-artifacts-pv.yaml <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: mlflow-artifacts-gcsfuse
spec:
  capacity:
    storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  mountOptions:
    - implicit-dirs
  claimRef:
    namespace: colorflow
    name: mlflow-artifacts
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: ${MLFLOW_ARTIFACT_BUCKET}
EOF

cat > k8s/overlays/gke/checkpoints-pv.yaml <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-checkpoints-gcsfuse
spec:
  capacity:
    storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  mountOptions:
    - implicit-dirs
  claimRef:
    namespace: colorflow
    name: model-checkpoints
  csi:
    driver: gcsfuse.csi.storage.gke.io
    volumeHandle: ${COLORFLOW_CHECKPOINT_BUCKET}
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
  volumeName: mlflow-artifacts-gcsfuse
EOF

cat > k8s/overlays/gke/checkpoints-pvc-patch.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
  namespace: colorflow
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  storageClassName: ""
  volumeName: model-checkpoints-gcsfuse
EOF

cat > k8s/overlays/gke/mlflow-gcsfuse-annotation-patch.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: colorflow
spec:
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
        gke-gcsfuse/ephemeral-storage-request: "1Gi"
        gke-gcsfuse/ephemeral-storage-limit: "1Gi"
EOF

cat > k8s/overlays/gke/mlserver-gcsfuse-annotation-patch.yaml <<'EOF'
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlserver
  namespace: colorflow
spec:
  template:
    metadata:
      annotations:
        gke-gcsfuse/volumes: "true"
EOF

cat > k8s/jobs/gke/trainer/gcsfuse-annotation-patch.yaml <<'EOF'
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
EOF

cat > k8s/tools/gke/uploader/gcsfuse-annotation-patch.yaml <<'EOF'
apiVersion: v1
kind: Pod
metadata:
  name: artifact-uploader
  namespace: colorflow
  annotations:
    gke-gcsfuse/volumes: "true"
EOF
else
  artifact_nfs_root="${MLFLOW_ARTIFACT_NFS_PATH%/}"

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
  storageClassName: ""
  mountOptions:
    - nfsvers=3
  nfs:
    server: ${MLFLOW_ARTIFACT_NFS_SERVER}
    path: ${artifact_nfs_root}/mlruns
EOF

cat > k8s/overlays/gke/checkpoints-pv.yaml <<EOF
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-checkpoints-filestore
spec:
  capacity:
    storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  storageClassName: ""
  mountOptions:
    - nfsvers=3
  nfs:
    server: ${MLFLOW_ARTIFACT_NFS_SERVER}
    path: ${artifact_nfs_root}/checkpoints
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

cat > k8s/overlays/gke/checkpoints-pvc-patch.yaml <<EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-checkpoints
  namespace: colorflow
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: ${MLFLOW_ARTIFACT_PVC_SIZE}
  storageClassName: ""
  volumeName: model-checkpoints-filestore
EOF

rm -f \
  k8s/overlays/gke/mlflow-gcsfuse-annotation-patch.yaml \
  k8s/overlays/gke/mlserver-gcsfuse-annotation-patch.yaml \
  k8s/jobs/gke/trainer/gcsfuse-annotation-patch.yaml \
  k8s/tools/gke/uploader/gcsfuse-annotation-patch.yaml
fi

rm -f \
  k8s/overlays/gke/mlflow-no-artifacts-volume-patch.yaml \
  k8s/overlays/gke/mlserver-no-artifacts-volume-patch.yaml

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
else
cat <<EOF
  MLFLOW_ARTIFACT_BUCKET=${MLFLOW_ARTIFACT_BUCKET}
  COLORFLOW_CHECKPOINT_BUCKET=${COLORFLOW_CHECKPOINT_BUCKET}
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
