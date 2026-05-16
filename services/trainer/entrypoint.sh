#!/bin/sh
# Training/tuning entrypoint.
#
# Data provisioning is manual. The container expects the dataset directory to
# already exist, either because the image was built after `dvc pull`, because the
# directory is mounted in at runtime, or because the entrypoint pulls it.

set -e

DVC_ROOT="${DVC_ROOT:-/app/storage/mlops-coco}"
DATA_DIR="${DATA_DIR:-${COLORFLOW_DATA_DIR:-/app/storage/mlops-coco/images}}"
DVC_PULL_DATA="${DVC_PULL_DATA:-false}"
DVC_PULL_TARGET="${DVC_PULL_TARGET:-images.dvc}"
GCS_DATA_URI="${GCS_DATA_URI:-}"

if [ -n "$GCS_DATA_URI" ] && { [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; }; then
  echo "[entrypoint] data missing at $DATA_DIR, syncing from $GCS_DATA_URI" >&2
  mkdir -p "$DATA_DIR"
  python - "$GCS_DATA_URI" "$DATA_DIR" <<'PY'
from pathlib import Path
import sys

from google.cloud import storage


uri = sys.argv[1]
target_root = Path(sys.argv[2])

if not uri.startswith("gs://"):
    raise ValueError(f"Unsupported GCS URI: {uri}")

bucket_name, _, prefix = uri[5:].partition("/")
client = storage.Client()
downloaded = 0

for blob in client.list_blobs(bucket_name, prefix=prefix):
    if blob.name.endswith("/"):
        continue
    relative_name = blob.name[len(prefix):].lstrip("/") if prefix else blob.name
    destination = target_root / relative_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(destination)
    downloaded += 1

print(f"[entrypoint] synced {downloaded} objects from {uri} to {target_root}", file=sys.stderr)
PY
fi

if [ "$DVC_PULL_DATA" = "true" ] && { [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; }; then
  echo "[entrypoint] data missing at $DATA_DIR, pulling $DVC_PULL_TARGET via DVC from $DVC_ROOT" >&2
  cd "$DVC_ROOT"

  git init > /dev/null 2>&1 # DVC requires a Git repository -> dummy repo
  dvc pull "$DVC_PULL_TARGET"
  cd /app
fi

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] expected pre-populated data at $DATA_DIR" >&2
  echo "[entrypoint] run 'dvc pull' manually before building the image, enable DVC_PULL_DATA, or mount the dataset into the container" >&2
  exit 1
fi

exec python "$@"
