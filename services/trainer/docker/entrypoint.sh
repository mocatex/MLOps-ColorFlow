#!/bin/sh
# Training/tuning entrypoint.
#
# Data provisioning is manual. The container expects /app/data/images to already
# exist, either because the image was built after `dvc pull`, because the
# directory is mounted in at runtime, or because the entrypoint pulls it.

set -e

DATA_DIR="${DATA_DIR:-/app/data/images}"
DVC_PULL_DATA="${DVC_PULL_DATA:-false}"
DVC_PULL_TARGET="${DVC_PULL_TARGET:-data/images.dvc}"

if [ "$DVC_PULL_DATA" = "true" ] && { [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; }; then
  echo "[entrypoint] data missing at $DATA_DIR, pulling $DVC_PULL_TARGET via DVC" >&2
  dvc pull "$DVC_PULL_TARGET"
fi

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] expected pre-populated data at $DATA_DIR" >&2
  echo "[entrypoint] run 'dvc pull' manually before building the image, enable DVC_PULL_DATA, or mount the dataset into the container" >&2
  exit 1
fi

exec python "$@"
