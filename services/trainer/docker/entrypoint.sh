#!/bin/sh
# Training/tuning entrypoint.
#
# Data provisioning is manual. The container expects /app/data/images to already
# exist, either because the image was built after `dvc pull` or because the
# directory is mounted in at runtime.

set -e

DATA_DIR="${DATA_DIR:-/app/data/images}"

if [ ! -d "$DATA_DIR" ] || [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "[entrypoint] expected pre-populated data at $DATA_DIR" >&2
  echo "[entrypoint] run 'dvc pull' manually before building the image or mount the dataset into the container" >&2
  exit 1
fi

exec python "$@"
