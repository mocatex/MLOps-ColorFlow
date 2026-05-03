#!/bin/sh
# Hybrid entrypoint for training/tuning containers.
#
# Behaviour:
#   - If /app/data/images is non-empty, assume data is already provisioned
#     (bind mount on host, PVC in K8s, initContainer-populated emptyDir, ...)
#     and skip `dvc pull`.
#   - Otherwise run `dvc pull` to materialise the DVC-tracked dataset.
#
# Override the auto-detection with DVC_PULL=force | skip | auto (default auto).

set -e

DATA_DIR="${DATA_DIR:-/app/data/images}"
DVC_PULL="${DVC_PULL:-auto}"

case "$DVC_PULL" in
  force)
    echo "[entrypoint] DVC_PULL=force — pulling data"
    dvc pull --quiet
    ;;
  skip)
    echo "[entrypoint] DVC_PULL=skip — not pulling"
    ;;
  auto)
    if [ -d "$DATA_DIR" ] && [ -n "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
      echo "[entrypoint] data present at $DATA_DIR — skipping dvc pull"
    else
      echo "[entrypoint] data missing at $DATA_DIR — running dvc pull"
      dvc pull --quiet
    fi
    ;;
  *)
    echo "[entrypoint] unknown DVC_PULL=$DVC_PULL (expected auto|force|skip)" >&2
    exit 1
    ;;
esac

exec python "$@"
