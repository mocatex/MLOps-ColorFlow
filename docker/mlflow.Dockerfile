# MLflow tracking server with Postgres backend + GCS artifact support.
# Backend store and artifact root are configured via env vars at runtime,
# so the same image works for local dev (sqlite + ./mlruns) and for K8s
# (postgres + gs://...).

FROM ghcr.io/mlflow/mlflow:v3.11.1

# psycopg2 → Postgres backend store driver
# google-cloud-storage + gcsfs → gs:// artifact store driver
RUN pip install --no-cache-dir \
        psycopg2-binary \
        google-cloud-storage \
        gcsfs

EXPOSE 5001

# Defaults are dev-friendly. Override these env vars in K8s / docker-compose:
#   MLFLOW_BACKEND_STORE_URI=postgresql://user:pass@host:5432/mlflow
#   MLFLOW_DEFAULT_ARTIFACT_ROOT=gs://mlops-coco/mlflow-artifacts
ENV MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db \
    MLFLOW_DEFAULT_ARTIFACT_ROOT=/mlruns \
    MLFLOW_ALLOWED_HOSTS="localhost,localhost:5001,127.0.0.1,127.0.0.1:5001,mlflow,mlflow:5001"

# Shell form so ${VAR} expansion works.
# --allowed-hosts is required by MLflow 3.x security middleware for any host
# other than `localhost`. Restrict in production (e.g. mlflow.svc.cluster.local).
CMD mlflow server \
        --host 0.0.0.0 \
        --port 5001 \
        --allowed-hosts "${MLFLOW_ALLOWED_HOSTS}" \
        --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
        --default-artifact-root ${MLFLOW_DEFAULT_ARTIFACT_ROOT}
