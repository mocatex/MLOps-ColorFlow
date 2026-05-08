# Start MLServer

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 uv run uvicorn app:app --host 0.0.0.0 --port 8080
```

This service exposes the inference API at `http://localhost:8080/v2/...`. The local UI can run separately on `http://localhost:8081` and will call MLServer there automatically.

## Model Loading Flow

MLServer keeps the serving flow deliberately narrow:

- on startup, it asks MLflow which model version the `champion` alias points to,
- it loads that version immediately,
- `/v2/health/live` reports whether the process is up,
- `/v2/health/ready` reports `ready` only after that model version has finished loading,
- inference uses the currently loaded model version.

The server keeps the loaded model in memory and only reloads when the `champion` alias points to a different model version.

With filesystem-backed artifacts on shared storage, `mlflow.pytorch.load_model(...)` resolves to a local path again, so there is no longer a need for the extra background thread and lock that were added for slow bucket-backed downloads.