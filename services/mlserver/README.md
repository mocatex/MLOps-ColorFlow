# Start MLServer

```bash
MLFLOW_TRACKING_URI=http://localhost:5001 uv run uvicorn app:app --host 0.0.0.0 --port 8080
```

This service exposes the inference API at `http://localhost:8080/v2/...`. The local UI can run separately on `http://localhost:8081` and will call MLServer there automatically.