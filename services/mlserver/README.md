# Start MLServer

```bash
cd services/mlserver
export MLFLOW_TRACKING_URI=http://127.0.0.1:5001
uv run uvicorn app:app --host 0.0.0.0 --port 8080
# to start in the background and write logs to /tmp/mlserver.log
uv run uvicorn app:app --host 0.0.0.0 --port 8080 > /tmp/mlserver.log 2>&1 &
# get the background PID
echo $!
# Stop the MLServer when you’re done inspecting it:
kill <PID>
# or if you lost the PID:
kill -9 $(lsof -nP -iTCP:8080 -sTCP:LISTEN -t)
```

Run these commands from `services/mlserver`, because `app.py` lives there. If you run `uvicorn app:app` from the repository root, Python cannot import the `app` module and Uvicorn exits with `Could not import module "app"`.

Use `127.0.0.1` here rather than `localhost` because `localhost` can resolve to `::1` first and reach a different listener on port `5001`, while the local file-backed MLflow UI is the IPv4 listener on `127.0.0.1:5001`.

This service exposes the inference API at `http://127.0.0.1:8080/v2/...`. The local UI can run separately on `http://127.0.0.1:8081` and will call MLServer there automatically.

## Model Loading Flow

MLServer keeps the serving flow deliberately narrow:

- on startup, it asks MLflow which model version the `champion` alias points to,
- it loads that version immediately,
- `/v2/health/live` reports whether the process is up,
- `/v2/health/ready` reports `ready` only after that model version has finished loading,
- inference uses the currently loaded model version.

The server keeps the loaded model in memory and only reloads when the `champion` alias points to a different model version.

With filesystem-backed artifacts on shared storage, `mlflow.pytorch.load_model(...)` resolves to a local path again, so there is no longer a need for the extra background thread and lock that were added for slow bucket-backed downloads.