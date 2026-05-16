Inspect the MLflow tracking server by running the following command in the terminal. 

```bash
# From the project root directory, run:
uv run mlflow ui --backend-store-uri "file://$PWD/storage/mlops-flow" --port 5001
# to start mlflow in the background and write logs to /tmp/mlflow-ui.log
uv run mlflow ui --backend-store-uri "file://$PWD/storage/mlops-flow" --port 5001 > /tmp/mlflow-ui.log 2>&1 &
# get the background PID
echo $!
# Stop the MLServer when you’re done inspecting it:
kill <PID>
# or if you lost the PID:
kill -9 $(lsof -nP -iTCP:5001 -sTCP:LISTEN -t)

# Then open http://127.0.0.1:5001

# You can’t reliably use http://localhost:5001 because it can 
# resolve to ::1 first and hit a different listener on port 5001.

# Stop the MLflow UI when you’re done inspecting it:
kill -9 $(lsof -nP -iTCP:5001 -sTCP:LISTEN -t)
```