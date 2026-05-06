Inspect the MLflow tracking server by running the following command in the terminal. 

```bash
# From the project root directory, run:
uv run mlflow ui --backend-store-uri "file://$PWD/storage/mlops-flow" --port 5001
# Then open http://localhost:5001
```