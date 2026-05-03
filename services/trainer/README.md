# MLOps-ColorFlow
Our semester project for the module Machine Learning Operations at ZHAW

```bash
# create python environment and install dependencies
uv sync
# activate the python environment
source .venv/bin/activate
```

# Local Setup

**See [DVC](./DVC.MD) for Training set File download**

Materialize the dataset manually before building or running the trainer
> uv run dvc pull

Build the trainer image (~5 min on first build)
> docker compose build training
*(if building fails because of oauth)*
> docker logout ghcr.io

Set the external MLflow endpoint for local runs
> export MLFLOW_TRACKING_URI=http://localhost:5001

# Training & Hyper Parameter Search
one-shot training
> docker compose run --rm training
one-shot HPO     
> docker compose run --rm tuning

Override Hydra config from the CLI:
> docker compose run --rm training python train.py training.epochs=2 data.batch_size=8
