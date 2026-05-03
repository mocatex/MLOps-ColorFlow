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

**Run once locally for GCloud project setup**
gcloud config set project mlops-colorflow

Build the dep layer (~5 min)
> docker compose build base
*(if building fails because of oauth)*
> docker logout ghcr.io

Build & Start MLFlow with postgres tracking stack
> docker compose up -d postgres mlflow

MLflow UI
> open http://localhost:5001

# Training & Hyper Parameter Search
one-shot training
> docker compose run --rm training
one-shot HPO     
> docker compose run --rm tuning

Override Hydra config from the CLI:
> docker compose run --rm training python train.py training.epochs=2 data.batch_size=8
