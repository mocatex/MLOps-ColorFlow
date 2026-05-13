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
> export MLFLOW_TRACKING_URI=http://127.0.0.1:5001

# Training & Hyper Parameter Search
one-shot training
> docker compose run --rm training
one-shot HPO     
> docker compose run --rm tuning

Override Hydra config from the CLI:
> docker compose run --rm training python train.py training.epochs=2 data.batch_size=8

Initialize GAN training if `pretrain_best.pt` is already your preferred supervised checkpoint:

```bash
cd services/trainer
uv run python train.py \
  training.checkpoint.resume_pretrain=../../storage/mlops-checkpoints/pretrain_best.pt \
  training.pretrain.enabled=false
```

# Prevent Sleep and enhance Performance on macOS

Apple Silicon laptops may throttle performance or sleep during long runs. Use `caffeinate` to keep training running without interruption.

```bash
cd services/trainer
sudo caffeinate -dimsu nice -n -10 uv run python train.py \
  training.checkpoint.resume_pretrain=../../storage/mlops-checkpoints/pretrain_best.pt \
  training.pretrain.enabled=false
```

`caffeinate -dimsu` is a macOS utility to prevent the system from sleeping while the command is running. If you are on another OS, just run the command without it.

```
-i: idle sleep. prevent the system from sleeping due to inactivity
-d: display sleep. prevent the display from sleeping
-m: disk sleep. prevent the system from sleeping due to disk inactivity
-s: system sleep. prevent the system from sleeping due to user inactivity
-u: declares user activity. This option is recommended when used with -d to prevent the display from sleeping due to user inactivity.
```

```bash
# on macOS, check the current power mode to ensure that the system won't sleep during training
# 0 = normal, 1 = low power, 2 = high performance
pmset -g | grep powermode
# if it is 0, you can switch to high performance mode for the duration of training
# this will increase CPU and GPU performance at the cost of higher power consumption 
# and potentially more fan noise, but it can significantly reduce training time
sudo pmset -a powermode 2
# after training, switch back to normal mode
sudo pmset -a powermode 0
```

