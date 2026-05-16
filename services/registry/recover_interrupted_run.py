"""One-off recovery for a training run that was killed mid-epoch.

The trainer logs the generator model + selection metrics + run status only
after `train_model` returns cleanly (see services/trainer/train.py).
A KeyboardInterrupt skips that block, so the run lands in MLflow as
status=FAILED with no `generator/` artifact and no `selection_score`.

This script reads the best checkpoint that was already produced during
training, recreates the generator, logs it under artifact_path="generator",
backfills `best_val_loss_G_L1` + `selection_score` from the run's
`val_loss_G_L1` history, sets the `checkpoint_uri` tag, and terminates the
run as FINISHED so register.py can pick it up.
"""

import argparse
import os
import sys
from pathlib import Path

import mlflow
import torch
from mlflow import MlflowClient

# Make the trainer package importable so we can rebuild the generator class
# the checkpoint was saved with.
TRAINER_SRC = Path(__file__).resolve().parents[1] / "trainer" / "src"
sys.path.insert(0, str(TRAINER_SRC))

from colorflow.models import build_backbone_unet  # noqa: E402


def resolve_tracking_uri() -> str:
    explicit = os.environ.get("MLFLOW_TRACKING_URI")
    if explicit:
        return explicit
    local_store = Path(__file__).resolve().parents[2] / "storage" / "mlops-flow"
    if local_store.exists():
        return local_store.as_uri()
    raise SystemExit("Set MLFLOW_TRACKING_URI or ensure storage/mlops-flow exists.")


def find_checkpoint(client: MlflowClient, run_id: str, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise SystemExit(f"Checkpoint override does not exist: {override}")
        return override

    artifact_uri = client.get_run(run_id).info.artifact_uri
    if artifact_uri.startswith("file://"):
        candidate = Path(artifact_uri[len("file://") :]) / "checkpoints" / "gan_best.pt"
        if candidate.exists():
            return candidate

    fallback = Path(__file__).resolve().parents[2] / "storage" / "mlops-checkpoints" / "gan_best.pt"
    if fallback.exists():
        return fallback

    raise SystemExit(
        "Could not locate gan_best.pt. Pass --checkpoint <path> explicitly."
    )


def best_val_l1_from_history(client: MlflowClient, run_id: str) -> float:
    history = client.get_metric_history(run_id, "val_loss_G_L1")
    if not history:
        raise SystemExit("Run has no val_loss_G_L1 metric history; cannot recover.")
    return min(m.value for m in history)


def recover(run_id: str, checkpoint_path: Path | None) -> None:
    tracking_uri = resolve_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    run = client.get_run(run_id)
    print(f"Recovering run {run_id} (status={run.info.status}) at {tracking_uri}")

    ckpt_path = find_checkpoint(client, run_id, checkpoint_path)
    print(f"Loading checkpoint: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    generator_state = state["generator_state_dict"]

    image_size = int(run.data.params.get("data.image_size_1", 256))
    layers_to_cut = int(run.data.params.get("model.generator.layers_to_cut", -2))
    input_channels = int(run.data.params.get("model.generator.input_channels", 1))
    output_channels = int(run.data.params.get("model.generator.output_channels", 2))

    generator = build_backbone_unet(
        device=torch.device("cpu"),
        input_channels=input_channels,
        output_channels=output_channels,
        size=image_size,
        layers_to_cut=layers_to_cut,
    )
    generator.load_state_dict(generator_state)
    generator.eval()

    best_val = best_val_l1_from_history(client, run_id)
    print(f"best_val_loss_G_L1 = {best_val}")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(
            {
                "best_val_loss_G_L1": best_val,
                "selection_score": -best_val,
            }
        )
        mlflow.set_tag("checkpoint_uri", ckpt_path.as_uri())
        mlflow.pytorch.log_model(generator, artifact_path="generator")

    client.set_terminated(run_id, status="FINISHED")
    print(f"Run {run_id} marked FINISHED with generator artifact + selection metrics.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id", help="MLflow run_id to recover")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Override path to gan_best.pt (defaults to the run's artifact copy)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    recover(args.run_id, args.checkpoint)
