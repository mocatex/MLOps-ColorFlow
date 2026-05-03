from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim

from colorflow.checkpointing import Checkpointer
from colorflow.data import build_dataloaders
from colorflow.models import MainModel, build_backbone_unet
from colorflow.tracking import build_tracker
from colorflow.train import (
    pretrain_generator,
    restore_gan,
    restore_pretrain,
    train_model,
)
from colorflow.utils import resolve_device


def run(cfg: DictConfig) -> float:
    """Execute the training pipeline for an already-composed config.

    Split out from :func:`main` so other entry points (HPO sweeps, Dagster ops)
    can drive training without re-entering Hydra.
    """
    device = resolve_device(cfg.device)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot = OmegaConf.to_container(cfg, resolve=True)
    train_loader, val_loader = build_dataloaders(cfg.data, seed=cfg.seed)

    with build_tracker(cfg.tracking) as tracker:
        tracker.set_tags({"stage": "train", "model": "pix2pix"})
        tracker.log_params(config_snapshot)

        generator = _maybe_pretrain(
            cfg, train_loader, val_loader, device, tracker, config_snapshot
        )

        gan_ckpt = Checkpointer(
            output_dir=cfg.training.checkpoint.dir,
            prefix="gan",
            monitor=cfg.training.checkpoint.monitor_gan,
            mode=cfg.training.checkpoint.mode,
            save_every=cfg.training.checkpoint.save_every,
            keep_last=cfg.training.checkpoint.keep_last,
        )

        model = MainModel(
            model_cfg=cfg.model,
            training_cfg=cfg.training,
            device=device,
            generator=generator,
        )

        start_epoch = 0
        if cfg.training.checkpoint.resume_gan:
            start_epoch = restore_gan(model, cfg.training.checkpoint.resume_gan, device)

        result = train_model(
            model,
            train_loader,
            val_loader,
            epochs=cfg.training.epochs,
            checkpointer=gan_ckpt,
            tracker=tracker,
            config_snapshot=config_snapshot,
            sample_dir=output_dir / "samples",
            start_epoch=start_epoch,
        )

        # Emit a run-level selection metric for the model-registry job.
        # The registry currently ranks runs by descending score, so negate the
        # validation loss to keep "higher is better" semantics there.
        best_val_l1 = float(result["best_val_loss_G_L1"])
        tracker.log_metrics(
            {
                "best_val_loss_G_L1": best_val_l1,
                "selection_score": -best_val_l1,
            }
        )

        # Log the best generator as an MLflow PyTorch model so mlserver-mlflow
        # can serve it without a custom runtime. Only meaningful for the MLflow
        # backend; NoopTracker.log_pytorch_model is a no-op.
        best_path = output_dir / "checkpoints" / "gan_best.pt"
        if best_path.exists():
            state = Checkpointer.load(best_path, map_location=device)
            model.generator.load_state_dict(state["generator_state_dict"])
            tracker.log_pytorch_model(model.generator, artifact_path="generator")

        return best_val_l1


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> float:
    return run(cfg)


def _maybe_pretrain(cfg, train_loader, val_loader, device, tracker, config_snapshot):
    """Build the generator backbone, run L1 pretraining if enabled, return it.

    Returns ``None`` when the model config opts out of a pretrained backbone —
    MainModel will then construct its own randomly-initialized U-Net.
    """
    if not cfg.model.generator.use_pretrained_backbone:
        return None

    generator = build_backbone_unet(
        device=device,
        input_channels=cfg.model.generator.input_channels,
        output_channels=cfg.model.generator.output_channels,
        size=cfg.data.image_size_1,
        layers_to_cut=cfg.model.generator.layers_to_cut,
    )
    optimizer = optim.Adam(
        generator.parameters(),
        lr=cfg.training.pretrain.lr,
        weight_decay=cfg.training.pretrain.weight_decay,
    )

    start_epoch = 0
    if cfg.training.checkpoint.resume_pretrain:
        start_epoch = restore_pretrain(
            generator, optimizer, cfg.training.checkpoint.resume_pretrain, device
        )

    if not cfg.training.pretrain.enabled:
        return generator

    pretrain_ckpt = Checkpointer(
        output_dir=cfg.training.checkpoint.dir,
        prefix="pretrain",
        monitor=cfg.training.checkpoint.monitor_pretrain,
        mode=cfg.training.checkpoint.mode,
        save_every=cfg.training.checkpoint.save_every,
        keep_last=cfg.training.checkpoint.keep_last,
    )
    pretrain_generator(
        generator,
        train_loader,
        val_loader,
        optimizer,
        nn.L1Loss(),
        epochs=cfg.training.pretrain.epochs,
        device=device,
        checkpointer=pretrain_ckpt,
        tracker=tracker,
        config_snapshot=config_snapshot,
        start_epoch=start_epoch,
    )
    return generator


if __name__ == "__main__":
    main()
