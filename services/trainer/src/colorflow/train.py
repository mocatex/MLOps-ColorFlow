from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from torch import nn
from tqdm import tqdm

from colorflow.checkpointing import Checkpointer
from colorflow.tracking import NoopTracker, Tracker
from colorflow.utils import (
    AverageMeter,
    create_loss_meters,
    lab_to_rgb,
    update_losses,
)


@torch.no_grad()
def evaluate_generator_l1(generator, val_loader, criterion, device) -> float:
    generator.eval()
    meter = AverageMeter()
    for data in val_loader:
        L = data["L"].to(device)
        ab = data["ab"].to(device)
        preds = generator(L)
        loss = criterion(preds, ab)
        meter.update(loss.item(), L.size(0))
    generator.train()
    return meter.avg


@torch.no_grad()
def evaluate_main_model(model, val_loader) -> dict[str, float]:
    """Run a non-training forward pass and compute the same losses train uses."""
    model.generator.eval()
    model.discriminator.eval()
    meters = create_loss_meters()
    for data in val_loader:
        model.prepare_input(data)
        model.forward()
        gen_image = torch.cat([model.L, model.gen_output], dim=1)
        gen_preds = model.discriminator(gen_image)
        model.disc_loss_gen = model.GANloss(gen_preds, False)
        real_image = torch.cat([model.L, model.ab], dim=1)
        real_preds = model.discriminator(real_image)
        model.disc_loss_real = model.GANloss(real_preds, True)
        model.disc_loss = (model.disc_loss_gen + model.disc_loss_real) * 0.5
        model.loss_G_GAN = model.GANloss(gen_preds, True)
        model.loss_G_L1 = model.L1loss(model.gen_output, model.ab) * model.lambda_l1
        model.loss_G = model.loss_G_GAN + model.loss_G_L1
        update_losses(model, meters, count=data["L"].size(0))
    model.generator.train()
    model.discriminator.train()
    return {k: m.avg for k, m in meters.items()}


@torch.no_grad()
def save_sample_grid(model, val_loader, output_path: str | Path, n: int = 5) -> Path:
    """Save a 3-row grid: input L, generated RGB, real RGB."""
    data = next(iter(val_loader))
    model.generator.eval()
    model.prepare_input(data)
    model.forward()
    fake = lab_to_rgb(model.L, model.gen_output.detach())
    real = lab_to_rgb(model.L, model.ab)
    n = min(n, model.L.size(0))
    fig = plt.figure(figsize=(3 * n, 8))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1); ax.imshow(model.L[i][0].cpu(), cmap="gray"); ax.axis("off")
        ax = plt.subplot(3, n, i + 1 + n); ax.imshow(fake[i]); ax.axis("off")
        ax = plt.subplot(3, n, i + 1 + 2 * n); ax.imshow(real[i]); ax.axis("off")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    model.generator.train()
    return output_path


def pretrain_generator(
    generator,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    epochs: int,
    device,
    checkpointer: Checkpointer | None = None,
    tracker: Tracker | None = None,
    config_snapshot: dict[str, Any] | None = None,
    start_epoch: int = 0,
) -> dict[str, float]:
    """Supervised L1 pretraining of the generator. Returns final metric summary."""
    tracker = tracker or NoopTracker()
    best_val = float("inf")

    for epoch in range(start_epoch, epochs):
        loss_meter = AverageMeter()
        for data in tqdm(train_loader, desc=f"pretrain {epoch + 1}/{epochs}"):
            L, ab = data["L"].to(device), data["ab"].to(device)
            preds = generator(L)
            loss = criterion(preds, ab)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), L.size(0))

        val_l1 = evaluate_generator_l1(generator, val_loader, criterion, device)
        metrics = {"pretrain_train_l1": loss_meter.avg, "pretrain_val_l1": val_l1}
        tracker.log_metrics(metrics, step=epoch)
        best_val = min(best_val, val_l1)
        print(f"[pretrain] epoch {epoch + 1}/{epochs} train_l1={loss_meter.avg:.5f} val_l1={val_l1:.5f}")

        if checkpointer is not None:
            state = {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "optim_state_dict": optimizer.state_dict(),
                "metrics": metrics,
                "best_metric": best_val,
                "config": config_snapshot,
                "stage": "pretrain",
            }
            saved = checkpointer.save(state, epoch=epoch + 1, metrics=metrics)
            if "best" in saved:
                tracker.log_artifact(saved["best"], artifact_path="checkpoints")

    return {"best_pretrain_val_l1": best_val, "final_pretrain_train_l1": loss_meter.avg}


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int,
    checkpointer: Checkpointer | None = None,
    tracker: Tracker | None = None,
    config_snapshot: dict[str, Any] | None = None,
    sample_dir: str | Path | None = None,
    start_epoch: int = 0,
) -> dict[str, float]:
    """GAN training loop. Returns final metric summary; logs per-epoch sample grids."""
    tracker = tracker or NoopTracker()
    best_val = float("inf")
    train_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, epochs):
        loss_meter_dict = create_loss_meters()
        for data in tqdm(train_loader, desc=f"gan {epoch + 1}/{epochs}"):
            model.prepare_input(data)
            model.optimize()
            update_losses(model, loss_meter_dict, count=data["L"].size(0))

        train_metrics = {f"train_{k}": m.avg for k, m in loss_meter_dict.items()}
        val_metrics = {f"val_{k}": v for k, v in evaluate_main_model(model, val_loader).items()}
        metrics = {**train_metrics, **val_metrics}
        tracker.log_metrics(metrics, step=epoch)

        val_l1 = val_metrics.get("val_loss_G_L1", float("inf"))
        best_val = min(best_val, val_l1)
        print(
            f"[gan] epoch {epoch + 1}/{epochs} "
            f"train_G={train_metrics['train_loss_G']:.4f} "
            f"val_G_L1={val_l1:.4f}"
        )

        if sample_dir is not None:
            sample_path = save_sample_grid(model, val_loader, Path(sample_dir) / f"epoch_{epoch + 1:03d}.png")
            tracker.log_artifact(sample_path, artifact_path="samples")

        if checkpointer is not None:
            state = {
                "epoch": epoch,
                "generator_state_dict": model.generator.state_dict(),
                "discriminator_state_dict": model.discriminator.state_dict(),
                "gen_optim_state_dict": model.gen_optim.state_dict(),
                "disc_optim_state_dict": model.disc_optim.state_dict(),
                "metrics": metrics,
                "best_metric": best_val,
                "config": config_snapshot,
                "stage": "gan",
            }
            saved = checkpointer.save(state, epoch=epoch + 1, metrics=metrics)
            if "best" in saved:
                tracker.log_artifact(saved["best"], artifact_path="checkpoints")

    return {"best_val_loss_G_L1": best_val, **train_metrics}


def restore_pretrain(generator, optimizer, checkpoint_path, device) -> int:
    """Load a pretrain checkpoint into ``generator`` + ``optimizer``. Returns next epoch."""
    state = Checkpointer.load(checkpoint_path, map_location=device)
    generator.load_state_dict(state["generator_state_dict"])
    optimizer.load_state_dict(state["optim_state_dict"])
    return state["epoch"] + 1


def restore_gan(model, checkpoint_path, device) -> int:
    """Load a GAN checkpoint into ``model``. Returns next epoch."""
    state = Checkpointer.load(checkpoint_path, map_location=device)
    model.generator.load_state_dict(state["generator_state_dict"])
    model.discriminator.load_state_dict(state["discriminator_state_dict"])
    model.gen_optim.load_state_dict(state["gen_optim_state_dict"])
    model.disc_optim.load_state_dict(state["disc_optim_state_dict"])
    return state["epoch"] + 1
