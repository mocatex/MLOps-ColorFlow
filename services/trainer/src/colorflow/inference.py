from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import PIL
import torch
from omegaconf import DictConfig, OmegaConf
from skimage.color import rgb2lab
from torchvision import transforms

from colorflow.checkpointing import Checkpointer
from colorflow.models import build_backbone_unet
from colorflow.utils import lab_to_rgb


def load_l_channel(path, image_size: int = 256) -> torch.Tensor:
    """Load an RGB image, resize, convert to LAB, return the standardized L channel."""
    img = PIL.Image.open(path).convert("RGB")
    img = img.resize((image_size, image_size))
    arr = np.array(img)
    lab = rgb2lab(arr).astype("float32")
    lab = transforms.ToTensor()(lab)
    return lab[[0], ...] / 50 - 1


def load_generator_from_checkpoint(checkpoint_path, device) -> torch.nn.Module:
    """Rebuild the generator architecture from the config embedded in the checkpoint.

    Falls back to the inference-time Hydra config if the checkpoint predates the
    self-describing format. Accepts both pretrain-stage and GAN-stage checkpoints,
    and tolerates a bare-state-dict legacy format.
    """
    state = Checkpointer.load(checkpoint_path, map_location=device)
    embedded_cfg = state.get("config") if isinstance(state, dict) else None
    if embedded_cfg is None:
        raise ValueError(
            f"Checkpoint {checkpoint_path!s} has no embedded config. "
            "Use load_generator(cfg, device) and pass an inference config instead."
        )
    cfg = OmegaConf.create(embedded_cfg)
    generator = _build_generator(cfg, device)
    generator.load_state_dict(_extract_generator_state(state))
    generator.eval()
    return generator


def load_generator(cfg: DictConfig, device) -> torch.nn.Module:
    """Build the architecture from ``cfg`` and load weights from cfg.checkpoint_path.

    Use this when the checkpoint has no embedded config. For checkpoints saved by
    the current training pipeline, prefer :func:`load_generator_from_checkpoint`.
    """
    generator = _build_generator(cfg, device)
    state = Checkpointer.load(cfg.checkpoint_path, map_location=device)
    generator.load_state_dict(_extract_generator_state(state))
    generator.eval()
    return generator


def _build_generator(cfg: DictConfig, device) -> torch.nn.Module:
    return build_backbone_unet(
        device=device,
        input_channels=cfg.model.generator.input_channels,
        output_channels=cfg.model.generator.output_channels,
        size=cfg.data.image_size_1,
        layers_to_cut=cfg.model.generator.layers_to_cut,
    )


def _extract_generator_state(state: Any) -> dict[str, torch.Tensor]:
    """Pull the generator weights out of the various checkpoint formats."""
    if isinstance(state, dict) and "generator_state_dict" in state:
        return state["generator_state_dict"]
    if isinstance(state, dict) and any(k.startswith("generator.") for k in state):
        prefix = "generator."
        return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    return state  # bare state dict (legacy)


def colorize(generator, l_channel: torch.Tensor, device) -> np.ndarray:
    """Run the generator on an L tensor and return an HxWx3 RGB array in [0, 1]."""
    with torch.no_grad():
        l_batch = l_channel.unsqueeze(0).to(device)
        ab = generator(l_batch)
        return lab_to_rgb(l_batch, ab)[0]


def save_rgb(rgb_array: np.ndarray, output_path) -> None:
    img = (rgb_array * 255).clip(0, 255).astype(np.uint8)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.fromarray(img).save(output_path)
