from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    """Loads images, resizes/augments them, converts to standardized LAB space."""

    def __init__(self, paths, image_size=(256, 256), train=True):
        if train:
            self.transforms = transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                ]
            )
        else:
            self.transforms = transforms.Compose([transforms.Resize(image_size)])

        self.train = train
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        lab = transforms.ToTensor()(lab)
        l_channel = lab[[0], ...] / 50 - 1
        ab = lab[[1, 2], ...] / 128
        return {"L": l_channel, "ab": ab}


def split_paths(paths, external_data_size=None, train_size=None, seed=42):
    """Sample ``external_data_size`` paths and split into train/val.

    Both sizes are optional: missing ``external_data_size`` uses every path,
    missing ``train_size`` falls back to an 80/20 split. Useful for tiny
    DVC-tracked sample datasets where the COCO defaults wouldn't fit.
    """
    np.random.seed(seed)
    n_total = len(paths)
    if n_total == 0:
        raise ValueError("No image paths found — did you `dvc pull`?")
    n_use = min(external_data_size or n_total, n_total)
    paths_subset = np.random.choice(paths, n_use, replace=False)
    random_idxs = np.random.permutation(n_use)
    n_train = train_size if train_size is not None else max(1, int(0.8 * n_use))
    n_train = min(n_train, max(1, n_use - 1))  # always leave ≥1 val sample
    train_idxs = random_idxs[:n_train]
    val_idxs = random_idxs[n_train:]
    return paths_subset[train_idxs], paths_subset[val_idxs]


def fetch_coco_sample_paths():
    from fastai.data.external import URLs, untar_data

    path = untar_data(URLs.COCO_SAMPLE)
    path = str(path) + "/train_sample"
    return glob.glob(path + "/*.jpg")


def fetch_local_directory_paths(directory: str | Path, glob_pattern: str = "*.jpg") -> list[str]:
    """Glob ``directory`` for images. Intended for DVC-tracked local datasets.

    Raises a clear error if the directory is missing — most often this means
    the user hasn't run ``dvc pull`` to materialise the data yet.
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(
            f"Image directory not found: {directory!s}. "
            f"For DVC-tracked data, run `dvc pull` from the repo root first."
        )
    return sorted(str(p) for p in directory.glob(glob_pattern))


def build_dataloaders(cfg, seed=42):
    if cfg.source == "fastai_coco_sample":
        paths = fetch_coco_sample_paths()
    elif cfg.source == "local_directory":
        paths = fetch_local_directory_paths(cfg.path, cfg.get("glob_pattern", "*.jpg"))
    else:
        raise ValueError(f"Unknown data source: {cfg.source}")

    train_paths, val_paths = split_paths(
        paths,
        external_data_size=cfg.get("external_data_size"),
        train_size=cfg.get("train_size"),
        seed=seed,
    )
    image_size = (cfg.image_size_1, cfg.image_size_2)

    train_data = ImageDataset(paths=train_paths, image_size=image_size, train=True)
    valid_data = ImageDataset(paths=val_paths, image_size=image_size, train=False)

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        pin_memory=cfg.pin_memory,
        num_workers=cfg.num_workers,
    )
    return train_loader, valid_loader
