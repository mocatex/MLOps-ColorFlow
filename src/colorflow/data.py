import glob

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
        L = lab[[0], ...] / 50 - 1
        ab = lab[[1, 2], ...] / 128
        return {"L": L, "ab": ab}


def split_paths(paths, external_data_size, train_size, seed=42):
    np.random.seed(seed)
    paths_subset = np.random.choice(paths, external_data_size, replace=False)
    random_idxs = np.random.permutation(external_data_size)
    train_idxs = random_idxs[:train_size]
    val_idxs = random_idxs[train_size:]
    return paths_subset[train_idxs], paths_subset[val_idxs]


def fetch_coco_sample_paths():
    from fastai.data.external import URLs, untar_data

    path = untar_data(URLs.COCO_SAMPLE)
    path = str(path) + "/train_sample"
    return glob.glob(path + "/*.jpg")


def build_dataloaders(cfg, seed=42):
    if cfg.source == "fastai_coco_sample":
        paths = fetch_coco_sample_paths()
    else:
        raise ValueError(f"Unknown data source: {cfg.source}")

    train_paths, val_paths = split_paths(
        paths, cfg.external_data_size, cfg.train_size, seed=seed
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
