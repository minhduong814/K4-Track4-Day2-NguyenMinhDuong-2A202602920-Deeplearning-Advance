"""DeepWeeds data loading, split validation, transforms, and DataLoaders."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

NUM_CLASSES = 9
EXPECTED_IMAGES = 17_509
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read the author's unmodified train/val/test CSV files for one fold."""
    if fold not in range(5):
        raise ValueError(f"fold must be in 0..4, got {fold}")
    labels_dir = Path(labels_dir)
    frames = []
    # Các split chính thức hiện chỉ có Filename và Label; Species chỉ có trong
    # labels.csv ở một số phiên bản của repo DeepWeeds nên không được bắt buộc.
    required = {"Filename", "Label"}
    for split in ("train", "val", "test"):
        path = labels_dir / f"{split}_subset{fold}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DeepWeeds split: {path}")
        df = pd.read_csv(path)
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if df["Filename"].isna().any() or df["Label"].isna().any():
            raise ValueError(f"{path} contains empty Filename/Label values")
        df["Label"] = df["Label"].astype(int)
        if not df["Label"].between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{path}: Label must be in 0..{NUM_CLASSES - 1}")
        frames.append(df)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Fail fast on the mandatory split checks and return report-ready counts."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {images_dir}")

    frames = {"train": train_df, "val": val_df, "test": test_df}
    names: dict[str, set[str]] = {}
    per_class: dict[str, dict[int, int]] = {}
    for split, df in frames.items():
        required = {"Filename", "Label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{split} is missing columns: {sorted(missing)}")
        if df["Filename"].duplicated().any():
            duplicates = df.loc[df["Filename"].duplicated(), "Filename"].head().tolist()
            raise ValueError(f"{split} contains duplicate filenames, e.g. {duplicates}")
        names[split] = set(df["Filename"].astype(str))
        counts = df["Label"].astype(int).value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        per_class[split] = {int(k): int(v) for k, v in counts.items()}

    overlap = {
        "train_val": len(names["train"] & names["val"]),
        "train_test": len(names["train"] & names["test"]),
        "val_test": len(names["val"] & names["test"]),
    }
    if any(overlap.values()):
        raise ValueError(f"The official splits overlap: {overlap}")
    union = names["train"] | names["val"] | names["test"]
    if len(union) != EXPECTED_IMAGES:
        raise ValueError(f"Expected {EXPECTED_IMAGES:,} unique images, found {len(union):,}")

    missing_files = [name for name in sorted(union) if not (images_dir / name).is_file()]
    if missing_files:
        sample = missing_files[:10]
        raise FileNotFoundError(
            f"{len(missing_files)} CSV images are missing below {images_dir}; examples: {sample}"
        )

    report = {
        "n": {split: int(len(df)) for split, df in frames.items()},
        "per_class": per_class,
        "overlap": overlap,
        "union": len(union),
        "missing_files": 0,
    }
    print("Split sizes:", report["n"])
    for split in frames:
        readable = {CLASS_NAMES[k]: v for k, v in per_class[split].items()}
        print(f"{split} per class:", readable)
    print("Pairwise overlap:", overlap, "| union:", len(union), "| missing files: 0")
    return report


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Build deterministic evaluation transforms or a named training augmentation."""
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    aug = aug.lower()
    normalize = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        resize_size = int(round(img_size / 0.875))
        return transforms.Compose([
            transforms.Resize(resize_size, antialias=True),
            transforms.CenterCrop(img_size),
            *normalize,
        ])

    prefix: list = [transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), antialias=True),
                    transforms.RandomHorizontalFlip()]
    if aug == "basic":
        pass
    elif aug == "color":
        prefix.append(transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                             saturation=0.3, hue=0.08))
    elif aug == "trivial":
        prefix.append(transforms.TrivialAugmentWide())
    elif aug == "randaug":
        prefix.append(transforms.RandAugment(num_ops=2, magnitude=9))
    elif aug in {"none", "eval"}:
        prefix = [transforms.Resize(int(round(img_size / 0.875)), antialias=True),
                  transforms.CenterCrop(img_size)]
    else:
        raise ValueError(f"Unknown augmentation '{aug}'; use basic/color/trivial/randaug/none")
    return transforms.Compose([*prefix, *normalize])


class DeepWeedsDataset(Dataset):
    """Map a split DataFrame to ``(image_tensor, label, filename)`` samples."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        required = {"Filename", "Label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing columns: {sorted(missing)}")
        self.df = df.reset_index(drop=True).copy()
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
        except Exception as exc:
            raise RuntimeError(f"Could not read image {path}") from exc
        return image, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2):
    """Create an order-stable evaluation loader or a shuffled/balanced train loader."""
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    sampler_obj = None
    if sampler is not None:
        if not train:
            raise ValueError("A sampler is only valid for the training loader")
        if sampler != "balanced":
            raise ValueError("sampler must be None or 'balanced'")
        labels = df["Label"].astype(int).to_numpy()
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        if (counts == 0).any():
            raise ValueError("Balanced sampling requires every class in the training split")
        sample_weights = torch.as_tensor(1.0 / counts[labels], dtype=torch.double)
        sampler_obj = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    generator = torch.Generator()
    generator.manual_seed(torch.initial_seed())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train and sampler_obj is None),
        sampler=sampler_obj,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )
