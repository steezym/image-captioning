"""Data pipeline Flickr8k untuk pasangan gambar-caption."""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.vocab import Vocabulary

IMG_SIZE = 128
MAX_LEN = 30
SPLIT_SEED = 42


def parse_captions(captions_file: str | Path) -> dict[str, list[str]]:
    """Baca captions.txt dan pertahankan koma yang berada di dalam caption."""
    path = Path(captions_file)
    grouped: defaultdict[str, list[str]] = defaultdict(list)

    with path.open("r", encoding="utf-8-sig") as file:
        header = file.readline().strip()
        if header != "image,caption":
            raise ValueError(f"Header captions tidak valid: {header!r}")

        for line_number, raw_line in enumerate(file, start=2):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            try:
                image_name, caption = line.split(",", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Baris {line_number} tidak memiliki pemisah koma") from error

            image_name = image_name.strip()
            caption = caption.strip()
            if not image_name or not caption:
                raise ValueError(f"Baris {line_number} memiliki gambar/caption kosong")
            grouped[image_name].append(caption)

    if not grouped:
        raise ValueError("Tidak ada caption yang ditemukan")
    return dict(grouped)


@dataclass(frozen=True)
class ImageSplits:
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]


def split_image_names(
    image_names: Iterable[str],
    *,
    seed: int = SPLIT_SEED,
    val_size: int = 1000,
    test_size: int = 1000,
) -> ImageSplits:
    """Split deterministik berdasarkan nama gambar unik."""
    names = sorted(set(image_names))
    if val_size < 0 or test_size < 0:
        raise ValueError("Ukuran validation/test tidak boleh negatif")
    if len(names) < val_size + test_size:
        raise ValueError("Jumlah gambar tidak cukup untuk ukuran split yang diminta")

    random.Random(seed).shuffle(names)
    val = tuple(names[:val_size])
    test = tuple(names[val_size : val_size + test_size])
    train = tuple(names[val_size + test_size :])
    return ImageSplits(train=train, val=val, test=test)


def build_train_transform() -> transforms.Compose:
    """Transform training konservatif tanpa horizontal flip."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),  # (3, 128, 128), float32 dalam [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Transform validation/test yang sepenuhnya deterministik."""
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.ToTensor(),  # (3, 128, 128), float32 dalam [0, 1]
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ]
    )


class Flickr8kDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset datar: satu item adalah satu pasangan gambar-caption."""

    def __init__(
        self,
        images_dir: str | Path,
        captions_by_image: Mapping[str, Sequence[str]],
        image_names: Sequence[str],
        vocab: Vocabulary,
        *,
        transform=None,
        max_len: int = MAX_LEN,
    ):
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.transform = transform or build_eval_transform()
        self.max_len = max_len
        self.samples = [
            (image_name, caption)
            for image_name in image_names
            for caption in captions_by_image[image_name]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_name, caption = self.samples[index]
        image_path = self.images_dir / image_name
        with Image.open(image_path) as image:
            image_tensor = self.transform(image.convert("RGB"))  # (3, 128, 128)

        caption_tensor = torch.tensor(
            self.vocab.encode(caption, max_len=self.max_len), dtype=torch.int64
        )  # (30,)
        return image_tensor, caption_tensor


@dataclass
class DataPipeline:
    vocab: Vocabulary
    splits: ImageSplits
    captions_by_image: dict[str, list[str]]
    train_dataset: Flickr8kDataset
    val_dataset: Flickr8kDataset
    test_dataset: Flickr8kDataset
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader


def _captions_for_images(
    captions_by_image: Mapping[str, Sequence[str]], image_names: Sequence[str]
) -> Iterable[str]:
    for image_name in image_names:
        yield from captions_by_image[image_name]


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


def create_data_pipeline(
    images_dir: str | Path,
    captions_file: str | Path,
    *,
    batch_size: int = 64,
    min_freq: int = 5,
    max_len: int = MAX_LEN,
    seed: int = SPLIT_SEED,
    val_size: int = 1000,
    test_size: int = 1000,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataPipeline:
    """Bangun split, vocabulary, dataset, dan DataLoader Flickr8k."""
    images_path = Path(images_dir)
    if not images_path.is_dir():
        raise FileNotFoundError(f"Folder gambar tidak ditemukan: {images_path}")

    captions_by_image = parse_captions(captions_file)
    missing = sorted(name for name in captions_by_image if not (images_path / name).is_file())
    if missing:
        preview = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} gambar tidak ditemukan, contoh: {preview}")

    splits = split_image_names(
        captions_by_image,
        seed=seed,
        val_size=val_size,
        test_size=test_size,
    )

    # Vocabulary hanya melihat caption training untuk mencegah data leakage.
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build(_captions_for_images(captions_by_image, splits.train))

    train_dataset = Flickr8kDataset(
        images_path,
        captions_by_image,
        splits.train,
        vocab,
        transform=build_train_transform(),
        max_len=max_len,
    )
    eval_transform = build_eval_transform()
    val_dataset = Flickr8kDataset(
        images_path,
        captions_by_image,
        splits.val,
        vocab,
        transform=eval_transform,
        max_len=max_len,
    )
    test_dataset = Flickr8kDataset(
        images_path,
        captions_by_image,
        splits.test,
        vocab,
        transform=eval_transform,
        max_len=max_len,
    )

    generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": _seed_worker,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    return DataPipeline(
        vocab=vocab,
        splits=splits,
        captions_by_image=captions_by_image,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )
