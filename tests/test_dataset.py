"""Uji unit data pipeline memakai dataset sintetis."""

from pathlib import Path

import torch
from PIL import Image

from src.dataset import (
    Flickr8kDataset,
    build_eval_transform,
    create_data_pipeline,
    parse_captions,
    split_image_names,
)
from src.vocab import Vocabulary


def _write_synthetic_data(root: Path, image_count: int = 6) -> tuple[Path, Path]:
    images_dir = root / "Images"
    images_dir.mkdir()
    captions_file = root / "captions.txt"

    lines = ["image,caption"]
    for index in range(image_count):
        name = f"image_{index}.jpg"
        color = (index * 20, index * 10, index * 5)
        Image.new("RGB", (40 + index, 30 + index), color).save(images_dir / name)
        lines.append(f"{name},A common object number {index}.")
        lines.append(f"{name},Caption with a comma, preserved for image {index}.")

    captions_file.write_text("\n".join(lines), encoding="utf-8")
    return images_dir, captions_file


def test_parse_caption_preserves_commas(tmp_path):
    _, captions_file = _write_synthetic_data(tmp_path, image_count=1)
    grouped = parse_captions(captions_file)
    assert grouped["image_0.jpg"][1] == "Caption with a comma, preserved for image 0."


def test_split_is_reproducible_and_has_no_overlap():
    names = [f"image_{index}.jpg" for index in range(10)]
    first = split_image_names(names, seed=42, val_size=2, test_size=3)
    second = split_image_names(reversed(names), seed=42, val_size=2, test_size=3)

    assert first == second
    assert len(first.train) == 5
    assert len(first.val) == 2
    assert len(first.test) == 3
    assert set(first.train).isdisjoint(first.val)
    assert set(first.train).isdisjoint(first.test)
    assert set(first.val).isdisjoint(first.test)


def test_dataset_item_contract(tmp_path):
    images_dir, captions_file = _write_synthetic_data(tmp_path, image_count=1)
    grouped = parse_captions(captions_file)
    vocab = Vocabulary(min_freq=1)
    vocab.build(grouped["image_0.jpg"])
    dataset = Flickr8kDataset(
        images_dir,
        grouped,
        ["image_0.jpg"],
        vocab,
        transform=build_eval_transform(),
    )

    image, caption = dataset[0]
    assert image.shape == (3, 128, 128)
    assert image.dtype == torch.float32
    assert torch.isfinite(image).all()
    assert -1.0 <= image.min() <= image.max() <= 1.0
    assert caption.shape == (30,)
    assert caption.dtype == torch.int64


def test_pipeline_builds_train_only_vocab_and_batch_shapes(tmp_path):
    images_dir, captions_file = _write_synthetic_data(tmp_path, image_count=6)
    pipeline = create_data_pipeline(
        images_dir,
        captions_file,
        batch_size=3,
        min_freq=1,
        val_size=1,
        test_size=1,
    )

    # Token nomor milik gambar validation/test tidak boleh memperluas vocabulary.
    train_captions = " ".join(
        caption
        for name in pipeline.splits.train
        for caption in pipeline.captions_by_image[name]
    ).lower()
    held_out_numbers = {
        name.removeprefix("image_").removesuffix(".jpg")
        for name in (*pipeline.splits.val, *pipeline.splits.test)
    }
    for number in held_out_numbers:
        if number not in train_captions.split():
            assert number not in pipeline.vocab.token_to_id

    images, captions = next(iter(pipeline.train_loader))
    assert images.shape == (3, 3, 128, 128)
    assert images.dtype == torch.float32
    assert captions.shape == (3, 30)
    assert captions.dtype == torch.int64
