"""Smoke test Flickr8k penuh; dilewati bila dataset belum diekstrak."""

from pathlib import Path

import torch

from src.dataset import create_data_pipeline
from src.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX, tokenize


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def test_full_flickr8k_pipeline():
    images_dir = DATA_ROOT / "Images"
    captions_file = DATA_ROOT / "captions.txt"
    assert images_dir.is_dir(), f"Folder gambar tidak ditemukan: {images_dir}"
    assert captions_file.is_file(), f"Caption tidak ditemukan: {captions_file}"

    pipeline = create_data_pipeline(
        images_dir,
        captions_file,
        batch_size=8,
        num_workers=0,
    )

    assert len(pipeline.captions_by_image) == 8091
    assert len(list(images_dir.glob("*.jpg"))) == 8091
    assert len(pipeline.splits.train) == 6091
    assert len(pipeline.splits.val) == 1000
    assert len(pipeline.splits.test) == 1000
    assert len(pipeline.train_dataset) == 30455
    assert len(pipeline.val_dataset) == 5000
    assert len(pipeline.test_dataset) == 5000
    assert set(pipeline.splits.train).isdisjoint(pipeline.splits.val)
    assert set(pipeline.splits.train).isdisjoint(pipeline.splits.test)
    assert set(pipeline.splits.val).isdisjoint(pipeline.splits.test)
    assert pipeline.vocab.token_to_id["<pad>"] == PAD_IDX == 0
    assert pipeline.vocab.token_to_id["<start>"] == START_IDX == 1
    assert pipeline.vocab.token_to_id["<end>"] == END_IDX == 2
    assert pipeline.vocab.token_to_id["<unk>"] == UNK_IDX == 3

    train_tokens = {
        token
        for name in pipeline.splits.train
        for caption in pipeline.captions_by_image[name]
        for token in tokenize(caption)
    }
    held_out_tokens = {
        token
        for name in (*pipeline.splits.val, *pipeline.splits.test)
        for caption in pipeline.captions_by_image[name]
        for token in tokenize(caption)
    }
    held_out_only = held_out_tokens - train_tokens
    assert held_out_only.isdisjoint(pipeline.vocab.token_to_id)

    images, captions = next(iter(pipeline.train_loader))
    assert images.shape == (8, 3, 128, 128)
    assert images.dtype == torch.float32
    assert torch.isfinite(images).all()
    assert -1.0 <= images.min() <= images.max() <= 1.0
    assert captions.shape == (8, 30)
    assert captions.dtype == torch.int64
