"""Jalankan dan validasi data pipeline Flickr8k secara lokal."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import torch

from src.dataset import create_data_pipeline
from src.vocab import END_IDX, PAD_IDX, START_IDX, UNK_IDX


REPO_ROOT = Path(__file__).resolve().parent
VENV_DIR = REPO_ROOT / ".venv"
ARCHIVE_PATH = REPO_ROOT / "archive.zip"
DATA_DIR = REPO_ROOT / "data"
BATCH_SIZE = 8


def extract_if_needed(archive_path: Path, data_dir: Path) -> None:
    """Ekstrak dataset dengan pemeriksaan path bila file belum tersedia."""
    images_dir = data_dir / "Images"
    captions_file = data_dir / "captions.txt"
    if images_dir.is_dir() and captions_file.is_file():
        print(f"Dataset sudah tersedia: {data_dir}")
        return

    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive tidak ditemukan: {archive_path}")

    data_dir.mkdir(parents=True, exist_ok=True)
    target_root = data_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            destination = (data_dir / member.filename).resolve()
            if destination != target_root and target_root not in destination.parents:
                raise ValueError(f"Path ZIP tidak aman: {member.filename}")
        archive.extractall(data_dir)
    print(f"Ekstraksi selesai: {data_dir}")


def require_project_venv() -> None:
    """Tolak Python global agar versi dependency tidak tercampur."""
    expected = VENV_DIR.resolve()
    active = Path(sys.prefix).resolve()
    if active != expected:
        command = VENV_DIR / "Scripts" / "python.exe"
        raise SystemExit(
            "Interpreter yang aktif bukan .venv proyek. Jalankan:\n"
            f'  & "{command}" "{Path(__file__).resolve()}"'
        )


def main() -> None:
    require_project_venv()
    extract_if_needed(ARCHIVE_PATH, DATA_DIR)

    pipeline = create_data_pipeline(
        DATA_DIR / "Images",
        DATA_DIR / "captions.txt",
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    train = set(pipeline.splits.train)
    val = set(pipeline.splits.val)
    test = set(pipeline.splits.test)
    images, captions = next(iter(pipeline.train_loader))

    assert len(pipeline.captions_by_image) == 8091
    assert (len(train), len(val), len(test)) == (6091, 1000, 1000)
    assert train.isdisjoint(val) and train.isdisjoint(test) and val.isdisjoint(test)
    assert [PAD_IDX, START_IDX, END_IDX, UNK_IDX] == [0, 1, 2, 3]
    assert images.shape[1:] == (3, 128, 128)
    assert images.dtype == torch.float32 and torch.isfinite(images).all()
    assert images.min() >= -1 and images.max() <= 1
    assert captions.shape[1:] == (30,)
    assert captions.dtype == torch.int64

    print(f"Repo         : {REPO_ROOT}")
    print(f"Python       : {Path(sys.executable).resolve()}")
    print(f"Archive      : {ARCHIVE_PATH}")
    print(f"Data         : {DATA_DIR}")
    print(f"Gambar unik  : {len(pipeline.captions_by_image):,}")
    print(
        "Split gambar : "
        f"train={len(train):,}, val={len(val):,}, test={len(test):,}"
    )
    print(f"Vocabulary   : {len(pipeline.vocab):,} token")
    print(f"Image batch  : {tuple(images.shape)}, {images.dtype}")
    print(f"Image range  : [{images.min().item():.6f}, {images.max().item():.6f}]")
    print(f"Caption batch: {tuple(captions.shape)}, {captions.dtype}")
    print("LULUS: data pipeline lokal memenuhi kontrak README.")


if __name__ == "__main__":
    main()
