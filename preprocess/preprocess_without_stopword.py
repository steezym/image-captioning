
import os
import re
import pickle
from collections import Counter
from pathlib import Path
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent

CAPTIONS_FILE = BASE_DIR / "preprocess" / "data" / "captions.txt"
IMAGE_DIR = BASE_DIR / "preprocess"  / "data" / "Images"
OUTPUT_DIR = BASE_DIR / "preprocess" / "preprocessed"

IMAGE_SIZE = (128, 128)
MAX_LENGTH = 30
MIN_WORD_FREQUENCY = 5
BATCH_SIZE = 32
RANDOM_STATE = 42

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

PAD_ID = 0
START_ID = 1
END_ID = 2
UNK_ID = 3

SPECIAL_TOKENS = {
    PAD_TOKEN: PAD_ID,
    START_TOKEN: START_ID,
    END_TOKEN: END_ID,
    UNK_TOKEN: UNK_ID,
}

# Convert images to float32 tensors normalized to [-1, 1].
image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5)
    ),
])


def clean_caption(caption):
    caption = str(caption).lower()
    caption = re.sub(r"[^a-z\s]", " ", caption)
    words = caption.split()

    # Remove common English stopwords.
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were",
        "am", "be", "been", "being", "and", "or", "but",
        "of", "to", "in", "on", "at", "for", "with",
        "by", "from", "as", "this", "that"
    }

    words = [word for word in words if word not in stopwords]
    return words


def split_by_image(captions_df):
    """Split unique image IDs before assigning captions to each split."""
    image_ids = sorted(captions_df["image"].unique())

    if len(image_ids) <= 2000:
        raise ValueError(
            "The dataset must contain more than 2000 unique images."
        )

    train_val_ids, test_ids = train_test_split(
        image_ids,
        test_size=1000,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=1000,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = captions_df[captions_df["image"].isin(train_ids)].copy()
    val_df = captions_df[captions_df["image"].isin(val_ids)].copy()
    test_df = captions_df[captions_df["image"].isin(test_ids)].copy()

    return train_df, val_df, test_df


def build_vocabulary(train_df):
    """Build vocabulary from training captions only."""
    word_counts = Counter()

    for caption in train_df["caption"]:
        word_counts.update(clean_caption(caption))

    vocab = dict(SPECIAL_TOKENS)

    for word, count in sorted(word_counts.items()):
        if count >= MIN_WORD_FREQUENCY and word not in vocab:
            vocab[word] = len(vocab)

    return vocab


def encode_caption(caption, vocab):
    """Encode a caption into exactly MAX_LENGTH token IDs."""
    words = clean_caption(caption)

    token_ids = [START_ID]

    for word in words:
        token_ids.append(vocab.get(word, UNK_ID))

    token_ids.append(END_ID)

    # Preserve <end> when the caption exceeds the maximum length.
    if len(token_ids) > MAX_LENGTH:
        token_ids = token_ids[:MAX_LENGTH - 1] + [END_ID]

    # Pad shorter captions.
    token_ids += [PAD_ID] * (MAX_LENGTH - len(token_ids))

    return token_ids


class CaptionDataset(Dataset):
    def __init__(self, captions_df, image_dir, vocab):
        self.data = captions_df.reset_index(drop=True)
        self.image_dir = image_dir
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        image_path = os.path.join(
            self.image_dir,
            row["image"]
        )

        # Load RGB image and transform to (3, 128, 128).
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = image_transform(image)

        # Encode caption as int64 with shape (30,).
        caption_ids = encode_caption(row["caption"], self.vocab)
        caption = torch.tensor(caption_ids, dtype=torch.int64)

        return image, caption


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load captions.
    captions_df = pd.read_csv(CAPTIONS_FILE)

    required_columns = {"image", "caption"}
    if not required_columns.issubset(captions_df.columns):
        raise ValueError(
            "captions.txt must contain 'image' and 'caption' columns."
        )

    captions_df = captions_df.dropna(
        subset=["image", "caption"]
    ).copy()

    captions_df["image"] = captions_df["image"].astype(str)
    captions_df["caption"] = captions_df["caption"].astype(str)
    captions_df["caption"] = captions_df["caption"].apply(
    lambda caption: " ".join(clean_caption(caption))
)

    # Exclude rows whose image files are missing.
    captions_df = captions_df[
        captions_df["image"].map(
            lambda name: os.path.isfile(os.path.join(IMAGE_DIR, name))
        )
    ].reset_index(drop=True)

    # Split by unique image ID to prevent data leakage.
    train_df, val_df, test_df = split_by_image(captions_df)

    # Build vocabulary from training captions only.
    vocab = build_vocabulary(train_df)

    # Encode captions for all splits using the same vocabulary.
    for split_df in (train_df, val_df, test_df):
        split_df["caption_ids"] = split_df["caption"].apply(
            lambda caption: encode_caption(caption, vocab)
        )

    # Save preprocessing artifacts.
    with open(os.path.join(OUTPUT_DIR, "vocab.pkl"), "wb") as file:
        pickle.dump(vocab, file)

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    with open(os.path.join(OUTPUT_DIR, "splits.pkl"), "wb") as file:
        pickle.dump(splits, file)

    # Create datasets for model training.
    train_dataset = CaptionDataset(train_df, IMAGE_DIR, vocab)
    val_dataset = CaptionDataset(val_df, IMAGE_DIR, vocab)
    test_dataset = CaptionDataset(test_df, IMAGE_DIR, vocab)

    # Training batches can be passed to EncoderCNN and DecoderLSTM.
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print(f"Training images:   {train_df['image'].nunique()}")
    print(f"Validation images: {val_df['image'].nunique()}")
    print(f"Test images:       {test_df['image'].nunique()}")
    print(f"Vocabulary size:   {len(vocab)}")
    print(f"Batch size:        {BATCH_SIZE}")

    # Verify the output shapes.
    images, captions = next(iter(train_loader))

    print("Image batch shape:", images.shape)
    print("Image batch dtype:", images.dtype)
    print("Image range:", images.min().item(), images.max().item())
    print("Caption batch shape:", captions.shape)
    print("Caption batch dtype:", captions.dtype)

    print("Special token IDs:")
    for token, token_id in SPECIAL_TOKENS.items():
        print(f"  {token}: {vocab[token]}")

    print("Preprocessing completed.")

    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    train_loader, val_loader, test_loader, vocab = main()
