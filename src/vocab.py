"""Vocabulary dan tokenisasi caption Flickr8k."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence

PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNK_TOKEN = "<unk>"

PAD_IDX = 0
START_IDX = 1
END_IDX = 2
UNK_IDX = 3

SPECIAL_TOKENS = (PAD_TOKEN, START_TOKEN, END_TOKEN, UNK_TOKEN)

# Huruf/angka dipertahankan; apostrof hanya boleh berada di dalam kata.
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Tokenisasi deterministik tanpa unduhan resource NLP."""
    normalized = text.lower().replace("’", "'")
    return _TOKEN_PATTERN.findall(normalized)


class Vocabulary:
    """Pemetaan dua arah token dan ID untuk caption."""

    def __init__(self, min_freq: int = 5):
        if min_freq < 1:
            raise ValueError("min_freq minimal 1")

        self.min_freq = min_freq
        self.token_to_id: dict[str, int] = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)

    def build(self, captions: Iterable[str]) -> None:
        """Bangun ulang vocabulary dari kumpulan caption training."""
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(tokenize(caption))

        # Urutan alfabetis membuat ID stabil pada setiap eksekusi.
        words = sorted(word for word, count in counts.items() if count >= self.min_freq)
        self.id_to_token = [*SPECIAL_TOKENS, *words]
        self.token_to_id = {
            token: index for index, token in enumerate(self.id_to_token)
        }

    def encode(self, caption: str, max_len: int = 30) -> list[int]:
        """Ubah caption menjadi tepat ``max_len`` ID token."""
        if max_len < 2:
            raise ValueError("max_len minimal 2 untuk <start> dan <end>")

        content = tokenize(caption)[: max_len - 2]
        ids = [START_IDX]
        ids.extend(self.token_to_id.get(token, UNK_IDX) for token in content)
        ids.append(END_IDX)
        ids.extend([PAD_IDX] * (max_len - len(ids)))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """Ubah ID menjadi teks; berhenti di <end> dan abaikan token struktur."""
        words: list[str] = []
        for index in ids:
            index = int(index)
            if index == END_IDX:
                break
            if index in (PAD_IDX, START_IDX):
                continue
            if 0 <= index < len(self.id_to_token):
                words.append(self.id_to_token[index])
            else:
                words.append(UNK_TOKEN)
        return " ".join(words)

    def __len__(self) -> int:
        return len(self.id_to_token)
