"""Uji vocabulary dan encoding caption."""

from src.vocab import (
    END_IDX,
    END_TOKEN,
    PAD_IDX,
    PAD_TOKEN,
    START_IDX,
    START_TOKEN,
    UNK_IDX,
    UNK_TOKEN,
    Vocabulary,
    tokenize,
)


def test_reserved_token_ids():
    vocab = Vocabulary()
    assert vocab.token_to_id == {
        PAD_TOKEN: PAD_IDX,
        START_TOKEN: START_IDX,
        END_TOKEN: END_IDX,
        UNK_TOKEN: UNK_IDX,
    }


def test_training_only_vocabulary_and_frequency_threshold():
    vocab = Vocabulary(min_freq=2)
    vocab.build(["train train shared rare", "shared common common"])

    assert "train" in vocab.token_to_id
    assert "shared" in vocab.token_to_id
    assert "common" in vocab.token_to_id
    assert "rare" not in vocab.token_to_id
    assert vocab.encode("rare")[1] == UNK_IDX
    assert "validation-only" not in vocab.token_to_id
    assert vocab.encode("validation-only")[1:3] == [UNK_IDX, UNK_IDX]


def test_tokenization_preserves_internal_apostrophe():
    assert tokenize("Dog's, DON'T — run!") == ["dog's", "don't", "run"]


def test_encoding_is_deterministic_and_padded():
    vocab = Vocabulary(min_freq=1)
    vocab.build(["A dog runs"])

    first = vocab.encode("A dog runs", max_len=8)
    second = vocab.encode("A dog runs", max_len=8)

    assert first == second
    assert first[0] == START_IDX
    assert first[4] == END_IDX
    assert first[5:] == [PAD_IDX, PAD_IDX, PAD_IDX]
    assert vocab.decode(first) == "a dog runs"


def test_encoding_truncates_content_and_keeps_end():
    caption = "one two three four five six"
    vocab = Vocabulary(min_freq=1)
    vocab.build([caption])

    encoded = vocab.encode(caption, max_len=5)

    assert len(encoded) == 5
    assert encoded[0] == START_IDX
    assert encoded[-1] == END_IDX
    assert vocab.decode(encoded) == "one two three"


def test_word_ids_are_deterministic():
    vocab_a = Vocabulary(min_freq=1)
    vocab_b = Vocabulary(min_freq=1)
    vocab_a.build(["zebra apple mango"])
    vocab_b.build(["mango zebra apple"])
    assert vocab_a.token_to_id == vocab_b.token_to_id
