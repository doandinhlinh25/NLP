"""Train and persist the Chapter 17 Word2Vec experiment for Lab 03."""

from __future__ import annotations

import json
import re
from itertools import islice
from pathlib import Path

from gensim.models import Word2Vec


LAB_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = LAB_ROOT.parent / "data" / "c4-train.00000-of-01024-30K.json" / "c4-train.00000-of-01024-30K.json"
MODEL_PATH = Path(__file__).resolve().parent / "word2vec.model"
REPORT_PATH = Path(__file__).resolve().parent / "training_config.md"
DOCUMENT_KEY = "text"
DOCUMENT_LIMIT = 2_000
TOKEN_PATTERN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z']*\b")
VECTOR_SIZE = 100
CONTEXT_WINDOW = 5
MINIMUM_COUNT = 2
EPOCHS = 5
WORKER_COUNT = 1
RANDOM_SEED = 2026
WORD2VEC_TRAINING_MODE = 0
TRAINING_OBJECTIVE = "CBOW"


def is_valid_training_configuration(
    vector_size: int, window: int, minimum_count: int, epochs: int, worker_count: int
) -> bool:
    """Check whether all positive Word2Vec hyperparameters are valid.

    Args:
        vector_size: Number of dimensions in each learned embedding.
        window: Maximum distance of context words from each target word.
        minimum_count: Smallest corpus frequency retained by the model.
        epochs: Number of passes through the training corpus.
        worker_count: Number of worker threads used for training.

    Returns:
        ``True`` when every numeric hyperparameter is a positive integer.
    """
    parameters = (vector_size, window, minimum_count, epochs, worker_count)
    return all(isinstance(parameter, int) and parameter > 0 for parameter in parameters)


def is_valid_training_mode(training_mode: int) -> bool:
    """Check whether a Gensim Word2Vec objective mode is supported.

    Args:
        training_mode: Gensim objective selector, zero for CBOW or one for Skip-gram.

    Returns:
        ``True`` for the two supported Gensim Word2Vec objective values.
    """
    return training_mode in (0, 1)


def tokenize_text(text: str) -> list[str]:
    """Convert one raw corpus document to lowercase alphabetic word tokens.

    Args:
        text: Raw text read from a corpus record.

    Returns:
        Lowercase tokens preserving internal apostrophes.
    """
    return TOKEN_PATTERN.findall(text.lower())


def load_tokenized_documents(dataset_path: Path, document_limit: int) -> list[list[str]]:
    """Load the configured number of non-empty C4 documents as token lists.

    Args:
        dataset_path: Path to the course-provided JSONL corpus.
        document_limit: Maximum number of non-empty documents used for training.

    Returns:
        Tokenized documents in their original corpus order.

    Raises:
        FileNotFoundError: If the configured corpus does not exist.
        ValueError: If no tokenized documents are available.
    """
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Corpus not found: {dataset_path}")
    with dataset_path.open(encoding="utf-8") as dataset_file:
        tokenized_documents = (
            tokens
            for record in (json.loads(line) for line in dataset_file)
            if isinstance(record.get(DOCUMENT_KEY), str)
            for tokens in [tokenize_text(record[DOCUMENT_KEY])]
            if tokens
        )
        selected_documents = list(islice(tokenized_documents, document_limit))
    if not selected_documents:
        raise ValueError("The corpus does not contain tokenized documents.")
    return selected_documents


def train_word2vec(
    tokenized_documents: list[list[str]],
    vector_size: int = VECTOR_SIZE,
    window: int = CONTEXT_WINDOW,
    minimum_count: int = MINIMUM_COUNT,
    epochs: int = EPOCHS,
    worker_count: int = WORKER_COUNT,
    training_mode: int = WORD2VEC_TRAINING_MODE,
    random_seed: int = RANDOM_SEED,
) -> Word2Vec:
    """Train a reproducible Word2Vec model with explicit hyperparameters.

    Args:
        tokenized_documents: Ordered token sequences used as training sentences.
        vector_size: Number of dimensions in each learned word vector.
        window: Maximum target-to-context distance.
        minimum_count: Smallest corpus frequency retained by the vocabulary.
        epochs: Number of complete passes through the input corpus.
        worker_count: Number of worker threads used by Gensim.
        training_mode: Gensim mode, where zero selects CBOW and one Skip-gram.
        random_seed: Fixed random seed for repeatable initialization.

    Returns:
        A fitted Gensim Word2Vec model.

    Raises:
        ValueError: If the fixed training configuration or input corpus is invalid.
    """
    if not is_valid_training_configuration(
        vector_size,
        window,
        minimum_count,
        epochs,
        worker_count,
    ) or not is_valid_training_mode(training_mode):
        raise ValueError("Word2Vec training configuration is invalid.")
    if not tokenized_documents:
        raise ValueError("At least one tokenized document is required for training.")
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=minimum_count,
        workers=worker_count,
        sg=training_mode,
        seed=random_seed,
    )
    model.build_vocab(tokenized_documents)
    model.train(tokenized_documents, total_examples=model.corpus_count, epochs=EPOCHS)
    return model


def write_training_report(model: Word2Vec, document_count: int, output_path: Path) -> None:
    """Write the required Word2Vec configuration and training outcome.

    Args:
        model: Fitted Word2Vec model whose retained vocabulary is reported.
        document_count: Number of documents supplied to model training.
        output_path: Destination Markdown report.
    """
    report = f"""# Chapter 17 — Experiment 2: Word2Vec

## Cấu hình huấn luyện

| Thuộc tính | Giá trị |
| --- | --- |
| Corpus | C4 JSONL, {document_count:,} văn bản không rỗng đầu tiên |
| Tokenization | Lower-case, English alphabetic tokens có thể chứa apostrophe |
| Vocabulary sau `min_count` | {len(model.wv):,} từ |
| Training objective | {TRAINING_OBJECTIVE} (`sg={WORD2VEC_TRAINING_MODE}`) |
| `vector_size` | {VECTOR_SIZE} |
| `window` | {CONTEXT_WINDOW} |
| `min_count` | {MINIMUM_COUNT} |
| `epochs` | {EPOCHS} |
| `workers` | {WORKER_COUNT} |
| `seed` | {RANDOM_SEED} |

`workers = 1` và seed cố định được dùng để hạn chế khác biệt giữa các lần chạy. Model đã huấn luyện được lưu tại `word2vec.model`; Chapter 18 nạp chính file này để kiểm tra embedding của `doctor`.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Train Word2Vec and write the Chapter 17 deliverables."""
    tokenized_documents = load_tokenized_documents(DATASET_PATH, DOCUMENT_LIMIT)
    model = train_word2vec(tokenized_documents)
    model.save(str(MODEL_PATH))
    write_training_report(model, len(tokenized_documents), REPORT_PATH)
    print(f"Trained {TRAINING_OBJECTIVE} Word2Vec on {len(tokenized_documents):,} documents.")
    print(f"Vocabulary size: {len(model.wv):,}. Saved {MODEL_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
