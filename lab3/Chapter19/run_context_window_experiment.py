"""Compare Word2Vec similarity scores across context-window configurations."""

from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab3.Chapter17.run_word2vec_experiment import (  # noqa: E402
    CONTEXT_WINDOW,
    DATASET_PATH,
    DOCUMENT_LIMIT,
    EPOCHS,
    MINIMUM_COUNT,
    RANDOM_SEED,
    TRAINING_OBJECTIVE,
    VECTOR_SIZE,
    WORD2VEC_TRAINING_MODE,
    WORKER_COUNT,
    load_tokenized_documents,
    train_word2vec,
)


RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "experiment_results.md"
CONTEXT_WINDOWS = (2, CONTEXT_WINDOW, 10)
PAIR_LABELS = (("doctor", "physician"), ("doctor", "hospital"), ("cat", "dog"))
HYPERPARAMETER_EPOCHS = 3
CSV_FIELDS = ("window", "word_pair", "cosine_similarity", "vocabulary_size")
REPORT_DECIMAL_PLACES = 6
OUT_OF_VOCABULARY = "not_in_vocabulary"


def is_valid_context_windows(windows: tuple[int, ...]) -> bool:
    """Check whether a context-window comparison contains positive unique values.

    Args:
        windows: Context-window sizes to evaluate.

    Returns:
        ``True`` when all windows are positive integers and no value repeats.
    """
    return bool(windows) and len(windows) == len(set(windows)) and all(
        isinstance(window, int) and window > 0 for window in windows
    )


def calculate_similarity(model, first_word: str, second_word: str) -> float | None:
    """Return a Word2Vec cosine score when both requested words are retained.

    Args:
        model: Fitted Gensim Word2Vec model.
        first_word: First token in the requested pair.
        second_word: Second token in the requested pair.

    Returns:
        Cosine similarity, or ``None`` if either word is out of vocabulary.
    """
    if first_word not in model.wv or second_word not in model.wv:
        return None
    return float(model.wv.similarity(first_word, second_word))


def run_context_window_experiment(
    tokenized_documents: list[list[str]],
) -> tuple[dict[int, dict[str, float | None]], int]:
    """Train one Word2Vec model per window and collect selected similarities.

    Args:
        tokenized_documents: Shared ordered training corpus for all configurations.

    Returns:
        Similarities by context window and the shared retained vocabulary size.
    """
    models = {
        window: train_word2vec(
            tokenized_documents,
            vector_size=VECTOR_SIZE,
            window=window,
            minimum_count=MINIMUM_COUNT,
            epochs=HYPERPARAMETER_EPOCHS,
            worker_count=WORKER_COUNT,
            training_mode=WORD2VEC_TRAINING_MODE,
            random_seed=RANDOM_SEED,
        )
        for window in CONTEXT_WINDOWS
    }
    similarities = {
        window: {
            f"{first_word}–{second_word}": calculate_similarity(model, first_word, second_word)
            for first_word, second_word in PAIR_LABELS
        }
        for window, model in models.items()
    }
    vocabulary_size = len(next(iter(models.values())).wv)
    return similarities, vocabulary_size


def create_rows(
    similarities: dict[int, dict[str, float | None]], vocabulary_size: int
) -> list[dict[str, int | str | float]]:
    """Flatten per-window similarities into reproducible CSV result rows.

    Args:
        similarities: Similarities produced by ``run_context_window_experiment``.
        vocabulary_size: Retained Word2Vec vocabulary size shared by the models.

    Returns:
        One row per context window and selected word pair.
    """
    return [
        {
            "window": window,
            "word_pair": pair_label,
            "cosine_similarity": similarity if similarity is not None else OUT_OF_VOCABULARY,
            "vocabulary_size": vocabulary_size,
        }
        for window, pair_scores in similarities.items()
        for pair_label, similarity in pair_scores.items()
    ]


def write_results(rows: list[dict[str, int | str | float]], output_path: Path) -> None:
    """Write context-window comparison rows as CSV.

    Args:
        rows: Flat rows returned by ``create_rows``.
        output_path: Destination CSV path.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_score(score: float | None) -> str:
    """Format an optional similarity score for Markdown output.

    Args:
        score: Computed cosine score or ``None`` for an unavailable pair.

    Returns:
        Fixed-precision similarity or an out-of-vocabulary marker.
    """
    return f"{score:.{REPORT_DECIMAL_PLACES}f}" if score is not None else OUT_OF_VOCABULARY


def write_report(similarities: dict[int, dict[str, float | None]], output_path: Path) -> None:
    """Write a Markdown comparison of the context-window experiment.

    Args:
        similarities: Per-window word-pair similarities.
        output_path: Destination Markdown report.
    """
    headers = " | ".join(f"window = {window}" for window in CONTEXT_WINDOWS)
    alignment = " | ".join("---:" for _ in CONTEXT_WINDOWS)
    pair_rows = "\n".join(
        f"| {pair_label} | "
        + " | ".join(format_score(similarities[window][pair_label]) for window in CONTEXT_WINDOWS)
        + " |"
        for pair_label in similarities[CONTEXT_WINDOWS[0]]
    )
    report = f"""# Chapter 19 — Experiment 3: Context window

## Thiết lập

Ba model {TRAINING_OBJECTIVE} được huấn luyện trên cùng {DOCUMENT_LIMIT:,} văn bản C4, cùng `vector_size={VECTOR_SIZE}`, `min_count={MINIMUM_COUNT}`, `epochs={HYPERPARAMETER_EPOCHS}`, `workers={WORKER_COUNT}` và `seed={RANDOM_SEED}`. Chỉ `window` thay đổi; không lưu các model hyperparameter vì chúng có thể tái tạo bằng script.

## Similarity theo context window

| Word pair | {headers} |
| --- | {alignment} |
{pair_rows}

Window nhỏ nhấn mạnh context gần, trong khi window lớn đưa thêm thông tin chủ đề ở xa hơn. Vì vậy, sự thay đổi score trong bảng là kết quả quan sát được của corpus này, không phải bằng chứng rằng window lớn luôn tốt hơn. Cần so sánh từng cặp từ và đối chiếu với frequency/context trước khi kết luận.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run the Chapter 19 context-window experiment and save its reports."""
    if not is_valid_context_windows(CONTEXT_WINDOWS):
        raise ValueError("Context-window configuration is invalid.")
    tokenized_documents = load_tokenized_documents(DATASET_PATH, DOCUMENT_LIMIT)
    similarities, vocabulary_size = run_context_window_experiment(tokenized_documents)
    rows = create_rows(similarities, vocabulary_size)
    write_results(rows, RESULTS_PATH)
    write_report(similarities, REPORT_PATH)
    print(f"Compared context windows {CONTEXT_WINDOWS} on {len(tokenized_documents):,} documents.")
    print(f"Wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
