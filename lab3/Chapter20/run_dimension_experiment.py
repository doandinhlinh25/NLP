"""Compare Word2Vec capacity, cost, and simple retrieval across dimensions."""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab3.Chapter17.run_word2vec_experiment import (  # noqa: E402
    CONTEXT_WINDOW,
    DATASET_PATH,
    DOCUMENT_LIMIT,
    MINIMUM_COUNT,
    RANDOM_SEED,
    TRAINING_OBJECTIVE,
    WORD2VEC_TRAINING_MODE,
    WORKER_COUNT,
    load_tokenized_documents,
    train_word2vec,
)


RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "experiment_results.md"
VECTOR_DIMENSIONS = (50, 100, 300)
HYPERPARAMETER_EPOCHS = 3
SIMILARITY_PAIR = ("doctor", "physician")
ANALOGY_POSITIVE_WORDS = ("king", "woman")
ANALOGY_NEGATIVE_WORDS = ("man",)
RETRIEVAL_QUERY = ("medical", "treatment")
RETRIEVAL_DOCUMENTS = {
    "medical": ("doctor", "patient", "hospital", "treatment"),
    "technology": ("computer", "software", "program", "network"),
    "food": ("banana", "fruit", "recipe", "food"),
}
EXPECTED_RETRIEVAL_LABEL = "medical"
BYTES_PER_MEGABYTE = 1_024 * 1_024
CSV_FIELDS = (
    "dimension",
    "training_seconds",
    "vocabulary_size",
    "vector_memory_mb",
    "doctor_physician_similarity",
    "analogy_top1",
    "semantic_retrieval_top1",
    "semantic_retrieval_correct",
)
OUT_OF_VOCABULARY = "not_in_vocabulary"
REPORT_DECIMAL_PLACES = 6


def is_valid_dimensions(dimensions: tuple[int, ...]) -> bool:
    """Check whether requested embedding dimensions are positive and distinct.

    Args:
        dimensions: Vector sizes to compare.

    Returns:
        ``True`` when all dimensions are unique positive integers.
    """
    return bool(dimensions) and len(dimensions) == len(set(dimensions)) and all(
        isinstance(dimension, int) and dimension > 0 for dimension in dimensions
    )


def average_known_vectors(model, tokens: tuple[str, ...]) -> np.ndarray | None:
    """Average the vectors available for a token sequence.

    Args:
        model: Fitted Gensim Word2Vec model.
        tokens: Query or document tokens to represent.

    Returns:
        Mean embedding vector, or ``None`` when no token is in vocabulary.
    """
    known_vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(known_vectors, axis=0) if known_vectors else None


def dense_cosine_similarity(first_vector: np.ndarray, second_vector: np.ndarray) -> float:
    """Calculate cosine similarity between two dense NumPy vectors.

    Args:
        first_vector: First dense numeric vector.
        second_vector: Second dense numeric vector.

    Returns:
        Cosine score, or zero if either input has zero magnitude.
    """
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    return float(np.dot(first_vector, second_vector) / denominator) if denominator else 0.0


def retrieve_semantic_document(model) -> str:
    """Retrieve the best toy document using averaged word embeddings.

    Args:
        model: Fitted Gensim Word2Vec model.

    Returns:
        Label of the highest-scoring toy document, or an OOV marker.
    """
    query_vector = average_known_vectors(model, RETRIEVAL_QUERY)
    if query_vector is None:
        return OUT_OF_VOCABULARY
    document_scores = {
        label: dense_cosine_similarity(query_vector, document_vector)
        for label, tokens in RETRIEVAL_DOCUMENTS.items()
        for document_vector in [average_known_vectors(model, tokens)]
        if document_vector is not None
    }
    return max(document_scores, key=document_scores.get) if document_scores else OUT_OF_VOCABULARY


def evaluate_analogy(model) -> str:
    """Run the configured vector-arithmetic analogy when all words are known.

    Args:
        model: Fitted Gensim Word2Vec model.

    Returns:
        Top predicted analogy word or an OOV marker.
    """
    required_words = ANALOGY_POSITIVE_WORDS + ANALOGY_NEGATIVE_WORDS
    if any(word not in model.wv for word in required_words):
        return OUT_OF_VOCABULARY
    result = model.wv.most_similar(
        positive=list(ANALOGY_POSITIVE_WORDS),
        negative=list(ANALOGY_NEGATIVE_WORDS),
        topn=1,
    )
    return result[0][0]


def run_dimension_experiment(tokenized_documents: list[list[str]]) -> list[dict[str, int | float | str | bool]]:
    """Train, time, and evaluate one model for every requested dimension.

    Args:
        tokenized_documents: Shared corpus for all dimension configurations.

    Returns:
        One comparable measurement row for every vector dimension.
    """
    results: list[dict[str, int | float | str | bool]] = []
    for dimension in VECTOR_DIMENSIONS:
        training_start = perf_counter()
        model = train_word2vec(
            tokenized_documents,
            vector_size=dimension,
            window=CONTEXT_WINDOW,
            minimum_count=MINIMUM_COUNT,
            epochs=HYPERPARAMETER_EPOCHS,
            worker_count=WORKER_COUNT,
            training_mode=WORD2VEC_TRAINING_MODE,
            random_seed=RANDOM_SEED,
        )
        training_seconds = perf_counter() - training_start
        first_word, second_word = SIMILARITY_PAIR
        similarity = (
            float(model.wv.similarity(first_word, second_word))
            if first_word in model.wv and second_word in model.wv
            else OUT_OF_VOCABULARY
        )
        retrieval_label = retrieve_semantic_document(model)
        results.append(
            {
                "dimension": dimension,
                "training_seconds": training_seconds,
                "vocabulary_size": len(model.wv),
                "vector_memory_mb": model.wv.vectors.nbytes / BYTES_PER_MEGABYTE,
                "doctor_physician_similarity": similarity,
                "analogy_top1": evaluate_analogy(model),
                "semantic_retrieval_top1": retrieval_label,
                "semantic_retrieval_correct": retrieval_label == EXPECTED_RETRIEVAL_LABEL,
            }
        )
    return results


def write_results(rows: list[dict[str, int | float | str | bool]], output_path: Path) -> None:
    """Write all dimension measurements as a CSV artifact.

    Args:
        rows: Dimension experiment rows.
        output_path: Destination CSV file.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_measurement(value: float | str) -> str:
    """Format numeric or unavailable experiment output for Markdown.

    Args:
        value: Numeric measurement or explicit out-of-vocabulary marker.

    Returns:
        Fixed-precision value or the unchanged marker.
    """
    return f"{value:.{REPORT_DECIMAL_PLACES}f}" if isinstance(value, float) else value


def write_report(rows: list[dict[str, int | float | str | bool]], output_path: Path) -> None:
    """Write the Chapter 20 capacity-versus-cost comparison report.

    Args:
        rows: Dimension experiment rows.
        output_path: Destination Markdown report.
    """
    table_rows = "\n".join(
        f"| {row['dimension']} | {float(row['training_seconds']):.{REPORT_DECIMAL_PLACES}f} | "
        f"{float(row['vector_memory_mb']):.{REPORT_DECIMAL_PLACES}f} | "
        f"{format_measurement(row['doctor_physician_similarity'])} | {row['analogy_top1']} | "
        f"{row['semantic_retrieval_top1']} | {row['semantic_retrieval_correct']} |"
        for row in rows
    )
    report = f"""# Chapter 20 — Experiment 4: Embedding dimension

## Thiết lập

Các model {TRAINING_OBJECTIVE} cùng dùng {DOCUMENT_LIMIT:,} văn bản C4, `window={CONTEXT_WINDOW}`, `min_count={MINIMUM_COUNT}`, `epochs={HYPERPARAMETER_EPOCHS}`, `workers={WORKER_COUNT}` và `seed={RANDOM_SEED}`. Chỉ `vector_size` thay đổi.

Semantic retrieval là downstream task nhỏ: query `{' '.join(RETRIEVAL_QUERY)}` được biểu diễn bằng trung bình word vectors và xếp hạng ba pseudo-documents (`medical`, `technology`, `food`). Kết quả đúng khi document `medical` đứng đầu; đây chỉ là sanity check, không phải benchmark.

## Kết quả

| Dimension | Training seconds | Vector memory (MB) | doctor–physician | Analogy top-1 | Retrieval top-1 | Retrieval correct |
| ---: | ---: | ---: | ---: | --- | --- | --- |
{table_rows}

`Vector memory` là số byte thực của ma trận embedding (`model.wv.vectors.nbytes`), chưa tính toàn bộ overhead của object. Cột analogy chạy `king - man + woman`; output là pattern trong vector space, không phải bằng chứng model hiểu quan hệ như con người.

## Nhận xét

Dimension lớn tăng capacity và bộ nhớ gần tuyến tính, đồng thời thường tăng thời gian train. Tuy nhiên, quality không chắc chắn tăng theo dimension khi corpus nhỏ: cần xem similarity, analogy và retrieval trong cùng bảng thay vì chỉ chọn vector lớn nhất.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Run Chapter 20 and write dimension-comparison artifacts."""
    if not is_valid_dimensions(VECTOR_DIMENSIONS):
        raise ValueError("Embedding dimension configuration is invalid.")
    tokenized_documents = load_tokenized_documents(DATASET_PATH, DOCUMENT_LIMIT)
    rows = run_dimension_experiment(tokenized_documents)
    write_results(rows, RESULTS_PATH)
    write_report(rows, REPORT_PATH)
    print(f"Compared dimensions {VECTOR_DIMENSIONS} on {len(tokenized_documents):,} documents.")
    print(f"Wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
