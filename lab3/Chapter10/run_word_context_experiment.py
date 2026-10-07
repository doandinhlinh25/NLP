"""Run the Chapter 10 word-context representation experiment for Lab 03."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


LAB_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = LAB_ROOT.parent / "data" / "c4-train.00000-of-01024-30K.json" / "c4-train.00000-of-01024-30K.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "experiment_results.md"
DOCUMENT_KEY = "text"
DOCUMENT_LIMIT = 5_000
MINIMUM_TOKEN_FREQUENCY = 5
MAXIMUM_VOCABULARY_SIZE = 1_000
CONTEXT_WINDOWS = (1, 2, 5)
TOKEN_PATTERN = re.compile(r"(?u)\b[a-zA-Z][a-zA-Z']*\b")
TARGET_PAIRS = (("doctor", "physician"), ("doctor", "hospital"), ("cat", "dog"))
TARGET_WORDS = tuple(dict.fromkeys(word for pair in TARGET_PAIRS for word in pair))
CSV_FIELDS = (
    "window",
    "document_count",
    "vocabulary_size",
    "matrix_shape",
    "nonzero_values",
    "density",
    "sparsity",
    "word_pair",
    "cosine_similarity",
)
MATRIX_FORMAT = "csr"
EMPTY_SIMILARITY = "not_in_vocabulary"
REPORT_DECIMAL_PLACES = 6


@dataclass(frozen=True)
class ExperimentResult:
    """Store the measurements produced for one context-window configuration.

    Attributes:
        window: Number of tokens considered to each side of a target word.
        matrix: Sparse target-word by context-word co-occurrence matrix.
        vocabulary: Tokens defining the rows and columns of ``matrix``.
        similarities: Requested word-pair cosine similarities, when available.
    """

    window: int
    matrix: csr_matrix
    vocabulary: tuple[str, ...]
    similarities: dict[str, float | None]


def tokenize_text(text: str) -> list[str]:
    """Normalize one corpus document into lowercase English word tokens.

    Args:
        text: Raw text document obtained from the JSONL corpus.

    Returns:
        Lowercase alphabetic tokens, preserving internal apostrophes.
    """
    return TOKEN_PATTERN.findall(text.lower())


def load_tokenized_documents(dataset_path: Path, document_limit: int) -> list[list[str]]:
    """Load and tokenize a bounded non-empty prefix of the JSONL corpus.

    Args:
        dataset_path: Location of the course-provided C4 JSONL corpus.
        document_limit: Maximum number of non-empty documents to use.

    Returns:
        Tokenized documents in corpus order.

    Raises:
        FileNotFoundError: If the configured corpus file does not exist.
        ValueError: If the corpus yields no usable tokenized documents.
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


def build_vocabulary(
    tokenized_documents: list[list[str]],
    minimum_frequency: int,
    maximum_size: int,
    required_tokens: tuple[str, ...],
) -> tuple[str, ...]:
    """Build a deterministic high-frequency vocabulary for the experiment.

    Args:
        tokenized_documents: Corpus documents already converted to tokens.
        minimum_frequency: Smallest corpus frequency retained in the vocabulary.
        maximum_size: Largest permitted vocabulary size.
        required_tokens: Evaluation tokens retained when they satisfy the frequency threshold.

    Returns:
        Evaluation tokens followed by remaining tokens ranked by decreasing frequency
        and then alphabetically.
    """
    token_counts = Counter(token for document in tokenized_documents for token in document)
    retained_required_tokens = tuple(
        token for token in required_tokens if token_counts[token] >= minimum_frequency
    )
    required_token_set = frozenset(retained_required_tokens)
    remaining_capacity = maximum_size - len(retained_required_tokens)
    eligible_tokens = (
        (token, count)
        for token, count in token_counts.items()
        if count >= minimum_frequency and token not in required_token_set
    )
    ordered_tokens = sorted(eligible_tokens, key=lambda item: (-item[1], item[0]))
    frequent_tokens = tuple(token for token, _ in ordered_tokens[:remaining_capacity])
    return retained_required_tokens + frequent_tokens


def build_cooccurrence_matrix(
    tokenized_documents: list[list[str]], vocabulary: tuple[str, ...], window: int
) -> csr_matrix:
    """Build a sparse target-word by context-word count matrix.

    Each occurrence of a vocabulary token contributes one count for every
    vocabulary context token at most ``window`` positions away. The target
    occurrence itself is excluded from its context.

    Args:
        tokenized_documents: Corpus documents converted to token sequences.
        vocabulary: Ordered tokens defining both matrix dimensions.
        window: Number of neighboring positions considered on each side.

    Returns:
        A square CSR co-occurrence matrix indexed by ``vocabulary``.

    Raises:
        ValueError: If ``window`` is not positive or ``vocabulary`` is empty.
    """
    if window <= 0:
        raise ValueError("Context window must be positive.")
    if not vocabulary:
        raise ValueError("Vocabulary must not be empty.")
    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    cooccurrence_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    for document in tokenized_documents:
        document_length = len(document)
        for target_position, target_token in enumerate(document):
            target_index = vocabulary_index.get(target_token)
            context_start = max(target_position - window, 0)
            context_stop = min(target_position + window + 1, document_length)
            if target_index is not None:
                for context_position in range(context_start, context_stop):
                    context_token = document[context_position]
                    context_index = vocabulary_index.get(context_token)
                    if context_position != target_position and context_index is not None:
                        cooccurrence_counts[(target_index, context_index)] += 1
    row_indices = [coordinate[0] for coordinate in cooccurrence_counts]
    column_indices = [coordinate[1] for coordinate in cooccurrence_counts]
    values = list(cooccurrence_counts.values())
    vocabulary_size = len(vocabulary)
    return csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(vocabulary_size, vocabulary_size),
        dtype=np.int64,
    ).asformat(MATRIX_FORMAT)


def cosine_similarity(matrix: csr_matrix, first_index: int, second_index: int) -> float:
    """Calculate cosine similarity between two rows of a sparse matrix.

    Args:
        matrix: Matrix whose rows are word-context vectors.
        first_index: Row index of the first word vector.
        second_index: Row index of the second word vector.

    Returns:
        Cosine similarity, or zero if either row has zero magnitude.
    """
    first_vector = matrix.getrow(first_index)
    second_vector = matrix.getrow(second_index)
    dot_product = float(first_vector.multiply(second_vector).sum())
    first_norm = float(np.sqrt(first_vector.multiply(first_vector).sum()))
    second_norm = float(np.sqrt(second_vector.multiply(second_vector).sum()))
    denominator = first_norm * second_norm
    return dot_product / denominator if denominator else 0.0


def calculate_pair_similarities(
    matrix: csr_matrix, vocabulary: tuple[str, ...], word_pairs: tuple[tuple[str, str], ...]
) -> dict[str, float | None]:
    """Calculate requested cosine similarities when both words are retained.

    Args:
        matrix: Co-occurrence matrix generated for one context window.
        vocabulary: Tokens defining the rows of ``matrix``.
        word_pairs: Ordered word pairs selected for comparison.

    Returns:
        Pair labels mapped to cosine scores, or ``None`` for out-of-vocabulary pairs.
    """
    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    pair_labels = (f"{first_word}–{second_word}" for first_word, second_word in word_pairs)
    return {
        pair_label: (
            cosine_similarity(matrix, vocabulary_index[first_word], vocabulary_index[second_word])
            if first_word in vocabulary_index and second_word in vocabulary_index
            else None
        )
        for pair_label, (first_word, second_word) in zip(pair_labels, word_pairs)
    }


def calculate_density(matrix: csr_matrix) -> float:
    """Return the proportion of non-zero entries in a sparse matrix.

    Args:
        matrix: Matrix whose density is to be measured.

    Returns:
        Non-zero entries divided by the total number of matrix cells.
    """
    total_cells = matrix.shape[0] * matrix.shape[1]
    return matrix.nnz / total_cells if total_cells else 0.0


def run_experiment(
    tokenized_documents: list[list[str]], vocabulary: tuple[str, ...], windows: tuple[int, ...]
) -> list[ExperimentResult]:
    """Run co-occurrence construction and similarity checks for each window.

    Args:
        tokenized_documents: Shared tokenized corpus.
        vocabulary: Shared vocabulary used in every comparison.
        windows: Context-window values to evaluate.

    Returns:
        One immutable experiment result for each requested window.
    """
    matrices = [
        build_cooccurrence_matrix(tokenized_documents, vocabulary, window)
        for window in windows
    ]
    return [
        ExperimentResult(
            window=window,
            matrix=matrix,
            vocabulary=vocabulary,
            similarities=calculate_pair_similarities(matrix, vocabulary, TARGET_PAIRS),
        )
        for window, matrix in zip(windows, matrices)
    ]


def create_result_rows(results: list[ExperimentResult], document_count: int) -> list[dict[str, str | int | float]]:
    """Convert experiment measurements to flat CSV rows.

    Args:
        results: Co-occurrence experiment results for all window sizes.
        document_count: Number of corpus documents represented by each result.

    Returns:
        One row per window and selected word pair.
    """
    return [
        {
            "window": result.window,
            "document_count": document_count,
            "vocabulary_size": len(result.vocabulary),
            "matrix_shape": f"{result.matrix.shape[0]} × {result.matrix.shape[1]}",
            "nonzero_values": result.matrix.nnz,
            "density": calculate_density(result.matrix),
            "sparsity": 1 - calculate_density(result.matrix),
            "word_pair": pair_label,
            "cosine_similarity": similarity if similarity is not None else EMPTY_SIMILARITY,
        }
        for result in results
        for pair_label, similarity in result.similarities.items()
    ]


def write_results(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    """Write the reproducible per-window measurements to a CSV file.

    Args:
        rows: Flat experiment result rows returned by ``create_result_rows``.
        output_path: Destination CSV file.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def format_similarity(value: float | None) -> str:
    """Format an optional similarity measurement for a Markdown table.

    Args:
        value: Cosine similarity or ``None`` for an unavailable pair.

    Returns:
        A fixed-precision score or the out-of-vocabulary label.
    """
    return f"{value:.{REPORT_DECIMAL_PLACES}f}" if value is not None else EMPTY_SIMILARITY


def write_report(results: list[ExperimentResult], document_count: int, output_path: Path) -> None:
    """Write the Chapter 10 experiment report from generated measurements.

    Args:
        results: Co-occurrence experiment results for all evaluated windows.
        document_count: Number of documents used to build every matrix.
        output_path: Destination Markdown report.
    """
    matrix_rows = "\n".join(
        f"| {result.window} | {len(result.vocabulary)} | {result.matrix.shape[0]} × {result.matrix.shape[1]} | "
        f"{result.matrix.nnz:,} | {calculate_density(result.matrix):.{REPORT_DECIMAL_PLACES}f} | "
        f"{1 - calculate_density(result.matrix):.{REPORT_DECIMAL_PLACES}f} |"
        for result in results
    )
    similarity_rows = "\n".join(
        f"| {pair_label} | "
        + " | ".join(format_similarity(result.similarities[pair_label]) for result in results)
        + " |"
        for pair_label in results[0].similarities
    )
    window_headers = " | ".join(f"window = {result.window}" for result in results)
    window_alignment = " | ".join("---:" for _ in results)
    report = f"""# Chapter 10 — Experiment 1: Word-context representation

## Thiết lập

Thí nghiệm dùng {document_count:,} văn bản không rỗng đầu tiên của corpus C4. Văn bản được lower-case và tách bằng mẫu token tiếng Anh. Vocabulary dùng chung giữa các cấu hình, giữ tối đa {MAXIMUM_VOCABULARY_SIZE:,} token xuất hiện ít nhất {MINIMUM_TOKEN_FREQUENCY} lần. Các từ dùng để đánh giá (`{', '.join(TARGET_WORDS)}`) được giữ lại khi đạt ngưỡng tần suất, để có thể so sánh cùng một tập word pairs ở mọi window. Mỗi ma trận có hàng là target word, cột là context word và ô `(i, j)` là số lần context word `j` xuất hiện trong cửa sổ quanh target word `i`.

## Kích thước và độ thưa của ma trận

| Context window | Vocabulary size | Matrix shape | Non-zero values | Density | Sparsity |
| ---: | ---: | --- | ---: | ---: | ---: |
{matrix_rows}

Khi context window tăng, mỗi target word có nhiều context hơn. Vì vậy số ô khác 0 và density được dự đoán tăng, trong khi matrix shape giữ nguyên vì vocabulary được cố định. Dù vậy, ma trận vẫn là sparse representation: phần lớn tổ hợp word-context không cùng xuất hiện trong corpus.

## Cosine similarity của các cặp từ chọn trước

| Word pair | {window_headers} |
| --- | {window_alignment} |
{similarity_rows}

`not_in_vocabulary` nghĩa là ít nhất một từ không thuộc vocabulary đã giới hạn, nên không thể tính similarity cho cặp đó. Các số liệu chi tiết, gồm cả kết quả tái lập được, nằm trong `results.csv`.

## Nhận xét cần kiểm tra

Các giá trị trên chỉ mô tả vector count-based của sample corpus. Similarity cao phản ánh các context quan sát được, không tự động chứng minh hai từ là đồng nghĩa. Khi đối chiếu với prediction ở Chapter 9, cần kiểm tra tần suất từ và các context gốc trước khi kết luận về sự khác biệt giữa các window.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Execute Chapter 10 and write its CSV and Markdown deliverables."""
    tokenized_documents = load_tokenized_documents(DATASET_PATH, DOCUMENT_LIMIT)
    vocabulary = build_vocabulary(
        tokenized_documents,
        MINIMUM_TOKEN_FREQUENCY,
        MAXIMUM_VOCABULARY_SIZE,
        TARGET_WORDS,
    )
    results = run_experiment(tokenized_documents, vocabulary, CONTEXT_WINDOWS)
    rows = create_result_rows(results, len(tokenized_documents))
    write_results(rows, RESULTS_PATH)
    write_report(results, len(tokenized_documents), REPORT_PATH)
    print(f"Processed {len(tokenized_documents):,} documents and {len(vocabulary):,} vocabulary items.")
    print(f"Wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
