"""Inspect the Chapter 17 Word2Vec embedding using corpus-based evidence."""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from gensim.models import Word2Vec


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab3.Chapter17.run_word2vec_experiment import (  # noqa: E402
    CONTEXT_WINDOW,
    DATASET_PATH,
    DOCUMENT_LIMIT,
    MODEL_PATH,
    TRAINING_OBJECTIVE,
    load_tokenized_documents,
)


REPORT_PATH = Path(__file__).resolve().parent / "embedding_inspection.md"
RESULTS_PATH = Path(__file__).resolve().parent / "doctor_top5.csv"
INSPECTION_WORD = "doctor"
TOP_NEIGHBOR_COUNT = 5
SHARED_CONTEXT_COUNT = 3
CSV_FIELDS = (
    "rank",
    "word",
    "cosine_similarity",
    "corpus_frequency",
    "shared_local_contexts",
)
NO_SHARED_CONTEXT = "no_shared_context_in_window"
REPORT_DECIMAL_PLACES = 6


def is_valid_inspection_request(word: str, top_neighbor_count: int, context_count: int) -> bool:
    """Check whether an embedding-inspection request has usable parameters.

    Args:
        word: Vocabulary word selected for nearest-neighbor inspection.
        top_neighbor_count: Number of nearest neighbors requested from Word2Vec.
        context_count: Number of shared local context tokens to report.

    Returns:
        ``True`` when the word is non-empty and both counts are positive integers.
    """
    return bool(word) and all(
        isinstance(value, int) and value > 0
        for value in (top_neighbor_count, context_count)
    )


def load_trained_model(model_path: Path) -> Word2Vec:
    """Load the Word2Vec model produced by Chapter 17.

    Args:
        model_path: Persisted Gensim model path.

    Returns:
        The fitted Word2Vec model ready for vector queries.

    Raises:
        FileNotFoundError: If Chapter 17 has not generated the model yet.
    """
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Word2Vec model not found: {model_path}. Run Chapter 17 first."
        )
    return Word2Vec.load(str(model_path))


def count_tokens(tokenized_documents: list[list[str]]) -> Counter[str]:
    """Count token occurrences across all tokenized documents.

    Args:
        tokenized_documents: Corpus documents represented as token sequences.

    Returns:
        Corpus-wide frequency counter keyed by token.
    """
    return Counter(token for document in tokenized_documents for token in document)


def count_local_contexts(
    tokenized_documents: list[list[str]], word: str, window: int
) -> Counter[str]:
    """Count tokens appearing within a local window around one word.

    Args:
        tokenized_documents: Corpus documents represented as token sequences.
        word: Target token whose neighboring contexts are counted.
        window: Number of positions considered on both sides of each occurrence.

    Returns:
        Counter of neighboring tokens around all occurrences of ``word``.
    """
    context_counts: Counter[str] = Counter()
    for document in tokenized_documents:
        document_length = len(document)
        for target_position, target_token in enumerate(document):
            context_start = max(target_position - window, 0)
            context_stop = min(target_position + window + 1, document_length)
            if target_token == word:
                context_tokens = (
                    document[context_position]
                    for context_position in range(context_start, context_stop)
                    if context_position != target_position
                )
                context_counts.update(context_tokens)
    return context_counts


def format_shared_contexts(
    inspection_contexts: Counter[str], neighbor_contexts: Counter[str]
) -> str:
    """Format the strongest local contexts shared by a target and neighbor.

    Args:
        inspection_contexts: Context frequency counter for the inspected word.
        neighbor_contexts: Context frequency counter for one nearest neighbor.

    Returns:
        Up to three shared ``token (count)`` values, or a no-evidence label.
    """
    shared_contexts = inspection_contexts & neighbor_contexts
    common_contexts = shared_contexts.most_common(SHARED_CONTEXT_COUNT)
    return (
        ", ".join(f"{token} ({count})" for token, count in common_contexts)
        if common_contexts
        else NO_SHARED_CONTEXT
    )


def create_inspection_rows(
    model: Word2Vec, tokenized_documents: list[list[str]], word: str
) -> list[dict[str, str | int | float]]:
    """Build reproducible inspection rows for the top Word2Vec neighbors.

    Args:
        model: Fitted Word2Vec model to query.
        tokenized_documents: Training-corpus sample used for context evidence.
        word: In-vocabulary target word to inspect.

    Returns:
        Ranked neighbor rows containing score, frequency, and shared contexts.

    Raises:
        KeyError: If ``word`` is absent from the trained Word2Vec vocabulary.
    """
    if word not in model.wv:
        raise KeyError(f"Word not in trained vocabulary: {word}")
    token_frequencies = count_tokens(tokenized_documents)
    inspection_contexts = count_local_contexts(tokenized_documents, word, CONTEXT_WINDOW)
    neighbors = model.wv.most_similar(word, topn=TOP_NEIGHBOR_COUNT)
    neighbor_context_counts = {
        neighbor: count_local_contexts(tokenized_documents, neighbor, CONTEXT_WINDOW)
        for neighbor, _ in neighbors
    }
    return [
        {
            "rank": rank,
            "word": neighbor,
            "cosine_similarity": similarity,
            "corpus_frequency": token_frequencies[neighbor],
            "shared_local_contexts": format_shared_contexts(
                inspection_contexts,
                neighbor_context_counts[neighbor],
            ),
        }
        for rank, (neighbor, similarity) in enumerate(neighbors, start=1)
    ]


def write_results(rows: list[dict[str, str | int | float]], output_path: Path) -> None:
    """Write nearest-neighbor inspection measurements as CSV.

    Args:
        rows: Top-neighbor inspection rows.
        output_path: Destination CSV file.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, str | int | float]], inspection_frequency: int, output_path: Path
) -> None:
    """Write the Chapter 18 interpretation with concrete corpus evidence.

    Args:
        rows: Top-neighbor inspection rows used to construct the table.
        inspection_frequency: Corpus frequency of the inspected target word.
        output_path: Destination Markdown report.
    """
    result_rows = "\n".join(
        f"| {row['rank']} | {row['word']} | {float(row['cosine_similarity']):.{REPORT_DECIMAL_PLACES}f} | "
        f"{row['corpus_frequency']} | {row['shared_local_contexts']} |"
        for row in rows
    )
    report = f"""# Chapter 18 — Inspect embeddings

## Truy vấn

Model {TRAINING_OBJECTIVE} ở Chapter 17 được truy vấn bằng `model.wv.most_similar("{INSPECTION_WORD}", topn={TOP_NEIGHBOR_COUNT})`. Cosine score càng cao nghĩa là hai embedding có hướng gần nhau trong không gian vector của model này. Trong sample huấn luyện, `{INSPECTION_WORD}` xuất hiện {inspection_frequency} lần.

## Top-{TOP_NEIGHBOR_COUNT} của `{INSPECTION_WORD}`

| Rank | Word | Cosine similarity | Corpus frequency | Shared local contexts |
| ---: | --- | ---: | ---: | --- |
{result_rows}

`Shared local contexts` là các token xuất hiện trong phạm vi {CONTEXT_WINDOW} vị trí quanh cả `{INSPECTION_WORD}` và từ hàng tương ứng, với số đếm chung nhỏ nhất giữa hai phía. Đây là evidence trực tiếp từ sample corpus, không phải nhãn ngữ nghĩa được gán thủ công.

## Diễn giải

Top-5 quan sát được không phải các từ y khoa như dự đoán ở Chapter 9. Evidence trong bảng cho thấy shared contexts phần lớn là từ chức năng phổ biến như `the`, `to`, `of` và `your`, thay vì context y khoa đặc hiệu. Với chỉ {inspection_frequency} lần xuất hiện của `{INSPECTION_WORD}`, model có quá ít quan sát để học một vector ổn định; các từ có pattern ngữ cảnh chung có thể nhận score rất cao. Vì vậy, score cao không tự chứng minh quan hệ đồng nghĩa: kết quả này là một failure case cần được kiểm tra bằng frequency và context gốc, thay vì chỉ kết luận rằng model “học được semantic similarity”.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Generate the Chapter 18 inspection CSV and evidence-based report."""
    if not is_valid_inspection_request(
        INSPECTION_WORD,
        TOP_NEIGHBOR_COUNT,
        SHARED_CONTEXT_COUNT,
    ):
        raise ValueError("Embedding inspection configuration is invalid.")
    model = load_trained_model(MODEL_PATH)
    tokenized_documents = load_tokenized_documents(DATASET_PATH, DOCUMENT_LIMIT)
    rows = create_inspection_rows(model, tokenized_documents, INSPECTION_WORD)
    inspection_frequency = count_tokens(tokenized_documents)[INSPECTION_WORD]
    write_results(rows, RESULTS_PATH)
    write_report(rows, inspection_frequency, REPORT_PATH)
    print(f"Inspected {INSPECTION_WORD!r} and wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
