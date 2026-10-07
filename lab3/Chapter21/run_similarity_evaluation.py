"""Evaluate Chapter 17 Word2Vec scores against a small word-pair set."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lab3.Chapter17.run_word2vec_experiment import MODEL_PATH, TRAINING_OBJECTIVE  # noqa: E402
from lab3.Chapter18.run_embedding_inspection import load_trained_model  # noqa: E402


RESULTS_PATH = Path(__file__).resolve().parent / "word_similarity_results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "word_similarity_evaluation.md"
WORD_PAIRS = (
    ("doctor", "physician", "high", True),
    ("car", "automobile", "high", True),
    ("king", "queen", "high", True),
    ("cat", "dog", "medium", True),
    ("computer", "banana", "low", False),
)
HUMAN_RANK_BY_EXPECTATION = {"high": 3, "medium": 2, "low": 1}
CSV_FIELDS = (
    "word_1",
    "word_2",
    "expected_similarity",
    "is_semantically_related",
    "cosine_similarity",
    "model_rank",
)
OUT_OF_VOCABULARY = "not_in_vocabulary"
REPORT_DECIMAL_PLACES = 6


def is_valid_word_pairs(word_pairs: tuple[tuple[str, str, str, bool], ...]) -> bool:
    """Check whether every evaluation pair has two words and an expectation.

    Args:
        word_pairs: Pairs with expected qualitative similarity and relation label.

    Returns:
        ``True`` when all configured records have valid word strings and labels.
    """
    allowed_expectations = frozenset(HUMAN_RANK_BY_EXPECTATION)
    return bool(word_pairs) and all(
        bool(first_word)
        and bool(second_word)
        and expectation in allowed_expectations
        and isinstance(is_related, bool)
        for first_word, second_word, expectation, is_related in word_pairs
    )


def create_evaluation_rows(model) -> list[dict[str, str | float | bool | int]]:
    """Calculate model cosine scores and ranks for all selected word pairs.

    Args:
        model: Fitted Chapter 17 Word2Vec model.

    Returns:
        Evaluation rows ranked by decreasing available cosine similarity.
    """
    scored_rows = [
        {
            "word_1": first_word,
            "word_2": second_word,
            "expected_similarity": expectation,
            "is_semantically_related": is_related,
            "cosine_similarity": (
                float(model.wv.similarity(first_word, second_word))
                if first_word in model.wv and second_word in model.wv
                else OUT_OF_VOCABULARY
            ),
        }
        for first_word, second_word, expectation, is_related in WORD_PAIRS
    ]
    ordered_rows = sorted(
        scored_rows,
        key=lambda row: (
            not isinstance(row["cosine_similarity"], float),
            -float(row["cosine_similarity"])
            if isinstance(row["cosine_similarity"], float)
            else 0.0,
        ),
    )
    return [
        {**row, "model_rank": rank}
        for rank, row in enumerate(ordered_rows, start=1)
    ]


def write_results(rows: list[dict[str, str | float | bool | int]], output_path: Path) -> None:
    """Write word-pair scores and model ranks as CSV.

    Args:
        rows: Ranked word-similarity evaluation rows.
        output_path: Destination CSV file.
    """
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def calculate_spearman_correlation(rows: list[dict[str, str | float | bool | int]]) -> float | None:
    """Calculate rank correlation between expected and available model scores.

    Args:
        rows: Evaluation rows containing expected levels and cosine scores.

    Returns:
        Spearman correlation, or ``None`` when scores cannot be compared.
    """
    available_rows = [
        row for row in rows if isinstance(row["cosine_similarity"], float)
    ]
    if len(available_rows) < 2:
        return None
    expected_ranks = [HUMAN_RANK_BY_EXPECTATION[str(row["expected_similarity"])] for row in available_rows]
    model_scores = [float(row["cosine_similarity"]) for row in available_rows]
    return float(spearmanr(expected_ranks, model_scores).statistic)


def format_score(score: float | str) -> str:
    """Format a word-pair score for Markdown output.

    Args:
        score: Numeric cosine similarity or an OOV marker.

    Returns:
        Fixed-precision numeric text or unchanged marker.
    """
    return f"{score:.{REPORT_DECIMAL_PLACES}f}" if isinstance(score, float) else score


def write_report(rows: list[dict[str, str | float | bool | int]], output_path: Path) -> None:
    """Write the quantitative word-similarity evaluation report.

    Args:
        rows: Ranked model scores for the configured word-pair set.
        output_path: Destination Markdown report.
    """
    related_scores = [
        float(row["cosine_similarity"])
        for row in rows
        if bool(row["is_semantically_related"]) and isinstance(row["cosine_similarity"], float)
    ]
    unrelated_scores = [
        float(row["cosine_similarity"])
        for row in rows
        if not bool(row["is_semantically_related"]) and isinstance(row["cosine_similarity"], float)
    ]
    correlation = calculate_spearman_correlation(rows)
    table_rows = "\n".join(
        f"| {row['model_rank']} | {row['word_1']} – {row['word_2']} | {row['expected_similarity']} | "
        f"{format_score(row['cosine_similarity'])} | {row['is_semantically_related']} |"
        for row in rows
    )
    mean_related = sum(related_scores) / len(related_scores) if related_scores else None
    mean_unrelated = sum(unrelated_scores) / len(unrelated_scores) if unrelated_scores else None
    report = f"""# Chapter 21 — Evaluation: Word similarity

## Thiết lập

Các pair được chấm bằng cosine similarity từ model {TRAINING_OBJECTIVE} của Chapter 17. `Expected similarity` là thứ tự trực giác trước khi xem model: `high` > `medium` > `low`. `model_rank = 1` là score cao nhất.

## Xếp hạng của model

| Model rank | Word pair | Expected similarity | Cosine similarity | Semantically related |
| ---: | --- | --- | ---: | --- |
{table_rows}

## Kết quả định lượng

- Mean cosine của các cặp được xem là related: {format_score(mean_related) if mean_related is not None else OUT_OF_VOCABULARY}.
- Mean cosine của cặp unrelated: {format_score(mean_unrelated) if mean_unrelated is not None else OUT_OF_VOCABULARY}.
- Spearman correlation giữa mức trực giác (`high`, `medium`, `low`) và model score: {format_score(correlation) if correlation is not None else OUT_OF_VOCABULARY}.

Score và thứ hạng cần được so sánh với intuition, không chỉ nhận xét chung rằng kết quả “khá tốt”. Nếu cặp unrelated có score cao hoặc một cặp expected `high` có thứ hạng thấp, đó là evidence để phân tích ảnh hưởng của corpus nhỏ, frequency và context window ở các chapter tiếp theo.
"""
    output_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Evaluate the Chapter 17 model on the Chapter 21 word-pair set."""
    if not is_valid_word_pairs(WORD_PAIRS):
        raise ValueError("Word-similarity evaluation configuration is invalid.")
    model = load_trained_model(MODEL_PATH)
    rows = create_evaluation_rows(model)
    write_results(rows, RESULTS_PATH)
    write_report(rows, REPORT_PATH)
    print(f"Evaluated {len(rows)} word pairs and wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
