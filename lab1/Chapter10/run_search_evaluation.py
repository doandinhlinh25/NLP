"""Build, evaluate, and report a TF-IDF document search engine for Lab 01."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np


LAB_ROOT = Path(__file__).resolve().parents[1]
if str(LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(LAB_ROOT))

from Chapter8.run_experiments import (  # noqa: E402
    DATASET_PATH,
    DOCUMENT_PREVIEW_LENGTH,
    PipelineResult,
    TOP_DOCUMENT_COUNT,
    build_vectorizers,
    load_documents,
    rank_documents,
)


PIPELINE_NAME = "B_normalized"
RESULTS_PATH = Path(__file__).resolve().parent / "results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "search_evaluation.md"
QUERY_LABELS = {
    "barbecue cooking class": frozenset({0}),
    "mac disk utility": frozenset({1}),
    "natural language processing": frozenset({5699}),
    "python programming tutorial": frozenset({1814}),
    "japanese encephalitis vaccine": frozenset({1909}),
    "football match": frozenset({16456, 20399}),
    "climate change": frozenset({9209, 2231, 11682}),
}


def fit_search_index(documents: list[str]):
    """Fit the selected normalized TF-IDF search index.

    Args:
        documents: Corpus documents in original input order.

    Returns:
        A fitted vectorizer and L2-normalized sparse document matrix.
    """
    vectorizer = build_vectorizers()[PIPELINE_NAME]
    return vectorizer, vectorizer.fit_transform(documents).tocsr()


def calculate_metrics(retrieved_ids: list[int], relevant_ids: frozenset[int]) -> tuple[float, float, float]:
    """Calculate Precision@K, Recall@K, and reciprocal rank for one query.

    Args:
        retrieved_ids: Document identifiers ordered by decreasing similarity.
        relevant_ids: Manually labeled relevant document identifiers.

    Returns:
        Precision@K, Recall@K, and reciprocal rank in that order.
    """
    relevant_retrieved = sum(document_id in relevant_ids for document_id in retrieved_ids)
    first_relevant_rank = next(
        (rank for rank, document_id in enumerate(retrieved_ids, start=1) if document_id in relevant_ids),
        None,
    )
    precision = relevant_retrieved / len(retrieved_ids)
    recall = relevant_retrieved / len(relevant_ids)
    reciprocal_rank = 1 / first_relevant_rank if first_relevant_rank else 0.0
    return precision, recall, reciprocal_rank


def evaluate_queries(documents: list[str], vectorizer, matrix) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Retrieve documents and evaluate all manually labeled queries.

    Args:
        documents: Corpus documents to preview in the output.
        vectorizer: Fitted query TF-IDF vectorizer.
        matrix: Fitted document TF-IDF matrix.

    Returns:
        Retrieval rows for the CSV and one metric row per query.
    """
    result_proxy = PipelineResult(PIPELINE_NAME, vectorizer, matrix)
    rankings = {
        query: rank_documents(result_proxy, query, TOP_DOCUMENT_COUNT)
        for query in QUERY_LABELS
    }
    retrieval_rows = [
        {
            "record_type": "retrieval",
            "query": query,
            "rank": rank,
            "document_id": document_id,
            "similarity": similarity,
            "is_relevant": document_id in QUERY_LABELS[query],
            "document_preview": documents[document_id].replace("\n", " ")[:DOCUMENT_PREVIEW_LENGTH],
            "precision_at_5": "",
            "recall_at_5": "",
            "reciprocal_rank": "",
        }
        for query, ranking in rankings.items()
        for rank, (document_id, similarity) in enumerate(ranking, start=1)
    ]
    metric_rows = [
        {
            "record_type": "metric",
            "query": query,
            "rank": "",
            "document_id": "",
            "similarity": "",
            "is_relevant": "",
            "document_preview": "",
            "precision_at_5": precision,
            "recall_at_5": recall,
            "reciprocal_rank": reciprocal_rank,
        }
        for query, ranking in rankings.items()
        for precision, recall, reciprocal_rank in [calculate_metrics([document_id for document_id, _ in ranking], QUERY_LABELS[query])]
    ]
    return retrieval_rows, metric_rows


def write_results(retrieval_rows: list[dict[str, object]], metric_rows: list[dict[str, object]]) -> None:
    """Write ranked results and per-query metrics to one CSV file.

    Args:
        retrieval_rows: Top-five retrieval records for all queries.
        metric_rows: Precision, recall, and reciprocal-rank records.
    """
    field_names = (
        "record_type",
        "query",
        "rank",
        "document_id",
        "similarity",
        "is_relevant",
        "document_preview",
        "precision_at_5",
        "recall_at_5",
        "reciprocal_rank",
    )
    with RESULTS_PATH.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(retrieval_rows)
        writer.writerows(metric_rows)


def write_report(metric_rows: list[dict[str, object]]) -> None:
    """Write the aggregate metrics and evidence-based error analysis.

    Args:
        metric_rows: Per-query evaluation metrics used to calculate the means.
    """
    precision_values = np.array([float(row["precision_at_5"]) for row in metric_rows])
    recall_values = np.array([float(row["recall_at_5"]) for row in metric_rows])
    reciprocal_rank_values = np.array([float(row["reciprocal_rank"]) for row in metric_rows])
    query_rows = "\n".join(
        f"| {row['query']} | {float(row['precision_at_5']):.3f} | {float(row['recall_at_5']):.3f} | {float(row['reciprocal_rank']):.3f} |"
        for row in metric_rows
    )
    report = f"""# TF-IDF search, evaluation, and error analysis

## Search setup

The search engine uses Pipeline B (lowercase, Unicode accent normalization, punctuation exclusion, English stopword removal, word TF-IDF with L2 normalization). Query and document vectors share the same fitted vocabulary; their dot product is therefore cosine similarity. The manually labeled relevant identifiers are recorded in `run_search_evaluation.py`, and all top-five retrieved documents are in `results.csv`.

## Evaluation at K = {TOP_DOCUMENT_COUNT}

| Query | Precision@5 | Recall@5 | Reciprocal rank |
| --- | ---: | ---: | ---: |
{query_rows}
| **Mean** | **{precision_values.mean():.3f}** | **{recall_values.mean():.3f}** | **{reciprocal_rank_values.mean():.3f}** |

Thus the final aggregate measures are **P@5 = {precision_values.mean():.3f}**, **Recall@5 = {recall_values.mean():.3f}**, and **MRR = {reciprocal_rank_values.mean():.3f}**. The set is intentionally small and manually judged; it demonstrates the evaluation procedure rather than claiming a benchmark-quality score.

## Error analysis

### Good case: `japanese encephalitis vaccine`

The labeled document is retrieved because the rare medical terms have strong lexical overlap with the query. These terms receive higher IDF than generic words such as `vaccine`, so the query-document cosine score is dominated by highly discriminative shared vocabulary.

### Good case: `climate change`

Several labeled documents occur in the top five. The repeated two-word phrase has direct lexical overlap with the documents and is widespread enough to retrieve multiple on-topic sources. This is a situation where a lexical model behaves reliably.

### Weak case: `barbecue cooking class`

Document 0 is the intended relevant document, but the query says `barbecue` while the document uses `BBQ`. Pipeline B does not know that these are equivalent, so it retrieves documents matching the generic terms `cooking` and `class` instead. The failure is primarily lexical matching and vocabulary representation, not a cosine-similarity bug.

### Weak case: `mac disk utility`

The labeled Mac OS X document appears below a generic Disk Drill download page. Both contain overlapping words, but TF-IDF cannot judge whether the full intent is troubleshooting the macOS utility. It favors lexical weight rather than task-level relevance.

## Most important failure and next representation

The `barbecue`/`BBQ` case is the clearest failure: two expressions refer to the same concept but have no shared word token. A word-embedding or contextual dense representation could place such expressions near one another from similar usage contexts; a hybrid lexical+dense retriever could preserve exact-match strengths while addressing this semantic gap.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    """Build the index, retrieve top-five results, and write evaluation artifacts."""
    documents = load_documents(DATASET_PATH)
    vectorizer, matrix = fit_search_index(documents)
    retrieval_rows, metric_rows = evaluate_queries(documents, vectorizer, matrix)
    write_results(retrieval_rows, metric_rows)
    write_report(metric_rows)
    print(f"Indexed {len(documents):,} documents with {PIPELINE_NAME}.")
    print(f"Wrote {RESULTS_PATH.name} and {REPORT_PATH.name}.")


if __name__ == "__main__":
    main()
