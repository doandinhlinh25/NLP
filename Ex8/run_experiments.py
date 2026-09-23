"""Run the sparse-representation and preprocessing experiments for Lab 01."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer


LAB_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = LAB_ROOT.parent / "data" / "c4-train.00000-of-01024-30K.json" / "c4-train.00000-of-01024-30K.json"
OUTPUT_PATH = Path(__file__).resolve().parent / "candidate_results.csv"
REPORT_PATH = Path(__file__).resolve().parent / "experiment_results.md"
DOCUMENT_KEY = "text"
DOCUMENT_PREVIEW_LENGTH = 180
TOP_TERM_COUNT = 20
TOP_DOCUMENT_COUNT = 5
VECTOR_DTYPE = np.float32
WORD_TOKEN_PATTERN = r"(?u)\b\w+\b"
NORMALIZED_TOKEN_PATTERN = r"(?u)\b[a-zA-Z][a-zA-Z']+\b"
CHARACTER_NGRAM_RANGE = (3, 5)
CHARACTER_MINIMUM_DOCUMENT_FREQUENCY = 3
CHARACTER_MAXIMUM_FEATURES = 100_000
QUERIES = (
    "barbecue cooking class",
    "mac disk utility",
    "natural language processing",
    "machine learning healthcare",
    "python programming tutorial",
    "covid vaccine",
    "football match",
    "climate change",
)
ABLATION_RELEVANCE_LABELS = {
    "barbecue cooking class": frozenset({0}),
    "mac disk utility": frozenset({1}),
    "natural language processing": frozenset({5699}),
    "python programming tutorial": frozenset({1814}),
    "japanese encephalitis vaccine": frozenset({1909}),
    "football match": frozenset({16456, 20399}),
    "climate change": frozenset({9209, 2231, 11682}),
}


@dataclass(frozen=True)
class PipelineResult:
    """Store one fitted TF-IDF pipeline and its document matrix."""

    name: str
    vectorizer: TfidfVectorizer
    matrix: csr_matrix


def load_documents(dataset_path: Path) -> list[str]:
    """Load non-empty text documents from a line-delimited JSON corpus.

    Args:
        dataset_path: Path to the JSONL corpus supplied for the lab.

    Returns:
        Corpus documents in their original input order.
    """
    with dataset_path.open(encoding="utf-8") as dataset_file:
        records = (json.loads(line) for line in dataset_file)
        return [record[DOCUMENT_KEY] for record in records if record.get(DOCUMENT_KEY)]


def build_vectorizers() -> dict[str, TfidfVectorizer]:
    """Create the three explicitly defined preprocessing pipelines.

    Returns:
        Named scikit-learn TF-IDF vectorizers for Pipeline A, B, and C.
    """
    return {
        "A_minimal": TfidfVectorizer(
            lowercase=True,
            token_pattern=WORD_TOKEN_PATTERN,
            dtype=VECTOR_DTYPE,
        ),
        "B_normalized": TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            stop_words=list(ENGLISH_STOP_WORDS),
            token_pattern=NORMALIZED_TOKEN_PATTERN,
            dtype=VECTOR_DTYPE,
        ),
        "C_subword": TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=CHARACTER_NGRAM_RANGE,
            min_df=CHARACTER_MINIMUM_DOCUMENT_FREQUENCY,
            max_features=CHARACTER_MAXIMUM_FEATURES,
            dtype=VECTOR_DTYPE,
        ),
    }


def fit_pipelines(documents: list[str]) -> dict[str, PipelineResult]:
    """Fit all preprocessing pipelines on a shared document corpus.

    Args:
        documents: Text documents used to build the TF-IDF index.

    Returns:
        Fitted vectorizers and sparse matrices, keyed by pipeline name.
    """
    vectorizers = build_vectorizers()
    return {
        name: PipelineResult(name, vectorizer, vectorizer.fit_transform(documents).tocsr())
        for name, vectorizer in vectorizers.items()
    }


def matrix_metrics(result: PipelineResult, document_count: int) -> dict[str, float | int]:
    """Calculate vocabulary, token, non-zero, and sparsity measurements.

    Args:
        result: A fitted preprocessing pipeline.
        document_count: Number of indexed documents.

    Returns:
        Numeric metrics required by the preprocessing ablation table.
    """
    vocabulary_size = len(result.vectorizer.vocabulary_)
    denominator = document_count * vocabulary_size
    return {
        "vocabulary_size": vocabulary_size,
        "average_nonzero_features": result.matrix.nnz / document_count,
        "nnz": result.matrix.nnz,
        "sparsity": 1 - (result.matrix.nnz / denominator),
    }


def calculate_query_oov(result: PipelineResult, query: str) -> float:
    """Calculate the share of analyzer features in a query absent from the vocabulary.

    Args:
        result: Fitted pipeline used to analyze the query.
        query: Search query to inspect.

    Returns:
        OOV feature ratio, with zero for a query producing no analyzer features.
    """
    analyzer = result.vectorizer.build_analyzer()
    query_features = analyzer(query)
    unknown_features = [feature for feature in query_features if feature not in result.vectorizer.vocabulary_]
    return len(unknown_features) / len(query_features) if query_features else 0.0


def rank_documents(result: PipelineResult, query: str, top_k: int) -> list[tuple[int, float]]:
    """Rank documents by cosine similarity to a TF-IDF query vector.

    Args:
        result: Fitted L2-normalized TF-IDF document index.
        query: Text query supplied by the user.
        top_k: Number of highest-scoring results to return.

    Returns:
        Zero-based document identifiers paired with decreasing similarity scores.
    """
    query_vector = result.vectorizer.transform([query])
    similarities = (result.matrix @ query_vector.T).toarray().ravel()
    candidate_indices = np.argpartition(-similarities, top_k - 1)[:top_k]
    ordered_indices = candidate_indices[np.argsort(-similarities[candidate_indices])]
    return [(int(index), float(similarities[index])) for index in ordered_indices]


def select_top_features(result: PipelineResult, document_id: int, top_k: int) -> list[tuple[str, float]]:
    """Return the most heavily weighted TF-IDF features of one document.

    Args:
        result: Fitted pipeline containing the selected document vector.
        document_id: Zero-based identifier of the document to inspect.
        top_k: Number of terms to return.

    Returns:
        Feature names and weights sorted from highest to lowest weight.
    """
    feature_names = result.vectorizer.get_feature_names_out()
    row = result.matrix.getrow(document_id)
    sorted_positions = row.data.argsort()[::-1][:top_k]
    return [(feature_names[row.indices[position]], float(row.data[position])) for position in sorted_positions]


def mean_precision_at_k(result: PipelineResult, relevance_labels: dict[str, frozenset[int]], top_k: int) -> float:
    """Calculate mean Precision@K from fixed manual relevance labels.

    Args:
        result: Fitted pipeline to evaluate.
        relevance_labels: Query-to-relevant-document mapping.
        top_k: Number of retrieved documents evaluated per query.

    Returns:
        Arithmetic mean of Precision@K across all labeled queries.
    """
    precision_values = [
        sum(document_id in relevant_ids for document_id, _ in rank_documents(result, query, top_k)) / top_k
        for query, relevant_ids in relevance_labels.items()
    ]
    return sum(precision_values) / len(precision_values)


def select_frequency_extremes(result: PipelineResult, top_k: int) -> tuple[list[tuple[str, int]], list[tuple[str, float]]]:
    """Find terms with the largest document frequency and largest IDF.

    Args:
        result: Fitted pipeline to inspect.
        top_k: Number of terms returned for each list.

    Returns:
        A pair containing high-document-frequency terms and high-IDF terms.
    """
    feature_names = result.vectorizer.get_feature_names_out()
    document_frequencies = np.asarray(result.matrix.getnnz(axis=0)).ravel()
    common_indices = np.argsort(-document_frequencies)[:top_k]
    idf_indices = np.argsort(-result.vectorizer.idf_)[:top_k]
    common_terms = [(feature_names[index], int(document_frequencies[index])) for index in common_indices]
    rare_terms = [(feature_names[index], float(result.vectorizer.idf_[index])) for index in idf_indices]
    return common_terms, rare_terms


def write_candidates(
    documents: list[str], results: dict[str, PipelineResult], output_path: Path
) -> None:
    """Write top-five retrieval candidates for manual relevance labeling.

    Args:
        documents: Indexed documents in original order.
        results: Fitted pipeline results to search.
        output_path: Destination CSV path.
    """
    rows = [
        (result.name, query, rank, document_id, similarity, documents[document_id].replace("\n", " ")[:DOCUMENT_PREVIEW_LENGTH])
        for result in results.values()
        for query in QUERIES
        for rank, (document_id, similarity) in enumerate(rank_documents(result, query, TOP_DOCUMENT_COUNT), start=1)
    ]
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("pipeline", "query", "rank", "document_id", "similarity", "document_preview"))
        writer.writerows(rows)


def write_report(
    documents: list[str], results: dict[str, PipelineResult], report_path: Path
) -> None:
    """Write a concise report for the sparse-representation experiment.

    Args:
        documents: Indexed documents used by the experiment.
        results: Fitted pipeline results.
        report_path: Destination Markdown path.
    """
    minimal_result = results["A_minimal"]
    metrics = {name: matrix_metrics(result, len(documents)) for name, result in results.items()}
    common_terms, rare_terms = select_frequency_extremes(minimal_result, TOP_TERM_COUNT)
    top_features = select_top_features(minimal_result, 0, TOP_TERM_COUNT)
    mean_oov = {
        name: sum(calculate_query_oov(result, query) for query in QUERIES) / len(QUERIES)
        for name, result in results.items()
    }
    search_precision = {
        name: mean_precision_at_k(result, ABLATION_RELEVANCE_LABELS, TOP_DOCUMENT_COUNT)
        for name, result in results.items()
    }
    table_rows = "\n".join(
        f"| {name} | {value['vocabulary_size']:,} | {value['average_nonzero_features']:.2f} | {value['nnz']:,} | {value['sparsity']:.6%} | {mean_oov[name]:.2%} | {search_precision[name]:.3f} |"
        for name, value in metrics.items()
    )
    common_list = "\n".join(f"- `{term}`: df = {frequency:,}" for term, frequency in common_terms)
    rare_list = "\n".join(f"- `{term}`: idf = {idf:.4f}" for term, idf in rare_terms)
    feature_list = "\n".join(f"- `{term}`: tf-idf = {weight:.4f}" for term, weight in top_features)
    report = f"""# Experiment 1 and preprocessing ablation

## Corpus and sparse representation

- Documents: **{len(documents):,}**
- Pipeline A matrix shape: **({len(documents):,}, {metrics['A_minimal']['vocabulary_size']:,})**
- Pipeline A non-zero values: **{metrics['A_minimal']['nnz']:,}**
- Pipeline A sparsity: **{metrics['A_minimal']['sparsity']:.6%}**

Every document vector has one position for every corpus vocabulary item so that all document vectors inhabit the same feature space. It remains sparse because any individual document contains only a small subset of those items.

## Pipeline comparison

| Pipeline | Vocabulary size | Avg. non-zero features/document | nnz | Matrix sparsity | Mean query-feature OOV | Mean P@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{table_rows}

Pipeline A is lowercase plus word tokenization. Pipeline B additionally normalizes accents, excludes punctuation through its word-token rule, and removes the scikit-learn English stopword list. Pipeline C uses character 3–5 grams (subword features), with `min_df={CHARACTER_MINIMUM_DOCUMENT_FREQUENCY}` and `max_features={CHARACTER_MAXIMUM_FEATURES:,}` to keep the 30K-document experiment feasible. Its OOV rate is measured over character n-gram features, so it is not directly identical to word-token OOV; it is included to show the subword robustness effect. Mean P@5 uses the same seven manually labeled queries as Exercise 10, so it is a small diagnostic comparison rather than a general benchmark.

## Pipeline A vocabulary inspection

### 20 highest document-frequency terms

{common_list}

### 20 highest-IDF terms

{rare_list}

### 20 highest TF-IDF terms in document 0

Document 0 preview: {documents[0].replace(chr(10), ' ')[:DOCUMENT_PREVIEW_LENGTH]}

{feature_list}

## Interpretation

A high-document-frequency term is not necessarily high TF-IDF: its IDF is low because it is shared by many documents. A high-IDF term is also not automatically high TF-IDF in every document, because the term must occur in the selected document and have enough local TF. Lowercasing merges capitalization variants and therefore reduces duplicate vocabulary entries. Stopword removal reduces frequent function words, but can remove useful short query terms or phrases; it is a modeling choice, not an automatic improvement. Removing punctuation can also discard information such as `C++`, version strings, URLs, emoticons, decimal numbers, and hyphenated entities. In the measured table, Pipeline B is the sparsest and Pipeline C is the least sparse because each document produces many overlapping character n-grams. The highest Mean P@5 should be chosen for this small labeled set; it does not automatically establish the best preprocessing policy for every retrieval task.
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    """Execute all 30K-corpus representation experiments and write artifacts."""
    documents = load_documents(DATASET_PATH)
    results = fit_pipelines(documents)
    write_candidates(documents, results, OUTPUT_PATH)
    write_report(documents, results, REPORT_PATH)
    print(f"Indexed {len(documents):,} documents from {DATASET_PATH}")
    print(f"Wrote {OUTPUT_PATH.name} and {REPORT_PATH.name}")


if __name__ == "__main__":
    main()
