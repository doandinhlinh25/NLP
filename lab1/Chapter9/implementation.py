"""Minimal, dependency-free TF-IDF implementation used for Lab 01."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from math import log, sqrt


TOKEN_SEPARATOR = " "
EXPECTED_TOLERANCE = 1e-9
TEST_DOCUMENTS = (
    "cat eats fish",
    "dog eats fish",
    "cat likes fish",
)


def _tokenize(document: str) -> list[str]:
    """Return lowercase whitespace-delimited tokens from one document."""
    return document.lower().split(TOKEN_SEPARATOR)


def build_vocabulary(documents: Iterable[str]) -> list[str]:
    """Build a deterministic, alphabetically sorted vocabulary from documents.

    Args:
        documents: Text documents used to construct the vocabulary.

    Returns:
        The unique lowercase tokens in alphabetical order.
    """
    return sorted({token for document in documents for token in _tokenize(document)})


def compute_counts(documents: Iterable[str], vocabulary: Sequence[str]) -> list[list[int]]:
    """Create a term-count vector for every document.

    Args:
        documents: Text documents to represent.
        vocabulary: Ordered tokens defining every vector position.

    Returns:
        One integer count vector per document, following ``vocabulary`` order.
    """
    document_counters = [Counter(_tokenize(document)) for document in documents]
    return [[counter[token] for token in vocabulary] for counter in document_counters]


def compute_tf(counts: Sequence[Sequence[int]]) -> list[list[float]]:
    """Normalize each count vector by its total number of tokens.

    Args:
        counts: One raw term-count vector per document.

    Returns:
        Term-frequency vectors with values summing to one for non-empty documents.
    """
    totals = [sum(row) for row in counts]
    return [
        [count / total if total else 0.0 for count in row]
        for row, total in zip(counts, totals)
    ]


def compute_idf(counts: Sequence[Sequence[int]]) -> list[float]:
    """Calculate unsmoothed inverse document frequency for every vocabulary term.

    Args:
        counts: One raw term-count vector per document.

    Returns:
        IDF values computed as ``log(number_of_documents / document_frequency)``.

    Raises:
        ValueError: If no documents are provided or a vocabulary term is absent.
    """
    document_count = len(counts)
    if document_count == 0:
        raise ValueError("IDF requires at least one document.")
    document_frequencies = [sum(count > 0 for count in column) for column in zip(*counts)]
    if any(frequency == 0 for frequency in document_frequencies):
        raise ValueError("Every vocabulary term must occur in at least one document.")
    return [log(document_count / frequency) for frequency in document_frequencies]


def compute_tfidf(
    term_frequencies: Sequence[Sequence[float]], idf: Sequence[float]
) -> list[list[float]]:
    """Multiply each document TF vector elementwise by an IDF vector.

    Args:
        term_frequencies: Term-frequency vectors for the documents.
        idf: IDF values in the same vocabulary order.

    Returns:
        TF-IDF vectors without a final L2 normalization step.

    Raises:
        ValueError: If a TF vector does not match the IDF vector length.
    """
    expected_length = len(idf)
    if any(len(row) != expected_length for row in term_frequencies):
        raise ValueError("Each TF vector must have the same length as IDF.")
    return [[value * weight for value, weight in zip(row, idf)] for row in term_frequencies]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity between two equally sized numeric vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The cosine similarity, or zero when either vector has zero magnitude.

    Raises:
        ValueError: If the vectors have different dimensions.
    """
    if len(left) != len(right):
        raise ValueError("Cosine similarity requires vectors of equal length.")
    numerator = sum(first * second for first, second in zip(left, right))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    denominator = left_norm * right_norm
    return numerator / denominator if denominator else 0.0


def verify_against_sklearn(documents: Sequence[str]) -> None:
    """Verify the custom unsmoothed TF-IDF convention against scikit-learn.

    Scikit-learn's unsmoothed ``TfidfTransformer`` stores ``log(N / df) + 1``.
    This function removes that final offset before comparing it with the Lab 01
    convention, which is exactly ``log(N / df)``.

    Args:
        documents: Documents used for the custom and reference calculations.

    Raises:
        AssertionError: If the two implementations differ beyond tolerance.
    """
    from numpy import asarray
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

    vocabulary = build_vocabulary(documents)
    counts = compute_counts(documents, vocabulary)
    term_frequencies = compute_tf(counts)
    custom_idf = compute_idf(counts)
    custom_tfidf = compute_tfidf(term_frequencies, custom_idf)
    vocabulary_mapping = {term: index for index, term in enumerate(vocabulary)}
    reference_counter = CountVectorizer(vocabulary=vocabulary_mapping, lowercase=True)
    reference_counts = reference_counter.transform(documents)
    reference_transformer = TfidfTransformer(norm=None, smooth_idf=False)
    reference_transformer.fit(reference_counts)
    reference_idf = reference_transformer.idf_ - 1
    reference_tf = reference_counts.toarray() / asarray(reference_counts.sum(axis=1))
    reference_tfidf = reference_tf * reference_idf
    assert abs(max(abs(first - second) for first, second in zip(custom_idf, reference_idf))) < EXPECTED_TOLERANCE
    assert abs(max(abs(first - second) for first, second in zip(sum(custom_tfidf, []), reference_tfidf.ravel()))) < EXPECTED_TOLERANCE


def run_unit_tests() -> None:
    """Run focused unit tests for all required Lab 01 functions."""
    vocabulary = build_vocabulary(TEST_DOCUMENTS)
    counts = compute_counts(TEST_DOCUMENTS, vocabulary)
    term_frequencies = compute_tf(counts)
    idf = compute_idf(counts)
    tfidf = compute_tfidf(term_frequencies, idf)
    cat_index = vocabulary.index("cat")
    fish_index = vocabulary.index("fish")
    expected_idf = log(3 / 2)
    assert vocabulary == ["cat", "dog", "eats", "fish", "likes"]
    assert counts == [[1, 0, 1, 1, 0], [0, 1, 1, 1, 0], [1, 0, 0, 1, 1]]
    assert abs(term_frequencies[0][cat_index] - (1 / 3)) < EXPECTED_TOLERANCE
    assert abs(sum(term_frequencies[0]) - 1.0) < EXPECTED_TOLERANCE
    assert abs(idf[cat_index] - expected_idf) < EXPECTED_TOLERANCE
    assert idf[fish_index] == 0.0
    assert abs(tfidf[0][cat_index] - (expected_idf / 3)) < EXPECTED_TOLERANCE
    assert tfidf[0][fish_index] == 0.0
    assert abs(cosine_similarity([1, 1, 1], [1, 1, 0]) - (2 / sqrt(6))) < EXPECTED_TOLERANCE
    verify_against_sklearn(TEST_DOCUMENTS)


if __name__ == "__main__":
    run_unit_tests()
    print("All unit tests passed.")
