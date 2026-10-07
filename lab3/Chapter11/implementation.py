"""Implement the Chapter 11 co-occurrence embedding primitives for Lab 03."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from math import sqrt

import numpy as np
from scipy.sparse import csr_matrix, issparse


DEFAULT_MINIMUM_FREQUENCY = 1
DEFAULT_CONTEXT_WINDOW = 1
DEFAULT_TOP_K = 5
ZERO_SIMILARITY = 0.0
EXCLUDED_SIMILARITY = float("-inf")
MATRIX_DTYPE = np.int64
TEST_TOLERANCE = 1e-12
TEST_CORPUS = (
    ("the", "cat", "eats", "fish"),
    ("the", "cat", "likes", "milk"),
    ("the", "dog", "eats", "meat"),
    ("the", "dog", "likes", "fish"),
)
TEST_VOCABULARY = ("the", "cat", "dog", "eats", "fish", "likes", "meat", "milk")


def is_valid_window(window: int) -> bool:
    """Check whether a context-window size can form a co-occurrence matrix.

    Args:
        window: Number of neighboring token positions on each side.

    Returns:
        ``True`` when the value is a positive integer; otherwise ``False``.
    """
    return isinstance(window, int) and window > 0


def is_valid_top_k(top_k: int) -> bool:
    """Check whether a nearest-neighbor result length is valid.

    Args:
        top_k: Requested number of neighboring words.

    Returns:
        ``True`` when the value is a positive integer; otherwise ``False``.
    """
    return isinstance(top_k, int) and top_k > 0


def build_vocabulary(
    tokenized_documents: Sequence[Sequence[str]],
    minimum_frequency: int = DEFAULT_MINIMUM_FREQUENCY,
    maximum_size: int | None = None,
) -> tuple[str, ...]:
    """Build a frequency-filtered, deterministic vocabulary from tokenized text.

    Tokens are ordered by descending corpus frequency and then alphabetically,
    which makes all subsequent matrix indexes reproducible.

    Args:
        tokenized_documents: Documents represented as ordered token sequences.
        minimum_frequency: Smallest total frequency retained in the vocabulary.
        maximum_size: Optional upper bound on the number of retained tokens.

    Returns:
        Ordered vocabulary tokens used for matrix rows and columns.

    Raises:
        ValueError: If the frequency threshold or maximum vocabulary size is invalid.
    """
    if minimum_frequency < DEFAULT_MINIMUM_FREQUENCY:
        raise ValueError("Minimum frequency must be positive.")
    if maximum_size is not None and maximum_size < DEFAULT_MINIMUM_FREQUENCY:
        raise ValueError("Maximum vocabulary size must be positive when provided.")
    token_counts = Counter(token for document in tokenized_documents for token in document)
    eligible_tokens = (
        (token, count)
        for token, count in token_counts.items()
        if count >= minimum_frequency
    )
    ordered_tokens = sorted(eligible_tokens, key=lambda item: (-item[1], item[0]))
    selected_tokens = ordered_tokens[:maximum_size]
    return tuple(token for token, _ in selected_tokens)


def build_cooccurrence_matrix(
    tokenized_documents: Sequence[Sequence[str]],
    vocabulary: Sequence[str],
    window: int = DEFAULT_CONTEXT_WINDOW,
) -> csr_matrix:
    """Construct a sparse word-context count matrix from a tokenized corpus.

    A target token receives one count for each vocabulary token positioned no
    farther than ``window`` places to its left or right. The target token itself
    is excluded from its own context at that position.

    Args:
        tokenized_documents: Documents represented as ordered token sequences.
        vocabulary: Ordered target and context vocabulary.
        window: Number of neighboring positions to consider on each side.

    Returns:
        Square CSR matrix whose rows are target words and columns are contexts.

    Raises:
        ValueError: If the window or vocabulary is invalid.
    """
    if not is_valid_window(window):
        raise ValueError("Context window must be a positive integer.")
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
        dtype=MATRIX_DTYPE,
    )


def cosine_similarity(
    first_vector: Sequence[float] | csr_matrix,
    second_vector: Sequence[float] | csr_matrix,
) -> float:
    """Calculate cosine similarity for dense vectors or sparse matrix rows.

    Args:
        first_vector: First numeric vector or one-row sparse vector.
        second_vector: Second numeric vector or one-row sparse vector.

    Returns:
        Cosine similarity, or zero when either vector has zero magnitude.

    Raises:
        ValueError: If vector dimensions differ or only one vector is sparse.
    """
    if issparse(first_vector) != issparse(second_vector):
        raise ValueError("Both vectors must use the same sparse or dense representation.")
    if issparse(first_vector) and issparse(second_vector):
        if first_vector.shape != second_vector.shape:
            raise ValueError("Cosine similarity requires equal vector dimensions.")
        dot_product = float(first_vector.multiply(second_vector).sum())
        first_norm = sqrt(float(first_vector.multiply(first_vector).sum()))
        second_norm = sqrt(float(second_vector.multiply(second_vector).sum()))
    else:
        first_values = np.asarray(first_vector, dtype=float).ravel()
        second_values = np.asarray(second_vector, dtype=float).ravel()
        if first_values.shape != second_values.shape:
            raise ValueError("Cosine similarity requires equal vector dimensions.")
        dot_product = float(np.dot(first_values, second_values))
        first_norm = float(np.linalg.norm(first_values))
        second_norm = float(np.linalg.norm(second_values))
    denominator = first_norm * second_norm
    return dot_product / denominator if denominator else ZERO_SIMILARITY


def most_similar(
    word: str,
    matrix: csr_matrix,
    vocabulary: Sequence[str],
    top_k: int = DEFAULT_TOP_K,
) -> list[tuple[str, float]]:
    """Return the most similar vocabulary words to a target word.

    Similarities are calculated with cosine similarity between sparse
    co-occurrence rows. The target word is excluded, and score ties are ordered
    alphabetically for reproducible output.

    Args:
        word: Target token that must be present in ``vocabulary``.
        matrix: Square target-word by context-word CSR matrix.
        vocabulary: Ordered tokens defining the rows of ``matrix``.
        top_k: Maximum number of neighbors to return.

    Returns:
        Neighbor tokens paired with cosine similarity in descending score order.

    Raises:
        KeyError: If the target word is absent from the vocabulary.
        ValueError: If the matrix shape or requested result length is invalid.
    """
    if not is_valid_top_k(top_k):
        raise ValueError("top_k must be a positive integer.")
    vocabulary_size = len(vocabulary)
    expected_shape = (vocabulary_size, vocabulary_size)
    if matrix.shape != expected_shape:
        raise ValueError("Matrix shape must match the square vocabulary size.")
    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    if word not in vocabulary_index:
        raise KeyError(f"Word not in vocabulary: {word}")
    target_index = vocabulary_index[word]
    target_vector = matrix.getrow(target_index)
    target_norm = sqrt(float(target_vector.multiply(target_vector).sum()))
    row_norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    dot_products = (matrix @ target_vector.T).toarray().ravel()
    denominators = row_norms * target_norm
    similarities = np.divide(
        dot_products,
        denominators,
        out=np.zeros_like(dot_products, dtype=float),
        where=denominators != ZERO_SIMILARITY,
    )
    similarities[target_index] = EXCLUDED_SIMILARITY
    ranked_indices = sorted(
        range(vocabulary_size),
        key=lambda index: (-similarities[index], vocabulary[index]),
    )
    available_indices = (
        index for index in ranked_indices if similarities[index] != EXCLUDED_SIMILARITY
    )
    return [
        (vocabulary[index], float(similarities[index]))
        for index in list(available_indices)[:top_k]
    ]


def run_unit_tests() -> None:
    """Run deterministic checks for the Chapter 11 core implementation.

    Raises:
        AssertionError: If a vocabulary, matrix, cosine, or ranking result differs
            from the expected calculation.
    """
    vocabulary = build_vocabulary(TEST_CORPUS)
    matrix = build_cooccurrence_matrix(TEST_CORPUS, vocabulary)
    vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
    cat_index = vocabulary_index["cat"]
    dog_index = vocabulary_index["dog"]
    eats_index = vocabulary_index["eats"]
    nearest_words = most_similar("cat", matrix, vocabulary, DEFAULT_TOP_K)
    assert vocabulary == TEST_VOCABULARY
    assert matrix.shape == (len(TEST_VOCABULARY), len(TEST_VOCABULARY))
    assert matrix[cat_index, eats_index] == 1
    assert abs(cosine_similarity(matrix.getrow(cat_index), matrix.getrow(dog_index)) - 1.0) < TEST_TOLERANCE
    assert nearest_words[0][0] == "dog"
    assert abs(nearest_words[0][1] - 1.0) < TEST_TOLERANCE
    assert abs(cosine_similarity((1.0, 2.0, 1.0), (2.0, 4.0, 2.0)) - 1.0) < TEST_TOLERANCE


def main() -> None:
    """Run the Chapter 11 implementation verification suite."""
    run_unit_tests()
    print("All Chapter 11 implementation checks passed.")


if __name__ == "__main__":
    main()
