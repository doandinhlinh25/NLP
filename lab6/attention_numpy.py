"""Triển khai scaled dot-product attention ổn định số bằng NumPy."""

from __future__ import annotations

import numpy as np


def stable_softmax(scores: np.ndarray, axis: int = -1) -> np.ndarray:
    """Tính softmax sau khi trừ score lớn nhất để tránh tràn số."""
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("scores phải khác rỗng và chứa giá trị hữu hạn")
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trả về scaled scores, weights theo hàng và context của Q, K, V."""
    queries = np.asarray(Q, dtype=np.float64)
    keys = np.asarray(K, dtype=np.float64)
    values = np.asarray(V, dtype=np.float64)
    if any(array.ndim != 2 for array in (queries, keys, values)):
        raise ValueError("Q, K và V phải là ma trận hai chiều")
    if queries.shape[1] == 0 or queries.shape[1] != keys.shape[1]:
        raise ValueError("Q và K phải có cùng số chiều đặc trưng khác không")
    if keys.shape[0] == 0 or keys.shape[0] != values.shape[0]:
        raise ValueError("K và V phải có cùng độ dài chuỗi khác không")
    if not all(np.all(np.isfinite(array)) for array in (queries, keys, values)):
        raise ValueError("Q, K và V phải chứa các số hữu hạn")
    scores = (queries @ keys.T) / np.sqrt(keys.shape[1])
    weights = stable_softmax(scores, axis=-1)
    context = weights @ values
    return scores, weights, context


def self_attention(
    X: np.ndarray, W_Q: np.ndarray, W_K: np.ndarray, W_V: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chiếu một chuỗi thành Q, K, V rồi tính self-attention."""
    inputs = np.asarray(X, dtype=np.float64)
    projections = tuple(np.asarray(matrix, dtype=np.float64) for matrix in (W_Q, W_K, W_V))
    if inputs.ndim != 2 or any(matrix.ndim != 2 or matrix.shape[0] != inputs.shape[1] for matrix in projections):
        raise ValueError("Số chiều của ma trận chiếu phải khớp với X")
    queries, keys, values = (inputs @ matrix for matrix in projections)
    scores, weights, context = scaled_dot_product_attention(queries, keys, values)
    return queries, keys, values, scores, weights, context


def verify_examples() -> bool:
    """Kiểm tra bài tính tay, độ ổn định số và các tính chất attention."""
    query = np.array([[1.0, 0.0]])
    keys = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    values = np.array([[1.0, 0.0], [0.0, 2.0], [2.0, 1.0]])
    scores, weights, context = scaled_dot_product_attention(query, keys, values)
    expected = stable_softmax(np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0))
    assert np.allclose(scores[0], [1 / np.sqrt(2), 0, 1 / np.sqrt(2)])
    assert np.allclose(weights[0], expected)
    assert np.allclose(context[0], expected @ values)
    assert np.allclose(stable_softmax(np.array([1000, 1001, 1002])), stable_softmax(np.array([0, 1, 2])))
    assert np.allclose(np.sum(weights, axis=1), 1)
    return True


if __name__ == "__main__":
    print("Kiểm tra attention đạt:", verify_examples())
