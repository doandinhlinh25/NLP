"""Tiền xử lý dữ liệu và chia tập cho LAB 04."""

import re
from collections.abc import Sequence

from sklearn.model_selection import train_test_split


def normalize_text(text: str) -> str:
    """Chuẩn hóa chữ hoa và khoảng trắng, đồng thời giữ từ phủ định."""
    normalized_value = "" if text is None else str(text)
    return re.sub(r"\s+", " ", normalized_value.strip().lower())


def is_valid_split_configuration(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float,
    validation_size: float,
) -> bool:
    """Trả về liệu cấu hình chia dữ liệu có hợp lệ hay không."""
    return (
        len(texts) == len(labels)
        and len(texts) > 0
        and test_size > 0
        and validation_size > 0
        and test_size + validation_size < 1
    )


def split_dataset(
    texts: Sequence[str],
    labels: Sequence[str],
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_state: int = 42,
) -> tuple[list[str], list[str], list[str], list[str], list[str], list[str]]:
    """Chia train/validation/test có stratification, với tỷ lệ mặc định 60/20/20."""
    if not is_valid_split_configuration(texts, labels, test_size, validation_size):
        raise ValueError("Dữ liệu hoặc tỷ lệ chia tập không hợp lệ.")

    text_values = list(texts)
    label_values = list(labels)
    train_validation_texts, test_texts, train_validation_labels, test_labels = train_test_split(
        text_values,
        label_values,
        test_size=test_size,
        random_state=random_state,
        stratify=label_values,
    )
    relative_validation_size = validation_size / (1 - test_size)
    train_texts, validation_texts, train_labels, validation_labels = train_test_split(
        train_validation_texts,
        train_validation_labels,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_validation_labels,
    )
    return (
        train_texts,
        validation_texts,
        test_texts,
        train_labels,
        validation_labels,
        test_labels,
    )
