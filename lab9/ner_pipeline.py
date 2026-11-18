"""Train and evaluate a baseline and pretrained token classifier on CoNLL-2003."""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.pipeline import make_pipeline
from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                          DataCollatorForTokenClassification, Trainer, TrainingArguments)

LAB_DIR = Path(__file__).resolve().parent
IGNORE_INDEX = -100
DEFAULT_MODEL = "distilbert/distilbert-base-cased"
SPLIT_NAMES = ("train", "validation", "test")
ENTITY_TYPES = ("PER", "ORG", "LOC", "MISC")
LENGTH_BUCKETS = ((1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 10**9, "4+"))


@dataclass(frozen=True)
class Settings:
    """Describe the reproducible NER experiment configuration."""
    model_name: str = DEFAULT_MODEL
    seed: int = 42
    max_length: int = 256
    epochs: int = 2
    batch_size: int = 8
    learning_rate: float = 2e-5
    train_limit: int = 0
    evaluation_limit: int = 0
    output_dir: Path = LAB_DIR


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_conll() -> tuple[Any, list[str]]:
    """Load CoNLL-2003 and return its canonical BIO label names."""
    dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
    names = dataset["train"].features["ner_tags"].feature.names
    return dataset, names


def limit_split(split: Any, limit: int, seed: int) -> Any:
    """Return a deterministically shuffled subset, or the full split at limit zero."""
    if limit <= 0 or limit >= len(split):
        return split
    return split.shuffle(seed=seed).select(range(limit))


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Apply first-subword supervision and ignore special and continuation tokens."""
    aligned: list[int] = []
    previous_word_id: int | None = None
    for word_id in word_ids:
        if word_id is None or word_id == previous_word_id:
            aligned.append(IGNORE_INDEX)
        else:
            aligned.append(int(word_labels[word_id]))
        previous_word_id = word_id
    return aligned


def tokenize_split(split: Any, tokenizer: Any, max_length: int) -> Any:
    """Tokenize pre-segmented sentences and align BIO labels to first subwords."""
    def encode_batch(batch: dict[str, Any]) -> dict[str, Any]:
        encoded = tokenizer(batch["tokens"], is_split_into_words=True,
                            truncation=True, max_length=max_length)
        aligned = []
        for row_index, labels in enumerate(batch["ner_tags"]):
            aligned.append(align_labels(encoded.word_ids(batch_index=row_index), labels))
        encoded["labels"] = aligned
        return encoded
    return split.map(encode_batch, batched=True, remove_columns=split.column_names)


def word_feature(words: list[str], index: int) -> dict[str, Any]:
    """Build sparse lexical and local-context features for one token."""
    word = words[index]
    lower = word.lower()
    features: dict[str, Any] = {
        "word.lower": lower, "word.shape": "".join("X" if char.isupper() else "x" if char.islower() else "d" if char.isdigit() else char for char in word),
        "word.isupper": word.isupper(), "word.istitle": word.istitle(),
        "word.has_digit": any(char.isdigit() for char in word),
        "prefix1": lower[:1], "prefix2": lower[:2], "suffix1": lower[-1:], "suffix2": lower[-2:],
        "bias": 1,
    }
    for offset in (-2, -1, 1, 2):
        neighbor_index = index + offset
        if 0 <= neighbor_index < len(words):
            neighbor = words[neighbor_index]
            features[f"{offset}.lower"] = neighbor.lower()
            features[f"{offset}.istitle"] = neighbor.istitle()
        else:
            features[f"{offset}.boundary"] = True
    return features


def flatten_features(split: Any, id_to_label: list[str]) -> tuple[list[dict[str, Any]], list[str], list[tuple[int, int]]]:
    """Flatten sentence token features, labels, and sentence offsets."""
    features: list[dict[str, Any]] = []
    labels: list[str] = []
    offsets: list[tuple[int, int]] = []
    for row in split:
        start = len(labels)
        words = row["tokens"]
        features.extend(word_feature(words, index) for index in range(len(words)))
        labels.extend(id_to_label[int(tag)] for tag in row["ner_tags"])
        offsets.append((start, len(labels)))
    return features, labels, offsets


def restore_sentences(flat_labels: list[str], offsets: list[tuple[int, int]]) -> list[list[str]]:
    """Restore token labels into sentence sequences using recorded offsets."""
    return [flat_labels[start:end] for start, end in offsets]


def run_baseline(train: Any, test: Any, label_names: list[str]) -> tuple[list[list[str]], Any]:
    """Fit a token-feature logistic-regression baseline and predict test sentences."""
    train_x, train_y, _ = flatten_features(train, label_names)
    test_x, test_y, test_offsets = flatten_features(test, label_names)
    model = make_pipeline(DictVectorizer(sparse=True), LogisticRegression(
        max_iter=300, class_weight="balanced", solver="lbfgs"))
    model.fit(train_x, train_y)
    predictions = model.predict(test_x).tolist()
    return restore_sentences(predictions, test_offsets), model


def make_token_metrics(gold: list[list[str]], predicted: list[list[str]], label_names: list[str]) -> dict[str, float]:
    """Calculate token-level micro precision, recall, and F1 excluding outside tags."""
    gold_flat = [label for sentence in gold for label in sentence]
    predicted_flat = [label for sentence in predicted for label in sentence]
    entity_labels = [name for name in label_names if name != "O"]
    true_positive = sum(actual == guess and actual in entity_labels for actual, guess in zip(gold_flat, predicted_flat))
    predicted_count = sum(label in entity_labels for label in predicted_flat)
    gold_count = sum(label in entity_labels for label in gold_flat)
    precision = true_positive / predicted_count if predicted_count else 0.0
    recall = true_positive / gold_count if gold_count else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def entity_spans(words: list[str], tags: list[str]) -> list[dict[str, Any]]:
    """Decode BIO tags into typed, exact-boundary entity spans."""
    spans: list[dict[str, Any]] = []
    current_type: str | None = None
    start = 0
    for index in range(len(tags) + 1):
        tag = tags[index] if index < len(tags) else "O"
        prefix, separator, entity_type = tag.partition("-")
        if not separator or prefix not in {"B", "I"}:
            prefix, entity_type = "O", ""
        if prefix == "B" or (current_type is not None and entity_type != current_type):
            if current_type is not None:
                spans.append({"start": start, "end": index, "type": current_type,
                              "text": " ".join(words[start:index])})
            current_type = entity_type if prefix == "B" else None
            start = index
        elif prefix == "I" and current_type is None:
            current_type, start = entity_type, index
        elif prefix == "O" and current_type is not None:
            spans.append({"start": start, "end": index, "type": current_type,
                          "text": " ".join(words[start:index])})
            current_type = None
    return spans


def score_by_type(gold: list[list[str]], predicted: list[list[str]], words: list[list[str]]) -> pd.DataFrame:
    """Compute exact-span precision, recall, and F1 for each entity type."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for tokens, actual_tags, predicted_tags in zip(words, gold, predicted):
        actual = {(span["start"], span["end"], span["type"]) for span in entity_spans(tokens, actual_tags)}
        guess = {(span["start"], span["end"], span["type"]) for span in entity_spans(tokens, predicted_tags)}
        for entity_type in ENTITY_TYPES:
            counts[entity_type]["tp"] += len(actual & guess & {span for span in actual | guess if span[2] == entity_type})
            counts[entity_type]["fp"] += sum(span[2] == entity_type for span in guess - actual)
            counts[entity_type]["fn"] += sum(span[2] == entity_type for span in actual - guess)
    rows = []
    for entity_type in ENTITY_TYPES:
        true_positive, false_positive, false_negative = (counts[entity_type][key] for key in ("tp", "fp", "fn"))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"entity": entity_type, "precision": precision, "recall": recall, "f1": f1,
                     "support": true_positive + false_negative})
    return pd.DataFrame(rows)


def length_analysis(words: list[list[str]], gold: list[list[str]], predicted: list[list[str]]) -> pd.DataFrame:
    """Calculate exact-match entity metrics by gold entity length."""
    counts: dict[str, Counter[str]] = {bucket[2]: Counter() for bucket in LENGTH_BUCKETS}
    for tokens, actual_tags, predicted_tags in zip(words, gold, predicted):
        actual = entity_spans(tokens, actual_tags)
        guess = entity_spans(tokens, predicted_tags)
        guess_keys = {(item["start"], item["end"], item["type"]) for item in guess}
        for entity in actual:
            bucket_name = next(name for lower, upper, name in LENGTH_BUCKETS if lower <= entity["end"] - entity["start"] <= upper)
            key = (entity["start"], entity["end"], entity["type"])
            counts[bucket_name]["gold"] += 1
            counts[bucket_name]["tp"] += key in guess_keys
        for entity in guess:
            bucket_name = next(name for lower, upper, name in LENGTH_BUCKETS if lower <= entity["end"] - entity["start"] <= upper)
            key = (entity["start"], entity["end"], entity["type"])
            if key not in {(item["start"], item["end"], item["type"]) for item in actual}:
                counts[bucket_name]["fp"] += 1
    rows = []
    for _, _, bucket_name in LENGTH_BUCKETS:
        true_positive, false_positive, gold_count = (counts[bucket_name][key] for key in ("tp", "fp", "gold"))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / gold_count if gold_count else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"entity_length": bucket_name, "precision": precision, "recall": recall, "f1": f1, "support": gold_count})
    return pd.DataFrame(rows)


def error_rows(words: list[list[str]], gold: list[list[str]], predicted: list[list[str]]) -> pd.DataFrame:
    """Collect false positive, false negative, and boundary-error entity examples."""
    errors: list[dict[str, str]] = []
    for tokens, actual_tags, predicted_tags in zip(words, gold, predicted):
        actual_spans = entity_spans(tokens, actual_tags)
        predicted_spans = entity_spans(tokens, predicted_tags)
        actual_keys = {(item["start"], item["end"], item["type"]) for item in actual_spans}
        predicted_keys = {(item["start"], item["end"], item["type"]) for item in predicted_spans}
        for predicted_entity in predicted_spans:
            if (predicted_entity["start"], predicted_entity["end"], predicted_entity["type"]) not in actual_keys:
                same_boundary = any(predicted_entity["start"] == item["start"] and predicted_entity["end"] == item["end"] for item in actual_spans)
                errors.append({"text": " ".join(tokens), "gold": " | ".join(item["text"] + "/" + item["type"] for item in actual_spans),
                               "prediction": predicted_entity["text"] + "/" + predicted_entity["type"],
                               "error_type": "wrong entity type" if same_boundary else "boundary or spurious entity"})
        for actual_entity in actual_spans:
            if (actual_entity["start"], actual_entity["end"], actual_entity["type"]) not in predicted_keys:
                overlaps = any(actual_entity["start"] < item["end"] and item["start"] < actual_entity["end"] for item in predicted_spans)
                errors.append({"text": " ".join(tokens), "gold": actual_entity["text"] + "/" + actual_entity["type"],
                               "prediction": " | ".join(item["text"] + "/" + item["type"] for item in predicted_spans),
                               "error_type": "boundary error" if overlaps else "false negative"})
    return pd.DataFrame(errors)


def tokenization_analysis(split: Any, tokenizer: Any, label_names: list[str], limit: int = 1000) -> pd.DataFrame:
    """Measure word-to-subword expansion by sentence length and entity status."""
    records = []
    for sentence_id, row in enumerate(split.select(range(min(len(split), limit)))):
        encoded = tokenizer(row["tokens"], is_split_into_words=True, truncation=True)
        word_ids = encoded.word_ids()
        piece_counts = Counter(word_id for word_id in word_ids if word_id is not None)
        for word_index, (word, tag_id) in enumerate(zip(row["tokens"], row["ner_tags"])):
            entity = label_names[int(tag_id)] != "O"
            records.append({"sentence_id": sentence_id, "sentence_words": len(row["tokens"]), "sentence_length": "short" if len(row["tokens"]) < 20 else "long",
                            "word": word, "word_length": len(word), "entity": entity,
                            "pieces": piece_counts[word_index], "subword_split": piece_counts[word_index] > 1})
    return pd.DataFrame(records)


def write_predictions(test: Any, label_names: list[str], predicted: list[list[str]], output_dir: Path) -> None:
    """Write human-readable gold and predicted token tags for the shared test set."""
    rows = []
    for row, prediction in zip(test, predicted):
        rows.append({"tokens": row["tokens"], "gold": [label_names[int(tag)] for tag in row["ner_tags"]], "prediction": prediction})
    (output_dir / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_ie_annotation_template(test: Any, predictions: list[list[str]], output_dir: Path) -> None:
    """Prepare a 30-sentence manual IE annotation sheet from held-out data."""
    records = []
    for index, row in enumerate(test.select(range(min(len(test), 30)))):
        spans = entity_spans(row["tokens"], predictions[index])
        extracted = {entity_type: [span["text"] for span in spans if span["type"] == entity_type]
                     for entity_type in ENTITY_TYPES}
        records.append({"sentence_id": index, "text": " ".join(row["tokens"]),
                        "gold_entities_json": "", "predicted_entities_json": json.dumps(extracted, ensure_ascii=False)})
    pd.DataFrame(records).to_csv(output_dir / "ie_annotation_template.csv", index=False)


def train_transformer(train: Any, validation: Any, test: Any, label_names: list[str], settings: Settings) -> tuple[list[list[str]], Any, Any]:
    """Fine-tune a pretrained Transformer for BIO token classification."""
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    model = AutoModelForTokenClassification.from_pretrained(
        settings.model_name, num_labels=len(label_names), id2label=dict(enumerate(label_names)), label2id=label_to_id)
    encoded = {split: tokenize_split(source, tokenizer, settings.max_length)
               for split, source in (("train", train), ("validation", validation), ("test", test))}
    arguments = TrainingArguments(output_dir=str(settings.output_dir / "trainer"),
        num_train_epochs=settings.epochs, per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.batch_size, learning_rate=settings.learning_rate,
        weight_decay=0.01, seed=settings.seed, eval_strategy="epoch", save_strategy="no",
        logging_strategy="epoch", report_to="none")
    trainer = Trainer(model=model, args=arguments, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"], data_collator=DataCollatorForTokenClassification(tokenizer))
    trainer.train()
    output = trainer.predict(encoded["test"])
    predicted_ids = output.predictions.argmax(axis=-1)
    predicted_labels = []
    for row_index, row in enumerate(test):
        labels = []
        previous_word_id = None
        word_count = len(row["tokens"])
        for token_index, word_id in enumerate(tokenizer(row["tokens"], is_split_into_words=True,
                                                       truncation=True, max_length=settings.max_length).word_ids()):
            if word_id is not None and word_id != previous_word_id and word_id < word_count:
                labels.append(label_names[int(predicted_ids[row_index][token_index])])
            previous_word_id = word_id
        labels.extend(["O"] * max(0, word_count - len(labels)))
        predicted_labels.append(labels[:word_count])
    model_dir = settings.output_dir / "checkpoints" / "token_classifier"
    model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(model_dir)
    return predicted_labels, trainer, tokenizer


def save_confusion(gold: list[list[str]], predicted: list[list[str]], label_names: list[str], output_dir: Path) -> None:
    """Save a BIO tag confusion matrix from token-level predictions."""
    actual = [tag for sentence in gold for tag in sentence]
    guesses = [tag for sentence in predicted for tag in sentence]
    matrix = confusion_matrix(actual, guesses, labels=label_names)
    display = ConfusionMatrixDisplay(matrix, display_labels=label_names)
    display.plot(cmap="Blues", xticks_rotation=90, values_format="d")
    display.figure_.set_size_inches(12, 10)
    display.figure_.tight_layout()
    display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)


def run(settings: Settings) -> None:
    """Run shared-split baseline and Transformer experiments and save deliverables."""
    seed_everything(settings.seed)
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    dataset, label_names = load_conll()
    splits = {name: limit_split(dataset[name], settings.train_limit if name == "train" else settings.evaluation_limit, settings.seed)
              for name in SPLIT_NAMES}
    gold = [[label_names[int(tag)] for tag in row["ner_tags"]] for row in splits["test"]]
    words = [row["tokens"] for row in splits["test"]]
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name, use_fast=True)
    token_stats = tokenization_analysis(splits["train"], tokenizer, label_names)
    token_stats.to_csv(settings.output_dir / "tokenization_analysis.csv", index=False)
    sentence_stats = token_stats.groupby(["sentence_id", "sentence_words", "sentence_length"], as_index=False).agg(
        token_count=("pieces", "sum"), subword_split_count=("subword_split", "sum"))
    sentence_stats["R_subwords_per_word"] = sentence_stats["token_count"] / sentence_stats["sentence_words"]
    sentence_stats.to_csv(settings.output_dir / "tokenization_summary.csv", index=False)
    started = time.perf_counter()
    baseline_predictions, _ = run_baseline(splits["train"], splits["test"], label_names)
    baseline_seconds = time.perf_counter() - started
    started = time.perf_counter()
    transformer_predictions, trainer, _ = train_transformer(splits["train"], splits["validation"], splits["test"], label_names, settings)
    transformer_seconds = time.perf_counter() - started
    models = (("baseline", baseline_predictions, baseline_seconds), ("pretrained_transformer", transformer_predictions, transformer_seconds))
    result_rows = []
    type_frames = []
    for model_name, predictions, elapsed in models:
        entity_scores = {"precision": precision_score(gold, predictions), "recall": recall_score(gold, predictions), "f1": f1_score(gold, predictions)}
        token_scores = make_token_metrics(gold, predictions, label_names)
        result_rows.append({"model": model_name, **entity_scores,
                            "token_precision": token_scores["precision"], "token_recall": token_scores["recall"],
                            "token_f1": token_scores["f1"], "seconds": elapsed})
        per_type = score_by_type(gold, predictions, words)
        per_type.insert(0, "model", model_name)
        type_frames.append(per_type)
    pd.DataFrame(result_rows).to_csv(settings.output_dir / "results.csv", index=False)
    pd.concat(type_frames, ignore_index=True).to_csv(settings.output_dir / "entity_metrics.csv", index=False)
    length_frame = length_analysis(words, gold, transformer_predictions)
    length_frame.to_csv(settings.output_dir / "entity_length_analysis.csv", index=False)
    axis = length_frame.plot(x="entity_length", y="f1", kind="bar", ylim=(0, 1), title="Exact entity F1 by gold entity length", legend=False)
    axis.set_xlabel("Gold entity length (words)")
    axis.set_ylabel("Exact span F1")
    axis.figure.tight_layout()
    axis.figure.savefig(settings.output_dir / "entity_length_analysis.png", dpi=160)
    plt.close(axis.figure)
    save_confusion(gold, transformer_predictions, label_names, settings.output_dir)
    errors = error_rows(words, gold, transformer_predictions)
    errors.to_csv(settings.output_dir / "error_analysis_examples.csv", index=False)
    write_predictions(splits["test"], label_names, transformer_predictions, settings.output_dir)
    write_ie_annotation_template(splits["test"], transformer_predictions, settings.output_dir)
    details = {"settings": {**asdict(settings), "output_dir": str(settings.output_dir)}, "labels": label_names,
               "split_sizes": {name: len(split) for name, split in splits.items()},
               "alignment": "Strategy A: supervise first subword only; special and continuation tokens use -100.",
               "trainer_metrics": trainer.state.log_history}
    (settings.output_dir / "run_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(pd.DataFrame(result_rows).to_string(index=False))


def main() -> None:
    """Parse CLI options and launch the NER experiments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--train-limit", type=int, default=0, help="Use zero to train on the full CoNLL train split.")
    parser.add_argument("--evaluation-limit", type=int, default=0, help="Use zero to use full validation and test splits.")
    arguments = parser.parse_args()
    run(Settings(model_name=arguments.model, epochs=arguments.epochs, batch_size=arguments.batch_size,
                 max_length=arguments.max_length, train_limit=arguments.train_limit,
                 evaluation_limit=arguments.evaluation_limit))


if __name__ == "__main__":
    main()
