"""Reproduce the IMDb sequence classification experiments for LAB 05."""

from __future__ import annotations

import csv
import html
import json
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset

from rnn_cell import verify_against_torch


@dataclass(frozen=True)
class Config:
    """Keep dataset, model, and experiment settings in one place."""

    data_path: Path = Path(__file__).resolve().parents[1] / "data" / "imdb" / "IMDB Dataset.csv"
    output_path: Path = Path(__file__).resolve().parent
    seed: int = 42
    train_count: int = 5000
    validation_count: int = 1000
    test_count: int = 1000
    vocabulary_size: int = 12000
    embedding_size: int = 64
    hidden_size: int = 64
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.001
    lengths: tuple[int, ...] = (50, 100, 200)
    primary_length: int = 100
    pad_id: int = 0
    unknown_id: int = 1
    test_fraction: float = 0.2
    validation_fraction: float = 0.25
    max_workers: int = 0
    torch_threads: int = 4
    histogram_bins: int = 60
    plot_dpi: int = 140
    examples_per_outcome: int = 3
    error_snippet_length: int = 420


TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|\d+")
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
MODEL_NAMES = ("RNN", "LSTM", "GRU")
LABEL_NAMES = ("negative", "positive")
LONG_RANGE_TEMPLATES = (
    "{cue} The movie had several scenes and a long story with many characters and turns.",
    "The movie had several scenes and a long story. {cue} It also had many characters and turns.",
    "The movie had several scenes and a long story with many characters and turns. {cue}",
)
LONG_RANGE_CUES = ("Ultimately it was excellent.", "Ultimately it was terrible.")
APPLICATION_REVIEWS = {
    "short": "I really enjoyed this movie.",
    "long": "The film begins slowly with several unnecessary scenes, but the acting becomes powerful and the ending is wonderful. I enjoyed the movie overall.",
    "negation": "I did not like this movie.",
    "contrast": "I thought the movie would be terrible, but it was surprisingly good.",
}
ORDER_PROBE_REVIEWS = (
    "this movie is not good",
    "this movie is good not",
    "the film is not bad",
    "the film is bad not",
)


def tokenize(text: str) -> list[str]:
    """Lowercase and tokenize a review while retaining negation words."""
    normalized = HTML_TAG_PATTERN.sub(" ", html.unescape(str(text))).lower()
    return TOKEN_PATTERN.findall(normalized)


def inspect_dataset(frame: pd.DataFrame, config: Config) -> dict:
    """Summarize the complete source dataset before any sample is selected."""
    lengths = frame["review"].fillna("").map(lambda value: len(tokenize(value)))
    summary = {
        "samples": int(len(frame)),
        "classes": frame["sentiment"].value_counts(dropna=False).to_dict(),
        "minimum_tokens": int(lengths.min()),
        "median_tokens": float(lengths.median()),
        "mean_tokens": float(lengths.mean()),
        "maximum_tokens": int(lengths.max()),
        "quantiles": {str(quantile): float(lengths.quantile(quantile)) for quantile in (0.5, 0.75, 0.9, 0.95)},
        "duplicate_documents": int(frame["review"].duplicated().sum()),
        "missing_text": int(frame["review"].isna().sum()),
        "missing_labels": int(frame["sentiment"].isna().sum()),
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(lengths.clip(upper=lengths.quantile(0.99)), bins=config.histogram_bins)
    axis.set(xlabel="Tokens per review (clipped at 99th percentile)", ylabel="Reviews", title="IMDb sequence lengths")
    figure.tight_layout()
    figure.savefig(config.output_path / "sequence_lengths.png", dpi=config.plot_dpi)
    plt.close(figure)
    return summary


def prepare_splits(frame: pd.DataFrame, config: Config) -> dict[str, pd.DataFrame]:
    """Split before sampling, then take balanced reproducible subsets."""
    clean = frame.dropna(subset=["review", "sentiment"]).drop_duplicates(subset="review")
    train_validation, test = train_test_split(clean, test_size=config.test_fraction, stratify=clean["sentiment"], random_state=config.seed)
    train, validation = train_test_split(train_validation, test_size=config.validation_fraction, stratify=train_validation["sentiment"], random_state=config.seed)
    counts = {"train": config.train_count, "validation": config.validation_count, "test": config.test_count}
    sources = {"train": train, "validation": validation, "test": test}
    return {
        name: source.groupby("sentiment", group_keys=False).sample(n=count // len(LABEL_NAMES), random_state=config.seed).sample(frac=1, random_state=config.seed).reset_index(drop=True)
        for name, source in sources.items()
        for count in (counts[name],)
    }


def build_vocabulary(texts: pd.Series, config: Config) -> dict[str, int]:
    """Fit token frequencies using training reviews only."""
    counts = Counter(token for text in texts for token in tokenize(text))
    return {token: index for index, (token, _) in enumerate(counts.most_common(config.vocabulary_size - 2), start=2)}


def encode(text: str, vocabulary: dict[str, int], max_length: int, config: Config) -> tuple[list[int], int]:
    """Right pad or truncate a review and return its effective length."""
    tokens = tokenize(text)
    ids = [vocabulary.get(token, config.unknown_id) for token in tokens[:max_length]]
    length = max(1, len(ids))
    return (ids + [config.pad_id] * (max_length - len(ids)), length)


class ReviewDataset(Dataset):
    """Expose encoded IMDb reviews and binary sentiment labels to PyTorch."""

    def __init__(self, frame: pd.DataFrame, vocabulary: dict[str, int], max_length: int, config: Config) -> None:
        """Encode all reviews once for a fixed sequence length."""
        encoded = [encode(text, vocabulary, max_length, config) for text in frame["review"]]
        self.tokens = torch.tensor([item[0] for item in encoded], dtype=torch.long)
        self.lengths = torch.tensor([item[1] for item in encoded], dtype=torch.long)
        self.labels = torch.tensor((frame["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy(), dtype=torch.float32)

    def __len__(self) -> int:
        """Return the number of encoded reviews."""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return token IDs, effective length, and sentiment label."""
        return self.tokens[index], self.lengths[index], self.labels[index]


class SequenceClassifier(nn.Module):
    """Classify reviews using an embedding and one recurrent layer."""

    def __init__(self, architecture: str, vocabulary_size: int, config: Config) -> None:
        """Build the selected recurrent architecture with shared dimensions."""
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_size, padding_idx=config.pad_id)
        recurrent_type = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[architecture]
        self.recurrent = recurrent_type(config.embedding_size, config.hidden_size, batch_first=True)
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        """Classify from the final real token rather than the padded tail."""
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, state = self.recurrent(packed)
        hidden = state[0] if isinstance(state, tuple) else state
        return self.classifier(hidden[-1]).squeeze(-1)


def score(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return positive-class precision, recall, F1, and accuracy."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": accuracy_score(labels, predictions), "precision": precision, "recall": recall, "f1": f1}


def evaluate(model: SequenceClassifier, loader: DataLoader, criterion: nn.Module) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate without gradients and preserve probabilities for analysis."""
    model.eval()
    losses = []
    labels = []
    probabilities = []
    with torch.no_grad():
        for tokens, lengths, target in loader:
            logits = model(tokens, lengths)
            losses.append(float(criterion(logits, target)))
            labels.extend(target.numpy())
            probabilities.extend(torch.sigmoid(logits).numpy())
    labels_array = np.asarray(labels, dtype=int)
    probability_array = np.asarray(probabilities)
    predictions = (probability_array >= 0.5).astype(int)
    metrics = score(labels_array, predictions)
    metrics["loss"] = float(np.mean(losses))
    return metrics, labels_array, predictions, probability_array


def train_model(architecture: str, max_length: int, splits: dict[str, pd.DataFrame], vocabulary: dict[str, int], config: Config) -> tuple[SequenceClassifier, dict, list[dict]]:
    """Train a recurrent classifier and return test metrics and learning history."""
    torch.manual_seed(config.seed)
    datasets = {name: ReviewDataset(frame, vocabulary, max_length, config) for name, frame in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train", num_workers=config.max_workers) for name, dataset in datasets.items()}
    model = SequenceClassifier(architecture, len(vocabulary) + 2, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses = []
        for tokens, lengths, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(tokens, lengths), targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation_metrics, _, _, _ = evaluate(model, loaders["validation"], criterion)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss": validation_metrics["loss"], "validation_f1": validation_metrics["f1"]})
    duration = time.perf_counter() - start
    test_metrics, labels, predictions, probabilities = evaluate(model, loaders["test"], criterion)
    tokens_total = sum(len(tokenize(text)) for text in splits["train"]["review"])
    tokens_retained = sum(min(len(tokenize(text)), max_length) for text in splits["train"]["review"])
    result = {
        "model": architecture, "max_length": max_length, "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "training_time_seconds": duration, "epochs": config.epochs, "vocabulary_size": len(vocabulary) + 2,
        "tokens_retained_percent": 100 * tokens_retained / tokens_total,
        "validation_loss": history[-1]["validation_loss"], **test_metrics,
    }
    return model, {"result": result, "labels": labels, "predictions": predictions, "probabilities": probabilities}, history


def run_baseline(splits: dict[str, pd.DataFrame], config: Config) -> tuple[dict, np.ndarray, TfidfVectorizer, LogisticRegression]:
    """Fit TF-IDF and logistic regression on the same sampled split."""
    vectorizer = TfidfVectorizer(max_features=config.vocabulary_size, tokenizer=tokenize, token_pattern=None, lowercase=False)
    start = time.perf_counter()
    train_features = vectorizer.fit_transform(splits["train"]["review"])
    classifier = LogisticRegression(max_iter=1000, random_state=config.seed)
    classifier.fit(train_features, splits["train"]["sentiment"])
    duration = time.perf_counter() - start
    test_features = vectorizer.transform(splits["test"]["review"])
    predictions = classifier.predict(test_features)
    labels = (splits["test"]["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy()
    predicted_binary = (predictions == LABEL_NAMES[1]).astype(int)
    return {
        "model": "TF-IDF + LR", "max_length": "", "parameters": int(train_features.shape[1] + 1),
        "training_time_seconds": duration, "epochs": "", "vocabulary_size": len(vectorizer.vocabulary_),
        "tokens_retained_percent": 100.0, "validation_loss": "", "loss": "", **score(labels, predicted_binary),
    }, predicted_binary, vectorizer, classifier


def probe_order(model: SequenceClassifier, vocabulary: dict[str, int], vectorizer: TfidfVectorizer, classifier: LogisticRegression, config: Config) -> list[dict]:
    """Compare two classifiers on reviews with identical word counts but changed order."""
    rows = []
    for review in ORDER_PROBE_REVIEWS:
        baseline_probability = classifier.predict_proba(vectorizer.transform([review]))[0, list(classifier.classes_).index(LABEL_NAMES[1])]
        rows.append({"review": review, "tfidf_positive_probability": float(baseline_probability), "gru_positive_probability": predict_text(model, review, vocabulary, config)})
    return rows


def predict_text(model: SequenceClassifier, text: str, vocabulary: dict[str, int], config: Config) -> float:
    """Return the positive sentiment probability for one raw review."""
    ids, length = encode(text, vocabulary, config.primary_length, config)
    model.eval()
    with torch.no_grad():
        return float(torch.sigmoid(model(torch.tensor([ids]), torch.tensor([length]))[0]))


def write_figures(results: list[dict], histories: dict[str, list[dict]], labels: np.ndarray, predictions: np.ndarray, config: Config) -> None:
    """Save learning curves, confusion matrix, and length trade-off charts."""
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, history in histories.items():
        if name in MODEL_NAMES:
            axis.plot([row["epoch"] for row in history], [row["train_loss"] for row in history], marker="o", label=f"{name} train")
            axis.plot([row["epoch"] for row in history], [row["validation_loss"] for row in history], marker="x", label=f"{name} validation")
    axis.set(xlabel="Epoch", ylabel="BCE loss", title="Learning curves at 100 tokens")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(config.output_path / "learning_curves.png", dpi=config.plot_dpi)
    plt.close(figure)
    matrix = confusion_matrix(labels, predictions)
    display = ConfusionMatrixDisplay(matrix, display_labels=LABEL_NAMES)
    display.plot(cmap="Blues", values_format="d")
    plt.title("Selected GRU test predictions")
    plt.tight_layout()
    plt.savefig(config.output_path / "confusion_matrix.png", dpi=config.plot_dpi)
    plt.close()
    length_rows = [row for row in results if row["model"] == "GRU"]
    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].plot([row["max_length"] for row in length_rows], [row["f1"] for row in length_rows], marker="o")
    axes[1].plot([row["max_length"] for row in length_rows], [row["training_time_seconds"] for row in length_rows], marker="o")
    axes[0].set(xlabel="Max length", ylabel="Test F1")
    axes[1].set(xlabel="Max length", ylabel="Training seconds")
    figure.tight_layout()
    figure.savefig(config.output_path / "length_tradeoff.png", dpi=config.plot_dpi)
    plt.close(figure)


def write_analysis(splits: dict[str, pd.DataFrame], predictions: np.ndarray, probabilities: np.ndarray, config: Config) -> None:
    """Save at least three correct and three incorrect test cases for review."""
    test = splits["test"]
    labels = (test["sentiment"] == LABEL_NAMES[1]).astype(int).to_numpy()
    rows = []
    for outcome, indices in (("correct", np.flatnonzero(labels == predictions)), ("incorrect", np.flatnonzero(labels != predictions))):
        for index in indices[:config.examples_per_outcome]:
            text = test.iloc[index]["review"]
            tokens = tokenize(text)
            reasons = []
            if any(token in tokens for token in ("not", "never", "no")):
                reasons.append("negation")
            if any(token in tokens for token in ("but", "however", "although")):
                reasons.append("contrast")
            if len(tokens) > config.primary_length:
                reasons.append("truncation possible")
            if not reasons:
                reasons.append("lexical/context ambiguity possible")
            rows.append((outcome, LABEL_NAMES[labels[index]], LABEL_NAMES[predictions[index]], float(probabilities[index]), len(tokens), ", ".join(reasons), text[:config.error_snippet_length].replace("|", "\\|")))
    lines = ["# Error analysis", "", "The observations below use held-out test reviews. The possible causes are hypotheses, not causal proof.", "", "| Outcome | True | Predicted | P(positive) | Tokens | Possible factors | Review excerpt |", "|---|---|---|---:|---:|---|---|"]
    lines.extend(f"| {outcome} | {truth} | {prediction} | {probability:.3f} | {length} | {reason} | {excerpt} |" for outcome, truth, prediction, probability, length, reason, excerpt in rows)
    lines.extend(["", "A mistake is not automatically caused by sequence modeling. Vocabulary coverage, label ambiguity, review length, truncation, and limited training data can also matter."])
    (config.output_path / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """Run inspection, comparable experiments, figures, and saved analyses."""
    config = Config()
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)
    frame = pd.read_csv(config.data_path)
    summary = inspect_dataset(frame, config)
    splits = prepare_splits(frame, config)
    vocabulary = build_vocabulary(splits["train"]["review"], config)
    baseline, _, vectorizer, baseline_classifier = run_baseline(splits, config)
    results = [baseline]
    histories = {}
    selected_model = None
    selected_evaluation = None
    for architecture in MODEL_NAMES:
        model, evaluation, history = train_model(architecture, config.primary_length, splits, vocabulary, config)
        results.append(evaluation["result"])
        histories[architecture] = history
        print(architecture, config.primary_length, evaluation["result"]["f1"], flush=True)
        if architecture == "GRU":
            selected_model = model
            selected_evaluation = evaluation
    for max_length in config.lengths:
        if max_length != config.primary_length:
            _, evaluation, history = train_model("GRU", max_length, splits, vocabulary, config)
            results.append(evaluation["result"])
            histories[f"GRU_{max_length}"] = history
            print("GRU", max_length, evaluation["result"]["f1"], flush=True)
    assert selected_model is not None and selected_evaluation is not None
    write_figures(results, histories, selected_evaluation["labels"], selected_evaluation["predictions"], config)
    write_analysis(splits, selected_evaluation["predictions"], selected_evaluation["probabilities"], config)
    with (config.output_path / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    probes = {
        "application": {name: predict_text(selected_model, review, vocabulary, config) for name, review in APPLICATION_REVIEWS.items()},
        "positions": {str(position): {cue: predict_text(selected_model, template.format(cue=cue), vocabulary, config) for cue in LONG_RANGE_CUES} for position, template in enumerate(LONG_RANGE_TEMPLATES)},
        "order": probe_order(selected_model, vocabulary, vectorizer, baseline_classifier, config),
    }
    metadata = {"dataset": summary, "split_sizes": {name: len(split) for name, split in splits.items()}, "config": {"seed": config.seed, "epochs": config.epochs, "batch_size": config.batch_size, "embedding_size": config.embedding_size, "hidden_size": config.hidden_size, "learning_rate": config.learning_rate, "primary_length": config.primary_length}, "cell_matches_torch": verify_against_torch(), "history": histories, "probes": probes, "confusion_matrix": confusion_matrix(selected_evaluation["labels"], selected_evaluation["predictions"]).tolist()}
    (config.output_path / "experiment_details.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    torch.save({"model": selected_model.state_dict(), "vocabulary": vocabulary, "config": metadata["config"]}, config.output_path / "gru_model.pt")
    print("Saved lab5 results", flush=True)


if __name__ == "__main__":
    main()
