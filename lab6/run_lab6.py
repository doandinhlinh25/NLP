"""Chạy lại thí nghiệm attention và phân loại IMDb của LAB 06."""

from __future__ import annotations

import csv
import html
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from torch import Tensor, nn
from torch.utils.data import DataLoader

from attention_numpy import self_attention, stable_softmax, verify_examples

LAB5_PATH = Path(__file__).resolve().parents[1] / "lab5"
sys.path.insert(0, str(LAB5_PATH))
from run_experiments import Config as Lab5Config  # noqa: E402
from run_experiments import ReviewDataset, build_vocabulary, prepare_splits, tokenize  # noqa: E402


@dataclass(frozen=True)
class Config:
    """Tập trung cấu hình thí nghiệm và tệp đầu ra."""

    output_path: Path = Path(__file__).resolve().parent
    data_path: Path = Path(__file__).resolve().parents[1] / "data" / "imdb" / "IMDB Dataset.csv"
    seed: int = 42
    max_length: int = 100
    embedding_size: int = 64
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.001
    torch_threads: int = 4
    plot_dpi: int = 150
    probability_threshold: float = 0.5
    pad_id: int = 0
    unknown_id: int = 1
    example_count: int = 3
    snippet_length: int = 280
    temperatures: tuple[float, ...] = (0.5, 1.0, 2.0)
    temperature_scores: tuple[float, ...] = (4.0, 2.0, 0.0)
    sequence_lengths: tuple[int, ...] = (10, 100, 500, 1000, 10000)
    demo_sentence: str = "The movie was surprisingly good despite a very slow beginning"
    demo_tokens: tuple[str, ...] = ("The", "movie", "was", "surprisingly", "good", "despite", "a", "very", "slow", "beginning")
    demo_feature_names: tuple[str, ...] = ("sentiment", "topic", "negativity", "other")
    demo_features: tuple[tuple[float, ...], ...] = (
        (0, 0, 0, 0.2), (0, 2, 0, 0), (0, 0, 0, 0.3), (0.4, 0, 0, 0),
        (2, 0, 0, 0), (0.2, 0, 0, 0), (0, 0, 0, 0.2), (0, 0, 0, 0.3),
        (0, 0, 1.4, 0), (0, 1.4, 0.2, 0),
    )
    sentiment_query: tuple[float, ...] = (2, 0, 0.5, 0)
    topic_query: tuple[float, ...] = (0, 2, 0, 0)
    semantic_sentences: tuple[str, ...] = (
        "The cat that I adopted yesterday is sleeping",
        "The book that I bought yesterday was expensive",
    )
    semantic_features: tuple[tuple[float, ...], ...] = (
        (0.1, 0.1, 0.1), (2.0, 0.0, 0.0), (0.1, 0.1, 0.1),
        (0.0, 0.0, 0.0), (1.5, 0.2, 0.0), (0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1), (1.2, 0.0, 0.8),
    )
    semantic_features_book: tuple[tuple[float, ...], ...] = (
        (0.1, 0.1, 0.1), (0.0, 2.0, 0.0), (0.1, 0.1, 0.1),
        (0.0, 0.0, 0.0), (0.2, 1.5, 0.0), (0.1, 0.1, 0.1),
        (0.1, 0.1, 0.1), (0.0, 1.2, 0.8),
    )
    application_texts: tuple[str, ...] = (
        "I like this movie.",
        "I do not like this movie.",
        "I thought this movie would be bad, but it was actually excellent.",
    )
    application_expected: tuple[str, ...] = ("positive", "negative", "positive")
    cue_tokens: tuple[str, ...] = ("good", "bad", "excellent", "terrible", "not", "but", "like", "love", "boring", "slow")
    label_names: tuple[str, ...] = ("negative", "positive")
    label_display_names: tuple[str, ...] = ("Tiêu cực", "Tích cực")
    query_display_names: tuple[str, ...] = ("cảm xúc", "chủ đề")


class AttentionClassifier(nn.Module):
    """Phân loại bài đánh giá từ masked self-attention và context tổng hợp."""

    def __init__(self, vocabulary_size: int, config: Config) -> None:
        """Tạo embedding, phép chiếu Q/K/V và classifier nhị phân."""
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_size, padding_idx=config.pad_id)
        self.query = nn.Linear(config.embedding_size, config.embedding_size)
        self.key = nn.Linear(config.embedding_size, config.embedding_size)
        self.value = nn.Linear(config.embedding_size, config.embedding_size)
        self.classifier = nn.Linear(config.embedding_size, 1)
        self.scale = config.embedding_size ** 0.5

    def forward(self, tokens: Tensor, lengths: Tensor, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Trả về logits và ma trận attention đã mask nếu được yêu cầu."""
        embedded = self.embedding(tokens)
        queries, keys, values = self.query(embedded), self.key(embedded), self.value(embedded)
        scores = queries @ keys.transpose(1, 2) / self.scale
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        valid = positions[None, :] < lengths[:, None]
        scores = scores.masked_fill(~valid[:, None, :], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        contexts = weights @ values
        pooled = (contexts * valid[:, :, None]).sum(dim=1) / lengths[:, None]
        logits = self.classifier(pooled).squeeze(-1)
        return (logits, weights) if return_attention else logits


def set_seed(config: Config) -> None:
    """Đặt seed cho quá trình tải dữ liệu và huấn luyện."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.torch_threads)


def save_heatmap(weights: np.ndarray, tokens: list[str], path: Path, title: str, config: Config) -> None:
    """Lưu heatmap attention giữa các token kèm nhãn."""
    figure, axis = plt.subplots(figsize=(max(7, len(tokens) * 0.65), max(6, len(tokens) * 0.6)))
    image = axis.imshow(weights, cmap="viridis", vmin=0, vmax=max(float(weights.max()), 0.1))
    axis.set(xticks=np.arange(len(tokens)), yticks=np.arange(len(tokens)), title=title, xlabel="Key / value", ylabel="Query")
    axis.set_xticklabels(tokens, rotation=45, ha="right")
    axis.set_yticklabels(tokens)
    figure.colorbar(image, ax=axis, label="Trọng số attention")
    figure.tight_layout()
    figure.savefig(path, dpi=config.plot_dpi)
    plt.close(figure)


def run_numpy_experiments(config: Config) -> dict:
    """Tạo kết quả query, scaling, temperature và heatmap minh họa."""
    assert verify_examples()
    keys = np.asarray(config.demo_features, dtype=float)
    queries = {"sentiment": np.asarray(config.sentiment_query), "topic": np.asarray(config.topic_query)}
    distributions = {}
    for name, query in queries.items():
        scores = keys @ query / np.sqrt(keys.shape[1])
        weights = stable_softmax(scores)
        distributions[name] = {"scores": scores.tolist(), "weights": weights.tolist()}
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, (name, values), display_name in zip(axes, distributions.items(), config.query_display_names):
        axis.bar(config.demo_tokens, values["weights"])
        axis.set(ylabel="Trọng số attention", title=f"Query: {display_name}", ylim=(0, 1))
    axes[-1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(config.output_path / "query_attention.png", dpi=config.plot_dpi)
    plt.close(figure)
    temperature = {}
    for value in config.temperatures:
        weights = stable_softmax(np.asarray(config.temperature_scores) / value)
        temperature[str(value)] = {"weights": weights.tolist(), "max_weight": float(weights.max()), "entropy_nats": float(-(weights * np.log(weights)).sum())}
    scaled = stable_softmax(np.asarray(config.temperature_scores) / np.sqrt(len(config.demo_feature_names)))
    unscaled = stable_softmax(np.asarray(config.temperature_scores))
    semantic = {}
    for sentence, feature_rows in zip(config.semantic_sentences, (config.semantic_features, config.semantic_features_book)):
        tokens = sentence.split()
        vectors = np.asarray(feature_rows)
        _, _, _, _, weights, _ = self_attention(vectors, np.eye(vectors.shape[1]), np.eye(vectors.shape[1]), np.eye(vectors.shape[1]))
        filename = "attention_heatmap.png" if sentence == config.semantic_sentences[0] else "attention_heatmap_book.png"
        save_heatmap(weights, tokens, config.output_path / filename, f"Self-attention minh họa: {tokens[1]}", config)
        semantic[sentence] = {"tokens": tokens, "weights": weights.tolist(), "source": "Đặc trưng 3 chiều tự thiết kế; chưa huấn luyện"}
    return {"query_experiment": distributions, "temperature": temperature, "scaling": {"unscaled": unscaled.tolist(), "scaled": scaled.tolist()}, "semantic": semantic, "sequence_cost": {str(length): length * length for length in config.sequence_lengths}}


def evaluate(model: AttentionClassifier, loader: DataLoader, config: Config) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Đánh giá classifier và trả về nhãn, dự đoán, xác suất."""
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for tokens, lengths, targets in loader:
            logits = model(tokens, lengths)
            labels.extend(targets.numpy().astype(int))
            probabilities.extend(torch.sigmoid(logits).numpy())
    truth = np.asarray(labels, dtype=int)
    probs = np.asarray(probabilities)
    predicted = (probs >= config.probability_threshold).astype(int)
    metrics = {"accuracy": float(accuracy_score(truth, predicted)), "f1": float(f1_score(truth, predicted)), "confusion_matrix": confusion_matrix(truth, predicted).tolist()}
    return metrics, truth, predicted, probs


def train_classifier(config: Config) -> tuple[AttentionClassifier, dict, dict, dict, dict]:
    """Huấn luyện trên tập con IMDb của Lab 05 và đánh giá tập test."""
    lab5_config = Lab5Config()
    frame = pd.read_csv(config.data_path)
    splits = prepare_splits(frame, lab5_config)
    vocabulary = build_vocabulary(splits["train"]["review"], lab5_config)
    datasets = {name: ReviewDataset(part, vocabulary, config.max_length, lab5_config) for name, part in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train") for name, dataset in datasets.items()}
    model = AttentionClassifier(len(vocabulary) + 2, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        losses = []
        for tokens, lengths, targets in loaders["train"]:
            optimizer.zero_grad()
            loss = criterion(model(tokens, lengths), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _, _, _ = evaluate(model, loaders["validation"], config)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation_accuracy": validation["accuracy"], "validation_f1": validation["f1"]})
    duration = time.perf_counter() - start
    metrics, truth, predictions, probabilities = evaluate(model, loaders["test"], config)
    metrics.update({"model": "Self-Attention", "max_length": config.max_length, "parameters": sum(parameter.numel() for parameter in model.parameters()), "training_time_seconds": duration, "epochs": config.epochs})
    display = ConfusionMatrixDisplay(np.asarray(metrics["confusion_matrix"]), display_labels=config.label_display_names)
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set(xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế")
    display.figure_.tight_layout()
    display.figure_.savefig(config.output_path / "confusion_matrix.png", dpi=config.plot_dpi)
    plt.close(display.figure_)
    torch.save({"model_state": model.state_dict(), "vocabulary": vocabulary, "config": {"embedding_size": config.embedding_size, "max_length": config.max_length}}, config.output_path / "attention_model.pt")
    return model, vocabulary, splits, {"metrics": metrics, "history": history, "truth": truth, "predictions": predictions, "probabilities": probabilities}, datasets


def inspect_text(model: AttentionClassifier, text: str, vocabulary: dict, config: Config) -> dict:
    """Dự đoán một văn bản và tính attention trung bình mỗi token nhận."""
    tokens = tokenize(text)[: config.max_length]
    ids = [vocabulary.get(token, config.unknown_id) for token in tokens]
    if not ids:
        raise ValueError("text has no recognized tokens")
    token_tensor = torch.tensor([ids], dtype=torch.long)
    length_tensor = torch.tensor([len(ids)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        logits, attention = model(token_tensor, length_tensor, return_attention=True)
    probability = float(torch.sigmoid(logits)[0])
    importance = attention[0].mean(dim=0).numpy()
    return {"tokens": tokens, "probability_positive": probability, "prediction": config.label_names[int(probability >= config.probability_threshold)], "mean_attention_received": importance.tolist()}


def save_application(model: AttentionClassifier, vocabulary: dict, config: Config) -> list[dict]:
    """Lưu dự đoán và biểu đồ attention cho các câu thử."""
    examples = []
    for index, (sentence, expected) in enumerate(zip(config.application_texts, config.application_expected), start=1):
        result = inspect_text(model, sentence, vocabulary, config)
        result.update({"text": sentence, "expected": expected, "correct": result["prediction"] == expected})
        examples.append(result)
        figure, axis = plt.subplots(figsize=(max(7, len(result["tokens"]) * 0.7), 3.5))
        axis.bar(result["tokens"], result["mean_attention_received"])
        axis.set(title=f"Ví dụ {index}: P(tích cực)={result['probability_positive']:.3f}", ylabel="Attention trung bình nhận được", xlabel="Token")
        axis.tick_params(axis="x", rotation=40)
        figure.tight_layout()
        figure.savefig(config.output_path / f"application_attention_{index}.png", dpi=config.plot_dpi)
        plt.close(figure)
    return examples


def save_error_analysis(model: AttentionClassifier, vocabulary: dict, splits: dict, evaluation: dict, config: Config) -> None:
    """Ghi ba dự đoán đúng và ba dự đoán sai trên tập test."""
    truth, predicted, probabilities = evaluation["truth"], evaluation["predictions"], evaluation["probabilities"]
    selected = np.concatenate((np.flatnonzero(truth == predicted)[: config.example_count], np.flatnonzero(truth != predicted)[: config.example_count]))
    lines = ["# Phân tích lỗi", "", "Sáu bài đánh giá IMDb thuộc tập test; xác suất tính cho nhãn tích cực. Attention là phân phối để quan sát, không phải giải thích nhân quả.", "", "| Kết quả | Nhãn thật | Dự đoán | P(tích cực) | Số token | Token nhận attention cao nhất | Yếu tố có thể ảnh hưởng | Trích đoạn bài đánh giá |", "|---|---|---|---:|---:|---|---|---|"]
    for index in selected:
        text = str(splits["test"].iloc[index]["review"])
        result = inspect_text(model, text, vocabulary, config)
        top = np.argsort(result["mean_attention_received"])[-3:][::-1]
        prominent = ", ".join(result["tokens"][position] for position in top)
        factors = []
        if "not" in result["tokens"] or "never" in result["tokens"]:
            factors.append("phủ định")
        if "but" in result["tokens"] or "however" in result["tokens"]:
            factors.append("tương phản")
        if len(tokenize(text)) > config.max_length:
            factors.append("cắt ngắn")
        if not factors:
            factors.append("diễn đạt mơ hồ / từ vựng")
        excerpt = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", text)))[: config.snippet_length].replace("|", "\\|")
        lines.append(f"| {'Đúng' if truth[index] == predicted[index] else 'Sai'} | {config.label_display_names[truth[index]]} | {config.label_display_names[predicted[index]]} | {probabilities[index]:.3f} | {len(tokenize(text))} | {prominent} | {', '.join(factors)} | {excerpt} |")
    lines.extend(["", "Các token được nêu là những cột có attention trung bình cao nhất sau khi lấy trung bình qua mọi vị trí query hợp lệ. Phép lấy trung bình có thể che khuất hành vi của từng query. Phủ định, tương phản, cắt ngắn và từ vựng chỉ là giả thuyết cần đối chiếu với văn bản và dự đoán; bảng này chưa xác định nguyên nhân gây lỗi."])
    (config.output_path / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")


def save_results(config: Config, evaluation: dict) -> None:
    """Ghi bảng tóm tắt để so sánh LSTM và attention."""
    with (LAB5_PATH / "results.csv").open(newline="", encoding="utf-8") as source:
        lstm = next(row for row in csv.DictReader(source) if row["model"] == "LSTM" and int(row["max_length"]) == config.max_length)
    attention = evaluation["metrics"]
    fields = ("model", "accuracy", "f1", "training_time_seconds", "parameters", "max_length", "epochs")
    with (config.output_path / "results.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerow({field: lstm[field] for field in fields})
        writer.writerow({field: attention[field] for field in fields})


def main() -> None:
    """Chạy toàn bộ thí nghiệm LAB 06 và lưu kết quả."""
    config = Config()
    set_seed(config)
    numpy_results = run_numpy_experiments(config)
    model, vocabulary, splits, evaluation, _ = train_classifier(config)
    applications = save_application(model, vocabulary, config)
    save_error_analysis(model, vocabulary, splits, evaluation, config)
    save_results(config, evaluation)
    details = {"config": {"seed": config.seed, "max_length": config.max_length, "embedding_size": config.embedding_size, "batch_size": config.batch_size, "epochs": config.epochs}, "numpy_experiments": numpy_results, "classification": {"metrics": evaluation["metrics"], "history": evaluation["history"], "applications": applications}}
    (config.output_path / "experiment_details.json").write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(details["classification"]["metrics"], indent=2))


if __name__ == "__main__":
    main()
