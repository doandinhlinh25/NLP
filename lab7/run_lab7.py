"""Run the reproducible calculations, experiments, and IMDb application for LAB 07."""

from __future__ import annotations

import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch import Tensor, nn
from torch.utils.data import DataLoader

LAB_DIRECTORY = Path(__file__).resolve().parent
REPOSITORY_DIRECTORY = LAB_DIRECTORY.parent
LAB5_DIRECTORY = REPOSITORY_DIRECTORY / "lab5"
sys.path.insert(0, str(LAB5_DIRECTORY))
from run_experiments import Config as Lab5Config  # noqa: E402
from run_experiments import ReviewDataset, build_vocabulary, prepare_splits, tokenize  # noqa: E402

from positional_encoding import sinusoidal_positional_encoding  # noqa: E402
from transformer_block import TransformerBlock  # noqa: E402


@dataclass(frozen=True)
class LabConfig:
    """Collect dataset, training, architecture, and output settings."""

    data_path: Path = REPOSITORY_DIRECTORY / "data" / "imdb" / "IMDB Dataset.csv"
    output_directory: Path = LAB_DIRECTORY
    seed: int = 42
    model_dimension: int = 128
    feedforward_dimension: int = 256
    vocabulary_size: int = 12000
    batch_size: int = 64
    epochs: int = 2
    learning_rate: float = 0.001
    primary_length: int = 100
    head_counts: tuple[int, ...] = (1, 2, 4, 8)
    sequence_lengths: tuple[int, ...] = (32, 64, 128, 256)
    dropout: float = 0.1
    thread_count: int = 4
    plot_dpi: int = 150
    inference_examples: tuple[str, ...] = (
        "I really enjoyed this movie.",
        "The film begins slowly, but the acting becomes powerful and the ending is wonderful.",
        "I did not like this movie.",
        "I expected a terrible film, but it was surprisingly good.",
        "The movie was not bad, although it was not especially memorable.",
    )


class TransformerClassifier(nn.Module):
    """Classify padded token sequences with learned embeddings and one encoder block."""

    def __init__(self, vocabulary_size: int, number_of_heads: int, config: LabConfig, padding_id: int = 0, pooling_strategy: str = "mean") -> None:
        """Create token embeddings, a Transformer block, and a binary classifier."""
        super().__init__()
        self.padding_id = padding_id
        self.model_dimension = config.model_dimension
        if pooling_strategy not in {"mean", "first_token"}:
            raise ValueError("pooling_strategy must be 'mean' or 'first_token'")
        self.pooling_strategy = pooling_strategy
        self.embedding = nn.Embedding(vocabulary_size, config.model_dimension, padding_idx=padding_id)
        self.block = TransformerBlock(config.model_dimension, number_of_heads, config.feedforward_dimension, config.dropout)
        self.classifier = nn.Linear(config.model_dimension, 1)

    def forward(self, tokens: Tensor, lengths: Tensor, *, return_attention: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """Return sentiment logits and optionally the block's attention maps."""
        padding_mask = tokens.eq(self.padding_id)
        embeddings = self.embedding(tokens)
        embeddings = embeddings + sinusoidal_positional_encoding(tokens.shape[1], self.model_dimension, device=tokens.device)
        if return_attention:
            encoded, attention = self.block(embeddings, padding_mask, return_attention=True)
        else:
            encoded = self.block(embeddings, padding_mask)
        if self.pooling_strategy == "first_token":
            pooled = encoded[:, 0]
        else:
            valid_tokens = (~padding_mask).unsqueeze(-1)
            pooled = (encoded * valid_tokens).sum(dim=1) / lengths.clamp_min(1).unsqueeze(-1)
        logits = self.classifier(pooled).squeeze(-1)
        if return_attention:
            return logits, attention
        return logits


class LSTMClassifier(nn.Module):
    """Classify the same padded IMDb sequences with a unidirectional LSTM."""

    def __init__(self, vocabulary_size: int, config: LabConfig, padding_id: int = 0) -> None:
        """Create an embedding, a single LSTM layer, and a binary classifier."""
        super().__init__()
        self.padding_id = padding_id
        self.embedding = nn.Embedding(vocabulary_size, config.model_dimension, padding_idx=padding_id)
        self.recurrent = nn.LSTM(config.model_dimension, config.model_dimension, batch_first=True)
        self.classifier = nn.Linear(config.model_dimension, 1)

    def forward(self, tokens: Tensor, lengths: Tensor) -> Tensor:
        """Return a sentiment logit from the last non-padding LSTM state."""
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.recurrent(packed)
        return self.classifier(hidden[-1]).squeeze(-1)


def score_predictions(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return accuracy and positive-class precision, recall, and F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(labels, predictions)), "precision": float(precision), "recall": float(recall), "f1": float(f1)}


def evaluate_model(model: TransformerClassifier, loader: DataLoader, criterion: nn.Module) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate a classifier and preserve labels, predictions, and probabilities."""
    model.eval()
    batch_losses: list[float] = []
    all_labels: list[float] = []
    all_probabilities: list[float] = []
    with torch.no_grad():
        for tokens, lengths, labels in loader:
            logits = model(tokens, lengths)
            batch_losses.append(float(criterion(logits, labels)))
            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(torch.sigmoid(logits).numpy().tolist())
    labels_array = np.asarray(all_labels, dtype=int)
    probabilities = np.asarray(all_probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    metrics = score_predictions(labels_array, predictions)
    metrics["loss"] = float(np.mean(batch_losses))
    return metrics, labels_array, predictions, probabilities


def train_model(number_of_heads: int, max_length: int, splits: dict[str, pd.DataFrame], vocabulary: dict[str, int], config: LabConfig, pooling_strategy: str = "mean") -> tuple[TransformerClassifier, dict[str, float], list[dict[str, float]], float, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Train one encoder classifier and return its test metrics and learning history."""
    torch.manual_seed(config.seed)
    datasets = {name: ReviewDataset(frame, vocabulary, max_length, Lab5Config()) for name, frame in splits.items()}
    loaders = {
        name: DataLoader(dataset, batch_size=config.batch_size, shuffle=(name == "train"), num_workers=0)
        for name, dataset in datasets.items()
    }
    model = TransformerClassifier(len(vocabulary) + 2, number_of_heads, config, pooling_strategy=pooling_strategy)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float]] = []
    start_time = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for tokens, lengths, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, lengths), labels)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        validation_metrics, _, _, _ = evaluate_model(model, loaders["validation"], criterion)
        history.append({"epoch": float(epoch), "training_loss": float(np.mean(train_losses)), "validation_loss": validation_metrics["loss"], "validation_accuracy": validation_metrics["accuracy"], "validation_f1": validation_metrics["f1"]})
    elapsed = time.perf_counter() - start_time
    test_metrics, labels, predictions, probabilities = evaluate_model(model, loaders["test"], criterion)
    return model, test_metrics, history, elapsed, (labels, predictions, probabilities)


def train_lstm(max_length: int, splits: dict[str, pd.DataFrame], vocabulary: dict[str, int], config: LabConfig) -> tuple[dict[str, float], float, int]:
    """Train the dimension-matched LSTM baseline on the shared IMDb splits."""
    torch.manual_seed(config.seed)
    datasets = {name: ReviewDataset(frame, vocabulary, max_length, Lab5Config()) for name, frame in splits.items()}
    loaders = {name: DataLoader(dataset, batch_size=config.batch_size, shuffle=name == "train", num_workers=0) for name, dataset in datasets.items()}
    model = LSTMClassifier(len(vocabulary) + 2, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    start_time = time.perf_counter()
    for _ in range(config.epochs):
        model.train()
        for tokens, lengths, labels in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tokens, lengths), labels)
            loss.backward()
            optimizer.step()
    elapsed = time.perf_counter() - start_time
    model.eval()
    targets: list[float] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for tokens, lengths, labels in loaders["test"]:
            probabilities.extend(torch.sigmoid(model(tokens, lengths)).numpy().tolist())
            targets.extend(labels.numpy().tolist())
    label_array = np.asarray(targets, dtype=int)
    probability_array = np.asarray(probabilities)
    metrics = score_predictions(label_array, (probability_array >= 0.5).astype(int))
    return metrics, elapsed, sum(parameter.numel() for parameter in model.parameters())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write experiment records with a stable union of columns."""
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def create_positional_experiment(config: LabConfig) -> dict[str, object]:
    """Compare permutation behavior with and without positional information."""
    generator = torch.Generator().manual_seed(config.seed)
    embedding_table = torch.randn(5, config.model_dimension, generator=generator)
    first_tokens = torch.tensor([2, 3, 4])
    permuted_tokens = torch.tensor([4, 3, 2])
    first_embedding = embedding_table[first_tokens].unsqueeze(0)
    permuted_embedding = embedding_table[permuted_tokens].unsqueeze(0)
    def attention_pool(representation: Tensor) -> Tensor:
        """Apply unprojected scaled self-attention and mean-pool its output."""
        scores = torch.matmul(representation, representation.transpose(-2, -1)) / math.sqrt(config.model_dimension)
        return torch.matmul(torch.softmax(scores, dim=-1), representation).mean(dim=1)

    no_position_first = attention_pool(first_embedding)
    no_position_permuted = attention_pool(permuted_embedding)
    position = sinusoidal_positional_encoding(len(first_tokens), config.model_dimension)
    with_position_first = attention_pool(first_embedding + position)
    with_position_permuted = attention_pool(permuted_embedding + position)
    result = {
        "first_sequence": ["I", "love", "NLP"],
        "permuted_sequence": ["NLP", "love", "I"],
        "no_position_max_absolute_difference": float((no_position_first - no_position_permuted).abs().max()),
        "with_position_max_absolute_difference": float((with_position_first - with_position_permuted).abs().max()),
        "positions_0_to_2_d4": sinusoidal_positional_encoding(3, 4).squeeze(0).tolist(),
    }
    role_first = embedding_table[torch.tensor([0, 1, 2])].unsqueeze(0)
    role_permuted = embedding_table[torch.tensor([2, 1, 0])].unsqueeze(0)
    result["subject_object_permutation"] = {
        "first_sequence": ["dog", "bites", "man"],
        "permuted_sequence": ["man", "bites", "dog"],
        "no_position_max_absolute_difference": float((attention_pool(role_first) - attention_pool(role_permuted)).abs().max()),
        "with_position_max_absolute_difference": float((attention_pool(role_first + position) - attention_pool(role_permuted + position)).abs().max()),
    }
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    difference_without = (no_position_first - no_position_permuted).abs().squeeze(0).numpy()
    difference_with = (with_position_first - with_position_permuted).abs().squeeze(0).numpy()
    axes[0].bar(range(len(difference_without)), difference_without)
    axes[0].set_title("Pooled self-attention, no PE")
    axes[1].bar(range(len(difference_with)), difference_with)
    axes[1].set_title("Pooled self-attention + PE")
    shared_limit = float(max(difference_with.max(), difference_without.max()) * 1.05)
    for axis in axes:
        axis.set(xlabel="Feature", ylabel="Absolute pooled-output difference")
        axis.set_ylim(0, shared_limit)
    figure.tight_layout()
    figure.savefig(config.output_directory / "positional_encoding.png", dpi=config.plot_dpi)
    plt.close(figure)
    return result


def write_attention_plot(model: TransformerClassifier, vocabulary: dict[str, int], config: LabConfig) -> dict[str, object]:
    """Visualize each head's attention on a short, hand-written sentence."""
    sentence = "the movie was surprisingly good despite the slow beginning"
    words = tokenize(sentence)
    ids = [vocabulary.get(word, 1) for word in words]
    token_tensor = torch.tensor([ids], dtype=torch.long)
    length_tensor = torch.tensor([len(ids)], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        _, attention = model(token_tensor, length_tensor, return_attention=True)
    maps = attention[0, :, :len(words), :len(words)].cpu().numpy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), squeeze=False)
    for head_index, axis in enumerate(axes.flat):
        image = axis.imshow(maps[head_index], cmap="viridis", vmin=0, vmax=float(maps.max()))
        axis.set(title=f"Head {head_index + 1}", xticks=range(len(words)), yticks=range(len(words)), xticklabels=words, yticklabels=words)
        axis.tick_params(axis="x", labelrotation=45, labelsize=8)
        axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.82, label="Attention weight")
    figure.savefig(config.output_directory / "attention_heads.png", dpi=config.plot_dpi, bbox_inches="tight")
    plt.close(figure)
    return {"sentence": sentence, "tokens": words, "head_attention_maps": maps.tolist()}


def run(config: LabConfig) -> None:
    """Run calculation demos, IMDb experiments, model evaluation, and report artifacts."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(config.thread_count)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    lab5_config = Lab5Config()
    lab5_config = Lab5Config(vocabulary_size=config.vocabulary_size, train_count=5000, validation_count=1000, test_count=1000, seed=config.seed)
    frame = pd.read_csv(config.data_path)
    splits = prepare_splits(frame, lab5_config)
    vocabulary = build_vocabulary(splits["train"]["review"], lab5_config)
    positional_results = create_positional_experiment(config)
    experiment_rows: list[dict[str, object]] = []
    details: dict[str, object] = {"seed": config.seed, "dataset_sizes": {name: len(part) for name, part in splits.items()}, "vocabulary_size": len(vocabulary), "positional_experiment": positional_results, "heads": {}, "sequence_lengths": {}, "histories": {}}
    primary_model: TransformerClassifier | None = None
    primary_outcome: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    primary_history: list[dict[str, float]] | None = None
    primary_metrics: dict[str, float] | None = None
    primary_elapsed = 0.0
    for head_count in config.head_counts:
        model, metrics, history, elapsed, outcome = train_model(head_count, config.primary_length, splits, vocabulary, config)
        if head_count == 4:
            primary_model, primary_metrics, primary_history, primary_elapsed, primary_outcome = model, metrics, history, elapsed, outcome
        row: dict[str, object] = {"experiment": "attention_heads", "heads": head_count, "head_dimension": config.model_dimension // head_count, "max_length": config.primary_length, "parameters": sum(parameter.numel() for parameter in model.parameters()), "training_time_seconds": elapsed, "epochs": config.epochs, **metrics}
        experiment_rows.append(row)
        details["heads"][str(head_count)] = {"metrics": metrics, "history": history, "training_time_seconds": elapsed, "parameters": row["parameters"]}
    for sequence_length in config.sequence_lengths:
        if sequence_length == config.primary_length:
            continue
        model, metrics, history, elapsed, _ = train_model(4, sequence_length, splits, vocabulary, config)
        row = {"experiment": "sequence_length", "heads": 4, "head_dimension": config.model_dimension // 4, "max_length": sequence_length, "parameters": sum(parameter.numel() for parameter in model.parameters()), "training_time_seconds": elapsed, "epochs": config.epochs, **metrics}
        experiment_rows.append(row)
        details["sequence_lengths"][str(sequence_length)] = {"metrics": metrics, "history": history, "training_time_seconds": elapsed, "attention_elements_per_head_per_example": sequence_length**2}
    _, first_token_metrics, _, first_token_elapsed, _ = train_model(4, config.primary_length, splits, vocabulary, config, pooling_strategy="first_token")
    experiment_rows.append({"experiment": "pooling_strategy", "model": "Transformer first_token", "heads": 4, "head_dimension": config.model_dimension // 4, "max_length": config.primary_length, "training_time_seconds": first_token_elapsed, "epochs": config.epochs, **first_token_metrics})
    details["pooling_comparison"] = {"mean": {"metrics": None, "training_time_seconds": None}, "first_token": {"metrics": first_token_metrics, "training_time_seconds": first_token_elapsed}}
    lstm_metrics, lstm_elapsed, lstm_parameters = train_lstm(config.primary_length, splits, vocabulary, config)
    experiment_rows.append({"experiment": "architecture_comparison", "model": "LSTM", "heads": "", "head_dimension": "", "max_length": config.primary_length, "parameters": lstm_parameters, "training_time_seconds": lstm_elapsed, "epochs": config.epochs, **lstm_metrics})
    if primary_model is None or primary_metrics is None or primary_history is None or primary_outcome is None:
        raise RuntimeError("the configured head counts must include four heads to produce primary artifacts")
    labels, predictions, probabilities = primary_outcome
    write_attention_plot(primary_model, vocabulary, config)
    figure, axis = plt.subplots(figsize=(7, 4))
    epochs = [int(item["epoch"]) for item in primary_history]
    axis.plot(epochs, [item["training_loss"] for item in primary_history], marker="o", label="Training loss")
    axis.plot(epochs, [item["validation_loss"] for item in primary_history], marker="o", label="Validation loss")
    axis.set(xlabel="Epoch", ylabel="Binary cross-entropy", title="Transformer learning curves", xticks=epochs)
    axis.legend()
    figure.tight_layout()
    figure.savefig(config.output_directory / "learning_curves.png", dpi=config.plot_dpi)
    plt.close(figure)
    display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(labels, predictions), display_labels=["negative", "positive"])
    figure, axis = plt.subplots(figsize=(5, 4))
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("IMDb Transformer test confusion matrix")
    figure.tight_layout()
    figure.savefig(config.output_directory / "confusion_matrix.png", dpi=config.plot_dpi)
    plt.close(figure)
    torch.save({"model_state": primary_model.state_dict(), "vocabulary": vocabulary, "config": asdict(config), "metrics": primary_metrics}, config.output_directory / "transformer_model.pt")
    application_results = []
    for sentence in config.inference_examples:
        encoded = [vocabulary.get(token, 1) for token in tokenize(sentence)[:config.primary_length]]
        effective_length = max(len(encoded), 1)
        encoded = encoded + [0] * (config.primary_length - len(encoded))
        with torch.no_grad():
            probability = float(torch.sigmoid(primary_model(torch.tensor([encoded]), torch.tensor([effective_length]))).item())
        application_results.append({"text": sentence, "positive_probability": probability, "prediction": "positive" if probability >= 0.5 else "negative"})
    details["primary"] = {"heads": 4, "head_dimension": config.model_dimension // 4, "max_length": config.primary_length, "metrics": primary_metrics, "training_time_seconds": primary_elapsed, "history": primary_history, "confusion_matrix": confusion_matrix(labels, predictions).tolist(), "application_examples": application_results, "test_probabilities": probabilities.tolist()}
    details["pooling_comparison"]["mean"] = {"metrics": primary_metrics, "training_time_seconds": primary_elapsed}
    details["architecture_comparison"] = {"lstm": {"metrics": lstm_metrics, "parameters": lstm_parameters, "training_time_seconds": lstm_elapsed}, "transformer": {"metrics": primary_metrics, "parameters": sum(parameter.numel() for parameter in primary_model.parameters()), "training_time_seconds": primary_elapsed}}
    details["shape_trace"] = {"input": [8, 100, config.model_dimension], "q_k_v": [8, 4, 100, config.model_dimension // 4], "attention_scores": [8, 4, 100, 100], "attention_output": [8, 100, config.model_dimension], "ffn_hidden": [8, 100, config.feedforward_dimension], "block_output": [8, 100, config.model_dimension]}
    details["calculation_examples"] = {"positional_encoding_d4_pos0_to_2": positional_results["positions_0_to_2_d4"], "ffn_x": [1, 2], "ffn_w1": [[1, 0], [0, 1]], "ffn_b1": [0, -1], "ffn_hidden_relu": [1, 1], "residual_example_x_plus_f_x": [2, 4]}
    write_csv(config.output_directory / "results.csv", experiment_rows)
    (config.output_directory / "experiment_details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    (config.output_directory / "error_analysis.md").write_text(create_error_analysis(splits["test"], labels, predictions, probabilities, config), encoding="utf-8")
    write_notebook(config, primary_metrics, experiment_rows, application_results)


def create_error_analysis(test_frame: pd.DataFrame, labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray, config: LabConfig) -> str:
    """Format three correct and three incorrect held-out examples for manual review."""
    records = test_frame.reset_index(drop=True).copy()
    records["actual"] = labels
    records["predicted"] = predictions
    records["positive_probability"] = probabilities
    records["correct"] = records["actual"] == records["predicted"]
    selected = pd.concat([records[records["correct"]].head(3), records[~records["correct"]].head(3)])
    output = ["# LAB 07 — Phân tích lỗi", "", "Các mẫu dưới đây lấy từ tập test cố định của IMDb. Attention weights mô tả phân phối nội bộ, không được xem là lời giải thích nhân quả.", "", "| Kết quả | Nhãn thật | Dự đoán | P(tích cực) | Mẫu văn bản |", "|---|---|---|---:|---|"]
    for _, sample in selected.iterrows():
        excerpt = " ".join(tokenize(sample["review"])[: config.primary_length])[:240].replace("|", "\\|")
        actual = "positive" if int(sample["actual"]) else "negative"
        predicted = "positive" if int(sample["predicted"]) else "negative"
        status = "Đúng" if bool(sample["correct"]) else "Sai"
        output.append(f"| {status} | {actual} | {predicted} | {sample['positive_probability']:.3f} | {excerpt}… |")
    output.extend(["", "## Nhận xét", "", "Các trường hợp sai cần được đọc cùng văn bản gốc để nhận diện phủ định, tương phản, độ dài/cắt cụt và từ hiếm. Mẫu nhỏ này gợi ý các hướng xem xét nhưng không đủ để khẳng định nguyên nhân duy nhất. Attention heatmap chỉ cho biết trọng số mô hình phân phối giữa token; trọng số cao không chứng minh token đó là lý do đưa ra nhãn.", ""])
    return "\n".join(output)


def write_notebook(config: LabConfig, primary_metrics: dict[str, float], experiment_rows: list[dict[str, object]], application_results: list[dict[str, object]]) -> None:
    """Create a concise notebook linking the runnable source and generated results."""
    try:
        import nbformat
    except ImportError as error:
        raise RuntimeError("Install nbformat to generate transformer_classification.ipynb") from error
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        nbformat.v4.new_markdown_cell("# LAB 07 — Transformer Architecture\n\nNotebook ghi lại kết quả được tạo bởi `run_lab7.py`. Chạy cell kế tiếp từ thư mục gốc repository để tái tạo toàn bộ thí nghiệm và hình."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport subprocess, sys\nsubprocess.run([sys.executable, str(Path('lab7/run_lab7.py'))], check=True)"),
        nbformat.v4.new_markdown_cell("## Transformer block tối giản\n\n`TransformerBlock` tự ghép Q/K/V projections, scaled dot-product multi-head attention, output projection, residual + LayerNorm, GELU feed-forward và residual + LayerNorm. Không dùng `TransformerEncoderLayer`. Positional encoding được cộng vào token embeddings trước block."),
        nbformat.v4.new_code_cell("from lab7.positional_encoding import sinusoidal_positional_encoding\nfrom lab7.transformer_block import TransformerBlock\nimport torch\nblock = TransformerBlock(128, 4, 256)\nx = torch.randn(8, 100, 128)\ny, attention = block(x, return_attention=True)\nprint('Q/K/V: (8, 4, 100, 32); scores:', tuple(attention.shape), 'output:', tuple(y.shape))"),
        nbformat.v4.new_markdown_cell("## Số liệu thí nghiệm lần chạy được lưu\n\nKết quả có thể khác theo môi trường và thời gian huấn luyện; xem `results.csv`, `experiment_details.json` và các hình đầu ra. Các phép so sánh giữ cố định split, vocabulary, embedding dimension, optimizer, batch size và epoch; ở thí nghiệm head chỉ thay số head, ở thí nghiệm độ dài chỉ thay max length."),
        nbformat.v4.new_code_cell("import pandas as pd\nfrom IPython.display import display, Image\ndisplay(pd.read_csv('lab7/results.csv'))\ndisplay(Image(filename='lab7/positional_encoding.png'))\ndisplay(Image(filename='lab7/attention_heads.png'))\ndisplay(Image(filename='lab7/learning_curves.png'))\ndisplay(Image(filename='lab7/confusion_matrix.png'))"),
        nbformat.v4.new_markdown_cell(f"### Kết quả mô hình chính\n\nBốn heads, độ dài {config.primary_length}, {config.epochs} epochs: accuracy **{primary_metrics['accuracy']:.3f}**, precision **{primary_metrics['precision']:.3f}**, recall **{primary_metrics['recall']:.3f}**, F1 **{primary_metrics['f1']:.3f}**. Đây là baseline học từ đầu trên tập con IMDb, không phải pretrained Transformer.\n\nApplication examples được lưu trong `experiment_details.json`: " + "; ".join(f"{item['prediction']} ({item['positive_probability']:.3f}) — {item['text']}" for item in application_results)),
    ]
    notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}
    nbformat.write(notebook, config.output_directory / "transformer_classification.ipynb")


if __name__ == "__main__":
    run(LabConfig())
