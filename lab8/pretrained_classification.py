from __future__ import annotations

import argparse
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer

LAB_DIR = Path(__file__).resolve().parent
ROOT = LAB_DIR.parent

@dataclass(frozen=True)
class Settings:
    """Store the shared reproducibility settings for all runs."""
    data_path: Path = ROOT / "data" / "imdb" / "IMDB Dataset.csv"
    model_name: str = "prajjwal1/bert-tiny"
    seed: int = 42
    batch_size: int = 16
    epochs: int = 1
    learning_rate: float = 2e-5
    max_length: int = 128
    train_limit: int = 2000
    validation_limit: int = 500
    test_limit: int = 500
    output_dir: Path = LAB_DIR
    threads: int = 4

class ReviewDataset(Dataset):
    """Hold tokenized reviews and binary sentiment labels."""
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        """Tokenize each review once for efficient repeated epochs."""
        self.encodings = tokenizer(frame.review.astype(str).tolist(), truncation=True,
                                   padding="max_length", max_length=max_length, return_tensors="pt")
        self.labels = torch.tensor(frame.label.astype(int).tolist(), dtype=torch.long)
    def __len__(self) -> int:
        """Return the number of reviews."""
        return len(self.labels)
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one encoded review and its label."""
        return {**{key: value[index] for key, value in self.encodings.items()}, "labels": self.labels[index]}

def seed_everything(seed: int) -> None:
    """Set random seeds for repeatable data sampling and model initialization."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def prepare_data(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create fixed stratified train, validation, and test splits from IMDb."""
    source = pd.read_csv(settings.data_path)
    source.columns = [str(column).strip().lower() for column in source.columns]
    source = source.rename(columns={"sentiment": "label"}).dropna(subset=["review", "label"])
    source = source.drop_duplicates(subset="review")
    source["label"] = source.label.map({"negative": 0, "positive": 1}).astype(int)
    training, test = train_test_split(source, test_size=0.2, random_state=settings.seed, stratify=source.label)
    train, validation = train_test_split(training, test_size=0.125, random_state=settings.seed, stratify=training.label)
    return (balanced_sample(train, settings.train_limit, settings.seed),
            balanced_sample(validation, settings.validation_limit, settings.seed),
            balanced_sample(test, settings.test_limit, settings.seed))

def balanced_sample(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Take a deterministic, approximately balanced sample from a split."""
    if limit >= len(frame): return frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    each = limit // 2
    selected = frame.groupby("label", group_keys=False).sample(n=each, random_state=seed)
    if limit % 2: selected = pd.concat([selected, frame.drop(selected.index).sample(n=1, random_state=seed)])
    return selected.sample(frac=1, random_state=seed).reset_index(drop=True)

def make_model(settings: Settings, mode: str, device: torch.device) -> tuple[torch.nn.Module, Any, int]:
    """Load pretrained or randomly initialized DistilBERT and optionally freeze its encoder."""
    tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    if mode == "scratch":
        config = AutoConfig.from_pretrained(settings.model_name, num_labels=2)
        model = AutoModelForSequenceClassification.from_config(config)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(settings.model_name, num_labels=2)
    if mode == "frozen":
        for parameter in model.base_model.parameters(): parameter.requires_grad = False
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    model.to(device)
    return model, tokenizer, parameter_count

def run_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device,
              optimizer: Any = None) -> tuple[float, list[int], list[int], list[list[float]]]:
    """Train for one epoch when an optimizer is given, then collect predictions."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0; labels_all: list[int] = []; predictions: list[int] = []; probabilities: list[list[float]] = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with torch.set_grad_enabled(training): output = model(**batch)
        if training:
            output.loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        total_loss += float(output.loss.detach()) * len(batch["labels"])
        probs = output.logits.detach().softmax(dim=1).cpu()
        labels_all.extend(batch["labels"].cpu().tolist()); predictions.extend(probs.argmax(dim=1).tolist())
        probabilities.extend(probs.tolist())
    return total_loss / max(len(loader.dataset), 1), labels_all, predictions, probabilities

def score(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    """Calculate accuracy, precision, recall, and weighted F1."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision),
            "recall": float(recall), "f1": float(f1)}

def run_experiment(settings: Settings, mode: str, train_frame: pd.DataFrame,
                   validation: pd.DataFrame, test: pd.DataFrame, max_length: int,
                   run_name: str) -> dict[str, Any]:
    """Train one model configuration and save its metrics and test predictions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, parameter_count = make_model(settings, mode, device)
    datasets = [ReviewDataset(part, tokenizer, max_length) for part in (train_frame, validation, test)]
    loaders = [DataLoader(dataset, batch_size=settings.batch_size, shuffle=index == 0)
               for index, dataset in enumerate(datasets)]
    optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=settings.learning_rate)
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(settings.epochs):
        train_loss, train_y, train_p, _ = run_epoch(model, loaders[0], device, optimizer)
        validation_loss, validation_y, validation_p, _ = run_epoch(model, loaders[1], device)
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_accuracy": score(train_y, train_p)["accuracy"],
                        "validation_accuracy": score(validation_y, validation_p)["accuracy"]})
    elapsed = time.perf_counter() - started
    _, y_true, y_pred, probabilities = run_epoch(model, loaders[2], device)
    metrics = score(y_true, y_pred)
    result: dict[str, Any] = {"run": run_name, "mode": mode, "model": settings.model_name,
        "max_length": max_length, "train_samples": len(train_frame), "validation_samples": len(validation),
        "test_samples": len(test), "batch_size": settings.batch_size, "epochs": settings.epochs,
        "learning_rate": settings.learning_rate, "seed": settings.seed, "device": str(device),
        "hardware": platform.processor() or platform.platform(), "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "training_seconds": elapsed, "parameters": parameter_count, **metrics, "history": history}
    destination = settings.output_dir / "checkpoints" / run_name
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination); tokenizer.save_pretrained(destination)
    prediction_frame = test[["review", "label"]].copy()
    prediction_frame["prediction"] = y_pred
    prediction_frame["confidence"] = [max(pair) for pair in probabilities]
    prediction_frame.to_csv(settings.output_dir / f"predictions_{run_name}.csv", index=False)
    result["predictions_file"] = f"predictions_{run_name}.csv"
    return result

def run_suite(settings: Settings, experiment: str) -> list[dict[str, Any]]:
    """Run a selected controlled comparison using fixed data splits."""
    seed_everything(settings.seed); torch.set_num_threads(settings.threads)
    train, validation, test = prepare_data(settings)
    plans: list[tuple[str, str, pd.DataFrame, int]] = []
    if experiment in {"all", "comparison"}:
        plans.extend([("scratch", "scratch", train, settings.max_length),
                      ("frozen", "frozen", train, settings.max_length),
                      ("fine_tuned", "fine_tune", train, settings.max_length)])
    if experiment in {"all", "data"}:
        plans.extend((f"data_{fraction}", "fine_tune", train.sample(frac=fraction, random_state=settings.seed), settings.max_length)
                     for fraction in (0.1, 0.25, 0.5, 1.0))
    if experiment in {"all", "length"}:
        plans.extend((f"length_{length}", "fine_tune", train, length) for length in (64, 128, 256))
    results = []
    for run_name, mode, training, length in plans:
        result = run_experiment(settings, mode, training, validation, test, length, run_name)
        results.append(result)
        pd.DataFrame(results).drop(columns=["history"], errors="ignore").to_csv(settings.output_dir / "results.csv", index=False)
        (settings.output_dir / "run_details.json").write_text(json.dumps({"settings": {**asdict(settings), "data_path": str(settings.data_path), "output_dir": str(settings.output_dir)}, "runs": results}, indent=2), encoding="utf-8")
        print(f"{run_name}: accuracy={result['accuracy']:.4f} f1={result['f1']:.4f} time={result['training_seconds']:.1f}s")
    if results: make_figures(settings.output_dir, results, test, plans)
    return results

def make_figures(output_dir: Path, results: list[dict[str, Any]], test: pd.DataFrame,
                 plans: list[tuple[str, str, pd.DataFrame, int]]) -> None:
    """Save data efficiency, confusion, and confidence analysis figures."""
    frame = pd.DataFrame(results)
    data_runs = frame[frame.run.str.startswith("data_")]
    if not data_runs.empty:
        axis = data_runs.plot(x="train_samples", y=["accuracy", "f1"], marker="o", title="IMDb data efficiency")
        axis.set_xlabel("Training reviews"); axis.set_ylabel("Score"); axis.figure.tight_layout()
        axis.figure.savefig(output_dir / "data_efficiency.png", dpi=160); plt.close(axis.figure)
    if not frame.empty:
        chosen = frame[frame.run == "fine_tuned"].iloc[0] if (frame.run == "fine_tuned").any() else frame.iloc[-1]
        pred_path = output_dir / chosen.predictions_file
        predictions = pd.read_csv(pred_path)
        display = ConfusionMatrixDisplay(confusion_matrix(predictions.label, predictions.prediction), display_labels=["negative", "positive"])
        display.plot(cmap="Blues"); display.figure_.tight_layout(); display.figure_.savefig(output_dir / "confusion_matrix.png", dpi=160); plt.close(display.figure_)
        correct = predictions.label == predictions.prediction
        bins = pd.cut(predictions.confidence, bins=[0, .6, .8, 1.000001], labels=["low", "medium", "high"], include_lowest=True)
        confidence = pd.DataFrame({"confidence_band": bins, "correct": correct}).groupby("confidence_band", observed=False).correct.agg(["count", "mean"])
        confidence.to_csv(output_dir / "confidence_analysis.csv")
        axis = confidence["mean"].plot(kind="bar", title="Accuracy by confidence band", ylim=(0, 1))
        axis.set_xlabel("Maximum predicted probability"); axis.set_ylabel("Accuracy"); axis.figure.tight_layout()
        axis.figure.savefig(output_dir / "confidence_analysis.png", dpi=160); plt.close(axis.figure)
        false_positive = predictions[(predictions.label == 0) & (predictions.prediction == 1)].nlargest(3, "confidence")
        false_negative = predictions[(predictions.label == 1) & (predictions.prediction == 0)].nlargest(3, "confidence")
        confident_wrong = predictions[(~correct)].nlargest(2, "confidence")
        selected = pd.concat([false_positive.assign(error_type="false positive"), false_negative.assign(error_type="false negative"), confident_wrong.assign(error_type="highest-confidence incorrect")]).drop_duplicates(subset=["review", "error_type"])
        selected.to_csv(output_dir / "error_examples.csv", index=False)

def main() -> None:
    """Parse run settings and execute the selected experiment suite."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("all", "comparison", "data", "length"), default="all")
    parser.add_argument("--model", default="prajjwal1/bert-tiny")
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--validation-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    arguments = parser.parse_args()
    settings = Settings(model_name=arguments.model, train_limit=arguments.train_limit,
        validation_limit=arguments.validation_limit, test_limit=arguments.test_limit,
        epochs=arguments.epochs, batch_size=arguments.batch_size, max_length=arguments.max_length)
    run_suite(settings, arguments.experiment)

if __name__ == "__main__": main()

