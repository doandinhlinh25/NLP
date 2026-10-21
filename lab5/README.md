# LAB 05 - Sequence Models for IMDb Sentiment

The assignment source is `W5.pdf`. The complete dataset is `../data/imdb/IMDB Dataset.csv`. The experiment script generates the notebook's metrics and charts from a reproducible, balanced 7,000-review subset; full-dataset descriptive statistics use all 50,000 source rows. The smaller training budget is explicit so that the five CPU recurrent runs can be reproduced on an ordinary computer. Results from this subset should not be compared directly with LAB 04's full-data scores.

## Run

```powershell
python -m pip install -r lab5/requirements.txt
python lab5/rnn_cell.py
python lab5/run_experiments.py
```

Open `lab5/sequence_classification.ipynb` to inspect the generated data, plots, and examples. Re-running its experiment cell recreates all output files. Python 3.11+ is recommended. The code fits the vocabulary and TF-IDF model on the training split only; validation and test remain separate. The same sample, tokenization, embedding and hidden dimensions, optimizer, batch size, epochs, and seed are used for RNN/LSTM/GRU. For the sequence length test, only the GRU maximum length changes.

## Deliverables

| File | Contents |
|---|---|
| `calculations.pdf` | RNN/LSTM arithmetic and theory |
| `prediction.md` | Predictions recorded before training |
| `rnn_cell.py` | Explicit RNN step and framework equivalence check |
| `sequence_classification.ipynb` | Executable experiment entry point and result inspection |
| `run_experiments.py` | Reproducible data, model, evaluation and chart pipeline |
| `results.csv` | Baseline, architecture and length results |
| `experiment_details.json` | Dataset audit, learning history, probes and confusion matrix |
| `sequence_lengths.png` | Full IMDb review length histogram |
| `learning_curves.png` | Training/validation losses |
| `length_tradeoff.png` | GRU performance and time versus length |
| `confusion_matrix.png` | GRU held-out test confusion matrix |
| `error_analysis.md` | Three correct and three incorrect held-out cases |
| `reflection.md` | Findings and limitations |

`gru_model.pt` holds the trained 100-token GRU and training vocabulary for later inference. The reported training time excludes CSV loading, tokenization, evaluation, and plotting. Precision, recall and F1 treat positive sentiment as the positive class. The test split is evaluated for each predefined experimental condition; it is not used to tune model settings.
