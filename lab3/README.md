# Lab 03 — Word Representations and Embeddings

Lab 03 chuyển từ biểu diễn sparse dựa trên count sang word-context representation, dense word embedding và đánh giá semantic similarity.

## Cấu trúc bài làm

| Phần trong đề | Tệp/thư mục thực hiện |
| --- | --- |
| Chapter 9 — Prediction trước experiment | `Chapter9/chapter9_prediction.md` |
| Chapter 10 — Experiment 1: word-context representation | `Chapter10/Ex10.ipynb`, `Chapter10/run_word_context_experiment.py`, `Chapter10/results.csv`, `Chapter10/experiment_results.md` |
| Chapter 11 — Core co-occurrence implementation | `Chapter11/implementation.py`, `Chapter11/verification.md` |
| Chapter 16 — CBOW và Skip-gram training examples | `Chapter16/cbow_skipgram.md` |
| Chapter 17 — Experiment 2: Word2Vec | `Chapter17/Ex17.ipynb`, `Chapter17/run_word2vec_experiment.py`, `Chapter17/training_config.md` |
| Chapter 18 — Inspect embeddings | `Chapter18/Ex18.ipynb`, `Chapter18/run_embedding_inspection.py`, `Chapter18/doctor_top5.csv`, `Chapter18/embedding_inspection.md` |
| Chapter 19 — Context-window experiment | `Chapter19/Ex19.ipynb`, `Chapter19/run_context_window_experiment.py`, `Chapter19/results.csv`, `Chapter19/experiment_results.md` |
| Chapter 20 — Embedding-dimension experiment | `Chapter20/Ex20.ipynb`, `Chapter20/run_dimension_experiment.py`, `Chapter20/results.csv`, `Chapter20/experiment_results.md` |
| Chapter 21 — Word-similarity evaluation | `Chapter21/Ex21.ipynb`, `Chapter21/run_similarity_evaluation.py`, `Chapter21/word_similarity_results.csv`, `Chapter21/word_similarity_evaluation.md` |

## Môi trường

- Python 3.10 trở lên
- Jupyter Notebook
- `numpy`, `scipy`, `gensim`

Cài đặt dependency từ thư mục gốc của repository:

```powershell
python -m pip install numpy scipy gensim jupyter
```

## Chạy lại phần code

Từ thư mục gốc của repository:

```powershell
python lab3/Chapter10/run_word_context_experiment.py
python lab3/Chapter11/implementation.py
python lab3/Chapter17/run_word2vec_experiment.py
python lab3/Chapter18/run_embedding_inspection.py
python lab3/Chapter19/run_context_window_experiment.py
python lab3/Chapter20/run_dimension_experiment.py
python lab3/Chapter21/run_similarity_evaluation.py
```

Hoặc mở notebook:

```powershell
jupyter notebook lab3/Chapter10/Ex10.ipynb
jupyter notebook lab3/Chapter17/Ex17.ipynb
jupyter notebook lab3/Chapter18/Ex18.ipynb
jupyter notebook lab3/Chapter19/Ex19.ipynb
jupyter notebook lab3/Chapter20/Ex20.ipynb
jupyter notebook lab3/Chapter21/Ex21.ipynb
```

Các script đọc corpus C4 ở `data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json`. Chapter 10 dùng 5.000 văn bản; Chapter 17–18 dùng 2.000 văn bản và tạo model, báo cáo, CSV tương ứng.
