# Lab 02 — Language Models

## Cấu trúc bài làm

| Nội dung đề bài | Bài làm |
| --- | --- |
| Exercise 1–5: tính tay và smoothing bằng tay | `Chapter1-Chapter5/` — làm sau |
| Exercise 6: prediction trước experiment | `Chapter6/` — làm sau |
| Experiment 1: corpus statistics | `Ex7/Ex7.ipynb`, `Ex7/run_corpus_statistics.py` |
| Core n-gram implementation | `Chapter8/ngram_lm.py` |
| Experiment 2–3: MLE, Laplace và perplexity | `Chapter9/Ex9.ipynb`, `Chapter9/run_smoothing_evaluation.py` |
| Next-word prediction và sentence ranking | `Chapter10/Ex10.ipynb`, `Chapter10/run_lm_applications.py` |
| Error analysis, reflection, learning check | `Chapter11/` — làm sau |
| Experiment theo đề Chapter 12 | `Chapter12/experiments.ipynb` |
| Chapter 13 — Experiment 1: Corpus statistics | `Chapter13/experiments.ipynb` |
| Chapter 14 — Core n-gram implementation | `Chapter14/experiments.ipynb` |
| Chapter 15 — Log probability | `Chapter15/experiments.ipynb` |
| Chapter 16 — Experiment 2: MLE vs Laplace | `Chapter16/experiments.ipynb` |
| Chapter 19–24 — Perplexity và next-word prediction | `Chapter19-Chapter24/experiments.ipynb` |

## Chạy lại phần code

Từ thư mục gốc repository:

```powershell
python lab2/Chapter8/ngram_lm.py
python lab2/Ex7/run_corpus_statistics.py
python lab2/Chapter9/run_smoothing_evaluation.py
python lab2/Chapter10/run_lm_applications.py
```

Corpus được đọc tại `data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json`. Các lệnh thực nghiệm mặc định dùng 10.000 documents và tạo CSV kết quả trong thư mục exercise tương ứng.
