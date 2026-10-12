# Natural Language Processing Laboratory Assignments

> **Sinh viên:** Nguyễn Thị Thanh Mai - 23000139  
> **Khóa học:** HUS NLP 2026  
> **Kho lưu trữ:** [HUS--NLP2026](https://github.com/jackfruit06/HUS---NLP2026)
>
---

Tổng hợp các bài thực hành môn **Xử lý Ngôn ngữ Tự nhiên** (NLP) — HUS NLP 2026. Các lab đi từ biểu diễn văn bản
dựa trên tần suất (TF-IDF), mô hình ngôn ngữ thống kê (N-gram) đến biểu diễn từ dạng vector (word embeddings),
và đều dùng chung một corpus 30,000 documents.

## Danh sách bài thực hành

| Lab | Nội dung | Kỹ thuật chính | Notebook |
|-----|----------|----------------|----------|
| [lab1](lab1/) | From Text Processing to Search | Tokenization, TF-IDF, cosine similarity, document retrieval | [experiments.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab1/experiments.ipynb) |
| [lab2](lab2/) | Language Models | N-gram LM (unigram/bigram/trigram), MLE, Laplace smoothing, perplexity, next-word prediction | [experiments.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab2/experiments.ipynb) |
| [lab3](lab3/) | Word Representations and Embeddings | Co-occurrence matrix, Word2Vec (Skip-gram), word similarity, word analogy, semantic search | [word_embedding.ipynb](https://colab.research.google.com/github/jackfruit06/HUS---NLP2026/blob/master/lab3/word_embedding.ipynb) |

## Cấu trúc kho lưu trữ

```
HUS---NLP2026/
├── README.md        <- file này
├── dataset          <- data
├── lab1/            <- LAB 01 — From Text Processing to Search
├── lab2/            <- LAB 02 — Language Models
└── lab3/            <- LAB 03 — Word Representations and Embeddings
```

Mỗi lab có `README.md` riêng mô tả cấu trúc thư mục, dữ liệu và cách chạy. Các phần chung của mỗi lab:

- `prediction` — dự đoán trước khi chạy experiment;
- `calculations` — bài tính tay;
- file `.py` — phần core tự cài đặt;
- notebook — các experiment, kèm output;
- `results.csv` — kết quả định lượng;
- `error analysis`, `reflection` — phân tích lỗi và tổng kết.

## Dữ liệu

- `/dataset/30k_documents.json` — corpus 30,000 documents (JSON Lines: `text`, `timestamp`, `url`), dùng chung cho cả 3 lab.
- `/dataset/IMDB Dataset.csv` — 50,000 review phim có nhãn sentiment, dùng cho downstream task của LAB 03.

## Môi trường

- Python 3.12
- Thư viện: `numpy`, `scipy`, `pandas`, `scikit-learn`, `gensim`, `jupyter`

```bash
pip install numpy scipy pandas scikit-learn gensim jupyter
```
