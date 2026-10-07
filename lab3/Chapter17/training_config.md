# Chapter 17 — Experiment 2: Word2Vec

## Cấu hình huấn luyện

| Thuộc tính | Giá trị |
| --- | --- |
| Corpus | C4 JSONL, 2,000 văn bản không rỗng đầu tiên |
| Tokenization | Lower-case, English alphabetic tokens có thể chứa apostrophe |
| Vocabulary sau `min_count` | 21,581 từ |
| Training objective | CBOW (`sg=0`) |
| `vector_size` | 100 |
| `window` | 5 |
| `min_count` | 2 |
| `epochs` | 5 |
| `workers` | 1 |
| `seed` | 2026 |

`workers = 1` và seed cố định được dùng để hạn chế khác biệt giữa các lần chạy. Model đã huấn luyện được lưu tại `word2vec.model`; Chapter 18 nạp chính file này để kiểm tra embedding của `doctor`.
