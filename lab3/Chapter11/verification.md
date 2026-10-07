# Chapter 11 — Core implementation verification

## Phạm vi cài đặt

`implementation.py` tự cài bốn hàm yêu cầu trong đề:

| Hàm | Vai trò |
| --- | --- |
| `build_vocabulary()` | Đếm token, lọc theo tần suất và tạo vocabulary có thứ tự xác định. |
| `build_cooccurrence_matrix()` | Tạo sparse target-word × context-word matrix theo context window. |
| `cosine_similarity()` | Tính cosine cho dense vector hoặc sparse row vector. |
| `most_similar()` | Xếp hạng các từ có cosine similarity cao nhất, ngoại trừ chính target word. |

Không sử dụng `Word2Vec(...)` hoặc mô hình embedding có sẵn trong phần này.

## Bộ kiểm tra

Bộ test dùng corpus nhỏ trong đề:

```text
the cat eats fish
the cat likes milk
the dog eats meat
the dog likes fish
```

Các điều kiện đã kiểm tra:

1. Vocabulary có thứ tự xác định theo tần suất rồi alphabet.
2. Ma trận với `window = 1` có kích thước đúng và có count `cat → eats = 1`.
3. Vector `cat` và `dog` giống nhau trong corpus kiểm tra nên cosine bằng `1.0`.
4. `most_similar("cat", ...)` trả về `dog` đầu tiên với score `1.0`.
5. Ví dụ vector `[1, 2, 1]` và `[2, 4, 2]` cho cosine bằng `1.0`.

Chạy lại từ thư mục gốc:

```powershell
python lab3/Chapter11/implementation.py
```

Kết quả mong đợi:

```text
All Chapter 11 implementation checks passed.
```
