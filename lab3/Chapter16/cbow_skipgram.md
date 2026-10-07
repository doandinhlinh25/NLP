# Chapter 16 — Prediction: CBOW và Skip-gram

## Câu và context window

Câu được dùng là:

```text
the cat eats fish
```

Với `window = 1`, mỗi target word chỉ dùng một từ đứng trước và một từ đứng sau làm context. Ở hai đầu câu, context chỉ có một từ.

## CBOW — Context → target word

CBOW nhận các context words làm input và dự đoán target word ở giữa.

| Input context | Target |
| --- | --- |
| `[cat]` | `the` |
| `[the, eats]` | `cat` |
| `[cat, fish]` | `eats` |
| `[eats]` | `fish` |

Ví dụ, từ hai context `the` và `eats`, CBOW cần dự đoán `cat`.

## Skip-gram — Target word → context words

Skip-gram đảo chiều: target word là input và từng context word tạo thành một training pair riêng.

| Input target | Output context |
| --- | --- |
| `the` | `cat` |
| `cat` | `the` |
| `cat` | `eats` |
| `eats` | `cat` |
| `eats` | `fish` |
| `fish` | `eats` |

Vì vậy, một target có hai hàng xóm tạo ra hai cặp Skip-gram. CBOW gộp hai hàng xóm thành một context để dự đoán target.

## Kết luận

Sự khác biệt cốt lõi là hướng của bài toán học: **CBOW dự đoán target từ context**, còn **Skip-gram dự đoán context từ target**. Với corpus nhỏ, kết quả cuối cùng còn phụ thuộc vào số lần xuất hiện của từ, `window`, số epoch và số chiều embedding.
