# LAB 04 — Tính toán trước khi chạy code

## Bài 1 — Binary classification

Cho `x = [1, 2, 0]`, `w = [0.5, -0.2, 0.4]`, `b = 0.1`.

- `z = w^T x + b =`
- `sigmoid(z) = 1 / (1 + e^(-z)) =`
- `P(y = 1 | x) =`

## Bài 2 — Prediction threshold

Cho `P(y = 1 | x) = 0.72`.

- Threshold `0.5`: dự đoán
- Threshold `0.8`: dự đoán
- Khi tăng threshold, precision thay đổi thế nào?
- Khi tăng threshold, recall thay đổi thế nào?

## Bài 3 — Confusion matrix

| | Predicted positive | Predicted negative |
|---|---:|---:|
| Actual positive | 80 | 20 |
| Actual negative | 10 | 90 |

- TP =
- FP =
- FN =
- TN =
- Precision = `TP / (TP + FP)` =
- Recall = `TP / (TP + FN)` =
- F1 = `2 * Precision * Recall / (Precision + Recall)` =
- Nếu đây là phát hiện spam, false positive là
- Nếu đây là phát hiện spam, false negative là

## Bài 4 — Accuracy trap

Dataset có 950 negative và 50 positive; model dự đoán mọi mẫu là negative.

- Accuracy =
- Model có phát hiện được positive không?
- Cần xem thêm metric nào? Giải thích:
