# LAB 07 — Transformer Architecture

Lab này ghép một Transformer Encoder block tối giản từ các phép chiếu Linear, softmax, dropout và LayerNorm. Core implementation không dùng `TransformerEncoderLayer`. Classifier được huấn luyện từ đầu trên cùng tập con IMDb, split, tokenization và vocabulary theo Lab 05; kết quả không phải pretrained Transformer.

## Chạy

Từ thư mục gốc repository, cài dependencies nếu cần rồi chạy:

```powershell
python -m pip install -r lab7/requirements.txt
python lab7/run_lab7.py
```

Mặc định dùng seed 42, tập con train/validation/test 5.000/1.000/1.000, vocabulary train-only, `d_model=128`, `d_ff=256`, batch 64, hai epoch và độ dài chính 100. Pipeline so sánh LSTM/Transformer, mean/first-token pooling, 1/2/4/8 heads, độ dài 32/64/128/256, kiểm tra hoán vị khi có/không positional encoding, và lưu biểu đồ, notebook, model cùng số liệu.

## Tệp bàn giao

| Tệp | Nội dung |
|---|---|
| `calculations.md` | Tính Q/K/V, multi-head, positional encoding, FFN, residual và shape trace |
| `prediction.md` | Sáu dự đoán được ghi trước thí nghiệm |
| `transformer_block.py` | Encoder block tự ghép với multi-head attention, residual, LayerNorm và FFN |
| `positional_encoding.py` | Sinusoidal encoding |
| `transformer_classification.ipynb` | Notebook xem lại và tái chạy thí nghiệm |
| `results.csv` | Kết quả số head và sequence length |
| `positional_encoding.png` | So sánh biểu diễn khi hoán vị sequence |
| `attention_heads.png` | Attention map của từng head |
| `learning_curves.png` | Train/validation loss |
| `confusion_matrix.png` | Confusion matrix trên test set |
| `error_analysis.md` | Ba mẫu đúng và ba mẫu sai lấy từ test set |
| `reflection.md` | Diễn giải kết quả và giới hạn |
| `experiment_details.json` | Cấu hình, số liệu, shape, lịch sử và ví dụ ứng dụng |
| `transformer_model.pt` | Trọng số mô hình và vocabulary |

Thời gian training phụ thuộc CPU. Dự đoán ví dụ không nên xem như đánh giá tổng quát. Attention weights là phân phối nội bộ, không phải giải thích nhân quả. Transformer có chi phí full attention bậc hai theo chiều dài chuỗi.
