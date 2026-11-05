# Calculation Exercise

## 1. MLM loss

Với một vị trí mask và token đúng `eats`, cross-entropy là `L = -ln P(eats | x)`:

| Xác suất của `eats` | Loss (nats) |
|---:|---:|
| 0.70 | 0.3567 |
| 0.95 | 0.0513 |
| 0.01 | 4.6052 |

Khi xác suất gán cho nhãn đúng tăng, loss giảm; khi xác suất đó gần 0, loss tăng mạnh. Cross-entropy phù hợp vì MLM là bài toán phân loại token trên vocabulary: nó tối ưu log-likelihood của token thật tại các vị trí bị che và phạt dự đoán tự tin sai.

## 2. Tokenization

Với tokenizer `prajjwal1/bert-tiny`, câu có 4 từ theo khoảng trắng. Chuỗi token là `[CLS], the, cats, are, playing, ., [SEP]`; IDs là `[101, 1996, 8870, 2024, 2652, 1012, 102]`. Có 5 content subtokens (the, cats, are, playing, .) và 2 special tokens. Với max_length=16, attention mask là `[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0]`. Token/subword IDs phụ thuộc vocabulary/model; notebook tạo lại các giá trị này bằng tokenizer thật.
