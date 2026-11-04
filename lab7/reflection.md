# LAB 07 — Kết quả và reflection

## Thiết lập

IMDb có 50.000 review cân bằng. Mỗi mô hình dùng cùng split cố định 5.000 train, 1.000 validation và 1.000 test; vocabulary 11.998 token chỉ được tạo từ train. Các lần so sánh dùng seed 42, batch 64, Adam với learning rate 0,001, hai epoch và test set không dùng để chọn tham số. Transformer dùng `d_model=128`, `d_ff=256`; LSTM dùng hidden size 128. Thời gian đo trên máy hiện tại, nên có thể thay đổi khi chạy lại.

## So sánh LSTM và Transformer

| Model | Parameters | Training time (s) | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| LSTM | 1.668.225 | 34,68 | 0,540 | 0,539 | 0,558 | 0,548 |
| Transformer, 4 heads + mean pooling | 1.668.609 | 22,19 | 0,708 | 0,733 | 0,654 | 0,691 |

Hai mô hình có số tham số gần bằng nhau và cùng độ dài 100. Ở lần chạy này Transformer có accuracy/F1 cao hơn, nhưng kết quả của một seed và hai epoch không chứng minh kiến trúc luôn tốt hơn. Cách tổng hợp sequence khác nhau (LSTM lấy hidden state cuối, Transformer mean-pool các token), song song so với tuần tự cũng khác; các yếu tố này ảnh hưởng kết quả và thời gian.

## Số attention heads

Giữ `d_model=128`, độ dài 100 và mọi cấu hình khác không đổi. Tổng số tham số giữ ở 1.668.609; `d_k` giảm từ 128 xuống 16 khi head tăng từ 1 lên 8. Test accuracy lần lượt là 0,680; 0,675; 0,708; 0,666 cho 1, 2, 4, 8 heads. F1 lần lượt 0,696; 0,739; 0,691; 0,558. Không có xu hướng accuracy tăng đều. Heatmap trong `attention_heads.png` hiển thị bốn phân phối riêng. Không gán nhãn cú pháp/ngữ nghĩa cho head; cần đo độ giống nhau hoặc bằng chứng khác trước khi gọi một head redundant.

## Độ dài chuỗi và chi phí

| Max length | Tokens retained (%) | Attention elements/head/sample | Training time (s) | Accuracy | F1 |
|---:|---:|---:|---:|---:|---:|
| 32 | 14,05 | 1.024 | 5,65 | 0,628 | 0,544 |
| 64 | 27,78 | 4.096 | 11,43 | 0,673 | 0,730 |
| 128 | 52,15 | 16.384 | 28,39 | 0,618 | 0,423 |
| 256 | 77,58 | 65.536 | 84,00 | 0,755 | 0,773 |

Số token giữ lại tăng cùng độ dài, và matrix mỗi head tăng gấp bốn mỗi khi T tăng gấp đôi. Trong lần chạy này thời gian huấn luyện tăng đáng kể, trong khi F1 không tăng đều ở các độ dài trung gian. Vì thế tăng max length có thể giữ thêm tín hiệu review nhưng đòi hỏi thêm bộ nhớ/tính toán và không đảm bảo metric tốt hơn trên mọi lần chạy.

## Positional information

Với sequence `I love NLP` và hoán vị `NLP love I`, thí nghiệm áp dụng scaled self-attention rồi mean pooling. Không có PE, max absolute difference giữa pooled output là `2,38×10⁻⁷` (sai số float); khi cộng PE, chênh lệch là `2,94×10⁻⁵`. Phép hoán vị có ý nghĩa câu `dog bites man`/`man bites dog` cho kết quả tương ứng `1,19×10⁻⁷` và `2,10×10⁻⁴`. Đây là bằng chứng số cho tính bất biến hoán vị trong cấu hình không vị trí và sự khác biệt sau thêm vị trí. Mức chênh phụ thuộc embedding ngẫu nhiên, trọng số minh họa và chiều vector; nó chứng minh cơ chế, không đo chất lượng classifier.

## Pooling

Giữ 4 heads và cấu hình train giống nhau, mean pooling đạt accuracy/F1 0,708/0,691; first-token pooling đạt 0,691/0,667. Đoạn code không thêm token `[CLS]`, do đó first-token ở đây chỉ là pooling vị trí đầu và không phải một CLS token đã học đặc biệt. Kết quả gợi ý pooling có thể ảnh hưởng phân loại; không đủ cơ sở để kết luận mean pooling luôn tốt hơn.

## Ứng dụng và error analysis

Classifier được thử trên câu ngắn, dài, phủ định, tương phản và cảm xúc mơ hồ; dự đoán cùng xác suất positive nằm trong `experiment_details.json`. Error analysis chọn ba đúng và ba sai trực tiếp từ test set. Cần xem xét phủ định, tương phản, từ hiếm và cắt cụt 100 token như các giả thuyết khi đọc lỗi, không coi đó là nguyên nhân đã được xác minh. Attention map thể hiện phân phối trọng số; attention cao không phải bằng chứng nhân quả cho nhãn.

## Xác minh sáu prediction ban đầu

| # | Dự đoán ban đầu | Quan sát | Kết luận |
|---:|---|---|---|
| 1 | Transformer có thể xử lý song song token trong layer hơn RNN. | Cấu trúc Q/K/V và attention tạo score toàn chuỗi cùng lúc; LSTM xử lý trạng thái tuần tự. Lần chạy 100 token Transformer mất 22,19 s so với 34,68 s của LSTM. | Phù hợp trong lần chạy; không khẳng định mọi thiết bị/độ dài. |
| 2 | Bỏ positional encoding làm mô hình không biểu diễn trực tiếp thứ tự. | Self-attention + mean pooling không PE cho pooled outputs gần bằng nhau sau hoán vị; thêm PE tạo chênh lệch. | Đúng với thí nghiệm. |
| 3 | Hai câu cùng từ nhưng khác thứ tự khó phân biệt nếu thiếu positional signal. | Hai hoán vị cho chênh lệch pooled output gần 0 nếu không PE. | Đúng với phép thử hoán vị. |
| 4 | Hai head không nhất thiết học pattern giống nhau. | Heatmap bốn head cho các phân phối theo token khác nhau. | Phù hợp quan sát; chưa đo similarity định lượng. |
| 5 | Tăng số head không đảm bảo accuracy tăng. | Accuracy 1/2/4/8 heads là 0,680/0,675/0,708/0,666. | Đúng trong thí nghiệm này. |
| 6 | Transformer không luôn nhanh hơn RNN ở mọi độ dài. | Transformer nhanh hơn ở lần đo chính 100 token, nhưng thời gian tăng từ 5,65 s ở 32 lên 84,00 s ở 256; chưa đo LSTM đầy đủ tại từng độ dài. | Có cơ sở từ xu hướng T², nhưng cần benchmark LSTM theo từng độ dài để xác minh so sánh trực tiếp. |

## Giới hạn và bước tiếp

| Giai đoạn | Giới hạn trước đó được xử lý | Giới hạn còn lại |
|---|---|---|
| TF-IDF | Từ ngữ được biểu diễn bằng đặc trưng có trọng số hữu ích cho phân loại/tìm kiếm. | Sparse, thiếu ngữ cảnh và thứ tự câu. |
| Word2Vec | Từ có vector dense và quan hệ ngữ nghĩa gần nhau hơn. | Mỗi từ thường có một vector tĩnh, chưa mô hình hóa câu. |
| LSTM | Hidden state tuần tự mô hình hóa thứ tự và ngữ cảnh. | Tính toán tuần tự, khó truyền thông tin xa và song song hóa. |
| Attention | Tương tác trực tiếp giữa token, kể cả token xa nhau. | Cần cách biểu diễn thứ tự; full attention có chi phí theo `T²`. |
| Transformer | Kết hợp multi-head attention, positional encoding, FFN, residual và LayerNorm. | Dài chuỗi tốn compute/bộ nhớ; bản học từ đầu cần dữ liệu và tối ưu nhiều. Pretrained Transformer là hướng tiếp theo. |

Implementation này dùng một Transformer block và train từ đầu trên tập con IMDb; kết quả không so trực tiếp với BERT pretrained. Khác biệt theo độ dài có thể đến từ seed/optimization và lượng token giữ lại. Bước tiếp theo hữu ích là nhiều seed, early stopping theo validation và benchmark LSTM cho từng độ dài.
