# LAB 07 — Dự đoán trước thí nghiệm

| # | Prediction | Reason | Confidence |
|---:|---|---|---|
| 1 | Transformer xử lý song song các token trong một layer tốt hơn RNN. | Self-attention tính phép chiếu và tương tác trên toàn chuỗi cùng lúc, còn RNN phụ thuộc trạng thái timestep trước. | Cao |
| 2 | Bỏ positional encoding thì kiến trúc không biểu diễn trực tiếp thứ tự token. | Self-attention chỉ dựa trên nội dung Q/K/V; hoán vị input hoán vị các biểu diễn tương ứng. | Cao |
| 3 | Với pooling bất biến hoán vị, hai câu cùng token nhưng khác thứ tự khó phân biệt nếu không có positional signal. | Attention không có đặc trưng vị trí; mean pooling xóa cả thứ tự hàng đầu ra. | Cao |
| 4 | Hai head không nhất thiết có attention pattern giống nhau. | Mỗi head có các projection Q/K/V độc lập và có thể học phân phối khác nhau. | Cao |
| 5 | Tăng số head không đảm bảo accuracy tăng. | Khi d_model cố định, mỗi head có dimension nhỏ hơn; lợi ích phụ thuộc dữ liệu, tối ưu hóa và regularization. | Cao |
| 6 | Transformer không luôn nhanh hơn RNN với mọi độ dài. | Parallelism có lợi, nhưng attention dùng bộ nhớ và phép tính bậc hai theo độ dài; chuỗi ngắn hoặc batch nhỏ có thể chịu overhead. | Cao |

Các dự đoán này được giữ nguyên trước khi chạy `run_lab7.py`; kết quả và xác minh sau thí nghiệm được ghi trong `reflection.md`.
