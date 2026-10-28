# Dự đoán cho năm câu hỏi thí nghiệm

| Câu hỏi | Dự đoán | Lý do | Mức tin cậy |
|---|---|---|---|
| 1. Một key có dot product lớn hơn | Softmax weight của key đó sẽ lớn hơn các key còn lại. | Hàm mũ và phép chuẩn hóa giữ nguyên thứ tự các score. | Cao |
| 2. Score `[1,1,1,1]` | Mỗi weight bằng `0,25`. | Các giá trị mũ bằng nhau nên mỗi phần tử nhận một phần bằng nhau. | Cao |
| 3. Score `[1,1,10]` | Gần như toàn bộ trọng số dồn vào key cuối. | Chênh lệch 9 đơn vị trước phép mũ rất lớn. | Cao |
| 4. Độ dài tăng từ 10 lên 1.000 | Ma trận full self-attention tăng từ 100 lên 1.000.000 phần tử, tức gấp 10.000 lần. | Số phần tử tỷ lệ với `T²`. | Cao |
| 5. Token con người cho là quan trọng | Attention đôi khi nhấn mạnh token khác. | Score phản ánh đặc trưng và nhiệm vụ của mô hình; weight không bắt buộc khớp với lời giải thích của con người. | Trung bình |

Các dự đoán trên suy ra từ công thức và được trình bày riêng với số liệu quan sát trong `experiment_details.json` và `reflection.md`.
