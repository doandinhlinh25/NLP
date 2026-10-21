# Dự đoán trước khi huấn luyện

| Câu hỏi | Dự đoán | Lý do | Mức độ tin cậy |
|---|---|---|---|
| TF-IDF + LR so với RNN + Embedding: xử lý trực tiếp thứ tự từ | RNN | Trạng thái RNN được cập nhật theo thứ tự token; TF-IDF unigram đếm từ mà không lưu vị trí. | Cao |
| RNN so với LSTM khi cần nắm bắt quan hệ xa | LSTM | Trạng thái ô nhớ có cổng có thể duy trì thông tin qua nhiều bước, giảm sự phụ thuộc vào các phép biến đổi tanh lặp lại. | Cao |
| Tăng độ dài đầu vào của RNN từ 50 lên 200 | Huấn luyện chậm hơn; mức cải thiện chưa chắc chắn | Nhiều bước hồi quy hơn làm tăng thời gian tính toán và kéo dài đường truyền gradient, đồng thời giữ lại nhiều ngữ cảnh hơn. | Cao về thời gian, trung bình về chất lượng |
| Cắt mọi bài đánh giá còn 30 token | Độ chính xác có thể giảm | Kết luận, ý đảo chiều hoặc từ phủ định có thể xuất hiện sau 30 token đầu. | Cao |
| LSTM có luôn chính xác hơn RNN không? | Không | Dữ liệu, thời lượng huấn luyện, dao động giữa các lần chạy và độ dài tác vụ đều ảnh hưởng đến kết quả. | Cao |
