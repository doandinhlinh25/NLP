# LAB 06 — Nhận xét và tổng kết

## Quan sát từ cơ chế attention

Trong ví dụ mười token với đặc trưng tự thiết kế, query sentiment gán cho `good` weight 0,420; query topic gán cho `movie` 0,380 và `beginning` 0,209. Key và value được giữ nguyên, chỉ query thay đổi. Kết quả minh họa rằng phân phối attention phụ thuộc vào query. Các vector này được thiết kế để quan sát cơ chế, không phải embedding đã học; không thể dùng biểu đồ để kết luận mô hình hiểu ngôn ngữ.

Với score gốc `[4,2,0]`, khi temperature giảm từ 2,0 xuống 0,5, weight lớn nhất tăng từ 0,665 lên 0,982 và entropy giảm từ 0,832 xuống 0,093 nat. Chia score cho `√4` cho hiệu ứng làm phẳng giống temperature 2,0 trong ví dụ này. Stable softmax xử lý được `[1000,1001,1002]` nhờ trừ giá trị lớn nhất trước khi lấy hàm mũ, trong khi tỷ lệ sau chuẩn hóa không đổi.

Ở heatmap minh họa câu có `cat`, query `cat` đặt weight cao nhất lên `cat` (0,399), `adopted` (0,224) và `sleeping` (0,159). Các hàng query `adopted` và `sleeping` cũng đặt weight lên `cat`. Câu có `book` được xây dựng bằng cách gán đặc trưng tương tự. Những ma trận này cho thấy cách đọc tương tác giữa token, không chứng minh một mô hình đã học mệnh đề quan hệ, ngữ pháp hay reasoning.

## Ứng dụng IMDb và so sánh với Lab 05

Classifier gồm embedding, các phép chiếu Q/K/V được học, full self-attention, masked mean pooling trên context và một lớp phân loại nhị phân. Mô hình được huấn luyện hai epoch trên cùng tập con IMDb cân bằng 5.000/1.000/1.000 cho train/validation/test và ngưỡng cắt 100 token như Lab 05. Từ vựng chỉ được xây từ tập train. F1 dùng nhãn sentiment tích cực làm lớp dương. Các chỉ số công bố được tính trên tập test riêng; validation dùng để theo dõi lúc huấn luyện. Phần này không dùng lớp attention dựng sẵn.

| Mô hình | Test accuracy | Test F1 | Thời gian huấn luyện | Số tham số |
|---|---:|---:|---:|---:|
| LSTM Lab 05, dùng final hidden state | 0,553 | 0,541 | 31,90 giây | 801.345 |
| Self-attention Lab 06, dùng mean context | 0,707 | 0,719 | Xem `results.csv` | 780.545 |

Confusion matrix của mô hình attention là `[[333,167],[126,374]]`, theo thứ tự nhãn tiêu cực rồi tích cực. Sau epoch 2, validation accuracy/F1 là 0,692/0,705. Attention đạt điểm cao hơn trong các lần chạy này; không thể từ đó kết luận kiến trúc này luôn tốt hơn LSTM. Thời gian chỉ tính vòng huấn luyện, phụ thuộc CPU và môi trường PyTorch; hai kiến trúc cũng có cách tính khác nhau. Giới hạn hai epoch và 100 token ảnh hưởng cả hai. Baseline TF-IDF của Lab 05 đạt accuracy 0,854, vẫn cao hơn trên tập con này.

Ba câu thử bắt buộc đều được dự đoán tiêu cực. `I like this movie.` có `P(tích cực)=0,008`, sai so với nhãn mong đợi. `I do not like this movie.` có 0,026 và được dự đoán đúng. Câu tương phản kết thúc bằng `actually excellent` có 0,004 và được dự đoán sai. Kết quả cho thấy lỗi ở câu tích cực ngắn và câu tương phản; việc đoán đúng một câu phủ định không chứng minh mô hình xử lý phủ định tốt. Các biểu đồ cột thể hiện attention trung bình mà mỗi token **nhận được** sau khi lấy trung bình qua các vị trí query. Chúng mô tả weight, không xác định nguyên nhân của dự đoán.

## Phân tích lỗi và giới hạn

`error_analysis.md` liệt kê sáu trường hợp thuộc tập test, gồm ba đúng và ba sai. Một bài tích cực bị đoán sai có đoạn `not as good ... but I was wrong ... far better`; mô hình chỉ gán xác suất tích cực 0,378 dù câu có ý tương phản và sửa lại đánh giá. Một bài tiêu cực bị đoán sai mở đầu bằng lời khen về quá trình sản xuất và nhận xác suất tích cực 0,590. Bài tích cực bị đoán sai khác dài 244 token và bị cắt ở 100 token đầu; tín hiệu cảm xúc có thể nằm ở phần sau. Trong bài tiêu cực bị đoán sai, các từ `stylish`, `heat`, `depth` nhận nhiều attention, phù hợp với giả thuyết mô hình chú ý tới lời khen, nhưng không chứng minh các token đó gây ra nhãn sai. Việc cắt ngắn cũng chỉ là một khả năng, chưa được xác định là nguyên nhân duy nhất. Phủ định, cách diễn đạt pha trộn, từ ngoài từ vựng, thời lượng huấn luyện ngắn và tối ưu hóa đều có thể góp phần.

Full self-attention tạo ma trận `T×T`: 1.000 token cần một triệu phần tử cho mỗi mẫu; 10.000 token cần 100 triệu. Mô hình đơn giản ở đây không thêm positional encoding. Nếu hoán vị các token mà không thay đổi việc cắt ngắn, đường xử lý attention cộng mean pooling của nó cho cùng đầu ra, nên không trực tiếp biểu diễn được khác biệt thứ tự như `not good` và `good not`. Đây là giới hạn đáng kể của cách triển khai trong lab. Transformer block về sau có thể thêm thông tin vị trí, multi-head attention, residual path và các lớp feed-forward. Attention tạo tương tác trực tiếp giữa token nhưng không xóa chi phí tính toán, cũng không biến weight thành bằng chứng về reasoning.
