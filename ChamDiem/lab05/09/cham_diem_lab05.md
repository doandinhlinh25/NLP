# Chấm điểm Lab 05 — Bài 9

- **Repo:** [mactuongmvt/NLP-Labs — labs/lab05](https://github.com/mactuongmvt/NLP-Labs/tree/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05).
- **Căn cứ:** Rubric trong `lab5/W5.pdf`; rà soát commit `89c028c3efc0fe147a4dbaa249619365f7baf247` ngày 28/10/2026.
- **Tổng điểm:** **90/100 = 9,0/10**. Mục vấn đáp cá nhân 3 phút chưa thể xác nhận từ repo nên tạm tính 0/5; có thể cập nhật sau kiểm tra trực tiếp.

## Điểm theo rubric

- **Theory & calculation — 10/10:** [`calculations.md`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/calculations.md) tính đúng RNN `tanh([0,67;0,30])≈[0,585;0,291]`, LSTM cell state `0,82`, giải thích hidden state, thứ tự, vanishing gradient và gates.
- **Prediction — 10/10:** [`prediction.md`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/prediction.md) có đủ 5 dự đoán, lý do và mức tin cậy.
- **Dataset & sequence analysis — 10/10:** [`sequence_classification.ipynb`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/sequence_classification.ipynb) đọc các split IMDb, báo lớp và độ dài, vẽ histogram, giải thích padding/truncation và tỷ lệ token được giữ. Repo cũng lưu các split tại `labs/lab04/data`.
- **RNN cell implementation — 15/15:** [`rnn_cell.py`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/rnn_cell.py) tự cài một bước RNN; notebook đối chiếu đầu ra với `torch.nn.RNNCell`.
- **RNN/LSTM/GRU experiment — 18/20:** Notebook huấn luyện đủ ba kiến trúc và TF-IDF baseline trên tập con cân bằng 4.000/1.000/2.000. Chỉ một seed và một cấu hình chính cho mỗi kiến trúc; chưa kiểm tra độ dao động qua nhiều lần chạy.
- **Sequence length experiment — 9/10:** [`results.csv`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/results.csv) có GRU 50/100/200 token, thời gian và F1. F1 giảm nhẹ từ 50 lên 100 rồi tăng ở 200, được nhận xét đúng; chưa có phân tích sai số giữa các seed.
- **Quantitative evaluation — 9/10:** Có accuracy, precision, recall, F1, thời gian, số tham số, [`learning_curves.png`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/learning_curves.png) và [`confusion_matrix.png`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/confusion_matrix.png). Một số mô hình được đánh giá trên test trong cùng quá trình so sánh; nên nói rõ bước chọn bằng validation để tránh dùng test làm căn cứ điều chỉnh.
- **Error analysis — 5/5:** [`error_analysis.md`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/error_analysis.md) phân tích 3 ca đúng, 3 ca sai, kèm nhãn, xác suất, độ dài và tác động có thể của truncation.
- **Reflection — 4/5:** [`reflection.md`](https://github.com/mactuongmvt/NLP-Labs/blob/89c028c3efc0fe147a4dbaa249619365f7baf247/labs/lab05/reflection.md) nêu baseline, gates, trade-off độ dài và attention; phần tổng kết khá ngắn so với lượng kết quả trong notebook.
- **Individual learning check — 0/5:** Không có bằng chứng về phần kiểm tra trực tiếp trong repo.

## Ưu điểm

- Notebook đã lưu output theo thứ tự các ô; có thể đối chiếu mã, bảng số và biểu đồ ngay trong một luồng.
- Tách baseline Lab 04 trên dữ liệu đầy đủ khỏi baseline chạy lại trên tập con Lab 05, tránh so trực tiếp hai tập test khác nhau.
- Báo cáo lỗi thận trọng: không tự động kết luận các lỗi là do cơ chế sequence modeling.

## Nhược điểm

- Kết luận so sánh kiến trúc và sequence length dựa trên một seed; cần nhiều lần chạy hoặc khoảng biến thiên để biết chênh lệch nhỏ có ổn định không.
- `reflection.md` có thể đối chiếu rõ hơn từng dự đoán ban đầu với kết quả đo và giải thích vì sao F1 của GRU ở 100 token thấp hơn ở 50 token.
- `README.md` còn một ký tự thừa ở cuối; nên chỉnh lại hướng dẫn và trình bày trước khi nộp.

## Bài học rút ra

- Khi đổi độ dài chuỗi, chất lượng không nhất thiết tăng đơn điệu: GRU F1 lần lượt `0,651`, `0,643`, `0,726` ở 50/100/200 token.
- Accuracy, precision, recall và F1 có thể xếp hạng mô hình khác nhau; GRU ở 100 token có accuracy cao nhất trong ba kiến trúc nhưng LSTM có F1 cao hơn.
- Baseline cùng split là mốc so sánh cần thiết; TF-IDF trên tập con đạt F1 `0,872`, cao hơn cả ba mô hình hồi quy trong cấu hình này.
