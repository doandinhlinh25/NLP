# Chấm điểm Lab 05 — Bài 8

- **Repo:** [doandinhlinh25/NLP — lab5](https://github.com/doandinhlinh25/NLP/tree/ddce87675b174b32385da814c0ac7eed88cba03b/lab5).
- **Căn cứ:** Rubric trong `lab5/W5.pdf`; rà soát commit `ddce87675b174b32385da814c0ac7eed88cba03b` ngày 28/10/2026.
- **Tổng điểm:** **90/100 = 9,0/10**. Mục vấn đáp cá nhân 3 phút chưa thể xác nhận từ repo nên tạm tính 0/5; có thể cập nhật khi giảng viên kiểm tra trực tiếp.

## Điểm theo rubric

- **Theory & calculation — 7/10:** [`calculations.pdf`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/calculations.pdf) tính đúng một bước RNN (`[0,67; 0,30]`, sau `tanh` xấp xỉ `[0,585; 0,291]`) và giải thích information flow, thứ tự từ. Hai trang viết tay chưa có phép tính LSTM gates; phần vanishing gradient chỉ được bàn thêm trong `reflection.md`.
- **Prediction — 10/10:** [`prediction.md`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/prediction.md) trả lời đủ 5 câu, có dự đoán, lý do và mức tin cậy.
- **Dataset & sequence analysis — 10/10:** [`experiment_details.json`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/experiment_details.json) và [`sequence_lengths.png`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/sequence_lengths.png) mô tả 50.000 review, phân bố lớp, độ dài, bản trùng; có giải thích padding/truncation và tỷ lệ token được giữ.
- **RNN cell implementation — 15/15:** [`rnn_cell.py`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/rnn_cell.py) viết phép cập nhật thủ công và đối chiếu với `torch.nn.RNNCell` bằng cùng tham số.
- **RNN/LSTM/GRU experiment — 18/20:** [`run_experiments.py`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/run_experiments.py) huấn luyện đủ ba kiến trúc và TF-IDF baseline trên cùng tập con. Ngân sách chỉ 2 epoch; điểm F1 của các mô hình hồi quy còn thấp, nên kết luận so sánh kiến trúc cần giữ trong phạm vi lần chạy này.
- **Sequence length experiment — 10/10:** [`results.csv`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/results.csv) có GRU ở 50/100/200 token, F1 và thời gian; [`length_tradeoff.png`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/length_tradeoff.png) trực quan hóa đánh đổi.
- **Quantitative evaluation — 10/10:** Bảng kết quả có accuracy, precision, recall, F1, thời gian và số tham số; có [`learning_curves.png`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/learning_curves.png) và [`confusion_matrix.png`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/confusion_matrix.png).
- **Error analysis — 5/5:** [`error_analysis.md`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/error_analysis.md) có 3 mẫu đúng, 3 mẫu sai từ test và trình bày nguyên nhân dưới dạng giả thuyết.
- **Reflection — 5/5:** [`reflection.md`](https://github.com/doandinhlinh25/NLP/blob/ddce87675b174b32385da814c0ac7eed88cba03b/lab5/reflection.md) liên hệ số liệu với độ dài, chi phí, giới hạn của RNN và bước chuyển sang attention.
- **Individual learning check — 0/5:** Không có bằng chứng về phần kiểm tra trực tiếp trong repo.

## Ưu điểm

- Pipeline tái lập được, dùng train/validation/test riêng và chỉ xây từ vựng trên train.
- Có baseline mạnh để tránh kết luận sai rằng RNN/LSTM/GRU luôn vượt TF-IDF; kết quả thực tế cho thấy TF-IDF F1 `0,856`, cao hơn GRU 100 token `0,601`.
- Trình bày đầy đủ đánh giá định lượng và 6 trường hợp phân tích lỗi có đối chiếu ngữ cảnh.

## Nhược điểm

- Bản tính tay mới có RNN và thứ tự từ; cần thêm phép tính `c_t=f_tc_{t-1}+i_t\tilde c_t` và giải thích cụ thể forget/input gates.
- Chỉ huấn luyện 2 epoch trên tập con 7.000 review, vì vậy kết quả các mô hình hồi quy chưa cho thấy năng lực hội tụ sau thời gian đủ dài.
- Notebook chưa lưu output; người chấm phải chạy script hoặc mở các tệp kết quả riêng để kiểm tra.

## Bài học rút ra

- Đánh giá mô hình chuỗi cần đặt cạnh baseline đơn giản, không chỉ so các kiến trúc neural với nhau.
- Tăng độ dài giữ thêm ngữ cảnh nhưng tăng thời gian tính toán; F1 và thời gian phải được báo cáo cùng nhau.
- Một dự đoán sai có thể liên quan đến truncation, cách diễn đạt hoặc huấn luyện ngắn; không nên quy một nguyên nhân duy nhất chỉ từ một ví dụ.
