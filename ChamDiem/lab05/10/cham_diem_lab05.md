# Chấm điểm Lab 05 — Bài 10

- **Repo:** [vutiendat2302/natural_language_processing — practice/lab05](https://github.com/vutiendat2302/natural_language_processing/tree/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05).
- **Căn cứ:** Rubric trong `lab5/W5.pdf`; rà soát commit `5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9` ngày 28/10/2026.
- **Tổng điểm:** **60/100 = 6,0/10**. Mục vấn đáp cá nhân chưa thể xác nhận từ repo nên tạm tính 0/5. Các chỉ số trong CSV được ghi nhận là số liệu **được báo cáo**, chưa được xác thực bằng pipeline huấn luyện có thể chạy lại.

## Điểm theo rubric

- **Theory & calculation — 8/10:** [`calculations.md`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/calculations.md) trình bày đúng bước RNN và phép tính LSTM `c_t=0,82`, cùng ý nghĩa hidden state/gates. Một số phát biểu quá tuyệt đối, chẳng hạn gradient “bị triệt tiêu hoàn toàn” sau 20–30 token, chưa có điều kiện hoặc phép đo hỗ trợ.
- **Prediction — 9/10:** [`prediction.md`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/prediction.md) có 5 dự đoán, lý do, confidence. Dự đoán giảm accuracy 10–15% khi cắt 30 token được nêu quá cụ thể khi chưa có phép thử tương ứng.
- **Dataset & sequence analysis — 6/10:** Notebook có code tải IMDb, chia train/validation và đếm lớp, nhưng không lưu output. Phần thống kê độ dài và histogram chủ yếu xuất hiện dưới dạng văn bản báo cáo, chưa có luồng tính toán tương ứng trong notebook.
- **RNN cell implementation — 15/15:** [`rnn_cell.py`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/rnn_cell.py) tự cài `SimpleRNNCell`; chạy hàm kiểm tra trong phiên rà soát cho kết quả khớp `torch.nn.RNNCell` với sai khác tối đa khoảng `2,98×10⁻⁸`.
- **RNN/LSTM/GRU experiment — 8/20:** [`sequence_classification.ipynb`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/sequence_classification.ipynb) định nghĩa ba kiến trúc nhưng không có vòng huấn luyện hoặc bước tạo [`results.csv`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/results.csv). CSV có các chỉ số nhưng không thể truy lại từ notebook đã nộp.
- **Sequence length experiment — 5/10:** CSV báo cáo LSTM ở 50/100/200 token và thời gian; notebook không có code chạy lại ba điều kiện đó.
- **Quantitative evaluation — 5/10:** Có CSV, [`learning_curves.png`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/learning_curves.png) và [`confusion_matrix.png`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/confusion_matrix.png), nhưng notebook chỉ đọc tệp có sẵn và không lưu output. Ô cuối còn đọc `probe_results.json` không có trong danh sách tệp của commit.
- **Error analysis — 2/5:** [`error_analysis.md`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/error_analysis.md) nêu 3 ca đúng và 3 ca sai, nhưng là các câu minh họa, chưa chỉ ra index/tập test/xác suất từ kết quả dự đoán đã lưu; các khẳng định về nguyên nhân như forget gate “chủ động” xử lý từ `but` chưa được kiểm chứng.
- **Reflection — 2/5:** [`reflection.md`](https://github.com/vutiendat2302/natural_language_processing/blob/5ebcc92ba72350fe580ec6d78f5acc121e8f3bb9/practice/lab05/reflection.md) trả lời nhiều câu lý thuyết, nhưng một số kết luận quá chắc chắn so với bằng chứng thực nghiệm có thể tái lập.
- **Individual learning check — 0/5:** Không có bằng chứng về phần kiểm tra trực tiếp trong repo; câu trả lời viết sẵn không thay thế được vấn đáp.

## Ưu điểm

- Tài liệu lý thuyết và bài tính tay phong phú, trình bày rõ công thức RNN và LSTM.
- RNN cell tự triển khai có phép đối chiếu thực chạy thành công với PyTorch.
- Có bảng chỉ số, learning curves và confusion matrix để người đọc thấy các kết quả mà tác giả báo cáo.

## Nhược điểm

- Notebook chưa thực sự huấn luyện RNN/LSTM/GRU hoặc tạo các tệp kết quả; toàn bộ ô code chưa lưu output. Vì vậy phần định lượng hiện không thể tái lập từ bài nộp.
- Ô cuối của notebook tham chiếu `probe_results.json` nhưng tệp này không có trong repo; `Run All` sẽ dừng ở ô đó nếu không bổ sung tệp.
- Phân tích lỗi cần lấy 3 mẫu đúng và 3 mẫu sai thật từ tập test, kèm nhãn, xác suất và đoạn văn bản có thể truy lại; tránh khẳng định cơ chế gates là nguyên nhân khi chưa đo.

## Bài học rút ra

- Có bảng số và hình chưa đủ để chứng minh thí nghiệm; cần mã huấn luyện, cấu hình, seed và output nối được tới từng kết quả.
- Phân tích lỗi đáng tin cậy phải xuất phát từ dự đoán thật trên tập test, sau đó mới nêu giả thuyết về ngữ cảnh, phủ định hoặc truncation.
- Nên dùng phát biểu có điều kiện khi nói về vanishing gradient, thứ tự từ và hiệu quả của LSTM/GRU; kiến trúc cho phép một hành vi nhưng không chứng minh mô hình đã học hành vi đó.
