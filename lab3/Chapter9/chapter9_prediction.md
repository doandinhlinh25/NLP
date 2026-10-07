# Chapter 9 — Prediction trước Experiment

> Các nhận định dưới đây là dự đoán được ghi lại **trước khi chạy mô hình**. Kết quả thực nghiệm có thể khác, tùy thuộc vào corpus, tần suất từ, cách tiền xử lý và siêu tham số huấn luyện.

## Prediction 1 — Các từ gần nhau nhất

**Prediction:** Cặp `doctor` và `physician`có độ tương đồng cao nhất vì cả hai thường cùng chỉ một nghề nghiệp hoặc vai trò trong lĩnh vực y tế. Từ `hospital` cũng có thể gần với `doctor` và `physician`, nhưng thấp hơn vì nó là một địa điểm/tổ chức, không phải từ đồng nghĩa. `banana` và `car` dự đoán có độ tương đồng thấp với nhóm từ y tế và với nhau.

**Reason:** Theo distributional hypothesis, các từ xuất hiện trong ngữ cảnh giống nhau sẽ có vector gần nhau. `doctor` và `physician` có khả năng cùng xuất hiện với các từ như `patient`, `treatment`, `disease` hoặc `hospital`. Trong khi đó, `banana` thường xuất hiện trong ngữ cảnh thực phẩm và `car` trong ngữ cảnh giao thông.

**Confidence:** Cao đối với quan hệ `doctor` – `physician`; trung bình đối với thứ hạng chính xác của các cặp còn lại vì thứ hạng phụ thuộc vào nội dung corpus.

## Prediction 2 — Tăng context window từ 2 lên 5

**Prediction:** Khi tăng context window từ `2` lên `5`, similarity sẽ thay đổi. Mô hình nhiều khả năng học được các quan hệ chủ đề rộng hơn, chẳng hạn `doctor` gần `hospital` hoặc `patient` hơn. Tuy nhiên, một số quan hệ xuất hiện gần nhau về mặt cú pháp có thể kém rõ hơn vì model thu nhận thêm các từ ở xa và ít liên quan trực tiếp.

**Reason:** Window bằng `2` ưu tiên các từ đứng rất gần target word, nên phản ánh local context mạnh hơn. Window bằng `5` bao quát nhiều từ hơn trong câu, cung cấp thêm thông tin về chủ đề nhưng cũng có thể đưa thêm nhiễu vào vector ngữ cảnh.

**Confidence:** Trung bình, vì tác động thực tế phụ thuộc vào độ dài câu, cấu trúc corpus và số lần các từ cùng xuất hiện.

## Prediction 3 — Tăng embedding dimension từ 50 lên 100 rồi 300

**Prediction:** Chất lượng embedding không chắc chắn tăng liên tục khi tăng số chiều. Vector `50` chiều có thể chưa đủ khả năng biểu diễn nhiều quan hệ ngữ nghĩa; `100` chiều có thể là điểm cân bằng tốt hơn; còn `300` chiều chỉ có khả năng tốt hơn khi corpus đủ lớn và đa dạng. Với corpus nhỏ, `300` chiều có thể tạo vector nhiễu, khó học ổn định, tốn thời gian huấn luyện và tăng kích thước mô hình.

**Reason:** Số chiều cao hơn làm tăng capacity của mô hình, nhưng cũng làm tăng lượng dữ liệu cần thiết để ước lượng các tham số đáng tin cậy. Vì vậy, dimension cần được chọn cùng với quy mô corpus, số epoch và mục tiêu đánh giá thay vì giả định rằng càng lớn càng tốt.

**Confidence:** Cao cho nhận định “không chắc chắn tăng”; trung bình cho dimension tốt nhất vì cần thực nghiệm để xác định.

## Prediction 4 — `doctor` và `physician` với corpus chỉ 100 câu

**Prediction:** `doctor` và `physician` không chắc chắn gần nhau nếu corpus chỉ có `100` câu. Nếu một hoặc cả hai từ xuất hiện ít, hoặc không có các ngữ cảnh tương tự, model sẽ không có đủ bằng chứng để học được quan hệ đồng nghĩa. Ngược lại, chúng vẫn có thể gần nhau nếu xuất hiện đủ nhiều trong các câu có context tương tự.

**Reason:** Word embedding học từ thống kê đồng xuất hiện. Corpus nhỏ làm số lần quan sát ít, vector dễ thiếu ổn định và các từ hiếm có ít context để so sánh. Do đó, ý nghĩa trực giác của từ không tự động bảo đảm rằng hai vector sẽ gần nhau.

**Confidence:** Cao, vì chất lượng của embedding phụ thuộc mạnh vào dữ liệu huấn luyện và tần suất xuất hiện của từ.

## Tiêu chí kiểm tra sau experiment

Sau khi chạy mô hình, cần so sánh các dự đoán trên với cosine similarity và danh sách từ gần nhất. Nếu kết quả khác dự đoán, cần kiểm tra tần suất từ, các câu chứa từ đó, context window, dimension và chất lượng tiền xử lý corpus trước khi kết luận.
