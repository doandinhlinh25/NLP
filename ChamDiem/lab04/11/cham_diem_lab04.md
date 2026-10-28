# Chấm điểm Lab 04 - Bài 11

**Bài nộp:** [Lavie306/HUS-NLP-2026 - Lab4](https://github.com/Lavie306/HUS-NLP-2026/tree/main/Lab4)  
**Tổng điểm:** **82/100 (8.2/10)**

Đối chiếu rubric trong `lab4/W4.pdf` với các tệp công khai tại nhánh `main` ngày 21/10/2026. Điểm vấn đáp cá nhân chỉ có thể xác nhận khi kiểm tra trực tiếp.

| Hạng mục W4 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 9/10 | [`Calculations.pdf`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/Calculations.pdf) viết tay đúng logit `0.2`, sigmoid khoảng `0.5498`, hai ngưỡng `0.5/0.8`, precision `0.8889`, recall `0.8`, F1 khoảng `0.8421`; nêu được bẫy accuracy 95%. Tuy nhiên không có `calculations.md` như danh mục nộp bắt buộc. |
| Prediction | 10/10 | [`prediction.md`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/prediction.md) có đủ 5 dự đoán, lý do và mức tin cậy. |
| Data inspection & split | 9/10 | [`classification.ipynb`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/classification.ipynb) kiểm tra 49.577 mẫu sau lọc trùng, 2 lớp gần cân bằng, missing và độ dài; chia phân tầng 29.745/9.916/9.916 trước khi fit. Phần thống kê trùng lặp chỉ báo `0` *sau* khi loại trùng, không cho biết số bản ghi trùng ban đầu. |
| Baseline implementation | 15/15 | Notebook dùng pipeline TF-IDF unigram + Logistic Regression, fit trên train, ghi thời gian `4.23s`, từ vựng `48.415` và đủ accuracy/precision/recall/macro F1 trên validation. |
| Representation comparison | 15/15 | Có Average Word2Vec + Logistic Regression học embedding trên train, thêm unigram+bigram và Linear SVM. Macro F1 validation của TF-IDF unigram+bigram `0.9017` cao hơn Average Word2Vec `0.8624`; có thí nghiệm đảo thứ tự `not good`/`good not`. |
| Quantitative evaluation | 11/15 | Notebook chọn mô hình bằng validation macro F1, rồi test mô hình được chọn: macro F1 `0.8993`, confusion matrix `[[4386,553],[445,4532]]`. Nhưng [`results.csv`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/results.csv) là kết quả từ [`preprocessing.py`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/preprocessing.py), đánh giá cả 4 mô hình trực tiếp trên test với cách chia khác; bigram F1 ở đây là `0.8966`. README trích bảng này trong khi notebook cho kết quả khác. Cần chỉ rõ run/split tương ứng và một bảng kết quả thống nhất. |
| Error analysis | 3/10 | [`error_analysis.md`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/error_analysis.md) chỉ mô tả nhóm lỗi chung; không trình bày 3 ca đúng và 3 ca sai kèm văn bản, nhãn thật, nhãn dự đoán và phân tích riêng từng ca. Báo cáo dẫn `misclassified.csv`, `experiment_summary.json` nhưng hai tệp không có trong thư mục nộp; số FP/FN `1478/1333` cũng không khớp confusion matrix notebook (`553/445`). |
| Model interpretation | 5/5 | Notebook trích top 10 trọng số dương/âm của Logistic Regression, gồm `great`, `excellent` và `worst`, `bad`, rồi liên hệ với tín hiệu cảm xúc. |
| Reflection | 5/5 | [`reflection.md`](https://github.com/Lavie306/HUS-NLP-2026/blob/main/Lab4/reflection.md) giải thích sparse/lexical so với dense/semantic, mất thứ tự khi lấy trung bình, lợi ích và chi phí của bigram, leakage và hướng sang mô hình ngữ cảnh. |
| Individual learning check | 0/5 | Repository không cung cấp bằng chứng về buổi kiểm tra cá nhân 3 phút; cần chấm trực tiếp mục này. |
| **Tổng** | **82/100** | **8.2/10** |

## Ưu điểm

- Notebook chia dữ liệu trước khi fit TF-IDF và Word2Vec; chọn mô hình bằng validation macro F1 rồi mới đánh giá test trong luồng notebook.
- Có bốn cấu hình mô hình, báo cáo nhiều metric, confusion matrix, trọng số đặc trưng và thí nghiệm về thứ tự từ.
- Bài tính tay và năm dự đoán trước thực nghiệm nhìn chung chính xác, có lý do rõ.

## Nhược điểm và hướng cải thiện

1. Đồng bộ `README.md`, `results.csv`, script và notebook về cùng split, seed, mô hình được chọn và metric; ghi rõ mỗi bảng thuộc run nào.
2. Viết lại `error_analysis.md` theo ít nhất 3 dự đoán đúng và 3 dự đoán sai thật, có trích đoạn, true/predicted label và nguyên nhân riêng; sửa số FP/FN, xóa hoặc bổ sung các tệp đang được dẫn nhưng chưa nộp.
3. Ghi số duplicate trước khi loại trùng và cung cấp `calculations.md` đúng tên deliverable.

## Học được gì từ repo

- Trên IMDb, thêm bigram cải thiện macro F1 validation của TF-IDF + Logistic Regression từ `0.8978` lên `0.9017`, trong khi Average Word2Vec chỉ đạt `0.8624` ở run notebook.
- Tách validation để chọn mô hình và giữ test cho lần đánh giá cuối giúp diễn giải kết quả đáng tin cậy; mọi báo cáo cần cùng một nguồn số liệu để người đọc đối chiếu được.
- Confusion matrix và ví dụ lỗi cụ thể cần đi cùng nhau: nhận xét chung về phủ định hoặc mỉa mai chưa thay thế được phân tích từng dự đoán.
