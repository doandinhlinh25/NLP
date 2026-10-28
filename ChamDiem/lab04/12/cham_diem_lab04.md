# Chấm điểm Lab 04 - Bài 12

**Bài nộp:** [ntrThanh/NLP-2026 - lab04](https://github.com/ntrThanh/NLP-2026/tree/master/lab04)  
**Tổng điểm:** **83/100 (8.3/10)**

Đối chiếu rubric trong `lab4/W4.pdf` với các tệp công khai tại nhánh `master` ngày 21/10/2026. Điểm vấn đáp cá nhân chỉ có thể xác nhận khi kiểm tra trực tiếp.

| Hạng mục W4 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 5/10 | [`calculations.pdf`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/calculations.pdf) tính đúng `z=0.2`, nhận diện TP=80, FP=10, FN=20, TN=90 và F1 khoảng `0.842`. Tuy nhiên chưa ghi giá trị sigmoid khoảng `0.5498`, Bài 2 chỉ xét ngưỡng `0.5` mà thiếu `0.8` và tác động lên precision/recall; công thức precision viết sai mẫu số dù kết quả số đúng. Bài 4 nêu cần xem micro F1, trong khi mô hình đoán toàn negative vẫn có micro F1 `0.95`; cần positive recall/F1 hoặc macro F1. Cũng thiếu `calculations.md` theo danh mục nộp. |
| Prediction | 10/10 | [`prediction.md`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/prediction.md) trả lời đủ 5 câu, mỗi câu có prediction, reason, confidence và phần đối chiếu sau chạy. |
| Data inspection & split | 10/10 | [`classification.ipynb`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/classification.ipynb) kiểm tra phân bố lớp, missing, duplicate và độ dài. Loại 418 bản trùng rồi chia phân tầng 70/15/15 thành 34.707/7.437/7.438; kiểm tra giao văn bản giữa ba tập bằng 0. TF-IDF và Word2Vec chỉ fit trên train. |
| Baseline implementation | 14/15 | TF-IDF + Logistic Regression chạy được, validation F1 lớp positive `0.8994`, test F1 `0.9028`; notebook in vocabulary `87.734` và thời gian khoảng `8.63s`. Các chỉ số thời gian/từ vựng không được lưu trong [`results.csv`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/results.csv), làm bảng tổng hợp chưa đủ như W4 yêu cầu. |
| Representation comparison | 15/15 | So sánh TF-IDF + LR, Average Word2Vec + LR (`0.8449` F1 trên test), thêm Linear SVM và unigram+bigram (`0.9074`). Word2Vec chỉ học trên train; notebook thử `this movie is good`/`this movie is not good`, và đo từ vựng bigram `555.050` cùng thời gian train. |
| Quantitative evaluation | 10/15 | Có accuracy, precision, recall, F1 và confusion matrix cho baseline. Tuy nhiên notebook tính test prediction cho cả bốn cấu hình và dùng test để so sánh/khẳng định bigram tốt nhất; validation chỉ được báo cáo cho baseline, chưa có bước chọn mô hình bằng validation trước khi xem test. `results.csv` là F1 lớp positive, chưa có macro metrics trong bảng so sánh. |
| Error analysis | 9/10 | [`error_analysis.md`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/error_analysis.md) có 3 ca đúng và 3 ca sai, trích đoạn, true/predicted label và giải thích cụ thể lexical ambiguity, mixed sentiment, cấu trúc nhượng bộ. Một số trọng số từ được gán trong diễn giải không nằm trong bảng top-feature của notebook, nên cần xuất bảng trọng số đầy đủ hoặc bỏ các con số chưa đối chiếu được. |
| Model interpretation | 5/5 | Notebook trích top 10 feature dương/âm với trọng số, chẳng hạn `great` `+8.6897` và `worst` `-10.4709`, và diễn giải chúng theo bài toán cảm xúc. |
| Reflection | 5/5 | [`reflection.md`](https://github.com/ntrThanh/NLP-2026/blob/master/lab04/reflection.md) giải thích khác biệt TF-IDF/Word2Vec, leakage, phủ định, bigram và mạch nối sang sequence models; có phần chuẩn bị câu hỏi vấn đáp. |
| Individual learning check | 0/5 | Phần trả lời viết sẵn trong reflection không chứng minh đã hoàn thành kiểm tra cá nhân 3 phút; mục này cần chấm trực tiếp. |
| **Tổng** | **83/100** | **8.3/10** |

## Ưu điểm

- Khảo sát dữ liệu kỹ, có bước loại trùng toàn bộ trước split và kiểm tra giao giữa train/validation/test.
- Thử nhiều representation và classifier, kèm thí nghiệm phủ định, bigram, trọng số đặc trưng và sáu ca phân tích cụ thể.
- Báo cáo rõ kích thước từ vựng và chi phí thời gian khi thêm bigram, giúp thấy sự đánh đổi hiệu năng và tài nguyên.

## Nhược điểm và hướng cải thiện

1. So sánh các cấu hình trên validation, chọn một cấu hình theo F1 hoặc macro F1, sau đó chỉ dùng test một lần; ghi rõ metric dùng để chọn.
2. Hoàn thiện bốn bài tính tay, đặc biệt ngưỡng `0.8`, giá trị sigmoid và bẫy micro F1 ở bài lệch lớp; nộp `calculations.md` đúng quy ước.
3. Thêm training time, vocabulary size và macro metrics vào bảng kết quả; làm rõ nguồn của các trọng số cụ thể trong phân tích lỗi.

## Học được gì từ repo

- Việc loại trùng trước split và kiểm tra giao văn bản là bước hữu ích để giảm nguy cơ cùng một review xuất hiện ở nhiều tập.
- Trên run được báo cáo, TF-IDF unigram+bigram đạt F1 lớp positive `0.9074`, cao hơn TF-IDF unigram `0.9028` nhưng từ vựng tăng từ `87.734` lên `555.050` và thời gian từ khoảng `8.63s` lên `29.55s`.
- Ví dụ `not good` cho thấy unigram và vector trung bình khó nắm quan hệ phủ định; đồng thời bài thực nghiệm nhắc rằng kết luận về mô hình tốt nhất phải dựa vào validation trước khi mở test.
