# Chấm điểm Lab 04 - Bài 22

**Bài nộp:** `lab4/`  
**Tổng điểm:** **92/100 (9.2/10)**

Đánh giá dựa trên `calculations_1.pdf`, `prediction.md`, `classification.ipynb`, `preprocessing.py`, `results.csv`, `confusion_matrix.png`, `error_analysis.md` và `reflection.md`. Mục vấn đáp cá nhân không thể xác nhận từ repository.

| Hạng mục W4 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 7/10 | PDF tính tay có Bài 1 đúng (`z = 0.2`, sigmoid xấp xỉ `0.55`) và trả lời đúng ngưỡng dự đoán. Bài 3 chưa tính đủ precision, recall, F1 từ confusion matrix; Bài 4 mới nêu recall bằng 0, chưa trình bày đầy đủ F1/macro metrics. Deliverable cũng đang là `calculations_1.pdf`, trong khi rubric yêu cầu `calculations.md`. |
| Prediction | 10/10 | Có đủ 5 dự đoán trước experiment, mỗi mục có Prediction, Reason và Confidence. Các nhận định về TF-IDF, Average Word2Vec, vocabulary, stopwords và accuracy/F1 phù hợp task IMDb. |
| Data inspection & split | 10/10 | Notebook kiểm tra missing text/label, duplicate, số lớp, class distribution và document length; biểu đồ class distribution được lưu. Split stratified 60/20/20 xảy ra trước mọi bước fit, nên tránh leakage. |
| Baseline implementation | 15/15 | Có pipeline `TfidfVectorizer` + `LogisticRegression`, fit bằng train và ghi training time, vocabulary size, accuracy, macro precision, macro recall, macro F1 ở validation. |
| Representation comparison | 15/15 | So sánh TF-IDF + Logistic Regression với Average Word2Vec + Logistic Regression; Word2Vec chỉ học trên train. Có thêm unigram + bigram và TF-IDF + Linear SVM. |
| Quantitative evaluation | 15/15 | `results.csv` báo cáo đủ metrics cho các model; Linear SVM được chọn theo validation macro F1 (`0.8981`) và test đúng một lần (`0.8975`). Có confusion matrix với 4 ô và phân tích false positive/false negative. |
| Error analysis | 10/10 | Có 3 ca phân loại đúng, 3 ca sai, nhãn thật/dự đoán và nguyên nhân: short text, lexical ambiguity, sarcasm, negation và mixed sentiment. |
| Model interpretation | 5/5 | Notebook trích xuất top TF-IDF features từ hệ số Logistic Regression; các feature positive/negative như `great`, `excellent`, `worst`, `bad` có ý nghĩa đúng với task. |
| Reflection | 5/5 | Trả lời đủ 7 câu, dùng số liệu thực nghiệm và giải thích đúng giới hạn của TF-IDF/Average Word2Vec, bigram, phủ định và context. |
| Individual learning check | 0/5 | Không có bằng chứng trong repository cho phần vấn đáp/kiểm tra cá nhân 3 phút; bài viết sẵn không thay thế được hạng mục này. |
| **Tổng** | **92/100** | **9.2/10** |

## Điểm mạnh

- Pipeline chia train/validation/test đúng thứ tự; vectorizer, Word2Vec và model chỉ fit trên train.
- Thí nghiệm có chiều sâu: baseline, Average Word2Vec, bigram và Linear SVM, với metrics macro thay vì chỉ accuracy.
- Linear SVM đạt macro F1 validation `0.8981` và test `0.8975`; lựa chọn model theo validation là hợp lý.
- Error analysis bám vào ví dụ thực tế, chỉ ra đúng failure do mỉa mai, phủ định và tín hiệu cảm xúc trái chiều.

## Điểm cần cải thiện

1. Khôi phục hoặc đổi tên `calculations_1.pdf` thành deliverable đúng quy ước, đồng thời bổ sung `calculations.md` nếu yêu cầu nộp bắt buộc theo W4.
2. Hoàn thiện Bài 3: ghi TP, FP, FN, TN và tính chính xác precision, recall, F1; hoàn thiện Bài 4 với macro F1/confusion matrix để chứng minh vì sao accuracy 95% gây hiểu nhầm.
3. Lưu bằng chứng cho individual learning check khi hoàn thành để nhận 5 điểm còn lại.
4. Ghi rõ phiên bản Python và các dependency đã chạy để việc tái lập kết quả thuận tiện hơn.

## Học được gì từ bài này

- Cùng một classifier nhưng representation khác nhau có thể tạo chênh lệch lớn: TF-IDF Logistic Regression đạt macro F1 validation `0.8918`, còn Average Word2Vec Logistic Regression đạt `0.8292`.
- Feature lexical mạnh cho sentiment IMDb, nhưng không thay thế được việc hiểu phủ định, mỉa mai và quan hệ giữa các từ.
- Đánh giá đáng tin cậy cần split không leakage, macro metrics, confusion matrix, model interpretation và phân tích từng lỗi; accuracy một mình không đủ.

