# Dự đoán trước experiment

**Dataset:**  
**Task:**  

## 1. TF-IDF + Logistic Regression

- Prediction: Regression có khả năng hoạt động tốt và là baseline mạnh.
- Reason: Dữ liệu phân loại văn bản thường có các từ/cụm từ đặc trưng cho từng nhãn; TF-IDF biểu diễn tốt các tín hiệu đó, còn Logistic Regression phù hợp với feature thưa và nhiều chiều.
- Confidence: Khá cao

## 2. Average Word2Vec có chắc tốt hơn TF-IDF?

- Prediction: Không chắc tốt hơn TF-IDF.
- Reason: Average Word2Vec mất word order và làm loãng tín hiệu từ quan trọng; TF-IDF giữ rõ các từ/cụm từ đặc trưng.
- Confidence: Cao

## 3. Tăng vocabulary size có chắc tăng accuracy?

- Prediction: Accuracy không chắc tăng.
- Reason: Có thể thêm thông tin hữu ích, nhưng cũng có thể thêm nhiễu, làm train chậm hơn và giảm khả năng tổng quát hóa.
- Confidence: Cao

## 4. Bỏ stopwords có chắc cải thiện performance?

- Prediction: Không chắc cải thiện.
- Reason: Stopwords đôi khi là nhiễu, nhưng cũng có thể mang thông tin ngữ nghĩa hoặc quan hệ ngữ pháp quan trọng.
- Confidence: Cao

## 5. Accuracy và F1 có thể cho kết luận khác nhau?

- Prediction: Có thể cho kết luận khác nhau.
- Reason: Accuracy tính tổng số dự đoán đúng, còn F1 cân bằng precision và recall, nên phản ánh tốt hơn hiệu quả ở từng lớp.
- Confidence: Rất cao.
