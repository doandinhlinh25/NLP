# Reflection — Representation và classification

**Dataset/task đã chọn:** IMDb Dataset - phân loại cảm xúc nhị phân cho review phim.

## 1. Representation/model nào phù hợp nhất với task? Vì sao?

TF-IDF unigram + Linear SVM tốt nhất trên validation với macro F1 `0.8981`; khi đánh giá một lần trên test, macro F1 là `0.8975`. Review IMDb thường có từ đánh giá trực tiếp như `great`, `excellent`, `worst` và `boring`; TF-IDF giữ rõ tín hiệu từ vựng đó, còn Linear SVM phù hợp với feature space thưa, nhiều chiều.

## 2. TF-IDF và Average Word2Vec tạo kết quả khác nhau thế nào?

TF-IDF + Logistic Regression đạt macro F1 validation `0.8918`, trong khi Average Word2Vec + Logistic Regression đạt `0.8292`. TF-IDF là biểu diễn sparse, nhấn mạnh từ hoặc phrase đặc trưng của review. Average Word2Vec là vector dense có tính ngữ nghĩa, nhưng phép trung bình làm yếu những từ quyết định sentiment và bỏ thứ tự từ.

## 3. Vì sao accuracy có thể gây hiểu nhầm khi class imbalance?

Một model có thể dự đoán toàn bộ là lớp phổ biến mà vẫn có accuracy cao. Vì vậy cần xem precision, recall, F1 theo lớp hoặc macro F1 và confusion matrix. IMDb sau làm sạch gần cân bằng (`49.81%` negative, `50.19%` positive), nên accuracy và macro F1 ở đây gần nhau; điều này không đúng với mọi dataset.

## 4. Average Word2Vec xử lý thứ tự từ và từ `not` thế nào?

Average Word2Vec chỉ cộng rồi lấy trung bình vector các từ nên không giữ thứ tự. Vì `not good` và `good not` chứa cùng token, vector trung bình của chúng giống nhau; mô hình không biết `not` đang phủ định `good`.

## 5. Bigram giúp được gì và làm feature space thay đổi ra sao?

Bigram tạo feature cho cụm như `not good`, nên có thể giữ một phần thông tin phủ định. Tuy nhiên vocabulary tăng từ `82,600` unigram lên `1,725,370` unigram + bigram và thời gian train tăng từ `6.51` giây lên `36.52` giây. Trong lần chạy này macro F1 validation của unigram + bigram là `0.8912`, không cao hơn unigram Logistic Regression (`0.8918`), nên chi phí feature space chưa đổi lấy cải thiện.

## 6. Confusion matrix cho thấy lỗi nào đáng chú ý nhất?

Trên test, confusion matrix theo thứ tự actual/predicted `negative`, `positive` là `[[4398, 542], [474, 4503]]`. Có 542 review negative bị dự đoán positive và 474 review positive bị dự đoán negative. False positive nhiều hơn nhẹ; các review có từ khen nhưng ý mỉa mai hoặc kết luận tiêu cực là nguồn lỗi đáng chú ý.

## 7. Kết quả nào dẫn tới contextual/sequence models ở lab tiếp theo?

Average Word2Vec không hiểu thứ tự, còn TF-IDF và Linear SVM dựa mạnh vào tín hiệu từ hoặc cụm từ. Những lỗi do phủ định, mỉa mai và sentiment trái chiều trong cùng review cho thấy cần mô hình sequence/context để biểu diễn quan hệ giữa các từ; đây là động lực cho RNN/LSTM/GRU và sau đó attention/Transformer.

