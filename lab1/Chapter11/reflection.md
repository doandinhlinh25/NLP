# Reflection — Lab 01

Prediction về vocabulary khoảng 200K là gần đúng: pipeline tối giản đo được 193,837 terms, thấp hơn dự đoán ban đầu. Prediction về sparsity là đúng, nhưng mức 99.912219% vẫn trực quan hơn nhiều so với chỉ nói “sparse”. Nó cho thấy vector có gần 194K chiều dù trung bình mỗi document chỉ có khoảng 170 feature khác 0.

Kết quả bất ngờ nhất là preprocessing không nhiều hơn theo một hướng duy nhất. Pipeline B giảm vocabulary còn 174,270 và là matrix thưa nhất (99.932066%) vì bỏ stopwords. Ngược lại, Pipeline C có OOV feature thấp hơn nhưng mỗi document có nhiều character n-gram nên matrix chỉ sparse 98.179959%. Vì vậy vocabulary nhỏ hơn không tự động đồng nghĩa search tốt hơn.

Evidence mạnh nhất là bảng ablation kết hợp Mean P@5 trên cùng tập nhãn nhỏ. Nó cho phép so sánh representation bằng số liệu thay vì giả định stopword removal hoặc subword luôn tốt. Hạn chế là bảy query và relevance labels còn nhỏ; không nên coi đây là benchmark cuối cùng.

Failure quan trọng nhất là query `barbecue cooking class`: document 0 nói về “BBQ class” nhưng không được Pipeline B đưa vào top 5. TF-IDF dựa vào token trùng nhau nên không hiểu `barbecue` và `BBQ` cùng nghĩa. Các document được lấy lên khớp với từ chung như `cooking` hoặc `class`, nhưng không đúng intent.

Nếu xây lại search engine, em sẽ giữ lexical index cho exact match, bổ sung synonym/abbreviation normalization và thử hybrid retrieval với embedding. AI được dùng để hỗ trợ khung code, unit test, và tổ chức thí nghiệm; số liệu ghi trong báo cáo được chạy trên corpus cục bộ và cần được em tự kiểm tra, giải thích trước khi nộp.
