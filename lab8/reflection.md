# Reflection (300–500 words)

Sau LAB 07, chúng ta đã biết cách xây dựng một Transformer. Sau LAB 08, tại sao trong thực tế người ta thường không bắt đầu bằng việc train Transformer từ đầu?

Trong thực tế, huấn luyện một Transformer từ đầu đòi hỏi lượng dữ liệu lớn, thời gian và tài nguyên tính toán đáng kể. Một tập downstream như IMDb chỉ cung cấp nhãn cho một lượng review hữu hạn. Nếu khởi tạo mọi trọng số ngẫu nhiên, mô hình phải đồng thời học cấu trúc ngôn ngữ và quy tắc phân loại sentiment từ số ví dụ đó. Điều này dễ dẫn đến overfitting và làm kết quả phụ thuộc mạnh vào cách lấy mẫu.

Pretraining tách việc học biểu diễn tổng quát khỏi mục tiêu cụ thể. Trong masked language modeling, mô hình dự đoán token bị che dựa trên ngữ cảnh. Quá trình này dùng văn bản lớn chưa cần nhãn, nhờ đó encoder học quan hệ giữa từ và ngữ cảnh trước khi biết task downstream. Khi fine-tuning, classification head ánh xạ biểu diễn sequence sang nhãn positive/negative; đồng thời encoder có thể cập nhật bằng gradient để thích nghi với sentiment. Feature extraction đóng băng encoder, nên ít tham số cần tối ưu và thường rẻ hơn, nhưng biểu diễn không điều chỉnh cho task.

Lợi ích chính là hiệu quả dữ liệu: mô hình pretrained thường cần ít nhãn downstream hơn để đạt chất lượng hữu ích. Đổi lại, model lớn làm inference và fine-tuning tốn bộ nhớ, compute và thời gian. Chi phí phải được đánh giá cùng accuracy, precision, recall và F1. Sequence length cũng là quyết định thực tế vì self-attention có chi phí xấp xỉ bậc hai theo độ dài chuỗi.

Pretraining không đảm bảo hiểu mọi miền. IMDb chứa review phim; model chuyển sang tin tức, văn bản khoa học hoặc review sản phẩm có thể gặp domain shift về từ vựng, phong cách và cách biểu đạt sentiment. Tokenizer có thể phân mảnh từ hiếm, truncation làm mất thông tin ở review dài, và fine-tuning quá mức có thể gây catastrophic forgetting. Vì vậy cần kiểm tra validation/test độc lập, phân tích false positives/negatives, hiệu chỉnh niềm tin và bổ sung dữ liệu đúng miền. Lần chạy gọn dùng BERT-tiny 4.39 triệu tham số, 500 review, một epoch và 100 test mẫu; F1 vì vậy nhiễu và chỉ mô tả cấu hình này. Ứng dụng cần cân bằng chất lượng, độ trễ, dung lượng và chi phí cập nhật. Kết quả lab chỉ hỗ trợ kết luận cho dataset, split, model và tài nguyên đã báo cáo; chúng không chứng minh model hiểu mọi ngữ cảnh.

*Bản reflection AI hỗ trợ theo yêu cầu; cập nhật thêm bằng quan sát thực tế từ lần chạy của bạn.*
