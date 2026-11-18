# Prediction before experiment

Hãy ghi dự đoán và lý do của bạn trước khi chạy model. Không xem kết quả thực nghiệm để điền phần này.

**Lưu ý về thời điểm:** run đã hoàn tất trước khi ghi phần này, nên các ý dưới đây là dự đoán lý thuyết hồi cứu, không phải dự đoán được ghi trước thí nghiệm. Mình không trình bày chúng như dự đoán tiên nghiệm; nếu rubric yêu cầu bằng chứng thời gian, cần báo TA đúng tình huống này.

1. Transformer pretrained so với baseline lexical features: dự đoán mô hình nào có entity-level F1 tốt hơn, và vì sao?

   **Dự đoán lý thuyết (ghi hồi cứu):** Pretrained Transformer có entity-level F1 cao hơn baseline vì biểu diễn theo ngữ cảnh kết hợp thông tin hai phía và kiến thức từ pretraining; baseline từ/cửa sổ cục bộ ít khái quát hơn với tên chưa thấy. Tuy nhiên, sample và một epoch có thể giới hạn lợi thế.

2. Entity nhiều token so với entity một token: entity nào dự đoán boundary chính xác hơn? Giải thích.

   **Dự đoán lý thuyết (ghi hồi cứu):** Entity một token có thể dễ bắt boundary hơn vì chỉ cần quyết định một vị trí; entity nhiều token cần đúng cả điểm bắt đầu, phần tiếp nối và điểm kết thúc. Ngoại lệ là tên đơn lẻ mơ hồ, trong khi cụm nhiều từ có ngữ cảnh rõ.

3. Entity hiếm trong train có nhiều khả năng bị bỏ sót/nhầm loại hơn không? Nêu cơ chế dự kiến.

   **Dự đoán lý thuyết (ghi hồi cứu):** Entity hiếm có nguy cơ bị bỏ sót hoặc nhầm loại cao hơn vì mô hình thấy ít ví dụ để học cách viết và ngữ cảnh; tên lạ còn có thể bị chia subword. Ngữ cảnh và pretraining có thể giảm nhưng không xóa hẳn vấn đề này.

**Đối chiếu sau run:** Transformer đạt entity F1 .6797, cao hơn baseline .6275; token F1 lần lượt .7631 và .7048. Entity 2 từ đạt F1 .799, cao hơn nhóm 1 từ .664; nhóm 3 từ .592, nhóm 4+ .500 nhưng chỉ support 6. Vì vậy dự đoán entity nhiều từ khó hơn không đúng với nhóm 2 từ trong mẫu này, nhưng xu hướng ở nhóm dài hơn không ổn định. MISC có F1 .402 và support 98; không có bảng tần suất từng entity hiếm, nên chưa thể kết luận riêng tác động rarity.
