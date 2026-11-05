# Predictions before experiments

## 1. Pretrained fine-tuning vs from scratch

**Prediction:** fine-tuned pretrained Transformer sẽ có F1 cao hơn model cùng kiến trúc khởi tạo ngẫu nhiên trên cùng lượng dữ liệu/epoch.

**Reason:** pretrained encoder bắt đầu với biểu diễn ngữ cảnh đã học từ corpus lớn, còn model scratch phải học biểu diễn và ranh giới sentiment từ tập downstream nhỏ.

**Expected evidence:** F1/accuracy tốt hơn và validation loss hội tụ thấp hơn cho fine-tuning. Thời gian từng bước và tổng thời gian cần đo vì phụ thuộc số tham số/thiết bị.

## 2. Frozen vs fine-tuned

**Prediction:** frozen encoder có thể phân loại tốt hơn ngẫu nhiên nhưng thường dưới fine-tuning (C).

**Reason:** biểu diễn pretrained có tín hiệu ngôn ngữ tổng quát; head học ánh xạ biểu diễn sang nhãn. Fine-tuning còn cho encoder thích nghi với sentiment IMDb.

**Expected evidence:** so sánh F1 và thời gian trong cùng split/epoch; frozen cập nhật ít tham số hơn.

## 3. 10% training data

**Prediction:** scratch model giảm mạnh hơn khi train set co xuống 10%.

**Reason:** pretrained weights đem theo biểu diễn đã học từ dữ liệu ngoài downstream set; scratch phụ thuộc trực tiếp hơn vào nhãn train để hình thành representation.

**Expected evidence:** đường cong F1 theo tỷ lệ dữ liệu suy giảm thoải mái hơn với pretrained. Pipeline đo data-efficiency cho fine-tuning; để đo chênh lệch mức giảm scratch phải chạy thêm các fractions với scratch.
