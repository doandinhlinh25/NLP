# Error Analysis

Các ví dụ bên dưới được trích từ `error_examples.csv`. Nhãn true/pred: 0 = negative, 1 = positive.

| Text (excerpt) | True | Pred | Confidence | Error type | Hypothesis |
|---|---:|---:|---:|---|---|
| “...fine actors...horrible script...powerful performances...improve...overall quality...final scene...incredibly poor...” | 0 | 1 | 0.570 | Mixed sentiment / context | Review alternates between praise and criticism; sentiment changes across clauses and the final negative judgment may be diluted by positive terms. |
| “...film had so much potential...good cast...that is about it...heartless...ignore this film...” | 0 | 1 | 0.539 | Sarcasm / contrast | Opens with praise, then retracts it sarcastically. The model may over-weight the early positive language. |
| “Poorly written...foggy plot...commendable cinematography...shallow...amateur acting...Grade: D+” | 0 | 1 | 0.538 | Mixed sentiment / lexical cue | A positive word (“commendable”) occurs within an otherwise negative review; likely insufficient aggregation of the full passage. |
| “...better than I thought...first few segments...muddled...movie takes off...adventures...” | 1 | 0 | 0.561 | Negation / long-range context | Sentiment turns from reserved criticism to positive appraisal later; full review context matters. |
| “...put off at first...intrigued...couldn't wait...pleased with everything...not especially upbeat...found it...” | 1 | 0 | 0.547 | Negation / sentiment shift | Negative and positive phrases coexist, with a positive conclusion after initial hesitation. |
| “...huge fan...couldn't wait...not disappointed...most powerful...great to see everyone...” | 1 | 0 | 0.545 | Negation / idiom | “Not disappointed” is a positive idiom containing a negation; a weak model can misread the polarity. |
| “...fine actors...horrible script...powerful performances...improve...final scene...incredibly poor...” | 0 | 1 | 0.570 | Top-confidence error | Highest-scoring wrong prediction, but 0.570 remains in the low confidence band (<0.6). |
| “...better than I thought...muddled...movie takes off...” | 1 | 0 | 0.561 | Top-confidence error | Another top-scoring mistake; still low confidence and not a high-confidence failure. |

Trong lần chạy này, confidence của cả 100 dự đoán đều thuộc nhóm thấp (<0.6); hai mẫu sai được chọn là các lỗi có confidence cao nhất (xấp xỉ 0.57), không phải các dự đoán tự tin theo ngưỡng thông thường. Vì vậy yêu cầu “2 examples model dự đoán rất tự tin nhưng sai” chưa đạt ở cấu hình smoke-run này.

## Câu hỏi sau thực nghiệm

- Q1: Data curve đo fine-tuning: F1 lần lượt 0.000, 0.667, 0.667, 0.495 ở 10%, 25%, 50%, 100%. Chưa chạy scratch ở các tỷ lệ nên chưa thể đối chiếu mức giảm giữa hai phương pháp.
- Q2: Trên toàn bộ train sample 500, pretrained fine-tune đạt F1 0.591 và accuracy 0.530, còn scratch đạt F1 0.667 và accuracy 0.500. Chỉ một lần chạy nhỏ, nên chưa có bằng chứng pretrained thắng; cũng không thể kết luận về hiểu ngôn ngữ nói chung.
- Q3: Frozen F1 0.000, accuracy 0.500; fine-tuning F1 0.591, accuracy 0.530. Trong cấu hình này frozen không gần fine-tuned, dù test sample nhỏ khiến kết quả bất ổn.
- Q4: Fine-tuning vượt frozen 0.591 F1 điểm; encoder thích nghi có thể giúp task, nhưng chênh lệch cũng chịu ảnh hưởng random initialization của head, ít epoch và sample nhỏ.
- Q5: Chưa có domain B nên không có observation out-of-domain. Cần train trên IMDb và test thêm tập news/product có label tương thích.
- Q6: F1 theo max length 64/128/256 lần lượt 0.622/0.355/0.167, thời gian 4.63/8.59/22.78 giây. 256 tốn nhiều hơn nhưng score thấp hơn trong lần chạy này; độ dài lớn không bảo đảm tốt hơn.

Gắn mọi diễn giải với số liệu/examples sau khi chạy. IMDb-only không đo domain shift.
