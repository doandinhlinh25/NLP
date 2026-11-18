# Tiêu chí chấm mẫu — LAB 09

**Căn cứ:** rubric và yêu cầu deliverables trong [`lab9/W9.pdf`](../../lab9/W9.pdf).  
**Tổng:** 100 điểm.  
**Mục đích:** hướng dẫn người chấm đối chiếu bài nộp; đây không phải điểm của một sinh viên cụ thể.

## Rubric

| Hạng mục | Điểm tối đa | Minh chứng cần xem |
|---|---:|---|
| Theory & calculation | 10 | `calculations.md`: shape tensor, giải thích token/entity metrics, precision/recall/F1 và các phép tính theo đề. |
| Prediction | 10 | `prediction.md`: dự đoán và lý do về Transformer so với baseline, entity length, rare entity; đối chiếu với kết quả. Kiểm tra thời điểm ghi dự đoán theo yêu cầu của đề. |
| BIO & label alignment | 15 | BIO đúng với entity spans; giải thích Strategy A/B/C; word-to-subword alignment nhất quán; hiểu `IGNORE`/`-100` trong loss. |
| Baseline implementation | 10 | Baseline đơn giản (ví dụ logistic regression với token features), chạy được và được so sánh trên cùng test set. |
| Transformer NER implementation | 15 | Token-classification head, fine-tuning, word IDs/alignment, labels, logits và dự đoán được thể hiện; không chỉ gọi sẵn một pipeline NER. |
| Tokenization experiment | 10 | Thống kê ít nhất 20 câu về words, tokens/subwords, tỉ lệ chia; so sánh câu ngắn/dài và entity/non-entity, có diễn giải. |
| Entity-level evaluation | 10 | Precision, recall, F1 theo exact entity span + type; báo cáo baseline và Transformer trên cùng test set; phân tích theo entity type và độ dài. |
| Error analysis | 10 | Tối thiểu 5 false positives, 5 false negatives, 5 boundary errors; có giải thích dựa trên câu và nhãn gold; nêu ít nhất 3 surface forms có loại entity khác nhau theo context. |
| Information Extraction application | 5 | Pipeline NER thành dữ liệu có cấu trúc; có tập gold nhỏ (đề gợi ý 30 câu) và đánh giá entity-level precision/recall/F1. |
| Reflection | 5 | Bài 300–500 từ giải thích contextual representations, `[CLS]` so với từng token, prediction head và mối liên hệ LAB 08–10; kết luận không vượt quá bằng chứng. |
| **Tổng** | **100** | |

## Cách áp dụng

- Chấm từng hạng mục trong giới hạn điểm rubric; ghi lại đường dẫn file và bằng chứng cụ thể trong phiếu chấm.
- Với kết quả thực nghiệm, kiểm tra cấu hình, split, seed và giới hạn mẫu. Không xem kết quả run rút gọn như kết quả trên toàn dataset.
- Đối chiếu token-level và exact entity-level metrics riêng; accuracy đơn lẻ không thay thế entity-level F1.
- Với error analysis, xác nhận từng ví dụ từ gold/prediction trước khi chấp nhận giải thích; candidate tự động không tự chứng minh nguyên nhân lỗi.
- Nếu prediction được viết sau khi đã xem kết quả, ghi nhận rõ và áp dụng rubric theo hướng dẫn giảng viên/TA; không xem nội dung hồi cứu là dự đoán tiên nghiệm.
- Nếu deliverable không có, không chạy được hoặc không đủ bằng chứng, ghi rõ điều đó và áp dụng thang điểm của người chấm nhất quán với rubric; không tự tạo minh chứng thay sinh viên.

## Checklist deliverables đối chiếu

- [ ] `calculations.md`
- [ ] `prediction.md`
- [ ] `ner_pipeline.py` hoặc phần cài đặt tương đương và baseline
- [ ] `tokenizer_alignment.ipynb` và `ner_finetuning.ipynb`
- [ ] `results.csv`, `entity_metrics.csv`, `entity_length_analysis.csv`
- [ ] `entity_length_analysis.png`, `confusion_matrix.png`
- [ ] `error_analysis.md` và `error_analysis_examples.csv`
- [ ] `information_extraction.py` và bằng chứng đánh giá IE
- [ ] `reflection.md`

Đề còn yêu cầu có thể xem trực tiếp `word_ids`, labels, logits và predictions. Hãy kiểm tra trong notebook hoặc log thực thi, không chỉ dựa vào việc tên file tồn tại.


