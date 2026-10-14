# Chấm điểm Lab 03 - Bài 22

**Bài nộp:** `lab3/`  
**Tổng điểm:** **75/100 (7.5/10)**

Đánh giá theo rubric W3 và các artefact hiện có trong `lab3/`. Phần individual learning check không thể xác nhận từ repository.

| Hạng mục W3 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 8/10 | Có PDF bài tính tay Chapter 6-8, nhưng không có `calculations.md`/báo cáo tách bạch để đối chiếu toàn bộ bài co-occurrence, cosine, sparse-dense và analogy theo W3. |
| Prediction | 10/10 | `Chapter9/chapter9_prediction.md` có 4 dự đoán cùng Reason/Confidence, nêu đúng trade-off context window, dimension và corpus nhỏ. |
| Co-occurrence implementation | 15/15 | `Chapter11/implementation.py` tự cài vocabulary, co-occurrence, cosine và most-similar; lệnh kiểm tra chạy thành công: `All Chapter 11 implementation checks passed.` |
| Word2Vec experiment | 15/15 | Có notebook/script, cấu hình training, top-5 cho `doctor` và diễn giải context trong Chapter 17-18. |
| Hyperparameter experiment | 15/15 | Chapter 19 so sánh window 2/5/10 và Chapter 20 so sánh dimension 50/100/300, có CSV cùng nhận xét trade-off. |
| Evaluation | 12/15 | Có evaluation word similarity định lượng ở Chapter 21. Chưa thấy artefact thực hiện/rút nhận xét riêng cho word analogy theo yêu cầu W3. |
| Application | 0/10 | Không có artefact semantic search tương ứng Mục 24 của W3. |
| Error analysis | 0/5 | Không có file/phần phân tích 3 similarity đúng và 3 sai/bất ngờ. |
| Reflection | 0/3 | Không có bảng so sánh representation và phân tích `bank`/contextual embedding theo Mục 27. |
| Individual learning check | 0/2 | Không có bằng chứng xác nhận phần kiểm tra cá nhân. |
| **Tổng** | **75/100** | **7.5/10** |

## Điểm mạnh

- Các thí nghiệm co-occurrence, Word2Vec và hai nhóm hyperparameter được tách thành script, notebook, CSV và báo cáo tương đối rõ.
- Core implementation thực sự tự cài đặt, không dùng Word2Vec thay cho phần co-occurrence bắt buộc.

## Điểm yếu

1. Bổ sung word analogy evaluation, semantic search, error analysis và reflection theo đúng deliverables W3.
2. Chuẩn hóa bài tính tay thành file `calculations.md` hoặc PDF được đặt tên/routing rõ trong README.
3. Lưu bằng chứng individual learning check sau khi thực hiện.


## Học được gì từ bài này

- Co-occurrence cho thấy trực tiếp mối liên hệ context - vector - cosine similarity; Word2Vec học vector dense để khái quát quan hệ đó tốt hơn trên corpus lớn.
- Chỉ báo cáo similarity chưa đủ: analogy, semantic search, error analysis và reflection giúp kiểm tra embedding có hữu ích trong những tình huống khác nhau hay không.
- Context window và embedding dimension luôn là trade-off giữa quan hệ cục bộ/chủ đề rộng, năng lực biểu diễn, dữ liệu huấn luyện và chi phí tính toán.
- Static embedding vẫn chỉ gán một vector cho mỗi từ; lỗi do đa nghĩa như `bank` là lý do cần contextual embeddings.
