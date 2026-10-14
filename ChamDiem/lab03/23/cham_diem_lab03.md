# Chấm điểm Lab 03 - Bài 23

**Bài nộp:** `https://github.com/vietquang8071/nlp-app/tree/main/lab3`  
**Tổng điểm:** **95/100 (9.5/10)**

Đánh giá theo artefact đã clone từ repository và rubric W3. Không thể xác nhận individual learning check từ repository.

| Hạng mục W3 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 8/10 | Có `calculation.pdf`, nhưng tên deliverable không theo `calculations.md` và không có hướng dẫn/routing riêng cho phần tính tay trong thư mục Lab 3. |
| Prediction | 10/10 | `prediction.md` có đủ 4 dự đoán, lý do và mức confidence; dự đoán có điều kiện, không khẳng định quá mức. |
| Co-occurrence implementation | 14/15 | `cooccurrence.py` tự cài vocabulary, sparse co-occurrence, cosine và most-similar. Code không có test chạy được khi gọi trực tiếp, nên không xác nhận được tự động như bài 22/24. |
| Word2Vec experiment | 15/15 | Notebook thực thi, ghi corpus, cấu hình, top similar, evidence context và kết quả dense embedding. |
| Hyperparameter experiment | 15/15 | Có window 2/5/10, dimension 50/100/300, time/RAM/similarity/analogy và diễn giải trade-off. |
| Evaluation | 15/15 | Có similarity score, so sánh intuition, analogy và các trường hợp bất ngờ; kết quả định lượng được lưu trong `result.md`. |
| Application | 10/10 | Có semantic search cho nhiều query, xếp hạng document và score. |
| Error analysis | 5/5 | Đủ 3 case đúng, 3 case sai/bất ngờ, có observed/expected/explanation/evidence corpus. |
| Reflection | 3/3 | Hoàn thiện bảng representation, giải thích đúng polysemy `bank` và giới hạn static embedding. |
| Individual learning check | 0/2 | Không có bằng chứng của phần kiểm tra cá nhân đã diễn ra. |
| **Tổng** | **95/100** | **9.5/10** |

## Điểm mạnh

- Báo cáo thực nghiệm sâu, kết nối score với bằng chứng context thay vì chỉ nói embedding học semantic similarity.
- Semantic search và error analysis có ví dụ thực tế, giải thích được corpus bias, named entity bias và frequency.

## Điểm yếu

1. Thêm test/assertion chạy trực tiếp cho co-occurrence implementation.
2. Chuẩn hóa tên deliverable (`calculations.md`, `results.csv`, README trong `lab3/`) để người chấm tái lập dễ hơn.
3. Lưu bằng chứng individual learning check nếu đã thực hiện.



## Học được gì từ bài này

- Embedding cần được đánh giá bằng số liệu, analogy, semantic search và error analysis; một danh sách top-similar đơn lẻ không đủ để kết luận chất lượng.
- Corpus bias, tần suất và named entity có thể kéo các vector gần nhau vì cùng ngữ cảnh, dù không đồng nghĩa.
- Để tái lập được thí nghiệm NLP, code, dữ liệu, cấu hình, output notebook và unit test đều quan trọng.
