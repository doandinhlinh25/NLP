# Chấm điểm Lab 03 - Bài 24

**Bài nộp:** `https://github.com/ntrThanh/NLP-2026/tree/master/lab03`  
**Tổng điểm:** **90/100 (9.0/10)**

Điểm tổng được hiệu chỉnh ở mức **9.0/10 theo yêu cầu chấm**. Nhận xét vẫn đối chiếu rubric W3 và artefact thực tế đã clone.

| Hạng mục W3 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 7/10 | Có `calculations.pdf` dạng scan, nhưng chưa có bản Markdown/text để dễ kiểm tra đầy đủ các bước co-occurrence, cosine, sparse/dense và analogy. |
| Prediction | 10/10 | `prediction.md` có đủ 4 dự đoán, Reason và Confidence, đồng thời nêu rõ phụ thuộc corpus/dimension/window. |
| Co-occurrence implementation | 15/15 | `cooccurrence.py` tự cài tokenize, vocabulary, ma trận co-occurrence, cosine và most-similar. Unit test chạy thành công trên toy corpus. |
| Word2Vec experiment | 15/15 | Notebook có output, ghi rõ CBOW, corpus, vector size, window, min count, epochs và top-similar. |
| Hyperparameter experiment | 15/15 | Có thí nghiệm window 2/5/10 và dimension 50/100/300, với time, RAM, similarity và analogy trong `results.csv`. |
| Evaluation | 11/15 | Có similarity score và analogy, nhưng đánh giá chưa có bảng xếp hạng/đối chiếu intuition-systematic tách riêng; phần diễn giải quantitative còn phân tán qua notebook và Markdown. |
| Application | 9/10 | Có semantic search và ranking kết quả, nhưng chưa có bộ relevance/tiêu chí đánh giá rõ để kiểm chứng chất lượng retrieval. |
| Error analysis | 5/5 | Đủ 3 ca đúng và 3 ca sai/bất ngờ, có observed, expected, explanation và evidence từ corpus. |
| Reflection | 3/3 | Giải thích đúng static embedding, polysemy `bank` và contextual representation; có bảng so sánh representation. |
| Individual learning check | 0/2 | Có câu trả lời ôn tập, nhưng không có bằng chứng cho một buổi kiểm tra cá nhân đã được xác nhận. |
| **Tổng** | **90/100** | **9.0/10** |

## Điểm mạnh

- Notebook được thực thi gần như đầy đủ: 27/27 code cell có execution count và 26 cell lưu output.
- Core co-occurrence có unit test; các thí nghiệm window, dimension, similarity, analogy và semantic search đều có artefact cụ thể.
- Error analysis và reflection có chất lượng, giải thích được frequency, domain bias, context window và polysemy.

## Điểm yếu

1. Chuyển phần tính tay scan thành deliverable dễ đối chiếu và đặt tên đúng quy ước `calculations.md`.
2. Tách evaluation thành bảng rank, expected/intuitive rank và nhận xét bất ngờ rõ ràng hơn.
3. Xác định tiêu chí relevance hoặc tạo bộ query-document nhỏ có nhãn cho semantic search.
4. Lưu minh chứng individual learning check nếu đã thực hiện.



## Học được gì từ bài này

- Co-occurrence và Word2Vec đều học từ context, nhưng Word2Vec nén quan hệ đó thành vector dense để so sánh semantic hiệu quả hơn.
- Tăng context window hoặc dimension không bảo đảm kết quả tốt hơn; cần đánh đổi với nhiễu, dữ liệu, thời gian và bộ nhớ.
- Semantic search cần tiêu chí relevance để kiểm tra truy hồi; similarity score cao không tự động chứng minh kết quả hữu ích.
- Static Word2Vec không giải quyết polysemy như bank; contextual embeddings tạo vector theo câu để khắc phục giới hạn này.
