# Chấm điểm Lab 02 — Bài 32

**Bài nộp:** `ChamDiem/lab02/32/lab02`  
**Tổng điểm:** **88/100 (8.8/10)**

Đánh giá dựa trên `calculations.pdf`, `prediction.md`, `ngram_lm.py`, `experiments.ipynb`, `results.csv`, `error_analysis.md` và `reflection.md`. Mục vấn đáp cá nhân không thể xác nhận từ repository.

| Hạng mục W2 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 7/10 | Bảng unigram, đa số conditional probability, Laplace và perplexity làm đúng. Tuy nhiên phần tính/xếp hạng câu kết luận `S1` cao hơn `S2`, trong khi hai câu có cùng xác suất bigram `1/24`; một giá trị conditional probability trong bảng cũng chưa nhất quán. Cách giải thích tổng xác suất sau `the` còn dựa vào quy tắc ngôn ngữ thay vì count của corpus. |
| Prediction | 9/10 | Có các dự đoán phù hợp về vocabulary, sparsity và PPL, đồng thời sửa lại dự đoán xếp hạng sau khi tính. Nên trình bày từng dự đoán với các nhãn Reason/Confidence rõ ràng để đối chiếu theo rubric hơn. |
| Core implementation | 18/20 | `ngram_lm.py` tự triển khai tokenization, vocabulary/`<UNK>`, đếm n-gram có boundary, MLE, Laplace, sentence/log probability, next-word prediction và perplexity. Mã sạch và đủ chức năng, nhưng chưa có test tự động/assertion cho giá trị biên và giá trị kỳ vọng. |
| Smoothing experiment | 15/15 | Có thí nghiệm MLE và Laplace, số liệu train/validation/test và hiện tượng zero probability được thể hiện trong notebook/CSV. |
| Perplexity evaluation | 15/15 | So sánh PPL unigram/bigram/trigram cho các split và liên hệ được với data sparsity. |
| Application | 10/10 | Có dự đoán next-word trên nhiều context, đối chiếu actual word và xếp hạng các câu ứng viên. |
| Error analysis | 9/10 | Đủ 2 ca đúng và 2 ca sai, có context, dự đoán, expected word và nguyên nhân. Phân tích còn ngắn; nên nêu thêm xác suất/rank hoặc bằng chứng count cho từng ca để thuyết phục hơn. |
| Reflection | 5/5 | Trả lời đủ 7 câu, nắm đúng trade-off context dài hơn, sparsity, smoothing và giới hạn n-gram. |
| Individual learning check | 0/5 | Không có bằng chứng cho phần vấn đáp/kiểm tra cá nhân đã được thực hiện; mã nguồn và câu trả lời viết sẵn không thay thế được mục này. |

## Điểm mạnh

- Phần code và notebook là một pipeline N-gram hoàn chỉnh, có dữ liệu thực nghiệm chứ không dừng ở ví dụ toy corpus.
- Kết quả PPL và smoothing được xuất thành CSV, giúp kiểm tra lại nhận xét dễ hơn.
- Người làm đã tự phát hiện và sửa lại kết luận tie của hai câu trong phần prediction; đây là hướng tư duy kiểm chứng kết quả tốt.

## Nhược điểm

Điểm yếu chính nằm ở độ chính xác và cách trình bày phần tính tay; phần phân tích lỗi và kiểm thử tự động vẫn chưa đủ sâu.

1. Sửa phần tính tay: viết lại đầy đủ xác suất của `S1` và `S2` để thấy cả hai đều bằng `1/24`, sau đó kết luận tie; kiểm tra lại mọi mẫu số conditional probability bằng count của context.
2. Tách rõ từng prediction thành **Prediction – Reason – Confidence – Result** để tránh lẫn dự đoán với kết quả đã quan sát.
3. Bổ sung assertion/unit tests cho toy corpus, zero-probability MLE, Laplace, empty input và PPL; mở rộng error analysis với xác suất/rank/count thực tế.
4. Lưu bằng chứng của individual learning check khi hoàn thành phần này.

## Học được gì từ bài này

- Tính tay là bước kiểm thử quan trọng cho code: một kết luận xếp hạng sai có thể phát hiện ngay bằng cách nhân đúng các conditional probability.
- Với mô hình N-gram, tổng các xác suất tiếp theo phải được suy ra từ phân bố đếm của corpus, không từ trực giác ngữ pháp.
- Kết quả MLE và Laplace cần đọc cùng nhau: MLE phản ánh dữ liệu đã thấy, còn smoothing giúp mô hình hoạt động trên context chưa gặp.
- Error analysis có giá trị nhất khi gắn dự đoán với số liệu cụ thể, từ đó phân biệt được lỗi do sparsity, context ngắn hay do lựa chọn tiền xử lý.
