# Chấm điểm Lab 02 — Language Models

**Điểm: 74/100 (7.4/10)**

## Phạm vi chấm

Chấm theo rubric trong `lab2/W2.pdf`: theory & calculation, prediction, core N-gram implementation, smoothing, perplexity, application, error analysis, reflection và individual learning check. Điểm dựa trên các PDF tính tay, notebook, mã nguồn và output đã lưu.

| Hạng mục rubric | Tối đa | Điểm | Căn cứ chấm |
| --- | ---: | ---: | --- |
| Theory & calculation | 10 | 8 | PDF Chapter 7 tính đúng vocabulary, unigram, bigram và xác suất câu `1/24`; PDF Chapter 11 tính đúng Laplace `1/15` và `4/15`. Tuy nhiên kết luận ranking bigram viết tay chưa nhất quán: S1 và S2 đều có xác suất `1/24`, như notebook Ex4 đã kiểm chứng. Chưa có lời giải số cụ thể cho bài tính perplexity ở Chapter 18. |
| Prediction | 10 | 9 | Có đủ 5 prediction về vocabulary, số n-gram, zero probability, training perplexity và corpus nhỏ; lý do nhìn chung đúng. Thiếu trường `Confidence` theo yêu cầu và chưa đối chiếu từng prediction với kết quả thí nghiệm ngay trong artefact. |
| Core implementation | 20 | 17 | `NGramLanguageModel` tự cài unigram/bigram/trigram, MLE/Laplace, conditional probability, log probability, perplexity, next-word prediction và continuation ranking. Điểm trừ vì implementation bị phân tán trong notebook thay vì một module core rõ ràng, thiếu unit test có assertion cho zero probability/smoothing/log probability, và các notebook Chapter 13–16 chưa lưu output chạy. |
| Smoothing experiment | 15 | 13 | `Chapter11-Chapter12/experiments_chapter12.ipynb` và `Chapter9/Ex9.ipynb` có bảng MLE/Laplace cho unigram, bigram, trigram trên train/validation/test. Đã thể hiện đúng MLE có PPL vô hạn trên validation/test khi gặp unseen n-gram. Cần bổ sung diễn giải trực tiếp dựa trên các số liệu của bảng, nhất là vì Laplace bậc cao có PPL lớn. |
| Perplexity evaluation | 15 | 13 | `Chapter19-Chapter24/experiments.ipynb` có bảng PPL train/validation/test và giải thích sparsity, overfitting, corpus size; `Chapter18/answers.ipynb` giải thích công thức, MLE/Laplace và quy trình train/validation/test đúng. Trừ điểm vì chưa giải riêng bài tính PPL với các xác suất cho sẵn, và phần giải thích “PPL train thường giảm khi tăng n” cần được đối chiếu cẩn thận với chính bảng Laplace đang có. |
| Application | 10 | 8 | Có đủ 5 contexts với top-3 next-word, xác suất và actual word; có sentence ranking cho ba candidates. Nhưng model được chọn là unigram nên prediction cho mọi context giống nhau (`the`), phần ranking trong output chưa cho thấy candidate tự nhiên được xếp cao hơn; cần phân tích rõ đây là giới hạn của cách chọn model/smoothing. |
| Error analysis | 10 | 4 | Có mô tả nguyên nhân chung trong `Chapter18/answers.ipynb` và một số output prediction. Chưa nộp đủ 2 prediction đúng + 2 prediction sai, mỗi case kèm context, prediction, expected word, probability và nguyên nhân như W2 yêu cầu. |
| Reflection | 5 | 2 | `Chapter18/answers.ipynb` có nhiều giải thích khái niệm liên quan (PPL, zero probability, OOV, smoothing), nhưng không trả lời đầy đủ 7 câu reflection của W2, đặc biệt câu context dài 100 từ/trigram. |
| Individual learning check | 5 | 0 | Không có bằng chứng có thể chấm cho phần vấn đáp/individual learning check. |
| **Tổng** | **100** | **74** | **7.4/10** |

## Điểm mạnh

- Bài tính tay chủ yếu đúng và có bước tính rõ ràng; notebook Ex4 còn tự kiểm chứng đúng trường hợp tie của bigram MLE.
- N-gram LM được tự cài đặt, không thay thế bằng thư viện language model; có xử lý MLE, Laplace, log probability và PPL.
- Có thí nghiệm thực tế 10,000 documents với thống kê unigram/bigram/trigram, hapax n-grams và biểu đồ frequency distribution.
- Bảng MLE/Laplace giải thích rõ zero-probability: MLE có PPL vô hạn trên dữ liệu chưa thấy, Laplace cho PPL hữu hạn.

## Nhược điểm

Những phần bị trừ điểm tập trung ở tính kiểm thử/tổ chức core, phần tính PPL bằng số, error analysis và reflection chưa hoàn thành theo cấu trúc W2.

1. Tạo một module core duy nhất (ví dụ `ngram_lm.py`) và unit test độc lập cho count, MLE, Laplace, zero probability, log probability, PPL và next-word distribution.
2. Hoàn thành bài tính perplexity bằng số theo Chapter 18; sửa ghi chú viết tay về ranking bigram để khẳng định S1 và S2 bằng nhau.
3. Viết phần diễn giải sát với bảng PPL: giải thích vì sao Laplace unigram được chọn ở validation, còn Laplace bigram/trigram có PPL lớn do vocabulary/context thưa.
4. Nộp error analysis đúng cấu trúc: 2 prediction đúng và 2 prediction sai; mỗi case cần context, top prediction, expected word, probability và nguyên nhân.
5. Trả lời đủ 7 câu reflection, đặc biệt giới hạn Markov của trigram khi context dài; chuẩn bị giải thích miệng các câu individual learning check.

## Học được gì từ bài này

- Count trong Lab 01 dùng để biểu diễn TF-IDF, còn trong Lab 02 dùng để ước lượng xác suất điều kiện của language model.
- Tăng n giúp model dùng context dài hơn nhưng làm số n-gram hiếm/unseen tăng nhanh; bậc cao không tự động tốt hơn trên validation/test.
- MLE gán xác suất 0 cho unseen n-gram, khiến xác suất câu bằng 0 và perplexity vô hạn; smoothing giải quyết zero probability bằng cách phân bổ lại xác suất cho toàn bộ vocabulary.
- Log probability biến phép nhân nhiều xác suất nhỏ thành phép cộng, giúp tránh numerical underflow và phù hợp để xếp hạng continuation.
- Perplexity chỉ so sánh được khi corpus đánh giá, tokenization, vocabulary và cách smoothing giống nhau; chọn model bằng validation, không dùng test để chọn cấu hình.
