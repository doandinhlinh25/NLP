# Chấm điểm Lab 02 — Bài 31

**Bài nộp:** `ChamDiem/lab02/31/lab02`  
**Tổng điểm:** **94/100 (9.4/10)**

Đánh giá được thực hiện từ các artefact trong repository: `calculations.md`, `prediction.md`, `ngram_lm.py`, `experiments.ipynb`, `results.csv`, `error_analysis.md` và `reflection.md`. Mục vấn đáp cá nhân không thể xác nhận chỉ bằng mã nguồn và tài liệu chuẩn bị.

| Hạng mục W2 | Điểm | Nhận xét có căn cứ |
|---|---:|---|
| Theory & calculation | 10/10 | Làm đủ unigram, bigram, probability câu, xếp hạng, zero probability, Laplace và perplexity. Các giá trị như `1/24`, `1/15`, `4/15`, `2.5198`, `3.4199` và lập luận đều nhất quán với toy corpus. |
| Prediction | 10/10 | Có 5 dự đoán trước khi chạy, kèm lý do và độ tin cậy; phần so sánh kết quả sau thực nghiệm rõ ràng. |
| Core implementation | 19/20 | `ngram_lm.py` tự cài đặt unigram/bigram/trigram, `<START>/<END>`, `<UNK>`, MLE, Laplace, log probability, sentence probability, next-word distribution và perplexity. Chạy script tạo được các giá trị kiểm tra toy corpus. Còn thiếu một bộ test tự động độc lập với assertion để khóa các corner case. |
| Smoothing experiment | 15/15 | Notebook và CSV có so sánh MLE/Laplace trên split train/validation/test; trình bày rõ zero probability của MLE và tác động của smoothing. |
| Perplexity evaluation | 15/15 | Báo cáo PPL unigram/bigram/trigram trên các split, đồng thời giải thích sparsity và khác biệt train so với validation/test. |
| Application | 10/10 | Có dự đoán từ tiếp theo cho nhiều context, đối chiếu actual token và xếp hạng câu bằng xác suất/log probability. |
| Error analysis | 10/10 | Có đúng 2 ca dự đoán đúng và 2 ca sai; từng ca ghi context, prediction/expected, probability và nguyên nhân (giới hạn context, sparsity). |
| Reflection | 5/5 | Trả lời đủ 7 câu; phần giải thích về smoothing, perplexity, sparsity và giới hạn ngữ cảnh là đúng trọng tâm. |
| Individual learning check | 0/5 | Có phần chuẩn bị vấn đáp tốt, nhưng repository không thể chứng minh đã thực hiện hoặc đạt phần vấn đáp cá nhân. |

## Điểm mạnh

- Bài nộp có chuỗi minh chứng liền mạch từ tính tay, dự đoán đến notebook, mã nguồn và CSV kết quả.
- Thí nghiệm làm mịn và PPL có số liệu cụ thể; không chỉ nêu kết luận chung chung.
- Error analysis bám sát output thực tế, không nhầm prediction với expected word.

## Nhược điểm

Phần kỹ thuật rất đầy đủ, nhưng chưa có test tự động độc lập và chưa có minh chứng cho phần vấn đáp cá nhân.

1. Bổ sung unit test có assertion cho count, xác suất MLE/Laplace, câu chứa n-gram chưa thấy, input rỗng và perplexity để việc kiểm chứng không chỉ dựa vào output in ra.
2. Khi nộp chính thức, lưu kèm bằng chứng hoặc kết quả của phần individual learning check để có thể chấm đủ 5 điểm còn lại.

## Học được gì từ bài này

- N-gram mạnh ở tính minh bạch: từ count có thể lần theo trực tiếp đến conditional probability, sentence score và PPL.
- Tăng bậc n giúp khai thác context gần nhưng làm dữ liệu thưa hơn; vì vậy PPL test không mặc định giảm khi chuyển từ bigram sang trigram.
- Laplace smoothing đổi sự đánh đổi giữa fit trên train và khả năng gán xác suất cho unseen n-gram, nên cần đọc kết quả trên validation/test thay vì chỉ train.
- Một error analysis tốt phải chỉ ra loại lỗi cụ thể (thiếu context, OOV hoặc sparsity), thay vì chỉ liệt kê từ dự đoán sai.
