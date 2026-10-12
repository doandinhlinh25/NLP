# Chấm điểm Lab 01 — From Text Processing to Search

**Điểm đề xuất: 64/100 (6.4/10)**

## Phạm vi và nguyên tắc chấm

Đối chiếu bài trong `lab1/` với rubric của `lab1/W1.pdf`. Điểm ghi nhận chất lượng artefact thực tế, khả năng tái lập, và việc tuân thủ AI Usage Policy. Lệnh sau đã chạy thành công:

```powershell
python implementation.py
```

Kết quả: `All tests passed.`

Tuy nhiên, đây **không phải** xác nhận rằng toàn bộ notebook có thể tái chạy từ `requirements.txt`: notebook import `tokenizers` nhưng dependency này không có trong `requirements.txt`, và data path được hard-code theo máy cá nhân.

| Hạng mục rubric | Tối đa | Điểm | Nhận xét và evidence |
| --- | ---: | ---: | --- |
| Theory recap | 10 | 8 | Có các giải thích đúng về chiều vector, IDF và TF-IDF trong `experiments.ipynb`. Nội dung ngắn, chưa thể hiện đầy đủ chiều sâu của phần recap; phần khai báo AI nói các câu trả lời phân tích trong notebook được AI soạn nháp. |
| Calculation exercises & prediction | 10 | 5 | `calculations.md` làm đúng count vector, TF, IDF, TF-IDF và cosine; `prediction.md` có ba prediction hợp lý. Nhưng cả hai file đều yêu cầu tự làm trước khi chạy code, trong khi phần AI declaration xác nhận AI đã soạn `calculations.md` và `prediction.md`; chỉ ghi nhận một phần điểm cho kết quả cuối cùng. |
| Core implementation | 20 | 15 | `implementation.py` chạy thành công, có các hàm TF-IDF/cosine cơ bản và một số assertion. Trừ điểm vì test chưa kiểm tra đúng giá trị cho tất cả hàm, cosine không kiểm tra kích thước vector, không xử lý count rỗng, không có đối chiếu implementation với thư viện như yêu cầu Part E, và API phụ thuộc vào input đã tokenized ở một số hàm. |
| Experiment 1 — Sparse representation | 15 | 12 | Có kết quả trên 30,000 document: shape `(30000, 193540)`, nnz và sparsity; có inspection terms. Điểm tái lập bị giảm vì `CORPUS_PATH` là đường dẫn tuyệt đối `E:/hus/...`, không dùng đường dẫn tương đối/documented dataset location. |
| Experiment 2 — Preprocessing ablation | 15 | 8 | Có ba pipeline A/B/C và bảng vocabulary, average tokens, sparsity. Thiếu OOV rate và search-performance comparison trong bảng, dù rubric yêu cầu ablation có các chỉ số đó. Pipeline B không thực hiện accent normalization như mô tả; fresh install sẽ lỗi vì thiếu package `tokenizers`; nhận xét P@5 trong reflection không có bảng/kết quả tương ứng để kiểm chứng. |
| Search application & evaluation | 15 | 10 | Có search TF-IDF, top-5, relevance labels, Precision@5, Recall@5, MRR và `results.csv`. Hạn chế: search phụ thuộc global state trong notebook, CSV không lưu metric tổng hợp/preview để audit dễ dàng, và relevance labels được khai báo là do AI chọn/gán. |
| Error analysis & next representation | 10 | 4 | Có 2 good case, 2 weak case và nêu được giới hạn lexical matching. Nhưng declaration ghi AI soạn error analysis và chọn/gán nhãn relevance, nên phần chứng minh năng lực phân tích độc lập không đạt yêu cầu AI policy. |
| Reflection | 5 | 2 | Reflection có nêu prediction sai, trade-off preprocessing và hướng cải tiến. Tuy nhiên file ghi rõ “Không dùng AI cho phần nội dung phân tích/reflection”, trong khi AI declaration xác nhận AI đã soạn reflection; chỉ chấm điểm phần người nộp nói đã review/xác nhận hiểu nội dung. |
| **Tổng** | **100** | **64** | **6.4/10** |

## Điểm tốt

- Bài tính tay có công thức và kết quả đúng, đặc biệt cosine `2 / sqrt(6) ≈ 0.816`.
- Core implementation không gọi `TfidfVectorizer` để thay thế phần tự cài đặt, và script chạy thành công.
- Notebook đã lưu output thực nghiệm cho Part D, search examples và evaluation.
- Error analysis nhận diện đúng hạn chế chính của lexical TF-IDF: overlap từ không thể thay thế hiểu ngữ cảnh/nghĩa.

## Nhược điểm

Các điểm yếu chính là dependency và đường dẫn dữ liệu không portable, ablation/evaluation chưa đủ evidence, cùng vấn đề tuân thủ AI policy ở nhiều phần cần làm độc lập.

1. Làm lại độc lập các phần rubric cấm dùng AI: calculation, prediction, relevance labeling, error analysis và reflection. Chỉ giữ khai báo AI cho phần code nếu chính sách môn học cho phép.
2. Thay `CORPUS_PATH` tuyệt đối bằng đường dẫn tương đối từ thư mục project; bổ sung hướng dẫn lấy data trong README.
3. Thêm `tokenizers` vào `requirements.txt`, cài môi trường mới và chạy lại toàn bộ notebook từ đầu; lưu đầy đủ outputs.
4. Hoàn thiện ablation: công bố OOV rate, P@5 cho từng pipeline, cách gán relevance labels và giải thích điều kiện so sánh công bằng.
5. Mở rộng core tests: kiểm tra exact values cho vocabulary/count/IDF/TF-IDF/cosine, input rỗng và sai chiều; đối chiếu numerically với scikit-learn theo cùng quy ước IDF.
6. Ghi mean P@5, mean Recall@5 và MRR vào một report/CSV riêng, kèm document preview hoặc rationale cho từng relevance label.

## Ghi chú về AI policy

Điểm trên đã áp dụng giảm điểm có kiểm soát thay vì tự động cho 0 toàn bài. Nếu giảng viên áp dụng nghiêm điều khoản “Không sử dụng AI trong” cho các phần đã tự khai báo, các mục Calculation & prediction, Error analysis và Reflection có thể bị chấm **0 điểm hoàn toàn** hoặc bị xử lý theo quy định học phần.

## Học được gì từ bài này

- Kết quả đúng chưa đủ: với bài học thuật, phần tính tay, prediction, gán relevance và reflection phải thể hiện năng lực độc lập của người nộp, đồng thời tuân thủ AI Usage Policy.
- Unit test có giá trị khi kiểm tra đúng giá trị kỳ vọng, edge case và đối chiếu với reference implementation; assertion quá tổng quát khó phát hiện lỗi thuật toán.
- Ablation chỉ công bằng khi các pipeline được đánh giá trên cùng query, cùng relevance labels và có metric được định nghĩa rõ ràng, đặc biệt với OOV giữa word và subword.
- Tập relevance sinh từ exact phrase match sẽ ưu ái lexical TF-IDF; relevance cần được gán độc lập dựa trên nội dung/document preview.
- Code thí nghiệm cần portable: dependency đầy đủ, data path tương đối và notebook đã chạy từ môi trường mới là điều kiện để số liệu đáng tin cậy.
