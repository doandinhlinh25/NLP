# Chấm điểm Lab 01 — From Text Processing to Search

**Điểm: 97/100 (9.7/10)**

## Phạm vi chấm

Chấm theo rubric trong `lab1/W1.pdf`, đối chiếu với các deliverable trong `lab1/`. Điểm dựa trên artefact đã nộp, mã nguồn và kết quả đã lưu. Phần tính tay trong PDF được kiểm tra theo sự hiện diện, cấu trúc lời giải và tính nhất quán với Exercise 6; không thể xác nhận từng phép tính trong PDF chỉ bằng việc chạy mã.

| Hạng mục rubric | Tối đa | Điểm | Căn cứ chấm |
| --- | ---: | ---: | --- |
| Theory recap | 10 | 10 | `Chapter11/learning_check.md` trả lời đầy đủ 7 ý: sparsity, IDF, TF-IDF, cosine, preprocessing và failure case. |
| Calculation exercises & prediction | 10 | 10 | Có bài tính tay Exercise 1–5 trong PDF; `ex6_prediction.md` tính đúng TF, IDF, TF-IDF, cosine và ranking D1 > D2 > D3. Prediction về vocabulary, sparsity và search được nêu trước experiment. |
| Core implementation | 20 | 19 | `Chapter9/implementation.py` có đủ vocabulary, count, TF, IDF, TF-IDF, cosine, kiểm tra kích thước vector và unit test/đối chiếu scikit-learn. Trừ 1 điểm vì chưa tái chạy được tại thời điểm chấm: môi trường `.venv` thiếu `numpy`, nên không xác nhận được dòng “All unit tests passed.” bằng thực thi hiện tại. |
| Experiment 1 — Sparse representation | 15 | 15 | `Chapter7/experiment.ipynb` có đủ 7 cell đã thực thi, kiểm tra dataset, pipeline TF-IDF, shape, sparsity và inspection vocabulary/TF-IDF. |
| Experiment 2 — Preprocessing ablation | 15 | 14 | Có ba pipeline A/B/C, bảng so sánh vocabulary, nnz, sparsity, OOV và P@5 trong `Chapter8/experiment_results.md`; cách giải thích trade-off hợp lý. Trừ 1 điểm vì `Ex8.ipynb` không lưu execution output, nên kết quả chủ yếu được xác minh qua report/CSV và script. |
| Search application & evaluation | 15 | 14 | Search TF-IDF, top-5 retrieval, P@5, Recall@5 và MRR được cài đặt rõ; `results.csv` và `search_evaluation.md` có đầy đủ kết quả. Trừ 1 điểm vì `Chapter10/Ex10.ipynb` chưa lưu output thực thi. |
| Error analysis & next representation | 10 | 10 | Phân tích hai good case và hai weak case, xác định đúng failure `barbecue`/`BBQ`, đồng thời đề xuất dense/hybrid retrieval có lý do. |
| Reflection | 5 | 5 | Reflection có đối chiếu prediction–kết quả, dùng số liệu ablation, nêu giới hạn tập nhãn và khai báo vai trò AI. |
| **Tổng** | **100** | **97** | **9.7/10** |

## Điểm mạnh

- Cấu trúc nộp bài rõ ràng; `README.md` map chính xác từng yêu cầu sang tệp tương ứng và có hướng dẫn tái chạy.
- Core TF-IDF không chỉ gọi thư viện: có cài đặt độc lập, unit test và giải thích khác biệt quy ước IDF với scikit-learn.
- Báo cáo experiment không chỉ liệt kê số liệu mà có diễn giải hợp lý về sparsity, stopword removal, subword feature và OOV.
- Evaluation có nhãn relevance, kết quả truy hồi cụ thể, metrics tổng hợp và error analysis bám sát evidence.

## Nhược điểm

Khả năng tái lập chưa trọn vẹn: môi trường chấm thiếu dependency và hai notebook thí nghiệm chưa lưu output thực thi.

1. Cài dependency theo README rồi chạy lại toàn bộ notebook, đặc biệt `Chapter8/Ex8.ipynb` và `Chapter10/Ex10.ipynb`; lưu output trong notebook trước khi nộp.
2. Ghi lại lệnh/chữ ký môi trường đã chạy (Python và phiên bản `numpy`, `scipy`, `scikit-learn`) để người chấm tái lập kết quả trực tiếp.
3. Với Experiment 2, nêu tách bạch hơn rằng OOV của character n-gram không hoàn toàn so sánh trực tiếp với OOV word-token; báo cáo hiện đã lưu ý điều này, nhưng nên nhấn mạnh ngay ở bảng kết quả.

## Ghi chú xác minh

Lệnh `python Chapter9/implementation.py` đã được thử trong workspace hiện tại nhưng dừng ở `ModuleNotFoundError: No module named 'numpy'`. Đây là thiếu dependency của môi trường đang chấm, không có bằng chứng cho thấy thuật toán hoặc unit test trong bài sai; vì vậy chỉ trừ điểm nhỏ ở khả năng tái lập ngay lập tức.

## Học được gì từ bài này

- Một hệ thống TF-IDF hoàn chỉnh cần đi theo chuỗi: biểu diễn sparse → tiền xử lý → weighting → search → đánh giá → phân tích lỗi; mỗi bước đều có thể thay đổi kết quả cuối.
- Sparse matrix giúp biểu diễn vocabulary rất lớn một cách khả thi, vì mỗi document chỉ kích hoạt một phần nhỏ feature.
- Đánh giá retrieval cần nhãn relevance, P@K, Recall@K và MRR; chỉ nhìn similarity score không đủ để kết luận search tốt.
- Failure `barbecue`/`BBQ` cho thấy lexical TF-IDF mạnh ở exact match nhưng không hiểu synonym/semantic relation; đây là lý do phù hợp để chuyển sang embedding hoặc hybrid retrieval.
- Một bài thí nghiệm tốt phải tái lập được: lưu output notebook, dependency và cách chạy quan trọng không kém kết quả số.
