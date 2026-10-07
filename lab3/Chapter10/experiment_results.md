# Chapter 10 — Experiment 1: Word-context representation

## Thiết lập

Thí nghiệm dùng 5,000 văn bản không rỗng đầu tiên của corpus C4. Văn bản được lower-case và tách bằng mẫu token tiếng Anh. Vocabulary dùng chung giữa các cấu hình, giữ tối đa 1,000 token xuất hiện ít nhất 5 lần. Các từ dùng để đánh giá (`doctor, physician, hospital, cat, dog`) được giữ lại khi đạt ngưỡng tần suất, để có thể so sánh cùng một tập word pairs ở mọi window. Mỗi ma trận có hàng là target word, cột là context word và ô `(i, j)` là số lần context word `j` xuất hiện trong cửa sổ quanh target word `i`.

## Kích thước và độ thưa của ma trận

| Context window | Vocabulary size | Matrix shape | Non-zero values | Density | Sparsity |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 1000 | 1000 × 1000 | 199,694 | 0.199694 | 0.800306 |
| 2 | 1000 | 1000 × 1000 | 344,979 | 0.344979 | 0.655021 |
| 5 | 1000 | 1000 × 1000 | 581,983 | 0.581983 | 0.418017 |

Khi context window tăng, mỗi target word có nhiều context hơn. Vì vậy số ô khác 0 và density được dự đoán tăng, trong khi matrix shape giữ nguyên vì vocabulary được cố định. Dù vậy, ma trận vẫn là sparse representation: phần lớn tổ hợp word-context không cùng xuất hiện trong corpus.

## Cosine similarity của các cặp từ chọn trước

| Word pair | window = 1 | window = 2 | window = 5 |
| --- | ---: | ---: | ---: |
| doctor–physician | 0.731670 | 0.787011 | 0.883575 |
| doctor–hospital | 0.757166 | 0.811473 | 0.857180 |
| cat–dog | 0.501695 | 0.747732 | 0.751333 |

`not_in_vocabulary` nghĩa là ít nhất một từ không thuộc vocabulary đã giới hạn, nên không thể tính similarity cho cặp đó. Các số liệu chi tiết, gồm cả kết quả tái lập được, nằm trong `results.csv`.

## Nhận xét cần kiểm tra

Các giá trị trên chỉ mô tả vector count-based của sample corpus. Similarity cao phản ánh các context quan sát được, không tự động chứng minh hai từ là đồng nghĩa. Khi đối chiếu với prediction ở Chapter 9, cần kiểm tra tần suất từ và các context gốc trước khi kết luận về sự khác biệt giữa các window.
