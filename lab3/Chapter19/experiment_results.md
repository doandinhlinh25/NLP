# Chapter 19 — Experiment 3: Context window

## Thiết lập

Ba model CBOW được huấn luyện trên cùng 2,000 văn bản C4, cùng `vector_size=100`, `min_count=2`, `epochs=3`, `workers=1` và `seed=2026`. Chỉ `window` thay đổi; không lưu các model hyperparameter vì chúng có thể tái tạo bằng script.

## Similarity theo context window

| Word pair | window = 2 | window = 5 | window = 10 |
| --- | ---: | ---: | ---: |
| doctor–physician | 0.872723 | 0.837317 | 0.663592 |
| doctor–hospital | 0.724713 | 0.691830 | 0.418925 |
| cat–dog | 0.897773 | 0.844520 | 0.778680 |

Window nhỏ nhấn mạnh context gần, trong khi window lớn đưa thêm thông tin chủ đề ở xa hơn. Vì vậy, sự thay đổi score trong bảng là kết quả quan sát được của corpus này, không phải bằng chứng rằng window lớn luôn tốt hơn. Cần so sánh từng cặp từ và đối chiếu với frequency/context trước khi kết luận.
