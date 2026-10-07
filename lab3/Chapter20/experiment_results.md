# Chapter 20 — Experiment 4: Embedding dimension

## Thiết lập

Các model CBOW cùng dùng 2,000 văn bản C4, `window=5`, `min_count=2`, `epochs=3`, `workers=1` và `seed=2026`. Chỉ `vector_size` thay đổi.

Semantic retrieval là downstream task nhỏ: query `medical treatment` được biểu diễn bằng trung bình word vectors và xếp hạng ba pseudo-documents (`medical`, `technology`, `food`). Kết quả đúng khi document `medical` đứng đầu; đây chỉ là sanity check, không phải benchmark.

## Kết quả

| Dimension | Training seconds | Vector memory (MB) | doctor–physician | Analogy top-1 | Retrieval top-1 | Retrieval correct |
| ---: | ---: | ---: | ---: | --- | --- | --- |
| 50 | 11.485828 | 4.116249 | 0.836147 | hall | medical | True |
| 100 | 13.995805 | 8.232498 | 0.837317 | hall | technology | False |
| 300 | 21.258880 | 24.697495 | 0.843532 | chapters | technology | False |

`Vector memory` là số byte thực của ma trận embedding (`model.wv.vectors.nbytes`), chưa tính toàn bộ overhead của object. Cột analogy chạy `king - man + woman`; output là pattern trong vector space, không phải bằng chứng model hiểu quan hệ như con người.

## Nhận xét

Dimension lớn tăng capacity và bộ nhớ gần tuyến tính, đồng thời thường tăng thời gian train. Tuy nhiên, quality không chắc chắn tăng theo dimension khi corpus nhỏ: cần xem similarity, analogy và retrieval trong cùng bảng thay vì chỉ chọn vector lớn nhất.
