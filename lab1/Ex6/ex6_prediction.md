# Exercise 6 — Prediction before running code

Với query `medical image classification`:

1. **Document có similarity cao nhất:** D1, vì query và D1 giống hệt nhau.
2. **Document có similarity thấp nhất:** D3, vì không có term nào trùng với query.
3. **Term có IDF thấp:** `medical` và `image`, vì cả hai xuất hiện trong D1 và D2. `classification`, `analysis`, `natural`, `language`, `processing` chỉ xuất hiện trong một document nên có IDF cao hơn.
4. **Bỏ IDF, chỉ dùng count vector:** dự đoán thứ hạng không đổi: D1 đứng đầu, D2 đứng sau do cùng hai term (`medical`, `image`), D3 cuối. Tuy nhiên khoảng cách điểm giữa D1 và D2 sẽ nhỏ hơn vì `classification` không còn được tăng trọng số nhờ IDF.

Đây là dự đoán được ghi trước khi chạy notebook `Ex6.ipynb`.
