# Dự đoán về TF-IDF và tìm kiếm

## Prediction 1 — Vocabulary

Nếu corpus có **30K documents**, vocabulary ước tính có khoảng **200K unique terms**.

## Prediction 2 — Sparsity

Ma trận TF-IDF sẽ **sparse** (thưa thớt).

- Kích thước ma trận: `30.000 × 200.000`.
- Tổng số phần tử: `30.000 × 200.000 = 6.000.000.000`.
- Mỗi document ước tính chỉ chứa khoảng **100–300 terms**.

Do đó, số phần tử khác `0` trên mỗi document chỉ chiếm một phần rất nhỏ so với 200K cột trong vocabulary. Tỷ lệ phần tử bằng `0` trong ma trận sẽ rất lớn.

## Prediction 3 — Search

Với một query bất kỳ, các documents đứng đầu kết quả tìm kiếm chưa chắc là những documents **gần nghĩa** nhất với query.

Nguyên nhân là TF-IDF ưu tiên các document có nhiều terms trùng với query. Nếu dữ liệu chứa nhiều terms rác, hoặc các terms chưa được lọc tốt để phân biệt documents, kết quả được đề xuất có thể không thực sự liên quan đến query.
