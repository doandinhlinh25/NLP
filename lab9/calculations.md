# Calculations and BIO exercises

## Exercise 1 — BIO

Tự điền nhãn BIO theo các cụm entity được cho trong W9:

| Token | BIO label |
|---|---|
| Elon | B-PER |
| Musk | I-PER |
| founded | O |
| SpaceX | B-ORG |
| in | O |
| California | B-LOC |
| . | O |

## Exercise 2 — Entity boundary

Nếu mọi token chỉ mang nhãn `ORG`, ta biết chúng thuộc loại tổ chức nhưng không biết entity bắt đầu và kết thúc ở đâu; các tổ chức kề nhau cũng có thể bị nhập thành một. `B-ORG` đánh dấu token đầu của một entity mới, còn `I-ORG` nối token vào entity đang mở. Với “University of California Berkeley”, chuỗi `B-ORG I-ORG I-ORG I-ORG` giữ nguyên ranh giới của cả cụm và phân biệt nó với từ ORG đứng cạnh.

## Exercise 3 — Tensor shapes

Với batch `B=8`, sequence length `T=128`, hidden size `d=768`, số nhãn `C=9`:

- Hidden states `H`: `(B, T, d) = (8, 128, 768)`.
- Token logits `Z = HW + b`: `(B, T, C) = (8, 128, 9)`.
- Token labels `Y`: `(B, T) = (8, 128)`.

## Ý nghĩa alignment và ignore index

Pipeline trong `ner_pipeline.py` dùng Strategy A: word đầu tiên của mỗi word giữ nguyên BIO label; continuation subword và special token mang `-100`. Loss `CrossEntropyLoss` của token classifier bỏ qua vị trí `-100`. Các subword vẫn có mặt trong input và representation theo ngữ cảnh; chỉ không có mục tiêu giám sát riêng.

## Số liệu tokenization lần chạy thực tế

Run dùng 1,000 câu đầu của tập train để khảo sát tokenizer. Các giá trị dưới đây là thống kê mô tả trực tiếp từ `tokenization_analysis.csv` và `tokenization_summary.csv`. Entity words trong sample thường bị tách nhiều hơn: tỷ lệ word bị tách là 43.0% so với 14.9% ở non-entity. Tên riêng hiếm, dài hoặc ít quen thuộc với vocabulary có thể tạo nhiều subword pieces; đây là xu hướng trong sample, không áp dụng cho mọi từ entity. Câu ngắn có R trung bình 2.154, câu dài 1.183; R là tỷ số token/word nên không đồng nghĩa câu ngắn có số token tuyệt đối cao hơn.

| Nhóm word | Số word | Pieces trung bình / word | Tỉ lệ word bị tách |
|---|---:|---:|---:|
| Non-entity | 9,224 | 1.301 | 0.149 |
| Entity | 2,833 | 1.888 | 0.430 |

| Nhóm câu | Số câu | Words trung bình | Tokens trung bình | `R` trung bình |
|---|---:|---:|---:|---:|
| Short (<20 words) | 801 | 7.714 | 12.984 | 2.154 |
| Long (≥20 words) | 199 | 29.538 | 34.910 | 1.183 |

## Entity-level metrics của Transformer

Metric là exact span + type; `support` là số gold entity trong test split. Nhận xét thận trọng theo support được ghi ngay sau bảng.

| Nhãn | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| PER | 0.856 | 0.875 | 0.865 | 257 |
| ORG | 0.610 | 0.639 | 0.624 | 252 |
| LOC | 0.698 | 0.732 | 0.715 | 231 |
| MISC | 0.430 | 0.378 | 0.402 | 98 |

| Độ dài gold entity (word) | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| 1 | 0.611 | 0.726 | 0.664 | 508 |
| 2 | 0.926 | 0.703 | 0.799 | 283 |
| 3 | 0.700 | 0.512 | 0.592 | 41 |
| 4+ | 0.500 | 0.500 | 0.500 | 6 |

PER tốt nhất (F1=.865; support 257), tiếp theo LOC (.715; 231), ORG (.624; 252), MISC thấp nhất (.402; 98). Không thể quy nguyên nhân chỉ cho support: MISC ít mẫu hơn, nhãn đa dạng và ranh giới/loại hình khó hơn; ORG cũng thường mơ hồ theo ngữ cảnh. Theo độ dài, F1 của entity 2 từ (.799; 283) cao hơn entity 1 từ (.664; 508), còn 3 từ giảm (.592; 41). Bucket 4+ chỉ có 6 entity nên F1=.500 rất không ổn định. Dữ liệu này không ủng hộ quy tắc đơn giản “entity càng dài càng dễ/khó”; exact span đòi hỏi mọi token và loại đều đúng.
