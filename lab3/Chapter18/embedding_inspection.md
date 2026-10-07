# Chapter 18 — Inspect embeddings

## Truy vấn

Model CBOW ở Chapter 17 được truy vấn bằng `model.wv.most_similar("doctor", topn=5)`. Cosine score càng cao nghĩa là hai embedding có hướng gần nhau trong không gian vector của model này. Trong sample huấn luyện, `doctor` xuất hiện 45 lần.

## Top-5 của `doctor`

| Rank | Word | Cosine similarity | Corpus frequency | Shared local contexts |
| ---: | --- | ---: | ---: | --- |
| 1 | baseboards | 0.966536 | 27 | the (15), to (13), your (11) |
| 2 | unexpected | 0.963113 | 14 | the (7), to (5), of (4) |
| 3 | correct | 0.959862 | 63 | the (22), to (21), you (10) |
| 4 | swimming | 0.957038 | 29 | the (13), your (8), and (8) |
| 5 | correctly | 0.955030 | 25 | to (11), the (10), you (8) |

`Shared local contexts` là các token xuất hiện trong phạm vi 5 vị trí quanh cả `doctor` và từ hàng tương ứng, với số đếm chung nhỏ nhất giữa hai phía. Đây là evidence trực tiếp từ sample corpus, không phải nhãn ngữ nghĩa được gán thủ công.

## Diễn giải

Top-5 quan sát được không phải các từ y khoa như dự đoán ở Chapter 9. Evidence trong bảng cho thấy shared contexts phần lớn là từ chức năng phổ biến như `the`, `to`, `of` và `your`, thay vì context y khoa đặc hiệu. Với chỉ 45 lần xuất hiện của `doctor`, model có quá ít quan sát để học một vector ổn định; các từ có pattern ngữ cảnh chung có thể nhận score rất cao. Vì vậy, score cao không tự chứng minh quan hệ đồng nghĩa: kết quả này là một failure case cần được kiểm tra bằng frequency và context gốc, thay vì chỉ kết luận rằng model “học được semantic similarity”.
