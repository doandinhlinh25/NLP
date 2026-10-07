# Chapter 21 — Evaluation: Word similarity

## Thiết lập

Các pair được chấm bằng cosine similarity từ model CBOW của Chapter 17. `Expected similarity` là thứ tự trực giác trước khi xem model: `high` > `medium` > `low`. `model_rank = 1` là score cao nhất.

## Xếp hạng của model

| Model rank | Word pair | Expected similarity | Cosine similarity | Semantically related |
| ---: | --- | --- | ---: | --- |
| 1 | king – queen | high | 0.938508 | True |
| 2 | car – automobile | high | 0.849378 | True |
| 3 | cat – dog | medium | 0.844520 | True |
| 4 | doctor – physician | high | 0.837317 | True |
| 5 | computer – banana | low | 0.798302 | False |

## Kết quả định lượng

- Mean cosine của các cặp được xem là related: 0.867430.
- Mean cosine của cặp unrelated: 0.798302.
- Spearman correlation giữa mức trực giác (`high`, `medium`, `low`) và model score: 0.670820.

Score và thứ hạng cần được so sánh với intuition, không chỉ nhận xét chung rằng kết quả “khá tốt”. Nếu cặp unrelated có score cao hoặc một cặp expected `high` có thứ hạng thấp, đó là evidence để phân tích ảnh hưởng của corpus nhỏ, frequency và context window ở các chapter tiếp theo.
