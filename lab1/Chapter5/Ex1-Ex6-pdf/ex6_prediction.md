# Exercise 6 — Prediction

> **Ghi chú:** Nội dung tính toán được tự gõ tay; AI chỉ hỗ trợ format và chỉnh cách diễn đạt để làm rõ ý nghĩa.

## 1. Dữ liệu và vocabulary

| Label | Text |
| --- | --- |
| D1 | medical image classification |
| D2 | medical image analysis |
| D3 | natural language processing |
| Q | medical image classification |

Sau tokenization, vocabulary theo thứ tự dùng để minh họa là:

[medical, image, classification, analysis, natural, language, processing]

Mỗi văn bản có ba token. Count vector tương ứng:

| Text | Count vector |
| --- | --- |
| D1 | [1, 1, 1, 0, 0, 0, 0] |
| D2 | [1, 1, 0, 1, 0, 0, 0] |
| D3 | [0, 0, 0, 0, 1, 1, 1] |
| Q | [1, 1, 1, 0, 0, 0, 0] |

## 2. Tính document frequency và IDF

Số document là $N = 3$. Dùng công thức:

$$
\operatorname{idf}(t) = \ln\left(\frac{N}{df(t)}\right)
$$

| Term | $df(t)$ | $idf(t)$ |
| --- | ---: | ---: |
| medical | 2 | $\ln(3/2) \approx 0.405465$ |
| image | 2 | $\ln(3/2) \approx 0.405465$ |
| classification | 1 | $\ln(3) \approx 1.098612$ |
| analysis | 1 | $\ln(3) \approx 1.098612$ |
| natural | 1 | $\ln(3) \approx 1.098612$ |
| language | 1 | $\ln(3) \approx 1.098612$ |
| processing | 1 | $\ln(3) \approx 1.098612$ |

Vì medical và image xuất hiện trong hai trên ba document, chúng có IDF thấp nhất.

## 3. TF-IDF vectors

Với mỗi văn bản, $tf(t, d) = 1/3$ cho mỗi term xuất hiện đúng một lần. Đặt:

$$
a = \frac{\ln(3/2)}{3} \approx 0.135155,
\qquad
b = \frac{\ln(3)}{3} \approx 0.366204
$$

Theo vocabulary ở trên:

| Text | TF-IDF vector |
| --- | --- |
| D1 | [a, a, b, 0, 0, 0, 0] |
| D2 | [a, a, 0, b, 0, 0, 0] |
| D3 | [0, 0, 0, 0, b, b, b] |
| Q | [a, a, b, 0, 0, 0, 0] |

### Similarity giữa Q và D1

Q và D1 có cùng vector, nên:

$$
\cos(Q, D1) = 1
$$

### Similarity giữa Q và D2

Q và D2 chỉ cùng hai term medical và image:

$$
Q \cdot D2 = a^2 + a^2 = 2a^2
$$

$$
\lVert Q \rVert = \lVert D2 \rVert = \sqrt{2a^2 + b^2}
$$

$$
\cos(Q, D2)
= \frac{2a^2}{2a^2 + b^2}
\approx 0.214099
$$

### Similarity giữa Q và D3

Q và D3 không có term chung:

$$
Q \cdot D3 = 0
\quad \Rightarrow \quad
\cos(Q, D3) = 0
$$

Vì vậy, ranking theo TF-IDF là: D1 (1.000000) > D2 (0.214099) > D3 (0.000000).

## 4. So sánh với count vector

Với count vector:

$$
Q \cdot D2 = 1 + 1 = 2
$$

$$
\lVert Q \rVert = \lVert D2 \rVert = \sqrt{3}
$$

$$
\cos_{\text{count}}(Q, D2)
= \frac{2}{\sqrt{3}\sqrt{3}}
= \frac{2}{3}
\approx 0.666667
$$

Tương tự, $\cos_{\text{count}}(Q, D1) = 1$ và $\cos_{\text{count}}(Q, D3) = 0$. Ranking count vector là: D1 (1.000000) > D2 (0.666667) > D3 (0.000000).

## 5. Kết luận trả lời đề bài

1. Document có similarity cao nhất là **D1**, vì D1 giống hệt query.
2. Document có similarity thấp nhất là **D3**, vì D3 không có term chung với query.
3. Term có IDF thấp là **medical** và **image**.
4. Bỏ IDF không làm thay đổi ranking: vẫn là **D1 > D2 > D3**. Tuy nhiên, D2 có count similarity 0.666667 nhưng TF-IDF similarity chỉ còn 0.214099, vì classification có IDF cao và chỉ xuất hiện trong D1/query.
