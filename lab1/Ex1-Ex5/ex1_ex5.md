# Exercises 1–5

## Exercise 1

```text
D1 = [1, 0, 1, 1, 0]
D2 = [0, 1, 1, 1, 0]
D3 = [1, 0, 0, 1, 1]
```

## Exercise 2

```text
TF(cat, D1) = 1/3
TF(fish, D1) = 1/3
TF(eats, D1) = 1/3
TF(t, D1) = 1/3 + 1/3 + 1/3 = 1
```

## Exercise 3

`IDF(t) = log(N / df(t))`

- `IDF(cat) = log(3 / df(cat)) = log(3 / 2) ≈ 0.18`
- `IDF(dog) = log(3 / df(dog)) = log(3 / 1) ≈ 0.48`
- `IDF(eats) = log(3 / df(eats)) = log(3 / 2) ≈ 0.18`
- `IDF(fish) = log(3 / df(fish)) = log(3 / 3) = 0`
- `IDF(like) = log(3 / df(like)) = log(3 / 1) ≈ 0.48`

> `IDF(fish)` nhỏ nhất vì `fish` xuất hiện nhiều nhất.

## Exercise 4

```text
TF-IDF(cat) = 1/3 × 0.18 = 0.06
TF-IDF(eats) = 1/3 × 0.18 = 0.06
TF-IDF(fish) = 1/3 × 0 = 0
```

Vì `fish` xuất hiện ở mọi tài liệu nên nó không có giá trị phân biệt giữa các tài liệu.

## Exercise 5

```text
Cos(x, y) = (x · y) / (||x||₂ · ||y||₂)
          = 2 / (√3 · √2) ≈ 0.82
```

Vì cosine similarity đo độ giống nhau về hướng của hai vector nên không quan tâm đến tỉ lệ số term trùng nhau.
