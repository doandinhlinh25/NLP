# LAB 07 — Tính toán và lý thuyết

## 1. Một attention head

Với `X ∈ R^(3×4)` và `WQ, WK, WV ∈ R^(4×2)`, phép chiếu cho `Q, K, V ∈ R^(3×2)`. Score là `S = QKᵀ/√2 ∈ R^(3×3)`. Softmax theo từng hàng tạo `A ∈ R^(3×3)` với tổng mỗi hàng bằng 1; `Z = AV ∈ R^(3×2)`. Phép chiếu chỉ đổi không gian biểu diễn, phần attention vẫn là scaled dot-product attention của LAB 06.

## 2. Vì sao nối các head?

Hai head có kích thước `T×2`; nối theo feature tạo `T×4`. Mỗi head có thể học phép chiếu và phân phối chú ý riêng. Nối giữ lại các đặc trưng cùng lúc để ma trận `WO ∈ R^(4×4)` học cách phối hợp chúng về dimension mô hình 4. Nếu cộng hai head, các chiều tương ứng bị trộn ngay và không còn giữ nguyên hai nhóm đặc trưng để phép chiếu đầu ra kết hợp.

## 3. Positional encoding, d=4

Với `PE(pos, 2i)=sin(pos/10000^(2i/d))` và `PE(pos, 2i+1)=cos(pos/10000^(2i/d))`, hai tần số cho d=4 là `1` và `0.01`:

| pos | PE vector (xấp xỉ) |
|---:|---|
| 0 | `[0.000000, 1.000000, 0.000000, 1.000000]` |
| 1 | `[0.841471, 0.540302, 0.010000, 0.999950]` |
| 2 | `[0.909297, -0.416147, 0.020000, 0.999800]` |

Input token là embedding cộng vector ở vị trí tương ứng. Không có positional signal, self-attention không phân biệt hoán vị token nếu cũng hoán vị đầu ra trước pooling bất biến hoán vị.

## 4. FFN

`x=[1,2]`, `W1=I`, `b1=[0,-1]`: `W1x+b1=[1,1]`, vì vậy `h=ReLU([1,1])=[1,1]`. Đề không cho `W2` hoặc `b2`, do đó không thể tính một vector `y` số cụ thể; kết quả tổng quát là `y=W2[1,1]+b2`. FFN áp dụng biến đổi phi tuyến riêng tại từng vị trí, không tính tương tác giữa các token.

## 5. Residual và normalization

Residual cho `y=x+F(x)`, giữ đường truyền trực tiếp của input và cho phép block học phần hiệu chỉnh. Với `x=[1,2]`, `F(x)=[1,2]` thì tổng là `[2,4]`; chỉ dùng `F(x)` sẽ bỏ đường truyền gốc. LayerNorm chuẩn hóa theo feature của mỗi token, đặt sau phép cộng trong block này.

## 6. Shape trace của core implementation

Với `B=8`, `T=100`, `d_model=128`, `H=4`, `d_k=d_model/H=32`:

| Tensor | Shape |
|---|---|
| Input và embedding + PE | `(8, 100, 128)` |
| Q, K, V sau tách head | `(8, 4, 100, 32)` |
| Score `QKᵀ/√d_k` và attention weights | `(8, 4, 100, 100)` |
| Context sau gộp head và output projection | `(8, 100, 128)` |
| FFN hidden (`d_ff=256`) | `(8, 100, 256)` |
| Block output | `(8, 100, 128)` |

Attention scores có `H×T×T` phần tử mỗi mẫu: `T=128` cho `4×16,384=65,536`; `T=256` cho `4×65,536=262,144`. Riêng mỗi head, tăng T từ 128 lên 256 làm ma trận tăng từ 16,384 lên 65,536 phần tử, tức bốn lần. Bộ nhớ và thời gian full attention tăng bậc hai theo T.

## 7. Tóm tắt khái niệm

- Transformer dùng self-attention để các vị trí trao đổi thông tin trực tiếp, tính song song theo token trong một layer.
- Multi-head dùng nhiều bộ chiếu song song; cố định `d_model`, tăng số head làm `d_k` giảm chứ không tăng tổng representation dimension.
- Positional encoding bổ sung thứ tự mà self-attention không có sẵn.
- FFN tăng khả năng biến đổi phi tuyến theo từng vị trí.
- Residual + LayerNorm hỗ trợ truyền thông tin và ổn định biểu diễn/training.
- Transformer vẫn tốn attention `O(T²)` bộ nhớ/tính toán và cần positional information để mã hóa thứ tự.
