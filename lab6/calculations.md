# LAB 06 — Lý thuyết và bài tính

## 1. Động cơ và vai trò của Q/K/V

Classifier hồi quy dùng hidden state cuối phải truyền tín hiệu xuất hiện sớm qua mọi bước ở giữa. Attention cho phép query so trực tiếp với tất cả key rồi kết hợp các value tương ứng. **Query** biểu diễn thông tin cần tìm, **key** biểu diễn đặc trưng dùng để so khớp, còn **value** chứa thông tin được tổng hợp. Score là mức độ phù hợp, chưa phải xác suất; softmax biến chúng thành các weight không âm có tổng bằng một. Context là tổng có trọng số của các value. Weight lớn nhất không bắt buộc ứng với từ mà con người cho là quan trọng nhất.

## 2. Tính tay

**Bài 1.** Với `q=[1,2]` và ba hàng của `K` là `[1,0]`, `[0,1]`, `[1,1]`, các dot product lần lượt là `1·1+2·0=1`, `1·0+2·1=2`, `1·1+2·1=3`. Vậy `s=[1,2,3]`.

**Bài 2.** Trừ giá trị lớn nhất khỏi `[1,2,3]` được `[-2,-1,0]`. Các giá trị hàm mũ xấp xỉ `[0,135335; 0,367879; 1]`, tổng `1,503215`. Do đó `softmax(s)≈[0,090031; 0,244728; 0,665241]`, tổng bằng `1,000000` sau làm tròn. Hàm mũ tăng nghiêm ngặt nên score lớn nhất nhận weight lớn nhất.

**Bài 3.** `c=0,1[1,0]+0,2[0,1]+0,7[2,2]=[1,5; 1,6]`. Value thứ ba có hệ số lớn nhất và đóng góp `[1,4; 1,4]`, lớn nhất theo từng tọa độ trong ví dụ này.

**Bài 4.** Với `q=[1,0]` và ba key đã cho, score chưa scaling là `[1,0,1]`. Các giá trị mũ là `[e,1,e]`, mẫu số `2e+1`; weight xấp xỉ `[0,422319; 0,155362; 0,422319]`. Context là `0,422319[1,0]+0,155362[0,2]+0,422319[2,1]≈[1,266956; 0,733044]`. Bài này dùng dot product chưa scaling đúng theo yêu cầu.

Ví dụ score `[2,1,0]` ở phần softmax của đề cho weight `[0,665241; 0,244728; 0,090031]`; token thứ nhất nhận attention lớn nhất.

## 3. Scaling, temperature và độ ổn định số

Với `d_k=4`, score gốc `[4,2,0]` trở thành `[2,1,0]` khi chia cho `√4=2`. Softmax của score gốc là `[0,866813; 0,117310; 0,015876]`; sau scaling là `[0,665241; 0,244728; 0,090031]`. Scaling làm phân phối phẳng hơn do thu nhỏ chênh lệch giữa các score. Nói chung, phép chia cho `√d_k` hạn chế sự tăng phương sai dot product khi nhiều chiều cùng đóng góp.

Với cùng score gốc `[4,2,0]`, entropy tính bằng `−Σ p_i ln(p_i)` theo đơn vị nat:

| Temperature | Weights | Weight lớn nhất | Entropy | Nhận xét |
|---:|---|---:|---:|---|
| 0,5 | `[0,981690; 0,017980; 0,000329]` | 0,981690 | 0,093035 | Tập trung mạnh |
| 1,0 | `[0,866813; 0,117310; 0,015876]` | 0,866813 | 0,441057 | Phân phối gốc |
| 2,0 | `[0,665241; 0,244728; 0,090031]` | 0,665241 | 0,832396 | Phẳng hơn |

Với `[1000,1001,1002]`, tính hàm mũ trực tiếp gây tràn số. Trừ `1002` cho từng score được `[-2,-1,0]` và vẫn cho xác suất `[0,090031; 0,244728; 0,665241]`. Với mọi hằng số `c`, `exp(s_i−c)/Σ_j exp(s_j−c)=exp(s_i)/Σ_j exp(s_j)` vì thừa số chung triệt tiêu. Vì vậy, trừ giá trị lớn nhất giữ nguyên kết quả softmax trong số học chính xác.

## 4. Ví dụ self-attention

Đề cho kích thước nhưng không cho giá trị cụ thể của `X`, `W_Q`, `W_K`, `W_V`. Để tính được kết quả có thể kiểm tra, chọn ba hàng `X` ứng với `I`, `like`, `cats` là `[1,0]`, `[0,1]`, `[1,1]` và cả ba ma trận chiếu là ma trận đơn vị `2×2`. Khi đó `Q=K=V=X`:

```text
QKᵀ = [[1,0,1],
       [0,1,1],
       [1,1,2]]

QKᵀ/√2 ≈ [[0,7071; 0; 0,7071],
           [0; 0,7071; 0,7071],
           [0,7071; 0,7071; 1,4142]]

A ≈ [[0,4011; 0,1978; 0,4011],
     [0,1978; 0,4011; 0,4011],
     [0,2483; 0,2483; 0,5035]]

AV ≈ [[0,8022; 0,5989],
      [0,5989; 0,8022],
      [0,7517; 0,7517]]
```

Mỗi hàng của `A` có tổng bằng một. Hàng đầu cho thấy `I` kết hợp thông tin từ cả `I`, `like`, `cats`; các hàng khác nhau vì query khác nhau. Đây là ví dụ minh họa tự chọn, không phải hành vi ngôn ngữ đã học.

## 5. Chi phí và giới hạn

| Số token `T` | Ma trận attention | Số phần tử | Bộ nhớ float32 cho một ma trận |
|---:|---:|---:|---:|
| 10 | 10×10 | 100 | 400 B |
| 100 | 100×100 | 10.000 | 40 KB |
| 500 | 500×500 | 250.000 | 1 MB |
| 1.000 | 1.000×1.000 | 1.000.000 | 4 MB |
| 10.000 | 10.000×10.000 | 100.000.000 | 400 MB |

Bảng chỉ tính một ma trận cho một mẫu và một head; chưa gồm Q/K/V, gradient, trạng thái optimizer và batch. Khi `T` tăng gấp đôi, kích thước ma trận tăng gấp bốn. Full self-attention cho phép hai token tương tác trực tiếp và có thể tính song song theo vị trí, còn xử lý hồi quy đi lần lượt từng bước. Đổi lại, self-attention có chi phí score và bộ nhớ bậc hai, đồng thời cần biểu diễn vị trí để phân biệt thứ tự. Classifier đơn giản của lab chưa có positional encoding nên khả năng nhạy với thứ tự bị hạn chế. Attention weights chỉ cho thấy phân phối trọng số, không chứng minh giải thích ngôn ngữ hay reasoning.
