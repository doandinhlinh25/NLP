# Learning check

1. **Vì sao TF-IDF sparse?** Vocabulary có hàng trăm nghìn chiều nhưng một document chỉ chứa một phần rất nhỏ terms; các vị trí còn lại có TF-IDF bằng 0.
2. **Vì sao term phổ biến có IDF thấp?** `df(t)` gần bằng số document `N`, nên `log(N / df(t))` gần 0. Term đó không giúp phân biệt document.
3. **Vì sao IDF cao chưa chắc TF-IDF cao?** Term phải xuất hiện trong document và có TF khác 0. Nếu không xuất hiện thì TF-IDF vẫn bằng 0.
4. **Vì sao cosine phù hợp?** Nó so sánh hướng của vector nên giảm ảnh hưởng của độ dài document và ưu tiên mức độ overlap có trọng số.
5. **Vì sao preprocessing đổi search result?** Nó đổi token, vocabulary, document frequency và query vector; do đó các feature đóng góp vào cosine similarity cũng đổi.
6. **Một failure case đã quan sát:** Query `barbecue cooking class` không tìm được document 0 về `BBQ class` trong Top-5 do hai cách gọi không chia sẻ token.
7. **Failure đó cần representation nào tiếp theo?** Embedding/dense retrieval có thể học sự gần nhau về ngữ nghĩa giữa `barbecue` và `BBQ`; có thể kết hợp với lexical TF-IDF thành hybrid retrieval.
