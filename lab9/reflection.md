# Reflection (300–500 words)

Tự viết 300–500 từ trả lời: Tại sao cùng một pretrained Transformer có thể dùng cho sequence classification và token classification chỉ bằng cách thay prediction head?

Bài viết cần giải thích contextual representation, `[CLS]` so với biểu diễn của từng token, cách head ánh xạ biểu diễn thành nhãn đầu ra, và mối liên hệ với LAB 08–10. Hãy dùng kết quả và giới hạn thí nghiệm LAB 09 làm dẫn chứng, không khẳng định vượt quá dữ liệu.

**Bản reflection của tôi:**

**Bản nháp reflection (khoảng 370 từ):**

Một pretrained Transformer có thể phục vụ nhiều bài toán vì phần backbone biến chuỗi đầu vào thành các contextual representation, còn prediction head quyết định ta đọc những biểu diễn đó thành dạng đầu ra nào. Ở mỗi lớp, vector của một token được cập nhật bằng self-attention, nên nó chứa thông tin từ chính token và các token liên quan trong câu. Cùng một từ vì vậy có thể có vector khác nhau tùy câu; “Paris” trong câu về thủ đô và trong tên một đội thể thao không nhất thiết được biểu diễn giống nhau. Pretraining cung cấp điểm khởi đầu giàu ngôn ngữ, sau đó fine-tuning điều chỉnh backbone và head theo nhãn của tác vụ.

Trong sequence classification như LAB 08, ta cần một nhãn cho toàn câu. Với kiến trúc BERT-style, vector `[CLS]` được dùng như biểu diễn tóm tắt để một classification head ánh xạ nó thành logits của các lớp, chẳng hạn positive/negative. DistilBERT cũng có biểu diễn token đầu vào và một head phù hợp cho phân loại chuỗi; chi tiết lấy vector có thể tùy kiến trúc. Trong token classification ở LAB 09, đầu ra cần một nhãn cho từng token, nên linear head được áp dụng lên từng vector: với hidden states `H` có kích thước `T × d`, phép `HW+b` tạo logits `T × C`. Softmax hoặc argmax theo chiều nhãn cho dự đoán BIO tại mỗi vị trí. Head thay đổi vì đơn vị dự đoán thay đổi: một vector gộp cho sequence so với từng vector token cho NER.

Thí nghiệm LAB 09 dùng DistilBERT cased, 2,000 câu train, 500 validation/test và một epoch. Entity-level F1 của Transformer là .6797, so với .6275 của baseline logistic regression dùng đặc trưng từ và cửa sổ lân cận; token F1 lần lượt là .7631 và .7048. Kết quả mẫu này phù hợp với khả năng contextual representation giúp gán nhãn token, nhưng run rút gọn không chứng minh mức cải thiện đó sẽ giữ nguyên trên toàn CoNLL-2003 hoặc miền khác. Nhãn MISC chỉ đạt F1 .402, thấp hơn PER .865; việc thay head không tự giải quyết mất cân bằng hay nhập nhằng nhãn. Alignment cũng cần xử lý vì annotation ở cấp word còn tokenizer tạo subword; thí nghiệm chỉ tính loss ở subword đầu tiên.

LAB 10 tiếp tục cùng ý tưởng backbone nhưng yêu cầu dự đoán vị trí bắt đầu và kết thúc của answer span. Như vậy, sequence classification, NER và QA khác nhau ở head, đơn vị đầu ra và cách chấm nhãn; biểu diễn ngữ cảnh chung giúp chuyển giao, nhưng không làm các tác vụ trở thành một bài toán giống nhau. Đây là diễn giải từ cấu hình và kết quả của run rút gọn, không phải kết luận về mọi Transformer hoặc mọi dữ liệu.

## Individual learning check

1. Sequence classification khác token classification thế nào?
2. BIO tagging dùng để làm gì?
3. `B-PER` khác `I-PER` thế nào?
4. Vì sao subword tokenizer tạo label alignment problem?
5. `IGNORE` trong loss có ý nghĩa gì?
6. Vì sao NER không nên chỉ đánh giá bằng token accuracy?
7. Entity-level F1 khác token-level F1 thế nào?
8. Vì sao entity boundary khó?
9. Vì sao contextual representation hữu ích cho NER?
10. NER được dùng như thế nào trong Information Extraction?

**Câu trả lời của tôi:**
1. Sequence classification gán một nhãn cho toàn chuỗi; token classification gán một nhãn cho từng token.
2. BIO mã hóa entity theo ranh giới và loại: B bắt đầu entity, I tiếp tục entity, O nằm ngoài entity.
3. `B-PER` là token đầu của tên người; `I-PER` là token tiếp theo trong cùng tên.
4. Annotation thường ở mức word nhưng tokenizer có thể chia một word thành nhiều subword, nên phải xác định subword nào nhận nhãn gốc.
5. `IGNORE` (thường là -100) khiến vị trí đó không đóng góp vào loss; token vẫn có trong input và vẫn tham gia biểu diễn ngữ cảnh.
6. Token accuracy có thể che lỗi boundary: đúng một phần các từ vẫn có thể làm entity hoàn chỉnh sai; nhãn O áp đảo cũng làm accuracy trông cao.
7. Token F1 so khớp nhãn trên từng vị trí; entity-level F1 yêu cầu span và loại entity khớp chính xác.
8. Entity nhiều từ đòi hỏi đúng điểm bắt đầu, điểm kết thúc và nhãn loại; từ sát cạnh hoặc dấu câu dễ bị gộp/tách sai.
9. Contextual representation dùng ngữ cảnh hai phía để phân biệt cùng một surface form có thể là người, địa điểm hay tổ chức.
10. NER trích span có loại từ văn bản; IE có thể nhóm chúng thành trường dữ liệu có cấu trúc như PERSON, ORGANIZATION, LOCATION.

