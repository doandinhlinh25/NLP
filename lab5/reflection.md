# LAB 05 - Tổng kết và suy ngẫm

## Bộ dữ liệu và độ dài chuỗi

Tệp CSV IMDb có 50.000 bài đánh giá, chia đều giữa hai nhãn tích cực và tiêu cực. Có 418 nội dung đánh giá trùng khớp hoàn toàn, không có văn bản hoặc nhãn bị thiếu. Sau khi bộ tách từ loại bỏ thẻ HTML và tách từ, độ dài nằm trong khoảng 6–2.477 token (trung vị 173, trung bình 231,3, phân vị 75 là 280, phân vị 90 là 452). `sequence_lengths.png` cho thấy phần đuôi dài về bên phải của phân phối. Các bản trùng được loại bỏ trước khi chia tập; sau đó thí nghiệm lấy các tập con cân bằng gồm 5.000 mẫu huấn luyện, 1.000 mẫu xác thực và 1.000 mẫu kiểm thử. Đây là tập con giới hạn theo tài nguyên tính toán, vì vậy không thể đối chiếu trực tiếp điểm số với LAB 04 dùng toàn bộ dữ liệu.

Độ dài tối đa cần được chọn dựa trên phân phối độ dài, lượng nội dung có ý nghĩa được giữ lại, vị trí thường xuất hiện tín hiệu cảm xúc và tài nguyên tính toán. Việc đệm (padding) không bổ sung thông tin ngôn ngữ; chuỗi được đóng gói (packed sequence) giúp trạng thái cuối không tiếp tục cập nhật qua các token PAD. Cắt ngắn chuỗi sẽ loại bỏ phần đuôi. Trong tập huấn luyện con, các ngưỡng 50, 100 và 200 token lần lượt giữ lại 21,5%, 41,6% và 68,9% tổng số token. Ngưỡng 30 token có thể bỏ sót kết luận hoặc ý tương phản xuất hiện muộn sau từ “but”.

## Mô hình cơ sở và các kiến trúc

Tất cả các hàng dưới đây dùng cùng tập kiểm thử tách riêng gồm 1.000 bài đánh giá. Precision, recall và F1 được tính cho nhãn tích cực. Thời gian huấn luyện không bao gồm bước tiền xử lý và đánh giá.

| Mô hình | Độ dài | Số tham số | Thời gian huấn luyện (giây) | Độ chính xác | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | Toàn bộ | 12.001 | 2,9 | 0,854 | 0,843 | 0,870 | 0,856 |
| RNN | 100 | 776.385 | 17,2 | 0,494 | 0,493 | 0,430 | 0,459 |
| LSTM | 100 | 801.345 | 31,9 | 0,553 | 0,556 | 0,526 | 0,541 |
| GRU | 100 | 793.025 | 12,2 | 0,573 | 0,564 | 0,642 | 0,601 |

Dự đoán rằng LSTM có thể xử lý quan hệ xa tốt hơn RNN cơ bản phù hợp với F1 cao hơn trong thí nghiệm này, nhưng thí nghiệm không tách riêng khả năng suy luận trên khoảng cách xa. Ở độ dài 100, GRU có F1 cao nhất trong các mô hình hồi quy. Mô hình cơ sở TF-IDF vượt trội rõ rệt với tập dữ liệu nhỏ và chỉ hai epoch huấn luyện. Vì vậy, kết quả không ủng hộ nhận định rằng mô hình hồi quy luôn chính xác hơn. Số tham số đã tính cả lớp embedding; với cùng kích thước, phần hồi quy của GRU có ít tham số hơn LSTM.

Phép thử thứ tự giữ nguyên số lần xuất hiện của từng từ. TF-IDF cho cùng xác suất tích cực 0,4769 với “this movie is not good” và “this movie is good not”; GRU lần lượt cho 0,4383 và 0,4469. Với “the film is not bad” và “the film is bad not”, TF-IDF đều cho 0,0284, còn GRU lần lượt cho 0,4926 và 0,4822. Như vậy, GRU đã huấn luyện có nhạy với thứ tự trong phép thử nhỏ này, đúng như cơ chế tính toán của nó. Tuy nhiên, các điểm số gần nhau và không chứng minh mô hình hiểu phủ định một cách đáng tin cậy. Cả hai cặp câu đều có cách diễn đạt thiếu tự nhiên nên không thể dùng để kết luận về hiệu năng tổng quát.

## Độ dài và chi phí

Khi giữ nguyên GRU và thay đổi độ dài, F1 tăng từ 0,565 ở 50 token lên 0,601 ở 100 token và 0,654 ở 200 token. Thời gian huấn luyện tương ứng là 6,7; 12,2 và 167,6 giây. Giá trị loss trên tập xác thực ở epoch 2 lần lượt là 0,694; 0,689 và 0,677. Nhiều ngữ cảnh hơn đã giúp tăng F1 trong ba lần chạy này, nhưng lần chạy với chuỗi dài nhất tốn thời gian hơn nhiều. Đây là sự đánh đổi đo được trên CPU và cấu hình này, không phải bảo đảm rằng tăng độ dài luôn có lợi. Mức sử dụng bộ nhớ chưa được đo.

## Đường học và lỗi dự đoán

Ở độ dài 100, loss huấn luyện của RNN giảm từ 0,699 xuống 0,684, trong khi loss xác thực tăng từ 0,699 lên 0,700; đây có thể là dấu hiệu sớm của hiện tượng quá khớp. Loss huấn luyện và xác thực của LSTM và GRU đều giảm qua hai epoch. Hai điểm đo là quá ít để đánh giá diễn biến về sau. Ma trận nhầm lẫn của GRU là `[[252, 248], [179, 321]]`, với hàng và cột đều theo thứ tự tiêu cực rồi tích cực. Cả dự đoán dương tính giả và âm tính giả đều đáng chú ý; chỉ nhìn độ chính xác sẽ bỏ qua sự phân bố này.

`error_analysis.md` liệt kê ba bài đánh giá được dự đoán đúng và ba bài bị dự đoán sai trong tập kiểm thử, kèm xác suất, độ dài và các yếu tố có thể ảnh hưởng. Mẫu âm tính giả đầu tiên mở đầu bằng lời khen rõ ràng (“wonderful”, “terrific”) và dài 175 token. Vì các tín hiệu này nằm trước ngưỡng cắt 100 token, riêng việc cắt ngắn không đủ để giải thích lỗi. Mẫu âm tính giả thứ hai dài 719 token, kết hợp lời khen bộ phim với những từ nói về mất mát và đau buồn; sắc thái của từng từ và việc mất ngữ cảnh phía sau đều là các khả năng hợp lý. Mẫu âm tính giả thứ ba dài 909 token và có phần kể cốt truyện dài; đoạn đầu 100 token có thể bỏ sót nhận xét cuối. Đây là các giả thuyết, không phải kết luận nhân quả. Độ phủ từ vựng, thời lượng huấn luyện ngắn, nhãn khó phân định và quá trình tối ưu cũng có thể ảnh hưởng.

## Phép thử ứng dụng và quan hệ xa

GRU đã lưu cho xác suất tích cực lần lượt là 0,482 với đánh giá tích cực ngắn, 0,427 với đánh giá tích cực dài có ý kiến pha trộn, 0,456 với đánh giá tiêu cực có phủ định và 0,420 với đánh giá tương phản kiểu “terrible, but surprisingly good”. Với ngưỡng 0,5, mô hình xếp cả bốn vào lớp tiêu cực, trong đó có ba ví dụ tích cực. Các ngữ cảnh có thể có ảnh hưởng là “enjoyed”, “powerful ... wonderful”, “not like” và “terrible ... but ... good”; đầu ra không chứng minh mô hình đã dựa vào những từ nào. Bài đánh giá dài vẫn ngắn hơn ngưỡng cắt 100 token nên không thể quy lỗi của nó cho việc cắt ngắn.

Trong các phép thử tổng hợp về vị trí, thay “excellent” bằng “terrible” hầu như không làm đổi xác suất của GRU ở bất kỳ vị trí nào (chênh lệch dưới 0,01). Các ví dụ này cho thấy độ nhạy yếu với thời lượng huấn luyện hiện tại. Chúng không chứng minh hiện tượng tiêu biến gradient: cơ chế đó xuất phát từ tích các ma trận Jacobian qua nhiều bước, trong khi hành vi đo được cũng có thể do huấn luyện chưa đủ, từ vựng hoặc sự khác biệt về phân phối dữ liệu.

## Liên hệ với LAB 04 và attention

TF-IDF có biểu diễn thưa và bỏ qua thứ tự chính xác của từ trong mô hình unigram này. Word2Vec lấy trung bình có biểu diễn đặc nhưng cũng làm mất thứ tự. RNN, LSTM và GRU tạo các trạng thái phụ thuộc ngữ cảnh theo thứ tự chuỗi; LSTM và GRU bổ sung các cổng giúp duy trì thông tin qua nhiều bước. Tuy nhiên, mô hình hồi quy vẫn nén đoạn đầu ngày càng dài vào một trạng thái hữu hạn, đọc lần lượt từng vị trí và buộc các tương tác xa đi qua nhiều bước. Cơ chế attention cho phép truy cập trực tiếp hơn đến những vị trí liên quan và là động lực cho bài thực hành tiếp theo. Chỉ cải thiện kiến trúc không thể thay thế đủ dữ liệu, đánh giá cẩn thận và phân tích lỗi.
