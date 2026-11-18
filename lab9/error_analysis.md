# Error analysis

Sau khi chạy pipeline, dùng `error_analysis_examples.csv` để chọn ít nhất 5 false positives, 5 false negatives và 5 lỗi boundary. Bổ sung giải thích của chính bạn; một dòng có thể thuộc hơn một nhóm lỗi.

| Text | Gold | Prediction | Error type | Explanation |
|---|---|---|---|---|
| Thai rice vessels loading and movements at Dec 06 . | Thai/MISC | Thai/LOC | Sai loại | “Thai” chỉ quốc tịch/nhóm trong ngữ cảnh này, được gold gán MISC; mô hình chọn LOC, có thể do liên hệ quốc gia. |
| Yakomas are hounded in stronghold districts of Patasse 's Baya people while other tribes have fled areas in rebel hands . | Patasse/PER | Patasse/LOC | Sai loại | Tên người trong cấu trúc sở hữu bị nhầm thành địa điểm; ngữ cảnh lân cận về vùng/tribes có thể gây nhiễu. |
| Real Madrid 's Balkan strike force ... Barcelona in Spain 's old firm game ... | Barcelona/ORG | Barcelona/LOC | Sai loại | Đội thể thao và địa danh có cùng surface form; câu nói về trận đấu nhưng nhãn bị nghiêng sang LOC. |
| Wenchang ... , Xinhua said . | Xinhua/ORG | Xinhua/LOC | Sai loại | Xinhua là hãng tin trong cấu trúc “said”; dự đoán địa điểm có thể do tên riêng đơn lẻ và thiếu ngữ cảnh đủ mạnh. |
| Interfax ... work with Albright . | Albright/PER | Albright/ORG | Sai loại | Tên người bị gán tổ chức; ví dụ cho thấy lỗi loại dù boundary một token đúng. |
| SOCCER - LEADING SCOTTISH PREMIER DIVISION SCORERS . | SCOTTISH PREMIER DIVISION/MISC | Không có | Bỏ sót | Tiêu đề viết hoa và cụm danh từ dài có thể khó hơn câu văn thông thường; toàn span bị bỏ qua. |
| “Internet is a potential cash cow ...” | Internet/MISC | Không có | Bỏ sót | “Internet” thường là danh từ chung, nhưng annotation trong câu này là MISC; mô hình có thể ưu tiên cách dùng phổ biến. |
| ALPINE SKIING-GOETCHL WINS WORLD CUP DOWNHILL . | WORLD CUP/MISC | Không có | Bỏ sót | Entity nằm trong tiêu đề thể thao và bị bỏ; kiểu chữ/ghép dấu câu có thể gây khó cho boundary. |
| CRICKET - WOOLMER MAKES SENTIMENTAL RETURN TO KANPUR . | WOOLMER/PER | Không có | Bỏ sót | Tên người trong headline toàn chữ hoa không được nhận ra. |
| CRICKET - WOOLMER MAKES SENTIMENTAL RETURN TO KANPUR . | KANPUR/LOC | Không có | Bỏ sót | Địa danh cũng bị bỏ trong cùng tiêu đề, gợi ý ảnh hưởng phong cách headline; hai entity riêng biệt. |
| Mongolia ... arrived from Ulan Bator ... | Ulan Bator/LOC | Ulan/LOC | Sai boundary | Mô hình nhận đúng loại nhưng chỉ lấy token đầu, bỏ “Bator”; exact-span metric tính sai entity. |
| Real Madrid 's Balkan strike force ... | Real Madrid/ORG | Real/ORG; thiếu Madrid | Sai boundary | Tên tổ chức bị cắt sau token đầu, dù loại đúng. |
| Crude petroleum 2,557 2,832 15,838 17,648 | Không có entity | Crude/PER | Entity thừa + boundary | Số liệu dạng bảng/headline gây dự đoán người giả; false positive này phản ánh ngữ cảnh phi văn xuôi. |
| S. Doull c subs ( M. Wasim ) b Waqar 1 | M. Wasim/PER | Wasim/PER (hoặc M./PER trong candidate khác) | Sai boundary | Tên người bị chia/cắt sai ở token đầu hoặc cuối trong dòng thống kê thể thao; dữ liệu dạng bảng có thể làm ranh giới khó nhận ra. |
| Interfax, outlining Albright 's biography ... United Nations ... | United Nations/ORG | United/MISC | Sai boundary và loại | Tên tổ chức nhiều từ bị cắt còn “United” và gán sai loại; ngữ cảnh câu nhắc đến tổ chức quốc tế. |

Các ví dụ lấy từ `error_analysis_examples.csv`; cùng một câu có thể xuất hiện nhiều lần vì mỗi hàng là một span lỗi. Bảng này chọn 5 false positives (sai loại/entity thừa), 5 false negatives và 5 boundary errors; nhóm có thể giao nhau. Candidate tự động chỉ ra vị trí cần xem, còn giải thích là phân tích dựa trên câu và cần đối chiếu annotation gốc. Không thể kết luận chắc chắn lỗi do subword alignment chỉ từ bảng này; cần xem trực tiếp word_ids/subword của ví dụ.

## Same surface form in different contexts

Tìm tối thiểu ba trường hợp trong test set có surface form giống nhau nhưng nhãn entity khác nhau; giải thích ngữ cảnh làm thay đổi vai trò. Nếu không tìm được trong test sample, nêu rõ giới hạn và mở rộng việc rà soát.

| Surface form | Context A / gold | Context B / gold | Giải thích |
|---|---|---|---|
| BOSTON | “Hartford 4 BOSTON 2” / ORG (team) | “AT BOSTON” / LOC (city) | Trong bảng kết quả thể thao BOSTON là đội; trong “AT BOSTON” là địa điểm thi đấu. |
| INDIANAPOLIS | “INDIANAPOLIS 1996-12-06” / LOC | “INDIANAPOLIS 8 6 0 269 284” / ORG (team) | Một headline là địa danh; dòng kết quả khác dùng tên đội. |
| Philadelphia | “The loss by Philadelphia allowed ... Packers” / ORG (team) | “Philadelphia 6 DALLAS 3” / LOC theo gold | Cùng tên trong câu về đội NFL so với headline tỷ số; các nhãn CoNLL ở sample cần được giữ nguyên dù trực giác có thể khác. |
| OAKLAND | “OAKLAND 6 7 0 ...” / ORG (team) | “KANSAS CITY AT OAKLAND” / LOC | Ở bảng điểm là đội, còn sau “AT” là địa điểm trận đấu theo annotation. |
