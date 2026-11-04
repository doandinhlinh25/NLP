# LAB 07 — Phân tích lỗi

Các mẫu dưới đây lấy từ tập test cố định của IMDb. Attention weights mô tả phân phối nội bộ, không được xem là lời giải thích nhân quả.

| Kết quả | Nhãn thật | Dự đoán | P(tích cực) | Mẫu văn bản |
|---|---|---|---:|---|
| Đúng | positive | positive | 0.510 | lee chang dong's exceptional secret sunshine is the single most emotionally ravaging experience of the year it is an instantly sobering brutally honest character piece on the reverberations of loss and a graceful memento mori that resonates… |
| Đúng | positive | positive | 0.885 | i won't go to a generalization and say it's the best love story of all time as some have said that's fine people feel very deeply about this film you either love it i believe or you simply hate it i don't want to say the best of all because… |
| Đúng | positive | positive | 0.715 | it is always a well known and important directorial device to set up the atmosphere of a film within the first 5 minutes in the crucial opening scenes the film should assert itself and make the viewers take notice and get interested in the … |
| Sai | positive | negative | 0.262 | this is a wonderful film makes the 1950 s look beautifully stylish kim novak is intriguing and compelling as a modern day witch with one foot in manhattan and another in infinity all the supporting performances are terrific from jack lemmon… |
| Sai | positive | negative | 0.152 | i stumbled upon this movie by accident i mean how else could i find out it wasn't hyped at all by the studios nor did i even hear about it's release from my normally plugged in friends after throwing my money away on so many bad movies this… |
| Sai | positive | negative | 0.367 | the movie is a very good movie one of the best from yash raj films the direction is incredible the screenplay is brilliant the story is excellent it tells about rahul who is obssed of kiran his college friend he is a full blown psycho doing… |

## Nhận xét

Các trường hợp sai cần được đọc cùng văn bản gốc để nhận diện phủ định, tương phản, độ dài/cắt cụt và từ hiếm. Mẫu nhỏ này gợi ý các hướng xem xét nhưng không đủ để khẳng định nguyên nhân duy nhất. Attention heatmap chỉ cho biết trọng số mô hình phân phối giữa token; trọng số cao không chứng minh token đó là lý do đưa ra nhãn.
