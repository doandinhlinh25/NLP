# Phân tích lỗi classification

**Dataset/task:** IMDb Dataset - phân loại cảm xúc review phim thành `positive` hoặc `negative`.  
**Model/representation được chọn theo validation:** TF-IDF unigram + Linear SVM (macro F1 validation: `0.8981`).  

## Các ví dụ phân loại đúng

| STT | Text | Nhãn thật | Dự đoán | Giải thích |
|---:|---|---|---|---|
| 1 | I hope this group of film-makers never re-unites. | negative | negative | Câu ngắn nhưng có tín hiệu phủ định rõ: `never`. |
| 2 | A touching movie. It is full of emotions and wonderful acting. | positive | positive | Các từ `touching`, `wonderful` là tín hiệu positive mạnh. |
| 3 | A great film in its genre ... A must see. | positive | positive | `great` và `must see` phù hợp với feature positive có trọng số cao. |

## Các ví dụ phân loại sai

| STT | Text | Nhãn thật | Dự đoán | Nguyên nhân |
|---:|---|---|---|---|
| 1 | More suspenseful, more subtle, much, much more disturbing.... | negative | positive | **Short text / lexical ambiguity:** `suspenseful` và `subtle` dễ bị xem là positive, còn tín hiệu negative rất ít. |
| 2 | A real hoot, unintentionally ... Nothing about this movie rings true. And it's boring to boot. | negative | positive | **Sarcasm và tín hiệu trái chiều:** từ `hoot`, `sweet`, `lovable` lấn át ý mỉa mai và từ negative ở cuối. |
| 3 | It's not Citizen Kane, but it does deliver ... badly acted and directed, poorly scripted. Who cares? | positive | negative | **Negation / mixed sentiment:** `not`, `badly`, `poorly` khiến bag-of-words nghiêng về negative dù đánh giá tổng thể là positive. |

Các lỗi cho thấy TF-IDF tuyến tính nhạy với từ mang cảm xúc, nhưng không nắm chắc mỉa mai, quan hệ đối lập và ý nghĩa của toàn câu.

