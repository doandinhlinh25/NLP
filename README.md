# NLP — Bài thực hành Xử lý ngôn ngữ tự nhiên

Kho lưu trữ bài tập, mã nguồn, dữ liệu và kết quả thực nghiệm của môn **Xử lý ngôn ngữ tự nhiên và ứng dụng** (Học kỳ I, 2026).

## Bài nộp và vị trí chấm điểm

Các bài nộp của sinh viên nằm trực tiếp trong thư mục `lab1/` đến `lab9/`. Mỗi thư mục lab chứa đề bài/tài liệu, mã nguồn hoặc notebook, kết quả chạy và phần báo cáo tương ứng. README riêng trong từng lab là hướng dẫn chính về cách cài dependencies, chạy pipeline và tên deliverables.

| Thư mục | Nội dung |
| --- | --- |
| `lab1/` | Tiền xử lý văn bản, TF-IDF, tìm kiếm và đánh giá. |
| `lab2/` | Language models và N-gram. |
| `lab3/` | Word-context representation và word embeddings. |
| `lab4/` | Bài thực hành theo đề W4. |
| `lab5/` | RNN, LSTM, GRU và phân loại chuỗi. |
| `lab6/` | Bài thực hành theo đề W6. |
| `lab7/` | Bài thực hành theo đề W7. |
| `lab8/` | Sequence classification với pretrained Transformer. |
| `lab9/` | Named Entity Recognition và Information Extraction. |
| `data/` | Dữ liệu dùng trong các thực nghiệm. |
| `quiz/` | Tài liệu ôn tập và câu hỏi. |
| `ChamDiem/` | Hồ sơ chấm/rubric theo lab; xem hướng dẫn bên dưới. |

### Package chấm điểm `ChamDiem/`

`ChamDiem/` lưu các file ghi nhận việc chấm và nhận xét theo từng bài, không phải package Python để cài bằng `pip`. Cấu trúc hiện tại:

```text
ChamDiem/
├── lab01/<thư-mục-con>/cham_diem_lab01.md
├── lab02/<thư-mục-con>/cham_diem_lab02.md
├── lab03/<thư-mục-con>/cham_diem_lab03.md
├── lab04/<thư-mục-con>/cham_diem_lab04.md
└── lab05/<thư-mục-con>/cham_diem_lab05.md
```

Mỗi `cham_diem_labXX.md` ghi điểm, rubric/hạng mục, căn cứ đối chiếu và nhận xét. Một số thư mục còn có README hoặc requirements riêng cho bài đó. Hiện package có hồ sơ cho LAB 01–05; chưa có hồ sơ chấm LAB 06–09.

Khi chấm một bài, mở file trong `ChamDiem/labXX/` tương ứng để xem nhận xét, rồi đối chiếu các deliverables trong `labX/` với đề và README của lab. Các file kết quả như CSV, hình, notebook và báo cáo nằm cùng thư mục bài nộp, không nằm trong `ChamDiem/`.

## Cài đặt và chạy

Các lab có dependencies khác nhau, vì vậy hãy cài requirements và làm theo lệnh trong README của lab đang thực hiện. Ví dụ với LAB 09:

```powershell
python -m pip install -r lab9/requirements.txt
python lab9/ner_pipeline.py
```

Một số thực nghiệm cần tải dữ liệu hoặc pretrained model qua Internet ở lần chạy đầu. Hãy xem README của lab để biết cấu hình mặc định, cách chạy nhanh và giới hạn của kết quả.

## Dữ liệu và môi trường

- Python 3.10 trở lên là mốc tham khảo; phiên bản và dependencies cụ thể có thể khác giữa các lab.
- Dữ liệu thực nghiệm đặt trong `data/` hoặc được tải theo hướng dẫn của lab.
- Không commit virtual environment, cache, cấu hình IDE hoặc checkpoint model sinh ra khi train; các mục này được loại trừ trong `.gitignore`.

## Liên kết

- [Bài thực hành LAB 01](lab1/README.md)
- [Bài thực hành LAB 09](lab9/README.md)

