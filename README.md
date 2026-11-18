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
| `ChamDiem/` | Hồ sơ chấm mẫu và nhận xét theo lab; xem hướng dẫn bên dưới. |

### Package chấm điểm `ChamDiem/`

`ChamDiem/` lưu các ví dụ phiếu chấm và nhận xét theo từng bài, không phải package Python để cài bằng `pip`. Tiêu chí chính thức của mỗi lab nằm ngay trong thư mục lab tương ứng, thường là file `Wn.pdf` (ví dụ `lab9/W9.pdf`); áp dụng cùng cách tìm này cho LAB 01–09. Cấu trúc hồ sơ chấm mẫu hiện có:

```text
ChamDiem/
├── lab01/<thư-mục-con>/cham_diem_lab01.md
├── lab02/<thư-mục-con>/cham_diem_lab02.md
├── lab03/<thư-mục-con>/cham_diem_lab03.md
├── lab04/<thư-mục-con>/cham_diem_lab04.md
└── lab05/<thư-mục-con>/cham_diem_lab05.md
```

Mỗi `cham_diem_labXX.md` là ví dụ cách ghi điểm, căn cứ đối chiếu và nhận xét, không thay thế tiêu chí trong đề. Các file bài nộp và kết quả để chấm nằm trong thư mục lab tương ứng. Hiện `ChamDiem/` có ví dụ cho LAB 01–05; với các lab sau, hãy dùng tiêu chí trong file đề của chính lab đó.

Khi chấm, mở file `Wn.pdf` trong thư mục `labN/` để xem rubric/yêu cầu, sau đó đối chiếu mã nguồn, notebook, kết quả và báo cáo trong cùng thư mục. Có thể tham khảo `ChamDiem/` để xem cách trình bày một phiếu nhận xét mẫu.

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

