# NLP

Repository lưu trữ bài thực hành và tài liệu môn **Xử lý ngôn ngữ tự nhiên**.

## Cấu trúc dự án

| Thư mục / tệp | Nội dung |
| --- | --- |
| `lab1/` | Lab 01: tiền xử lý văn bản, biểu diễn thưa, TF-IDF, tìm kiếm và đánh giá. |
| `lab2/` | Tài liệu Lab 02 (`W2.pdf`). |
| `data/` | Corpus C4 mẫu 30K dùng cho các thực nghiệm. |
| `quiz/` | Tài liệu câu hỏi ôn tập. |

Chi tiết các bài từ Exercise 1 đến Exercise 11 của Lab 01 có trong [README của Lab 01](lab1/README.md).

## Yêu cầu

- Python 3.10 trở lên
- Jupyter Notebook (để mở các tệp `.ipynb`)
- Các thư viện Python: `numpy`, `scipy`, `scikit-learn`

## Cài đặt môi trường

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn jupyter
```

## Chạy Lab 01

Từ thư mục gốc của repository:

```powershell
python lab1/Ex9/implementation.py
python lab1/Ex8/run_experiments.py
python lab1/Ex10/run_search_evaluation.py
```

Để làm việc với notebook:

```powershell
jupyter notebook
```

Sau đó mở notebook tương ứng trong thư mục `lab1/`.

## Dữ liệu

Các chương trình Lab 01 sử dụng corpus tại:

```text
data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json
```

> Lưu ý: tệp dữ liệu có dung lượng lớn. Nếu repository tiếp tục bổ sung dữ liệu lớn, nên quản lý bằng Git LFS.

## Lưu ý đóng góp

Không commit môi trường ảo, cấu hình IDE hoặc tệp cache. Các mục này đã được loại trừ trong `.gitignore`.
