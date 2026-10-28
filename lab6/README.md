# LAB 06 — Attention Mechanism

Thư mục này hoàn thành các yêu cầu trong `W6.pdf`: tự triển khai attention bằng NumPy và xây dựng classifier sentiment bằng PyTorch trên dữ liệu IMDb. Toàn bộ số liệu và hình được tạo từ mã nguồn trong thư mục. Notebook có thể chạy lại để tái tạo kết quả.

## Cách chạy

Từ thư mục gốc của dự án, với Python 3.13:

```powershell
py -3.13 -m pip install -r lab6/requirements.txt
py -3.13 lab6/attention_numpy.py
py -3.13 lab6/run_lab6.py
py -3.13 lab6/build_notebook.py
```

Mở `lab6/self_attention.ipynb` để xem và chạy lại phần tính toán, thí nghiệm NumPy, heatmap self-attention, bảng chi phí theo độ dài chuỗi, classifier và các hình trực quan cho ba câu thử. `run_lab6.py` dùng lại cách tách token và đúng tập con IMDb có thể tái lập của Lab 05 để đối chiếu với LSTM. Thời gian huấn luyện phụ thuộc vào máy chạy.

## Tệp bàn giao

| Tệp | Nội dung |
|---|---|
| `calculations.md` | Tính score, softmax, context, scaling, temperature, self-attention và chi phí theo độ dài |
| `prediction.md` | Năm dự đoán, lý do và mức tin cậy |
| `attention_numpy.py` | Stable softmax, scaled dot-product attention và self-attention bằng NumPy |
| `self_attention.ipynb` | Notebook có thể chạy lại, đã lưu kết quả |
| `results.csv` | So sánh LSTM và attention trên tập kiểm thử IMDb |
| `attention_heatmap.png` | Ma trận attention minh họa cho câu có từ `cat` |
| `attention_heatmap_book.png` | Ma trận attention minh họa cho câu có từ `book` |
| `query_attention.png` | Phân phối khi thay query sentiment và topic |
| `confusion_matrix.png` | Confusion matrix của classifier attention |
| `application_attention_1.png` đến `_3.png` | Biểu đồ attention của ba câu thử |
| `error_analysis.md` | Ba dự đoán đúng và ba dự đoán sai trên tập kiểm thử |
| `reflection.md` | Nhận xét kết quả, đối chiếu và giới hạn |
| `experiment_details.json` | Cấu hình và số liệu chi tiết |
| `attention_model.pt` | Trọng số mô hình và từ vựng đã huấn luyện |
| `requirements.txt` | Các thư viện Python cần thiết |

Các heatmap minh họa dùng đặc trưng 3 chiều tự thiết kế; chúng không chứng minh mô hình đã học ngữ pháp. Classifier có Q/K/V được học nhưng chưa có positional encoding nên khả năng biểu diễn thứ tự còn hạn chế. Attention weights chỉ mô tả phân phối trọng số, không phải lời giải thích nhân quả cho dự đoán. Hàng LSTM trong `results.csv` lấy từ kết quả Lab 05; hàng attention là kết quả đo của Lab 06.
