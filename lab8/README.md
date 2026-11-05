# LAB 08 — Pretrained Transformers and BERT Fine-Tuning

## Chạy bài

Từ thư mục gốc:

```powershell
python -m pip install -r lab8/requirements.txt
python lab8/pretrained_classification.py --experiment all --train-limit 2000 --validation-limit 500 --test-limit 500 --epochs 1
```

Lần đầu cần Internet để tải model/tokenizer `prajjwal1/bert-tiny`. Có thể chọn `comparison`, `data` hoặc `length`. IMDb được chia stratified 80/10/10 theo seed 42; mọi cấu hình sử dụng chung validation/test. Kết quả gồm accuracy, precision, recall, F1, thời gian, confusion matrix, data-efficiency, confidence, checkpoints và ví dụ lỗi. Cấu hình thực tế được lưu trong `run_details.json`.

## Deliverables

- `pretrained_classification.py`: tokenizer, scratch, frozen, fine-tuning, data subsets, sequence lengths.
- `tokenization.ipynb`, `fine_tuning.ipynb`: tokenization và xem kết quả/inference.
- `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`.
- `results.csv`, `run_details.json`, plots và prediction/error CSV được tạo khi chạy.

## Giới hạn

Hiện repository chỉ có IMDb. Chưa có domain B độc lập, vì vậy không thể kết luận thực nghiệm về out-of-domain/domain shift; cần thêm tập news/product có nhãn tương thích. Chạy mặc định là cấu hình gọn, không phải benchmark quy mô lớn.

## Khai báo AI

AI-assisted code: pipeline thí nghiệm, huấn luyện, metrics và visualization.
Generated: `pretrained_classification.py`, notebook và tài liệu bài tập.
Modified: sample limits/configuration và báo cáo theo kết quả đo.
Verified: script, dependencies, và output được kiểm tra sau khi chạy.


## Chạy thử đã thực hiện

Máy chạy bằng CPU, model `prajjwal1/bert-tiny` (4,386,178 tham số), seed 42, batch 16, 1 epoch; split 500/100/100. Đây là kiểm tra pipeline trên test sample 100 review, nên số liệu có phương sai lấy mẫu lớn.

| Run | Train reviews | Max length | Accuracy | F1 | Train seconds |
|---|---:|---:|---:|---:|---:|
| scratch | 500 | 128 | 0.500 | 0.667 | 3.86 |
| frozen | 500 | 128 | 0.500 | 0.000 | 1.95 |
| fine_tuned | 500 | 128 | 0.530 | 0.591 | 3.49 |
| data_0.1 | 50 | 128 | 0.490 | 0.000 | 1.11 |
| data_0.25 | 125 | 128 | 0.500 | 0.667 | 2.16 |
| data_0.5 | 250 | 128 | 0.500 | 0.667 | 4.07 |
| data_1.0 | 500 | 128 | 0.510 | 0.495 | 8.05 |
| length_64 | 500 | 64 | 0.550 | 0.622 | 4.63 |
| length_128 | 500 | 128 | 0.600 | 0.355 | 8.59 |
| length_256 | 500 | 256 | 0.500 | 0.167 | 22.78 |

BERT-tiny fine-tuned đạt accuracy 0.530, F1 0.591; frozen đạt 0.500/0.000; scratch đạt 0.500/0.667. Các kết quả không tạo thành xu hướng ổn định ở sample 100 và một epoch; không nên diễn giải superiority từ run gọn này. Curves biểu diễn observations thực tế; tăng cỡ train/validation/test và epochs rồi chạy lại trước báo cáo chính thức. Domain-shift run chưa có vì repository không có domain B.

Hardware: Windows CPU (Intel64 Family 6 Model 186), Python 3.13.5, PyTorch 2.11.0+cpu. Full run configurations are recorded in un_details.json.

Trained checkpoints are generated under checkpoints/ when the script runs and are excluded from Git to avoid committing repeated model weights.
