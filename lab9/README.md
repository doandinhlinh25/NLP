# LAB 09 — Named Entity Recognition and Information Extraction

## Chạy bài

Từ thư mục gốc, cài `python -m pip install -r lab9/requirements.txt`, rồi chạy:

```powershell
python lab9/ner_pipeline.py
```

Lần đầu cần Internet để tải CoNLL-2003 và `distilbert/distilbert-base-cased`. Bài dùng đầy đủ các split chuẩn, seed 42, tối ưu đầu tiên bằng baseline logistic regression với feature từ/cửa sổ ngữ cảnh; Transformer được fine-tune cho token classification. Có thể dùng `--train-limit 2000 --evaluation-limit 500 --epochs 1` để kiểm tra nhanh, nhưng cần ghi rõ đây là run rút gọn. Cả hai model dùng cùng test set.

## Thiết kế và deliverables

- `ner_pipeline.py`: tải dữ liệu, baseline, tokenization, label alignment, fine-tuning, token/entity metrics, phân tích theo loại/độ dài, confusion matrix và ví dụ lỗi.
- `information_extraction.py`: nhóm kết quả NER thành PERSON, ORGANIZATION, LOCATION, MISCELLANEOUS.
- `tokenizer_alignment.ipynb`, `ner_finetuning.ipynb`: notebook hướng dẫn kiểm tra alignment và chạy/đọc thí nghiệm.
- `results.csv`, `entity_metrics.csv`, `entity_length_analysis.csv`, `entity_length_analysis.png`, `confusion_matrix.png`, `tokenization_analysis.csv`, `tokenization_summary.csv`, `error_analysis_examples.csv`, `predictions.json`, `ie_annotation_template.csv`, `run_details.json`: được tạo sau khi chạy pipeline.
- `calculations.md`, `prediction.md`, `error_analysis.md`, `reflection.md`: phần viết cần hoàn thiện bằng quan sát và lập luận của sinh viên.

## Label alignment

Dùng Strategy A trong đề: chỉ subword đầu tiên của mỗi word nhận BIO label; special tokens và các subword tiếp theo nhận `-100` (PyTorch ignore index), nên không tham gia loss. Khi đánh giá, dự đoán ở subword đầu tiên được ánh xạ lại về word để so sánh công bằng với baseline.

## Giới hạn

CoNLL-2003 dùng annotation tiếng Anh theo newswire và có các nhãn PER, ORG, LOC, MISC. Kết quả không tự động đại diện cho tiếng Việt hoặc các miền khác. Báo cáo cần ghi lại cấu hình thực tế, đặc biệt nếu dùng sample giới hạn.

## Kết quả lần chạy hiện tại

Chạy trên CPU với 2,000 train / 500 validation / 500 test, 1 epoch, DistilBERT cased, batch 8, max length 256, seed 42. Đây là run rút gọn; entity-level micro F1: baseline **0.6275**, Transformer **0.6797**. Token-level F1 tương ứng **0.7048** và **0.7631**. Thời gian train Transformer khoảng 222.5 giây (không tính bước baseline). Chi tiết và số theo từng loại entity nằm trong `results.csv`, `entity_metrics.csv` và `run_details.json`. Hãy dùng số đo này làm dữ liệu báo cáo của lần chạy, nhưng tự viết phần diễn giải và error analysis.

## Khai báo AI

- AI-assisted code: pipeline huấn luyện/đánh giá, baseline, alignment, visualization và IE boilerplate.
- Generated: mã thực nghiệm và khung tài liệu/notebook.
- Modified: cấu hình, cách chạy và deliverable theo W9.
- Verified: kiểm tra thủ công nội dung, logic pipeline và khớp tên deliverables; cập nhật kết quả thực nghiệm sau khi tự chạy.
