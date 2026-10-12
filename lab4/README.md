# LAB 04 - Text Classification

Khung nộp bài này bám sát `W4.pdf` và cấu trúc tham khảo của `NLP_course/lab04`. Khung chỉ
chuẩn bị pipeline và biểu mẫu báo cáo; các nhận xét, kết quả và dataset phải là của nhóm.

## Bắt đầu

1. Cài môi trường bằng `py -m pip install -r requirements.txt`.
2. Đặt CSV tại `data/data.csv`.
3. Mở `classification.ipynb` trong Jupyter hoặc VS Code.
4. Chỉnh `DATA_PATH`, `TEXT_COLUMN`, `LABEL_COLUMN` và, nếu cần, `POSITIVE_LABEL` ở cell cấu hình.
5. Chạy notebook tuần tự, sau đó điền các tệp Markdown bằng kết quả thật.

CSV phải có một cột văn bản và một cột nhãn. Ví dụ:

```csv
text,label
This film is excellent,positive
This film is boring,negative
```

## Cấu trúc bài nộp

```text
lab4/
├── W4.pdf
├── README.md
├── requirements.txt
├── calculations.md             # Bài tính tay
├── prediction.md               # Dự đoán trước experiment
├── classification.ipynb        # Toàn bộ pipeline và experiment
├── preprocessing.py            # Chuẩn hóa và split dữ liệu
├── results.csv                 # Notebook ghi metrics tại đây
├── class_distribution.png       # Notebook sinh ở Experiment 1
├── confusion_matrix.png        # Notebook sinh sau khi chạy
├── error_analysis.md
├── reflection.md
└── data/
    ├── README.md
    └── data.csv                # Dataset tự chọn, không commit nếu quá lớn
```

## Nội dung đã có trong notebook

- Dataset inspection: missing values, duplicate text, class distribution và độ dài văn bản.
- Train/validation/test split stratified (60/20/20), trước mọi bước `fit`.
- TF-IDF unigram + Logistic Regression baseline.
- Average Word2Vec + Logistic Regression.
- So sánh unigram với unigram + bigram; Linear SVM là model tùy chọn.
- Macro precision, recall, F1, confusion matrix, ví dụ đúng/sai và top TF-IDF features.

## Quy tắc chống data leakage

Luôn chia train/validation/test trước. `TfidfVectorizer` và Word2Vec chỉ được `fit` bằng
tập train; validation và test chỉ được `transform` hoặc dự đoán. Model được chọn theo macro F1
trên validation, rồi mới đánh giá test đúng một lần.
