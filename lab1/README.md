# Lab 01 — From Text Processing to Search

## Cấu trúc bài làm

| Nội dung đề bài | Bài làm |
| --- | --- |
| Exercise 1–5: tính tay | `Ex1-Ex5/ex1_ex5.md` |
| Exercise 6: prediction trên corpus nhỏ | `Ex6/Ex6.ipynb`, `Ex6/ex6_prediction.md` |
| Prediction trước experiment corpus 30K | `Ex6/prediction.md` |
| Experiment 1: vocabulary ban đầu | `Ex7/Ex7.ipynb` |
| Sparse representation và preprocessing ablation | `Ex8/Ex8.ipynb`, `Ex8/experiment_results.md` |
| Core TF-IDF implementation | `Ex9/implementation.py`, `Ex9/verification.md` |
| Search, evaluation và error analysis | `Ex10/Ex10.ipynb`, `Ex10/results.csv`, `Ex10/search_evaluation.md` |
| Reflection và learning check | `Ex11/reflection.md`, `Ex11/learning_check.md` |

## Chạy lại bài

Từ thư mục `lab1`, dùng Python 3 với `numpy`, `scipy`, và `scikit-learn` đã cài đặt:

```powershell
python Ex9/implementation.py
python Ex8/run_experiments.py
python Ex10/run_search_evaluation.py
```

Corpus được đọc tại `../data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json`; không được sao chép vào repository.

## AI contribution

AI hỗ trợ tạo khung implementation TF-IDF, gợi ý cấu trúc pipeline và code ghi kết quả. Phần tính toán, prediction, số liệu thực nghiệm, nhãn relevance, và diễn giải cần được người nộp tự kiểm tra trước khi nộp theo chính sách của đề.
