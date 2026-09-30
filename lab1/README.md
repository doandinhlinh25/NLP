# Lab 01 — From Text Processing to Search

Lab 01 đi từ biểu diễn văn bản thưa đến TF-IDF document search: tính tay, dự đoán, thí nghiệm representation, tự cài đặt TF-IDF, tìm kiếm, đánh giá và reflection.

## Cấu trúc bài làm

| Phần trong đề | Tệp/thư mục thực hiện |
| --- | --- |
| Exercise 1–5: tính tay | Chapter5/Ex1-Ex6-pdf/NLP_Lab01_Chapter_5_Ex1_Ex5_23001898_DoanDinhLinh.pdf |
| Exercise 6: prediction trên corpus nhỏ và kiểm chứng | Chapter5/Ex1-Ex6-pdf/ex6_prediction.md, Chapter5/Ex6-Code/Ex6.ipynb |
| Prediction trước experiment corpus 30K | Chapter6/prediction.md |
| Part D — Experiment 1: sparse representation | Chapter7/experiment.ipynb |
| Part F — Experiment 2: preprocessing ablation | Chapter8/Ex8.ipynb, Chapter8/run_experiments.py, Chapter8/experiment_results.md |
| Part E — Core TF-IDF implementation | Chapter9/implementation.py, Chapter9/verification.md |
| Part G–I — Search, evaluation, error analysis | Chapter10/Ex10.ipynb, Chapter10/run_search_evaluation.py, Chapter10/results.csv, Chapter10/search_evaluation.md |
| Reflection và learning check | Chapter11/reflection.md, Chapter11/learning_check.md |

## Môi trường

- Python 3.10 trở lên
- Jupyter Notebook
- numpy, scipy và scikit-learn

Cài đặt dependency từ thư mục gốc repository:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy scipy scikit-learn jupyter
~~~

## Thứ tự thực hiện

1. Hoàn thành phần tính tay và prediction trước khi xem kết quả code.
2. Chạy Experiment 1 ở Chapter7 để kiểm tra vocabulary, matrix shape, sparsity và vocabulary inspection.
3. Chạy preprocessing ablation ở Chapter8; lệnh này tạo lại candidate_results.csv và experiment_results.md.
4. Chạy Chapter9 để kiểm tra unit test của TF-IDF implementation.
5. Chạy Chapter10 để tạo lại results.csv và search_evaluation.md.
6. Tự hoàn thiện reflection và learning check ở Chapter11.

## Chạy lại phần code

Từ thư mục lab1:

~~~powershell
python Chapter9/implementation.py
python Chapter8/run_experiments.py
python Chapter10/run_search_evaluation.py
~~~

Mở notebook:

~~~powershell
jupyter notebook Chapter5/Ex6-Code/Ex6.ipynb
jupyter notebook Chapter7/experiment.ipynb
jupyter notebook Chapter8/Ex8.ipynb
jupyter notebook Chapter10/Ex10.ipynb
~~~

## Dữ liệu

Các thí nghiệm dùng corpus C4 tại:

~~~text
../data/c4-train.00000-of-01024-30K.json/c4-train.00000-of-01024-30K.json
~~~

Không sao chép corpus vào thư mục lab1 hoặc commit lại dữ liệu.

## AI contribution

AI được dùng để hỗ trợ cấu trúc code, kiểm tra lỗi, định dạng tài liệu và tổ chức thí nghiệm. Phần tính tay, prediction trước experiment, nhãn relevance, diễn giải kết quả, error analysis và reflection cần được người nộp tự kiểm tra, chỉnh sửa và chịu trách nhiệm trước khi nộp.
