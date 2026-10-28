"""Tạo notebook LAB 06 có thể chạy lại từ các ô code."""

from __future__ import annotations

from pathlib import Path

import nbformat


def build_notebook() -> Path:
    """Tạo notebook gồm bài tính, thí nghiệm và ứng dụng."""
    destination = Path(__file__).resolve().parent / "self_attention.ipynb"
    cells = [
        nbformat.v4.new_markdown_cell("# LAB 06 — Attention Mechanism\n\nNotebook chạy được từ thư mục gốc hoặc `lab6`. Các heatmap minh họa dùng đặc trưng tự thiết kế và được khai báo rõ. Classifier sentiment được huấn luyện theo cách chia tập IMDb của Lab 05. Xem `calculations.md` để theo dõi bài tính tay và `prediction.md` để xem các dự đoán cho thí nghiệm."),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport sys\nROOT = Path.cwd()\nLAB = ROOT / 'lab6' if (ROOT / 'lab6').exists() else ROOT\nsys.path.insert(0, str(LAB))\nfrom attention_numpy import stable_softmax, scaled_dot_product_attention, self_attention, verify_examples\nfrom run_lab6 import Config, run_numpy_experiments, main\nimport numpy as np\nimport pandas as pd\nfrom IPython.display import display, Image\nassert verify_examples()"),
        nbformat.v4.new_markdown_cell("## Q, K, V và stable softmax\n\nQuery biểu diễn thông tin cần tìm, key quyết định mức khớp, còn value cung cấp nội dung để tổng hợp. Mỗi hàng của ma trận score được chuẩn hóa theo các key. Trừ giá trị lớn nhất của hàng giúp tránh tràn số."),
        nbformat.v4.new_code_cell("Q = np.array([[1., 0.]])\nK = np.array([[1., 0.], [0., 1.], [1., 1.]])\nV = np.array([[1., 0.], [0., 2.], [2., 1.]])\nscores, weights, context = scaled_dot_product_attention(Q, K, V)\nprint('Score sau scaling:', scores)\nprint('Weights:', weights, 'tổng theo hàng:', weights.sum(axis=1))\nprint('Context:', context)\nprint('Score rất lớn:', stable_softmax(np.array([1000., 1001., 1002.])))"),
        nbformat.v4.new_markdown_cell("## Scaling và temperature\n\nVí dụ dùng score gốc `[4, 2, 0]` và `d_k=4`. Temperature thấp làm softmax tập trung hơn; temperature cao làm entropy tăng."),
        nbformat.v4.new_code_cell("demo = run_numpy_experiments(Config())\nprint('Chưa scaling:', demo['scaling']['unscaled'])\nprint('Đã scaling:', demo['scaling']['scaled'])\ntemperature_table = pd.DataFrame(demo['temperature']).T[['max_weight', 'entropy_nats']]\ndisplay(temperature_table.rename(columns={'max_weight': 'Weight lớn nhất', 'entropy_nats': 'Entropy (nat)'}))"),
        nbformat.v4.new_markdown_cell("## Phân phối thay đổi theo query\n\nCâu ví dụ dùng đặc trưng được gán rõ, không phải word embedding đã học. Query sentiment dự kiến nhấn mạnh `good`; query topic dự kiến nhấn mạnh `movie` và `beginning`. Thí nghiệm cho thấy sự phụ thuộc vào query nhưng không chứng minh mô hình hiểu ngôn ngữ."),
        nbformat.v4.new_code_cell("tokens = list(Config().demo_tokens)\ndisplay(pd.DataFrame({'token': tokens, **{f'{name}_score': result['scores'] for name, result in demo['query_experiment'].items()}, **{f'{name}_attention': result['weights'] for name, result in demo['query_experiment'].items()}}))\ndisplay(Image(filename=str(LAB / 'query_attention.png')))"),
        nbformat.v4.new_markdown_cell("## Self-attention và heatmap\n\nVới `I like cats`, các ma trận chiếu đơn vị và embedding 3×2 đã nêu cho phép kiểm tra từng phép tính. Hai heatmap còn lại dùng đặc trưng 3 chiều tự thiết kế. Hàng là query, cột là key. Hãy xem các hàng `cat`, `adopted`, `sleeping` mà không coi phân phối weight là bằng chứng hiểu ngữ pháp."),
        nbformat.v4.new_code_cell("X = np.array([[1., 0.], [0., 1.], [1., 1.]])\nQ, K, V, scores, weights, context = self_attention(X, np.eye(2), np.eye(2), np.eye(2))\nprint('Q=K=V:\\n', Q)\nprint('QK^T:\\n', Q @ K.T)\nprint('Weights:\\n', np.round(weights, 4))\nprint('Context:\\n', np.round(context, 4))\ndisplay(Image(filename=str(LAB / 'attention_heatmap.png')))\ncat_demo = demo['semantic'][Config().semantic_sentences[0]]\nfor token in ('cat', 'adopted', 'sleeping'):\n    position = cat_demo['tokens'].index(token)\n    print(token, sorted(zip(cat_demo['tokens'], cat_demo['weights'][position]), key=lambda item: item[1], reverse=True)[:3])\ndisplay(Image(filename=str(LAB / 'attention_heatmap_book.png')))"),
        nbformat.v4.new_markdown_cell("## Chi phí theo độ dài chuỗi\n\nMa trận full self-attention `T×T` có `T²` phần tử. Bảng chưa tính batch, tensor Q/K/V, gradient và trạng thái optimizer."),
        nbformat.v4.new_code_cell("display(pd.DataFrame([{'Độ dài': int(length), 'Số phần tử ma trận': count, 'Bộ nhớ float32 (byte)': 4 * count} for length, count in demo['sequence_cost'].items()]))"),
        nbformat.v4.new_markdown_cell("## Phân loại sentiment trên IMDb\n\nChạy ô dưới để huấn luyện lại mô hình embedding → self-attention → masked mean context → classifier. Ô tạo lại `results.csv`, confusion matrix và ba biểu đồ attention của phần ứng dụng. Mô hình chưa có positional encoding; xem giới hạn trong `reflection.md`."),
        nbformat.v4.new_code_cell("main()\ndisplay(pd.read_csv(LAB / 'results.csv'))\ndisplay(Image(filename=str(LAB / 'confusion_matrix.png')))\nfor index in range(1, 4):\n    display(Image(filename=str(LAB / f'application_attention_{index}.png')))"),
        nbformat.v4.new_markdown_cell("## Diễn giải kết quả\n\nĐối chiếu phân phối attention với nhãn thật và dự đoán. Một token nhận attention không chứng minh nó gây ra dự đoán. `error_analysis.md` ghi ba trường hợp đúng, ba trường hợp sai trên tập test và trình bày nguyên nhân có thể xảy ra dưới dạng giả thuyết. Hàng LSTM lấy từ Lab 05, dùng cùng các tập train/validation/test, độ dài chuỗi, kích thước embedding và số epoch; classifier dựa trên hidden state hồi quy có kiến trúc và chi phí tính toán khác."),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}})
    nbformat.write(notebook, destination)
    return destination


if __name__ == "__main__":
    print(build_notebook())
