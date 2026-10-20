# LAB01 — Phân loại Email Rác bằng Machine Learning

Dự án xây dựng hệ thống phân loại email thành **Ham (email bình thường, nhãn 0)** và **Spam (email rác, nhãn 1)** từ nội dung văn bản. Ba thuật toán được cài đặt thủ công trong `src/model.py`: Logistic Regression, Naive Bayes và Support Vector Machine (SVM).

## Thành viên nhóm

| STT | Họ và tên | MSSV | Phụ trách |
|---:|---|---|---|
| 1 | [Trần Thị Hạnh Nguyên] | [075306005389] | [Khởi tạo không gian làm việc] |
| 2 | [Phạm Minh Thái] | [080206015913] | [Đánh giá & Ráp nối Pipeline] |
| 3 | [Lương Thị Thu Minh] | [066306009846] | [Khám phá và Tiền xử lý (Chống rò rỉ dữ liệu)] |
| 4 | [Đặng Phước Thịnh] | [083206013709] | [Model Training] |
| 5 | [Phạm Văn Tiền] | [077206003281] | [Viết báo cáo & Tổng kết (Nghiệm thu)] |
| 6 | [Trần Mạnh Tấn] | [092206014273] | [Viết báo cáo & Tổng kết (Nghiệm thu)] |
| 7 | [Vũ Nguyễn Quốc Thái] | [075206023881] | [Model Training] |



## Cấu trúc dự án

```text
lab01_ML/
├── agents/                 # Quy tắc, mô tả và kế hoạch thực hiện
├── data/
│   ├── raw/                # Dữ liệu gốc: SpamAssassin.csv
│   └── processed/          # Các tập train/test được tạo khi chạy
├── docs/
│   └── Lab01_descript.md   # Báo cáo mô tả hệ thống và kết quả
├── models/                 # Vectorizer và mô hình đã huấn luyện
├── notebooks/
│   └── lab01.ipynb         # Notebook báo cáo thực nghiệm
├── reports/                # Metrics và ma trận nhầm lẫn
└── src/
    ├── eda.py              # Khám phá dữ liệu
    ├── preprocess.py       # Làm sạch, TF-IDF, chia train/test
    ├── model.py            # Cài đặt các thuật toán
    ├── train.py            # Huấn luyện và lưu mô hình
    ├── evaluate.py         # Tính metrics và tạo confusion matrix
    ├── predict.py          # Dự đoán email nhập từ terminal
    └── pipeline.py         # Điều phối quy trình
```

## Yêu cầu

- Python 3.10 trở lên (khuyến nghị dùng một phiên bản Python tương thích với các thư viện trong `requirements.txt`).
- Các thư viện được liệt kê trong `requirements.txt`.
- Dữ liệu `data/raw/SpamAssassin.csv` đã có trong repository.

## Cài đặt môi trường

Mở terminal tại thư mục gốc của dự án.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể dùng Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux / Git Bash

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Chạy dự án

### 1. Khám phá dữ liệu (không bắt buộc)

```bash
python src/eda.py
```

Lệnh này in kích thước dữ liệu, kiểm tra giá trị thiếu/trùng lặp, thống kê nhãn và lưu biểu đồ phân bố nhãn vào `reports/`.

### 2. Chạy toàn bộ pipeline

```bash
python src/pipeline.py
```

Pipeline thực hiện tuần tự:

1. Tiền xử lý dữ liệu, chia Train/Test và tạo đặc trưng TF-IDF.
2. Huấn luyện Logistic Regression, Naive Bayes và SVM.
3. Đánh giá trên tập Train/Test, lưu bảng metrics và các ma trận nhầm lẫn.

Có thể bỏ qua bước tiền xử lý nếu các tệp dữ liệu đã xử lý đang tồn tại và tương thích:

```bash
python src/pipeline.py --skip-preprocess
```

Có thể bỏ qua bước huấn luyện nếu các tệp mô hình đã tồn tại:

```bash
python src/pipeline.py --skip-preprocess --skip-train
```

**Lưu ý:** Hai tùy chọn bỏ qua chỉ dùng khi các tệp đầu vào tương ứng đã tồn tại; nếu thiếu, pipeline sẽ báo lỗi.

### 3. Dự đoán email mới

Sau khi đã chạy pipeline để tạo `models/tfidf_vectorizer.pkl` và ba tệp mô hình:

```bash
python src/predict.py
```

Nhập nội dung email khi được yêu cầu. Nhập `exit` để thoát.

### 4. Chạy notebook

Mở `notebooks/lab01.ipynb` bằng Jupyter Notebook hoặc VS Code. Nên chạy lần lượt các cell từ trên xuống dưới. Notebook gọi các module của dự án và tạo/đọc các tệp kết quả ở những thư mục tương ứng.

## Kết quả tham khảo

Các số liệu dưới đây được đọc từ `reports/metrics_summary.csv` có trong bản dự án. Chúng là kết quả đã lưu trong gói mã nguồn; nếu chạy lại pipeline, kết quả có thể thay đổi khi mã, dữ liệu hoặc môi trường thay đổi.

| Mô hình | Accuracy (Test) | Precision | Recall | F1 | TPR | FPR |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 92.91% | 84.07% | 94.75% | 89.09% | 94.75% | 7.91% |
| Naive Bayes | 94.09% | 95.63% | 84.53% | 89.74% | 84.53% | 1.70% |
| SVM | 72.04% | 75.41% | 12.71% | 21.75% | 12.71% | 1.82% |

Theo bảng metrics hiện có, **Naive Bayes có F1 cao nhất trên tập Test (89.74%)** và Accuracy 94.09%. Logistic Regression có Recall cao nhất (94.75%), phù hợp hơn khi ưu tiên phát hiện nhiều email Spam. SVM trong kết quả hiện tại có Recall thấp, nên cần xem xét lại cấu hình/triển khai trước khi sử dụng.

Xem phân tích, các chỉ số đầy đủ và ma trận nhầm lẫn tại [`docs/Lab01_descript.md`](docs/Lab01_descript.md).

## Lưu ý về các tệp được tạo

Thư mục `data/processed/` và `models/` trong gói có thể chỉ chứa tệp giữ chỗ. Chạy `python src/pipeline.py` để tạo các tập dữ liệu đã xử lý, vectorizer và mô hình trước khi chạy đánh giá hoặc dự đoán. Không đưa dữ liệu cá nhân hoặc email nhạy cảm vào kiểm thử nếu chưa được phép.

## Công nghệ sử dụng

- Python, NumPy, pandas
- scikit-learn (`TfidfVectorizer`, `train_test_split`)
- Matplotlib, Seaborn
- joblib / pickle để lưu các đối tượng đã xử lý và mô hình
- Jupyter Notebook
