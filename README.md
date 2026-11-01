# LAB01 — Phân loại Email Rác bằng Machine Learning

Dự án xây dựng hệ thống phân loại email thành **Ham (email bình thường, nhãn 0)** và **Spam (email rác, nhãn 1)** dựa trên nội dung văn bản. Ba thuật toán được triển khai trong `src/model.py` gồm Logistic Regression, Naive Bayes và Support Vector Machine (SVM).

## Thành viên nhóm

| STT | Họ và tên | MSSV | Phụ trách |
|---:|---|---|---|
| 1 | Trần Thị Hạnh Nguyên | 075306005389 | Khởi tạo không gian làm việc |
| 2 | Phạm Minh Thái | 080206015913 | Đánh giá & ráp nối Pipeline |
| 3 | Lương Thị Thu Minh | 066306009846 | Khám phá và tiền xử lý (chống rò rỉ dữ liệu) |
| 4 | Đặng Phước Thịnh | 083206013709 | Model Training |
| 5 | Phạm Văn Tiền | 077206003281 | Viết báo cáo & tổng kết (nghiệm thu) |
| 6 | Trần Mạnh Tấn | 092206014273 | Viết báo cáo & tổng kết (nghiệm thu) |
| 7 | Vũ Nguyễn Quốc Thái | 075206023881 | Model Training |

## Cấu trúc dự án

```text
lab01_ML/
├── agents/                  # Quy tắc, mô tả và kế hoạch thực hiện
├── data/
│   ├── raw/                 # Dữ liệu gốc: SpamAssassin.csv
│   └── processed/           # Các tập train/test được tạo khi chạy
├── docs/
│   ├── images/              # Hình ảnh minh họa và biểu đồ
│   └── Lab01_descript.md    # Báo cáo mô tả hệ thống và kết quả
├── models/                  # Vectorizer và mô hình đã huấn luyện
├── notebooks/
│   └── lab01.ipynb          # Notebook báo cáo thực nghiệm
├── reports/                 # Metrics và ma trận nhầm lẫn
├── src/
│   ├── eda.py               # Khám phá dữ liệu
│   ├── preprocess.py        # Làm sạch, TF-IDF, chia train/test
│   ├── model.py              # Cài đặt các thuật toán
│   ├── train.py              # Huấn luyện và lưu mô hình
│   ├── evaluate.py           # Tính metrics và tạo confusion matrix
│   ├── predict.py            # Dự đoán email nhập từ terminal
│   └── pipeline.py           # Điều phối quy trình
├── config.yaml               # Cấu hình thực nghiệm
├── requirements.txt          # Các thư viện cần thiết
└── README.md                 # Hướng dẫn dự án
```

## Yêu cầu

- Python 3.10 trở lên.
- Các thư viện được liệt kê trong `requirements.txt`.
- Dữ liệu `data/raw/SpamAssassin.csv` có sẵn trong repository.

## Cài đặt môi trường

Mở terminal tại thư mục gốc của dự án.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể sử dụng Command Prompt:

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

## Cấu hình thực nghiệm

Các tham số chính dưới đây được sử dụng trong cấu hình hiện tại của dự án. File `config.yaml` là nơi tham chiếu cấu hình cho các thành phần có hỗ trợ đọc tham số từ file.

| Thành phần | Tham số | Giá trị |
|---|---|---:|
| TF-IDF | `max_features` | 10000 |
| Logistic Regression | `n_iterations` | 1000 |
| SVM | `learning_rate` | 0.01 |
| SVM | `n_iterations` | 100 |

Lưu ý: Không phải mọi tham số trong chương trình đều nhất thiết được đọc từ `config.yaml`; một số giá trị có thể được khai báo trực tiếp trong mã nguồn. Khi thay đổi cấu hình, cần kiểm tra cả file cấu hình và phần triển khai tương ứng để bảo đảm các giá trị được áp dụng đúng.

## Chạy dự án

### 1. Khám phá dữ liệu (không bắt buộc)

```bash
python src/eda.py
```

Lệnh này thống kê kích thước dữ liệu, kiểm tra giá trị thiếu và trùng lặp, phân bố nhãn, đồng thời lưu biểu đồ phân bố nhãn và CSV thống kê vào các thư mục kết quả tương ứng.

### 2. Chạy toàn bộ pipeline

```bash
python src/pipeline.py
```

Pipeline thực hiện tuần tự:

1. Tiền xử lý dữ liệu, chia Train/Test và tạo đặc trưng TF-IDF.
2. Huấn luyện Logistic Regression, Naive Bayes và SVM.
3. Đánh giá mô hình trên tập Train và Test.
4. Lưu bảng metrics và các ma trận nhầm lẫn.

Có thể bỏ qua bước tiền xử lý nếu các tệp dữ liệu đã xử lý tồn tại và tương thích:

```bash
python src/pipeline.py --skip-preprocess
```

Có thể bỏ qua bước huấn luyện nếu các tệp mô hình đã tồn tại:

```bash
python src/pipeline.py --skip-preprocess --skip-train
```

**Lưu ý:** Các tùy chọn bỏ qua chỉ sử dụng được khi những tệp đầu vào tương ứng đã tồn tại và tương thích với cấu hình hiện tại; nếu thiếu tệp hoặc không tương thích, pipeline có thể báo lỗi hoặc cho ra kết quả không phù hợp.

### 3. Dự đoán email mới

Sau khi chạy pipeline để tạo `models/tfidf_vectorizer.pkl` và các tệp mô hình:

```bash
python src/predict.py
```

Nhập nội dung email khi được yêu cầu. Nhập `exit` để thoát.

### 4. Chạy notebook

Mở `notebooks/lab01.ipynb` bằng Jupyter Notebook hoặc VS Code. Nên chạy lần lượt các cell từ trên xuống dưới. Notebook sử dụng các module của dự án và tạo hoặc đọc kết quả từ những thư mục tương ứng.

## Kết quả thực nghiệm

Bảng dưới đây trình bày kết quả đánh giá ba mô hình trên tập Train và Test theo số liệu thực nghiệm.

| Mô hình | Tập dữ liệu | Accuracy | Precision | Recall | F1-score | TPR | FPR | TN | FP | FN | TP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Train | 95.35% | 90.08% | 95.30% | 92.62% | 95.30% | 4.62% | 3136 | 152 | 68 | 1380 |
| Logistic Regression | Test | 93.16% | 84.69% | 94.75% | 89.44% | 94.75% | 7.54% | 760 | 62 | 19 | 343 |
| Naive Bayes | Train | 99.62% | 98.97% | 99.79% | 99.38% | 99.79% | 0.46% | 3273 | 15 | 3 | 1445 |
| Naive Bayes | Test | 97.30% | 96.35% | 94.75% | 95.54% | 94.75% | 1.58% | 809 | 13 | 19 | 343 |
| SVM | Train | 71.94% | 88.89% | 9.39% | 16.99% | 9.39% | 0.52% | 3271 | 17 | 1312 | 136 |
| SVM | Test | 71.88% | 89.19% | 9.12% | 16.54% | 9.12% | 0.49% | 818 | 4 | 329 | 33 |

### Ý nghĩa các chỉ số

- **Accuracy:** Tỷ lệ dự đoán đúng trên tổng số mẫu.
- **Precision:** Tỷ lệ email được dự đoán là Spam thực sự thuộc lớp Spam.
- **Recall:** Tỷ lệ email Spam thực tế được mô hình phát hiện đúng.
- **F1-score:** Trung bình điều hòa giữa Precision và Recall.
- **TPR (True Positive Rate):** Tỷ lệ phát hiện đúng lớp Spam; trong bảng này bằng Recall.
- **FPR (False Positive Rate):** Tỷ lệ email Ham bị phân loại nhầm thành Spam.
- **TN (True Negative):** Số email Ham được phân loại đúng.
- **FP (False Positive):** Số email Ham bị phân loại nhầm thành Spam.
- **FN (False Negative):** Số email Spam bị phân loại nhầm thành Ham.
- **TP (True Positive):** Số email Spam được phân loại đúng.

### Nhận xét và so sánh các mô hình

#### 1. Logistic Regression

Trên tập Train, Logistic Regression đạt Accuracy 95.35% và F1-score 92.62%. Trên tập Test, Accuracy đạt 93.16%, Precision đạt 84.69%, Recall đạt 94.75% và F1-score đạt 89.44%.

Mô hình có khả năng phát hiện Spam khá tốt, chỉ bỏ sót 19 email Spam trên tập Test. Tuy nhiên, mô hình có 62 FP, tức là một số lượng đáng kể email bình thường bị phân loại nhầm thành Spam. FPR đạt 7.54%, cao hơn Naive Bayes.

**Đánh giá:** Logistic Regression phù hợp khi ưu tiên phát hiện nhiều email Spam, nhưng cần cải thiện Precision để giảm tình trạng email bình thường bị đưa vào thư mục Spam.

#### 2. Naive Bayes

Naive Bayes đạt kết quả tốt nhất trong ba mô hình trên tập Test, với Accuracy 97.30%, Precision 96.35%, Recall 94.75% và F1-score 95.54%.

Mô hình có 809 TN, 13 FP, 19 FN và 343 TP trên tập Test. FPR chỉ đạt 1.58%, thấp hơn đáng kể so với Logistic Regression. Điều này cho thấy mô hình vừa phát hiện tốt email Spam, vừa hạn chế phân loại nhầm email bình thường.

Trên tập Train, Naive Bayes đạt F1-score 99.38%, cao hơn mức 95.54% trên tập Test. Chênh lệch này cho thấy hiệu suất trên dữ liệu huấn luyện cao hơn dữ liệu chưa thấy, nhưng kết quả Test vẫn tốt.

**Đánh giá:** Naive Bayes là mô hình tốt nhất trong lần thực nghiệm này xét theo Accuracy, Precision và F1-score trên tập Test. Đây là lựa chọn phù hợp nếu cần cân bằng giữa phát hiện Spam và hạn chế cảnh báo nhầm.

#### 3. Support Vector Machine (SVM)

SVM đạt Accuracy 71.94% trên tập Train và 71.88% trên tập Test. Tuy nhiên, Recall chỉ đạt 9.39% trên Train và 9.12% trên Test; F1-score lần lượt là 16.99% và 16.54%.

Trên tập Test, mô hình chỉ phát hiện đúng 33 email Spam và bỏ sót 329 email Spam. Mặc dù Precision đạt 89.19% và FPR chỉ đạt 0.49%, Recall thấp cho thấy mô hình dự đoán rất ít email Spam thành công.

**Đánh giá:** SVM chưa phù hợp với mục tiêu phân loại Spam trong cấu hình thực nghiệm hiện tại. Cần kiểm tra cách triển khai, cách biểu diễn đặc trưng, tham số mô hình và ngưỡng phân loại trước khi sử dụng.

### So sánh tổng thể

| Tiêu chí trên tập Test | Logistic Regression | Naive Bayes | SVM |
|---|---:|---:|---:|
| Accuracy | 93.16% | **97.30%** | 71.88% |
| Precision | 84.69% | **96.35%** | 89.19% |
| Recall | **94.75%** | **94.75%** | 9.12% |
| F1-score | 89.44% | **95.54%** | 16.54% |
| FPR | 7.54% | 1.58% | **0.49%** |
| FP | 62 | **13** | 4 |
| FN | **19** | **19** | 329 |

Naive Bayes có Accuracy, Precision và F1-score cao nhất trên tập Test. Logistic Regression và Naive Bayes có Recall bằng nhau ở mức 94.75%, nhưng Naive Bayes có ít FP hơn đáng kể. SVM có FPR thấp nhất nhưng bỏ sót quá nhiều email Spam, vì vậy không thể chỉ dựa vào FPR để kết luận mô hình hoạt động tốt.

### Kết luận

Dựa trên kết quả thực nghiệm, **Naive Bayes là mô hình được đề xuất cho hệ thống phân loại email Spam**. Mô hình đạt Accuracy 97.30% và F1-score 95.54% trên tập Test, đồng thời có Precision 96.35% và FPR 1.58%. So với Logistic Regression, Naive Bayes giữ được Recall tương đương nhưng giảm số email bình thường bị phân loại nhầm thành Spam từ 62 xuống 13.

Logistic Regression vẫn là một lựa chọn đáng cân nhắc khi ưu tiên khả năng phát hiện Spam, nhưng Precision thấp hơn khiến mô hình tạo ra nhiều cảnh báo nhầm hơn. Trong khi đó, SVM cần được kiểm tra và điều chỉnh thêm vì Recall hiện tại quá thấp.

Kết quả này chỉ phản ánh bộ dữ liệu và cấu hình của lần thực nghiệm được cung cấp. Khi thay đổi dữ liệu, bước tiền xử lý hoặc tham số mô hình, hiệu suất có thể thay đổi. Nếu chạy lại pipeline, cần cập nhật bảng metrics và phần nhận xét theo kết quả mới.

## Lưu ý về các tệp được tạo

Thư mục `data/processed/` và `models/` trong gói có thể chỉ chứa tệp giữ chỗ. Chạy `python src/pipeline.py` để tạo các tập dữ liệu đã xử lý, vectorizer và mô hình trước khi chạy đánh giá hoặc dự đoán.

Không đưa dữ liệu cá nhân hoặc email nhạy cảm vào kiểm thử nếu chưa được phép.

## Công nghệ sử dụng

- Python, NumPy, pandas
- scikit-learn (`TfidfVectorizer`, `train_test_split`)
- Matplotlib, Seaborn
- joblib / pickle để lưu các đối tượng đã xử lý và mô hình
- Jupyter Notebook
