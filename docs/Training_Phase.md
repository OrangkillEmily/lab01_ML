# Training Phase — Huấn luyện mô hình

## 1. Purpose — Mục đích

Training Phase là giai đoạn huấn luyện các thuật toán Machine Learning nhằm xây dựng mô hình có khả năng phân loại email thành hai lớp:

- **Ham (0):** Email bình thường.
- **Spam (1):** Email rác.

Mục tiêu của giai đoạn này là sử dụng dữ liệu đã được tiền xử lý để mô hình học các đặc trưng và tìm ra những tham số phù hợp cho việc phân loại email.

Dự án sử dụng ba thuật toán:

- Logistic Regression
- Gaussian Naive Bayes
- Support Vector Machine (SVM)

Sau khi huấn luyện, các mô hình được lưu thành file `.pkl` để tái sử dụng trong Inference Phase mà không cần huấn luyện lại.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| `X_train.pkl` | Ma trận đặc trưng TF-IDF của tập huấn luyện |
| `y_train.pkl` | Nhãn tương ứng của email: Ham (0) hoặc Spam (1) |
| `config.yaml` | File cấu hình các tham số của mô hình |

Dữ liệu đầu vào được tạo từ Preprocessing Phase, bao gồm quá trình làm sạch văn bản, chuyển đổi văn bản thành vector TF-IDF và chia dữ liệu thành Train/Test theo tỷ lệ 80/20.

### Output

| Output | Description |
|---|---|
| `model_lr.pkl` | Mô hình Logistic Regression đã huấn luyện |
| `model_nb.pkl` | Mô hình Naive Bayes đã huấn luyện |
| `model_svm.pkl` | Mô hình SVM đã huấn luyện |
| Training logs | Thông tin thời gian và trạng thái huấn luyện |
| Training figures | Các biểu đồ theo dõi quá trình huấn luyện |

Các mô hình sau khi huấn luyện được lưu trong thư mục `models/` và được sử dụng lại trong Inference Phase.

## 3. How to do — Quy trình thực hiện

### Training workflow

```text
       X_train + y_train
              |
              v
        Load Training Data
              |
              v
         Load Config
              |
              v
       Initialize Models
              |
       +------+------+ 
       |      |      |
       v      v      v
      LR     NB     SVM
       |      |      |
       v      v      v
      fit()  fit()  fit()
       |      |      |
       +------+------+ 
              |
              v
       Trained Models
              |
              v
       Save Models (.pkl)
              |
              v
       Training Tracking
              |
              v
        Evaluation Phase
```

### Các bước thực hiện

**Step 1 — Load Training Data**

Đọc dữ liệu đặc trưng và nhãn từ `X_train.pkl` và `y_train.pkl`.

- X_train: Các đặc trưng TF-IDF của email.
- y_train: Nhãn tương ứng của từng email.

**Step 2 — Load Configuration**

Đọc các tham số được khai báo trong `config.yaml`.

| Model | Parameters |
|---|---|
| Logistic Regression | learning_rate = 0.1, n_iterations = 1000 |
| Naive Bayes | Sử dụng các tham số thống kê được ước lượng từ dữ liệu |
| SVM | learning_rate = 0.01, lambda_param = 0.01, n_iterations = 100 |

**Step 3 — Initialize Models**

Khởi tạo ba mô hình Logistic Regression, Naive Bayes và SVM với các tham số tương ứng.

**Step 4 — Train Models**

Gọi phương thức `fit(X_train, y_train)` để huấn luyện từng mô hình.

Trong quá trình này:

- Logistic Regression học các trọng số `weights` và `bias`.
- Naive Bayes ước lượng `mean`, `var`, `priors` và các thông tin xác suất theo lớp.
- SVM học `weights` và `bias` nhằm tìm ranh giới phân chia giữa hai lớp.

**Step 5 — Save Trained Models**

Sau khi huấn luyện hoàn tất, sử dụng `joblib.dump()` để lưu các mô hình thành file `.pkl`.

**Step 6 — Tracking and Visualization**

Ghi nhận trạng thái, thời gian thực thi và các thông tin của lần chạy.

Kết quả được lưu tại:

- `progress_results/`: Kết quả tiến trình.
- `logs/`: Nhật ký thực thi.
- `reports/figures/`: Biểu đồ theo dõi.
- `docs/experiment_logs.csv`: Lịch sử thực nghiệm.

Sau khi hoàn thành Training Phase, hệ thống chuyển sang Evaluation Phase để đánh giá hiệu quả của các mô hình.

## 4. Evaluation Metrics — Các chỉ số đánh giá

Sau khi huấn luyện, các mô hình được đánh giá bằng những chỉ số phân loại sau:

| Metric | Description |
|---|---|
| Accuracy | Tỷ lệ dự đoán chính xác trên tổng số mẫu |
| Precision | Tỷ lệ email Spam thực sự trong số email được dự đoán là Spam |
| Recall | Tỷ lệ email Spam được phát hiện chính xác |
| F1-score | Trung bình điều hòa giữa Precision và Recall |
| TPR | Tỷ lệ phát hiện đúng email Spam |
| FPR | Tỷ lệ email Ham bị dự đoán nhầm thành Spam |
| AUC | Đánh giá khả năng phân biệt hai lớp của mô hình |
| Confusion Matrix | Ma trận thể hiện số lượng TN, FP, FN và TP |

Các chỉ số được tính trong Evaluation Phase thông qua dữ liệu Train và Test.

Ngoài ra, hệ thống ghi nhận thời gian huấn luyện để theo dõi hiệu suất thực thi của từng mô hình.

## 5. Used Data — Dữ liệu sử dụng

Dữ liệu được sử dụng trong Training Phase có nguồn gốc từ bộ dữ liệu SpamAssassin.

Sau Preprocessing Phase, dữ liệu được chia thành hai tập:

| Dataset | Number of Samples | Purpose |
|---|---:|---|
| Training Set | 4,736 | Dùng để huấn luyện mô hình |
| Testing Set | 1,184 | Dùng để đánh giá mô hình |

Tổng số email sau khi loại bỏ dữ liệu trùng lặp là 5,920.

Các file dữ liệu:

- `X_train.pkl`: Đặc trưng của tập Train.
- `y_train.pkl`: Nhãn của tập Train.
- `X_test.pkl`: Đặc trưng của tập Test.
- `y_test.pkl`: Nhãn của tập Test.

**Lưu ý:** Tập Test không được sử dụng để huấn luyện mô hình. Nó được giữ riêng để đánh giá khả năng dự đoán trên dữ liệu chưa được mô hình sử dụng khi học.

## Code References

| File | Responsibility |
|---|---|
| `src/train.py` | Điều phối quá trình huấn luyện |
| `src/model.py` | Cài đặt các thuật toán Machine Learning |
| `src/evaluate.py` | Đánh giá hiệu suất mô hình |
| `src/tracking.py` | Theo dõi tiến trình thực thi |
| `config.yaml` | Cấu hình tham số |
