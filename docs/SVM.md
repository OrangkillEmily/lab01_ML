# Support Vector Machine (SVM)

## 1. Purpose

Sử dụng SVM để tìm siêu phẳng phân chia hai lớp email:

- `-1` nội bộ tương ứng với Ham (`0`).
- `1` tương ứng với Spam (`1`).

Mô hình được cài đặt bằng NumPy theo cách cập nhật Stochastic Gradient Descent trên Hinge Loss trong `src/model.py`. Mục tiêu là tối đa hóa khoảng lề giữa hai lớp đồng thời giới hạn vi phạm lề.

## 2. Input-Output

### Input

- Ma trận TF-IDF dạng sparse hoặc dense.
- Tập huấn luyện: `4724` email, tối đa `10000` đặc trưng.
- Tập kiểm tra: `1182` email, cùng không gian đặc trưng TF-IDF.
- Nhãn được chuyển từ `{0, 1}` thành `{-1, 1}` trong quá trình huấn luyện.

### Output

- Điểm quyết định `score = X @ weights + bias`.
- Nhãn `1` khi `score >= 0`; ngược lại nhãn `0`.
- `weights` là vector pháp tuyến của siêu phẳng.
- `bias` xác định độ dịch chuyển của siêu phẳng.
- Tệp mô hình được lưu tại `models/model_svm.pkl`.

## 3. How to do

1. Dùng dữ liệu TF-IDF đã được chia Train/Test và tạo vectorizer trên tập Train.
2. Chuyển nhãn `0/1` thành `-1/1`.
3. Khởi tạo `weights = 0` và `bias = 0`.
4. Huấn luyện trong `100` vòng lặp với `learning_rate=0.01` và `lambda_param=0.01`.
5. Duyệt từng mẫu theo SGD.
6. Nếu `y_i * score_i >= 1`, chỉ áp dụng cập nhật regularization:

```text
weights = weights - learning_rate * (2 * lambda_param * weights)
```

7. Nếu mẫu vi phạm margin, cập nhật cả `weights` và `bias`:

```text
weights = weights - learning_rate * (2 * lambda_param * weights - y_i * x_i)
bias = bias + learning_rate * y_i
```

8. Dự đoán bằng dấu của điểm quyết định.

## 4. Evaluation metrics

Dự án đánh giá bằng Accuracy, Precision, Recall, F1-score, TPR, FPR, AUC và Confusion Matrix.

Kết quả thực nghiệm hiện lưu trong `reports/metrics_summary.csv`:

| Tập dữ liệu | Accuracy | Precision | Recall | F1-score | TPR | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 71.94% | 88.89% | 9.39% | 16.99% | 9.39% | 0.52% | 0.9748 |
| Test | 71.88% | 89.19% | 9.12% | 16.54% | 9.12% | 0.49% | 0.9647 |

Confusion Matrix trên tập Test:

| | Dự đoán Ham | Dự đoán Spam |
|---|---:|---:|
| Thực tế Ham | TN = 818 | FP = 4 |
| Thực tế Spam | FN = 329 | TP = 33 |

Nhận xét: SVM ít tạo cảnh báo nhầm nhưng bỏ sót `329/362` email Spam trên tập Test, tương đương Recall chỉ `9.12%`. Với mục tiêu phát hiện Spam, mô hình trong cấu hình hiện tại chưa phù hợp; cần điều chỉnh tham số, quy trình huấn luyện hoặc cân bằng dữ liệu trước khi sử dụng.

## 5. Used data

- Dữ liệu gốc: `data/raw/SpamAssassin.csv`.
- Kích thước ban đầu: `6046` dòng.
- Cột văn bản: `data`.
- Cột nhãn: `label` với `0 = Ham`, `1 = Spam`.
- Phân bố nhãn gốc: `4150` Ham và `1896` Spam.
- Chia Train/Test: `80/20`, `random_state=42`, `stratify=y`.
- Đặc trưng: TF-IDF tối đa `10000`, n-gram `(1, 2)`, loại bỏ stop words tiếng Anh.
- Siêu tham số: `learning_rate=0.01`, `lambda_param=0.01`, `n_iterations=100`.
- Cấu hình tham chiếu: `config.yaml`.
