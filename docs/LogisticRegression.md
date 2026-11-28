# Logistic Regression

## 1. Purpose

Sử dụng Logistic Regression để phân loại email thành:

- `0`: Ham — email bình thường.
- `1`: Spam — email rác.

Mô hình được cài đặt hoàn toàn bằng NumPy trong `src/model.py`, không sử dụng Logistic Regression có sẵn của scikit-learn.

Trong dự án này, mô hình tính xác suất Spam bằng hàm sigmoid, sau đó dùng ngưỡng quyết định để trả về nhãn `0` hoặc `1`.

## 2. Input-Output

### Input

- Ma trận TF-IDF cho mỗi email.
- Kích thước tập huấn luyện: `(4724, 10000)` sau khi tiền xử lý.
- Kích thước tập kiểm tra: `(1182, 10000)`.
- Nhãn huấn luyện `y_train` là mảng nhị phân gồm `0` và `1`.

### Output

- Nhãn dự đoán: `0` (Ham) hoặc `1` (Spam).
- Mô hình hiện cài đặt trả về `1` khi xác suất sigmoid lớn hơn hoặc bằng `0.3`; ngược lại trả về `0`.
- Phương thức huấn luyện lưu `weights` theo từng đặc trưng TF-IDF và một giá trị `bias`.
- Tệp mô hình được lưu tại `models/model_lr.pkl`.

## 3. How to do

1. Nạp dữ liệu văn bản `data/raw/SpamAssassin.csv`.
2. Loại bỏ giá trị thiếu, email trùng lặp và email rỗng sau khi làm sạch.
3. Chuyển văn bản về chữ thường, bỏ thẻ HTML, URL, ký tự không phải chữ cái tiếng Anh và khoảng trắng thừa.
4. Chia dữ liệu theo tỷ lệ Train/Test `80/20`, `random_state=42`, đồng thời giữ tỷ lệ các lớp bằng `stratify`.
5. Fit `TfidfVectorizer` trên tập Train với tối đa `10000` đặc trưng, n-gram `(1, 2)` và `stop_words="english"`.
6. Khởi tạo `weights = 0` và `bias = 0`.
7. Huấn luyện bằng Gradient Descent trong `1000` vòng lặp với `learning_rate=0.1`.
8. Với mỗi vòng lặp, tính xác suất sigmoid, sai số dự đoán và gradient của `weights`, `bias`, sau đó cập nhật tham số.
9. Lưu mô hình và đánh giá trên cả tập Train và tập Test.

Công thức dự đoán:

```text
z = X @ weights + bias
probability = 1 / (1 + exp(-z))
prediction = 1 nếu probability >= 0.3, ngược lại 0
```

## 4. Evaluation metrics

Dự án tính Accuracy, Precision, Recall, F1-score, TPR, FPR, AUC và Confusion Matrix.

Kết quả thực nghiệm hiện lưu trong `reports/metrics_summary.csv`:

| Tập dữ liệu | Accuracy | Precision | Recall | F1-score | TPR | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 95.35% | 90.08% | 95.30% | 92.62% | 95.30% | 4.62% | 0.9716 |
| Test | 93.16% | 84.69% | 94.75% | 89.44% | 94.75% | 7.54% | 0.9597 |

Confusion Matrix trên tập Test:

| | Dự đoán Ham | Dự đoán Spam |
|---|---:|---:|
| Thực tế Ham | TN = 760 | FP = 62 |
| Thực tế Spam | FN = 19 | TP = 343 |

Nhận xét: mô hình phát hiện Spam tốt nhờ Recall cao, nhưng còn 62 email Ham bị phân loại nhầm thành Spam. Điều này làm Precision trên tập Test thấp hơn Naive Bayes.

## 5. Used data

- Dữ liệu gốc: `data/raw/SpamAssassin.csv`.
- Kích thước ban đầu: `6046` dòng, `3` cột.
- Cột đầu vào: `data`.
- Cột nhãn: `label`, trong đó `0` là Ham và `1` là Spam.
- Phân bố nhãn gốc: `4150` Ham và `1896` Spam.
- Tập Train sau tiền xử lý: `4724` mẫu.
- Tập Test sau tiền xử lý: `1182` mẫu.
- Đặc trưng đầu vào: tối đa `10000` đặc trưng TF-IDF dạng uni-gram và bi-gram.
- Cấu hình tham chiếu: `config.yaml`.
