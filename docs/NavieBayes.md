# Naive Bayes

## 1. Purpose

Sử dụng Gaussian Naive Bayes để phân loại email thành:

- `0`: Ham — email bình thường.
- `1`: Spam — email rác.

Mô hình ước lượng phân bố thống kê của từng đặc trưng TF-IDF cho mỗi lớp và giả định các đặc trưng độc lập có điều kiện với nhau. Mô hình được cài đặt hoàn toàn bằng NumPy trong `src/model.py`.

## 2. Input-Output

### Input

- Ma trận TF-IDF của từng email.
- Tập huấn luyện: `4724` email và tối đa `10000` đặc trưng.
- Tập kiểm tra: `1182` email.
- Nhãn `y_train` gồm lớp `0` và `1`.

### Output

- Nhãn dự đoán `0` hoặc `1`.
- `priors`: xác suất tiên nghiệm của từng lớp.
- `mean`: trung bình TF-IDF của từng đặc trưng trong từng lớp.
- `var`: phương sai TF-IDF của từng đặc trưng trong từng lớp.
- Tệp mô hình được lưu tại `models/model_nb.pkl`.

## 3. How to do

1. Tiền xử lý và tạo TF-IDF theo quy trình chung của dự án.
2. Xác định các lớp bằng `np.unique(y)`.
3. Với mỗi lớp `c`, lọc các email thuộc lớp đó.
4. Tính xác suất tiên nghiệm:

```text
P(c) = số email thuộc lớp c / tổng số email huấn luyện
```

5. Tính trung bình $\mu_{c,j}$ và phương sai $\sigma^2_{c,j}$ của từng đặc trưng `j` trong lớp `c`.
6. Đảm bảo phương sai tối thiểu là `1e-9` để tránh phép chia cho không.
7. Khi dự đoán một email, tính log xác suất Gaussian cho từng lớp:

```text
log likelihood = sum(-0.5 * (log(2*pi*var) + ((x - mean)^2 / var)))
```

8. Cộng log xác suất tiên nghiệm và log likelihood:

```text
posterior_score = log(P(c)) + log likelihood
```

9. Chọn lớp có posterior score cao nhất.

Naive Bayes không có vòng lặp huấn luyện hoặc learning rate trong triển khai này; các tham số được tính trực tiếp từ dữ liệu một lần.

## 4. Evaluation metrics

Dự án tính Accuracy, Precision, Recall, F1-score, TPR, FPR, AUC và Confusion Matrix.

Kết quả thực nghiệm hiện lưu trong `reports/metrics_summary.csv`:

| Tập dữ liệu | Accuracy | Precision | Recall | F1-score | TPR | FPR | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 99.62% | 98.97% | 99.79% | 99.38% | 99.79% | 0.46% | 0.999959 |
| Test | 97.30% | 96.35% | 94.75% | 95.54% | 94.75% | 1.58% | 0.993548 |

Confusion Matrix trên tập Test:

| | Dự đoán Ham | Dự đoán Spam |
|---|---:|---:|
| Thực tế Ham | TN = 809 | FP = 13 |
| Thực tế Spam | FN = 19 | TP = 343 |

Nhận xét: Naive Bayes là mô hình có kết quả tốt nhất trong lần thực nghiệm này. Mô hình chỉ phân loại nhầm `13` email Ham thành Spam và bỏ sót `19` email Spam trên tập Test.

## 5. Used data

- Dữ liệu gốc: `data/raw/SpamAssassin.csv`.
- Kích thước ban đầu: `6046` dòng, `3` cột.
- Cột đầu vào: `data`.
- Cột nhãn: `label`, trong đó `0 = Ham`, `1 = Spam`.
- Phân bố nhãn gốc: `4150` Ham và `1896` Spam.
- Chia Train/Test: `80/20`, `random_state=42`, `stratify=y`.
- Tập Train: `4724` mẫu.
- Tập Test: `1182` mẫu.
- Đặc trưng: TF-IDF tối đa `10000`, n-gram `(1, 2)`, stop words tiếng Anh.
- Không có hyperparameter huấn luyện trong `config.yaml` cho Naive Bayes.
