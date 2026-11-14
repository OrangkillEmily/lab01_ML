# Inference Phase — Dự đoán dữ liệu mới

## 1. Purpose — Mục đích

Inference Phase là giai đoạn sử dụng các mô hình Machine Learning đã được huấn luyện trong Training Phase để dự đoán nhãn cho những email mới.

Mục tiêu là xác định email đầu vào thuộc một trong hai lớp:

- **Ham (0):** Email bình thường.
- **Spam (1):** Email rác.

Khác với Training Phase, Inference Phase không thực hiện quá trình học tham số. Hệ thống chỉ sử dụng các mô hình đã được lưu để đưa ra kết quả dự đoán.

## 2. Input — Output

### Input

| Input | Description |
|---|---|
| Email Text | Nội dung email mới cần phân loại |
| `tfidf_vectorizer.pkl` | Bộ chuyển đổi văn bản thành vector TF-IDF đã được huấn luyện |
| `model_lr.pkl` | Logistic Regression đã huấn luyện |
| `model_nb.pkl` | Naive Bayes đã huấn luyện |
| `model_svm.pkl` | SVM đã huấn luyện |

### Output

| Output | Description |
|---|---|
| Predicted Label | Nhãn dự đoán Ham hoặc Spam |
| Prediction Result | Kết quả dự đoán từ từng mô hình |
| Processing Time | Thời gian xử lý và dự đoán |
| Inference History | Lịch sử các lần dự đoán |

Kết quả dự đoán được hiển thị cho người dùng và lưu vào `reports/inference_history.csv`.

## 3. How to do — Quy trình thực hiện

### Inference workflow

```text
         New Email Text
                |
                v
          Validate Input
                |
                v
           clean_text()
                |
                v
         Cleaned Email Text
                |
                v
       Load TF-IDF Vectorizer
                |
                v
       TF-IDF Transformation
                |
                v
          Feature Vector
                |
         +------+------+ 
         |      |      |
         v      v      v
        LR     NB     SVM
         |      |      |
         v      v      v
      predict predict predict
         |      |      |
         +------+------+ 
                |
                v
        Predicted Output
          Ham / Spam
                |
                v
        Display Prediction
                |
                v
        Save Inference History
```

### Các bước thực hiện

**Step 1 — Receive Input**

Nhận nội dung email mới từ người dùng thông qua chương trình hoặc giao diện ứng dụng.

**Step 2 — Load Trained Models**

Sử dụng `joblib.load()` để tải các mô hình đã huấn luyện và bộ chuyển đổi TF-IDF từ thư mục `models/`.

Các mô hình không cần huấn luyện lại vì tham số đã được lưu từ Training Phase.

**Step 3 — Text Preprocessing**

Sử dụng hàm `clean_text()` để làm sạch nội dung email.

Các thao tác bao gồm:

- Chuyển văn bản thành chữ thường.
- Loại bỏ HTML.
- Loại bỏ URL.
- Loại bỏ ký tự không cần thiết.
- Chuẩn hóa khoảng trắng.

**Step 4 — TF-IDF Transformation**

Sử dụng `tfidf_vectorizer.pkl` để chuyển email đã làm sạch thành vector TF-IDF.

Gọi:

```python
X = vectorizer.transform([email])
```

Không sử dụng `fit_transform()` vì vectorizer đã được huấn luyện từ dữ liệu Train.

**Step 5 — Prediction**

Đưa vector đặc trưng vào ba mô hình đã huấn luyện:

```python
lr_result = model_lr.predict(X)
nb_result = model_nb.predict(X)
svm_result = model_svm.predict(X)
```

Mỗi mô hình sẽ đưa ra một nhãn dự đoán tương ứng.

**Step 6 — Decode Prediction**

Chuyển đổi kết quả số thành nhãn dễ hiểu:

- 0 → Ham
- 1 → Spam

**Step 7 — Display and Save Results**

Hiển thị kết quả dự đoán và ghi nhận thông tin vào lịch sử Inference.

Các thông tin được lưu bao gồm:

- Nội dung email đầu vào.
- Kết quả dự đoán của từng mô hình.
- Thời gian thực hiện.
- Run ID.
- Thời gian xử lý.

## 4. Evaluation Metrics — Các chỉ số đánh giá

Trong Inference Phase, hệ thống tập trung theo dõi hiệu suất xử lý và kết quả dự đoán.

| Metric | Description |
|---|---|
| Cleaning Time | Thời gian làm sạch văn bản |
| TF-IDF Time | Thời gian chuyển đổi văn bản thành vector |
| Prediction Time | Thời gian mô hình đưa ra dự đoán |
| Total Time | Tổng thời gian xử lý một email |
| Feature Shape | Kích thước vector đặc trưng đầu vào |
| Predicted Label | Nhãn dự đoán Ham hoặc Spam |

**Lưu ý:** Accuracy, Precision, Recall, F1-score và AUC không thể tính trực tiếp cho một email mới nếu chưa biết nhãn thực tế của email đó.

Các chỉ số đánh giá phân loại được thực hiện trong Evaluation Phase khi có dữ liệu kiểm thử kèm nhãn thật.

## 5. Used Data — Dữ liệu sử dụng

Inference Phase sử dụng hai nhóm dữ liệu chính:

**1. New Input Data**

Email mới do người dùng cung cấp. Đây là dữ liệu cần dự đoán và không bắt buộc phải có nhãn.

**2. Trained Artifacts**

Các file được tạo từ Training và Preprocessing Phase:

- `tfidf_vectorizer.pkl`
- `model_lr.pkl`
- `model_nb.pkl`
- `model_svm.pkl`

Các artifact này giúp hệ thống xử lý email mới theo đúng không gian đặc trưng và tham số đã học trước đó.

Inference Phase không sử dụng dữ liệu mới để huấn luyện lại mô hình.

## Code References

| File | Responsibility |
|---|---|
| `src/predict.py` | Điều phối quá trình dự đoán |
| `src/preprocess.py` | Làm sạch nội dung email |
| `src/model.py` | Thực hiện dự đoán thông qua `predict()` |
| `src/tracking.py` | Theo dõi quá trình thực thi |
| `models/` | Lưu các mô hình và TF-IDF vectorizer đã huấn luyện |
| `reports/inference_history.csv` | Lưu lịch sử dự đoán |
