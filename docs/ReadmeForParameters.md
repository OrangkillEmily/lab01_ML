# GIẢI THÍCH SƠ BỘ VỀ CÁC PARAMS
`./src/model.py` chứa 3 thuật toán Supervised Learning
1 **Logistic Regression**
2 **Support Vector Machine - SVM** 
3 **Navie Bayes**

## 1. `LogisticRegression`

Mô hình học tuyển tính sử dụng hàm Sigmoid để dự đoán xác suất $P(y|X)$

### A. Model Parameters (Tham số cần tìm)
* **`self.weights`** (Vector Trọng số):
    * *Kiểu dữ liệu*: Mảng Numpy 1 chiều có kích thước `n_features`
    * *Ý nghĩa*: Đo lường mức dodọ ảnh hưởng của từng từ vựng (TF-IDF) tới xác suất đặc trưng của Spam (ví dụ: *free*, *click*, *gift*,...)
    * **Lý do toán học:** 
        * Ma trận đặc trưng đầu vào $X$ có kích thước `(n_samples, n_features)` (với `n_features` là số lượng từ vựng trong từ điển TF-IDF, ví dụ: 1000 từ).
        * Để thực hiện phép nhân ma trận hoặc tính tổng có trọng số $x_i \cdot w$, **mỗi một đặc trưng (từ vựng) $j$ bắt buộc phải đi kèm với đúng 1 trọng số $w_j$ tương ứng**.
        * Trọng số $w_j$ thể hiện mức độ đóng góp (tầm quan trọng) của từ vựng thứ $j$ vào việc quyết định email đó là Spam hay Ham. Do đó, nếu bài toán dùng 1000 từ vựng thì vector trọng số phải có đúng 1000 phần tử.

* **`self.bias`** (Trọng số lệch):
    * *Kiểu dữ liệu*: Số thực (`float`)
    * *Ý nghĩa* Xác định ngưỡng quyết định cơ sở khi tất cả đặc trưng đầu vào $X$ bằng 0.
    * **Lý do toán học:**
        * Bias $b$ đóng vai trò là hằng số dịch chuyển ngưỡng quyết định (ngưỡng cơ sở) cho **toàn bộ dữ liệu** chứ không phụ thuộc vào bất kỳ từ vựng cụ thể nào.
        * Phân loại nhị phân (Binary Classification - Spam vs Ham) chỉ tạo ra **1 ranh giới / 1 siêu phẳng duy nhất**, do đó chỉ cần đúng **1 giá trị vô hướng $b$** để nâng hoặc hạ siêu phẳng đó.
### B. Hyperparameters:
* **`learning_rate`**: Tốc độ học, điều chỉnh kích thước của bước nhảy mỗi khi cập nhật trọng số theo hướng Gradient Descent
* **`n_iterations`**: Số vòng lặp tối đa để Gradient Descent cập nhật trọng số

___ 

## 2. `SVM`
### A. Model Paremeters
* **`self.weights`** (Vector Pháp tuyến của Siêu phẳng *Hyperplane*):
    * *Kiểu dữ liệu*: Mảng Numpy 1 chiều có kích thước `n_features`
    * *Ý nghĩa*: Định hình góc quay của **Siêu phẳng** $w^T x - b = 0$. Độ lớn $\|w\|$ quy định độ rộng của khoảng lề ($\text{Margin} = \frac{2}{\|w\|}$).
* **`self.bias`** (Độ dịch chuyển):
    * *Kiểu dữ liệu*: Số thực
    * *Ý nghĩa*: Xác định khoảng cách từ gốc toạ độ đến siêu phẳng dọc theo hướng Vector pháp tuyến $w$

### B. Hyperparameters 

* **`learning_rate`** ($\alpha$): Tốc độ cập nhật trọng số trong thuật toán Stochastic Gradient Descent (SGD).
* **`lambda_param`** ($\lambda$): Hệ số điều hòa (Regularization parameter). Cân bằng giữa việc **phóng đại khoảng lề Margin** ($\frac{2}{\|w\|}$) và **giảm thiểu lỗi vi phạm lề** (Margin Violations).
* **`n_iterations`**: Số lượt duyệt qua tập dữ liệu để tối ưu hóa Hinge Loss.

## 3. `Navie Bayes`

### A. Model Parameters
* **`self.priors`** (Xác suất tiên nghiệm):
    * *Kiểu dữ liệu*: Mảng Numpy 1 Chiều. Kích thước `n_classes`
    * *Ý nghĩa*:Tỷ lệ xuất hiện tự nhiên của từng lớp: $P(y = \text{Spam})$ và $P(y = \text{Ham})$ trên tập huấn luyện.
    * **Lý do toán học:**
        * `self.priors` lưu xác suất xuất hiện ban đầu của từng lớp $P(y = c)$ trên tổng số email huấn luyện.
        * Bài toán có bao nhiêu lớp (`n_classes`) thì sẽ có bấy nhiêu giá trị xác suất tiên nghiệm (với phân loại nhị phân có 2 lớp: $P(\text{Ham})$ và $P(\text{Spam})$). Vì vậy kích thước là `(2,)`.
* **`self.mean`** (Giá trị trung bình):
    * *Kiểu dữ liệu*: Ma trận Numpy 2 chiều, kích thước `(n_classes x n_features)`
    * *Ý nghĩa*: Giá trị TF-IDF trung bình $\mu_{c, j}$ của từng từ vựng $j$ trong từng lớp $c$
* **`self.var`** (Phương sai):
    * *Kiểu dữ liệu*: Ma trận Numpy 2 chiều, kích thước `(n_classes x n_features)`
    * *Ý nghĩa*: Mức độ phân tán giá trị TF-IDF của từng từ vựng trong từng lớp

* **Lý do toán học:**
  * Thuật toán Naive Bayes giả định rằng: Mỗi từ vựng $j$ trong tập dữ liệu tuân theo một Phân phối chuẩn (Gaussian Distribution) riêng biệt **ứng với từng lớp $c$**.
  * Phân phối Gaussian được định nghĩa hoàn toàn bởi 2 tham số: Trung bình $\mu$ và Phương sai $\sigma^2$.
  * Để đo độ phân bố của từ vựng $j$ trong lớp $c$, ta phải tính:
    1. Giá trị TF-IDF trung bình của từ $j$ **khi email đó là Ham** ($\mu_{\text{ham}, j}$).
    2. Giá trị TF-IDF trung bình của từ $j$ **khi email đó là Spam** ($\mu_{\text{spam}, j}$).
  * Do đó, dữ liệu tạo thành một ma trận bảng 2 chiều: 
    * **Chiều Hàng (`n_classes`):** Phân chia theo lớp (Hàng 0 ứng với Ham, Hàng 1 ứng với Spam).
    * **Chiều Cột (`n_features`):** Giá trị thống kê cho từng từ vựng tương ứng.
  * Vì có 2 lớp và $M$ từ vựng, kích thước bắt buộc phải là `(2, n_features)`.
### B. Hyperparameters:
* **Không có siêu tham số vòng lặp** vì Naive Bayes tính toán tham số trực tiếp bằng các công thức thông kê tần suất trong một lượt duyệt qua duy nhất

## Tóm tắt: Model Parameters là những trọng số cần tìm trong hàm `fit`

