# HƯỚNG DẪN TỪNG BƯỚC & BÀN GIAO DỰ ÁN (STEP-BY-STEP & HANDOFF)

## 1. Rules From Project (Quy tắc làm việc & AI)
- **AI Role:** AI đóng vai trò trợ lý chuyên môn, con người đưa ra quyết định cuối cùng. Không tự ý đoán mò, phải hỏi lại nếu thiếu thông tin.
- **Triết lý Code (Vibe Coding):** Code đơn giản, tường minh, dễ hiểu, không "over-engineer". Tuyệt đối không dùng code nếu chưa hiểu rõ logic.
- **Yêu cầu cốt lõi:** Bắt buộc phải triển khai huấn luyện ĐỦ 3 mô hình: Logistic Regression, Naive Bayes, SVM.
- **Nguyên tắc Dữ liệu:** Chống rò rỉ dữ liệu (Data Leakage) tuyệt đối. Tập Test (20%) phải được cắt và tách biệt hoàn toàn từ bước Tiền xử lý.

## 2. Chosen Direction (Định hướng & Kiến trúc)
- **Nguồn dữ liệu:** Đọc từ `data/raw/SpamAssassin.csv`.
- **Quy trình ELT/ML:** Đọc raw CSV -> Tiền xử lý (Cleaning + TF-IDF) -> Chia Train/Test (80-20) -> Lưu xuống đĩa -> Built model-> Huấn luyện -> Đánh giá.
- **Tiền xử lý:** Loại bỏ thẻ HTML, dấu câu, stop-words tiếng Anh.
- **Log & Trace:** Thông báo chạy pipeline phải được in rõ ràng ra terminal.
- **Môi trường:** Quản lý bằng `venv` và `requirements.txt`.

## 3. Current Files And Roles (Cấu trúc file hiện tại)
- `agents/`: Chứa định hướng dự án (description, LAB, RULES, stepByStep).
- `data/raw/`: Chứa file gốc `SpamAssassin.csv` .
- `data/processed/`: Chứa ma trận dữ liệu đã làm sạch (`X_train`, `X_test`, `y_train`, `y_test`).
- `docs/Lab01_descript.md`: File báo cáo tài liệu dự án.
- `models/`: Chứa các trọng số mô hình đã huấn luyện (`.pkl`).
- `notebooks/lab01.ipynb`: report notebook cho lab 1.
- `reports/`: Nơi lưu các biểu đồ EDA và Confusion Matrix (dạng ảnh `.png`).
- `src/__init__.py`: Khai báo `src` là một package.
- `src/eda.py`: Trực quan hóa dữ liệu thô, lưu ảnh vào `reports/`.
- `src/preprocess.py`: Làm sạch text, chạy TF-IDF, cắt Train/Test và lưu vào `data/processed/`.
- `src/train.py`: Load `X_train`, `y_train` để huấn luyện 3 mô hình (LR, NB, SVM), lưu vào `models/`.
- `src/evaluate.py`: Load mô hình từ `models/` và tập Test để đánh giá, tính metrics, vẽ Confusion Matrix.
- `src/utils.py`: Chứa các hàm tiện ích dùng chung (đo thời gian, nạp/lưu file).
- `src/pipeline.py`: Kịch bản điều phối, gọi chạy tuần tự preprocess -> train -> evaluate.
- `src/predict.py`: Kịch bản test nhanh, nhập vào 1 email bất kỳ trên terminal để mô hình dự đoán.
- `src/model.py`: Xây dựng mô hình từ đầu (from scratch) không dùng thư viện có sẵn như scikit-learn.
- `models/`: Nơi lưu trữ các file mô hình phân loại đã được huấn luyện hoàn chỉnh (định dạng .pkl), dùng để mang ra đánh giá hoặc dự đoán email mới.

## 4. Install & Setup (Cài đặt môi trường & Chuẩn bị)

Để bắt đầu làm việc hoặc khi mới tải code từ Git về máy, mỗi thành viên bắt buộc phải thực hiện đúng 3 bước sau để đảm bảo máy tính không bị lỗi:

**Bước 4.1. Kích hoạt môi trường ảo (Virtual Environment)**
Tuyệt đối không cài trực tiếp thư viện vào Python của máy tính. Hãy tạo và bật môi trường ảo riêng cho dự án:
- Mở Terminal (Git Bash trên VS Code) và gõ:

  # Tạo môi trường ảo (Chỉ làm 1 lần duy nhất)
  python -m venv venv
  
  # Bật môi trường (Phải bật mỗi khi mở lại VS Code)
  source venv/Scripts/activate

  (Dấu hiệu thành công: Đầu dòng lệnh hiện chữ (venv)).

**Bước 4.2. Cài đặt kho thư viện**

Gõ lệnh sau để máy tự động tải đúng phiên bản các thư viện cần thiết (pandas, scikit-learn,...):

pip install -r requirements.txt

**Bước 4.3. Xác nhận Dữ liệu thô (Raw Data)**

Kiểm tra xem file SpamAssassin.csv đã có sẵn trong thư mục data/raw/ chưa. Nếu có rồi thì bạn đã sẵn sàng bắt đầu code!

## 5. Workflow Timeline (Tiến độ thực thi chi tiết 4 Giai đoạn)

**Giai đoạn 1: Khởi tạo không gian làm việc** (✅ Đã xong)
Đã dựng xong khung thư mục tĩnh (src/, data/raw/, reports/, models/).

Đã cấu hình .gitignore chặn thư mục ma trận nặng data/processed/ để không làm sập GitHub.

Đã chốt luật làm việc tại thư mục agents/.


**Giai đoạn 2: Khám phá và Tiền xử lý (Chống rò rỉ dữ liệu)**
Mục tiêu: Biến các email toàn chữ thập cẩm thành các ma trận số học và cất riêng tập Test.

*Nhiệm vụ 1 (File src/eda.py):*

Đọc file data/raw/SpamAssassin.csv.

In ra đếm xem có bao nhiêu email Spam, bao nhiêu email Ham. Vẽ biểu đồ tỷ lệ này và lưu ảnh vào reports/.

*Nhiệm vụ 2 (File src/preprocess.py):*

Làm sạch (Clean): Xóa toàn bộ thẻ HTML (<br>, <html>), dấu câu, khoảng trắng thừa, và lọc bỏ các từ tiếng Anh vô nghĩa (Stop-words).

Số hóa (Vectorize): Sử dụng thuật toán TfidfVectorizer để biến cột nội dung email thành ma trận số.

Cắt lớp (Train/Test Split): Dùng train_test_split cắt dữ liệu theo tỷ lệ 80% (Học) - 20% (Thi).

Đầu ra bắt buộc: Lưu 4 tập dữ liệu (X_train, X_test, y_train, y_test) dưới dạng file .pkl (hoặc .npy) vào thư mục data/processed/.


**Giai đoạn 3: Huấn luyện 3 Mô hình (Model Training)**
Mục tiêu: Đưa dữ liệu 80% vào dạy cho máy học cách nhận diện Spam bằng các thuật toán tự xây dựng từ đầu (from scratch).

**Nhiệm vụ 1 (File src/model.py):**
 Xây dựng thủ công các class cho 3 thuật toán bằng kiến thức Toán rời rạc & Đại số tuyến tính. Tuyệt đối không dùng thư viện có sẵn (như scikit-learn) cho các model này. Các class cần có cấu trúc:
Hàm __init__(self): Khởi tạo các siêu tham số (learning rate, số vòng lặp...).
Hàm fit(self, X, y): Viết vòng lặp để cập nhật trọng số (weights, bias) dựa trên đạo hàm (Gradient Descent) cho Logistic Regression/SVM, hoặc tính xác suất tiên nghiệm (Prior/Likelihood) cho Naive Bayes.
Hàm predict(self, X): Tính toán và trả về nhãn dự đoán (0 hoặc 1).

**Nhiệm vụ 2 (File src/train.py):**

Chỉ được phép đọc file X_train và y_train từ thư mục data/processed/.

Viết code khởi tạo và gọi hàm .fit() cho cả 3 thuật toán bắt buộc:

LogisticRegression()

MultinomialNB() (Naive Bayes)

SVC() (Support Vector Machine)

Đầu ra bắt buộc: Sau khi train xong, dùng thư viện joblib lưu 3 mô hình này thành 3 file riêng biệt vào thư mục models/ (VD: models/model_lr.pkl, models/model_nb.pkl, models/model_svm.pkl).


**Giai đoạn 4: Đánh giá & Ráp nối Pipeline**
Mục tiêu: Đưa 20% dữ liệu đã giấu ra để làm bài kiểm tra và tính điểm.

*Nhiệm vụ 1 (File src/evaluate.py):*

Tải 3 file mô hình từ thư mục models/ lên.

Tải tập X_test, y_test từ data/processed/ lên.

Chạy hàm .predict() cho từng mô hình để xem chúng đoán đúng bao nhiêu %.

Tính các chỉ số đo lường: Accuracy, Precision, Recall, F1-score, TPR, FPR.

Dùng seaborn vẽ Ma trận nhầm lẫn (Confusion Matrix) cho từng mô hình và lưu 3 tấm ảnh vào thư mục reports/.

*Nhiệm vụ 2 (File src/pipeline.py):*

Viết code để khi gõ python src/pipeline.py, máy tính sẽ tự động gọi chạy từ từ: preprocess -> train -> evaluate mà không cần con người bấm từng lệnh riêng lẻ.


**Giai đoạn 5: Viết báo cáo & Tổng kết (Nghiệm thu)**
*Mục tiêu: Đóng gói toàn bộ kết quả của dự án thành tài liệu hoàn chỉnh để nộp bài.*

- **Nhiệm vụ 1 (Tài liệu hóa):** Mở file `docs/Lab01_descript.md`, trình bày lại luồng xử lý của hệ thống. Copy các thông số (Accuracy, F1-score, TPR, FPR, Precision, Recall) và chèn các hình ảnh Ma trận nhầm lẫn từ thư mục `reports/` vào tài liệu để minh chứng.
- **Nhiệm vụ 2 (Báo cáo Notebook):** Nếu giảng viên yêu cầu nộp định dạng file `.ipynb`, hãy mở `notebooks/lab01.ipynb`, kết hợp code và text giải thích (Markdown) tạo thành một luồng chạy liền mạch từ trên xuống dưới.
- **Nhiệm vụ 3 (Mặt tiền dự án):** Hoàn thiện file `README.md` ở thư mục gốc. Ghi rõ danh sách thành viên nhóm và ghi chú câu lệnh `python src/pipeline.py` để giảng viên hoặc bất kỳ ai cũng có thể tự chạy code.