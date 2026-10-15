"""
Gói mã nguồn (Source package) cho dự án Phân loại Email Rác (Spam Email Classifier).

Thư mục `src/` được thiết kế theo cấu trúc module hóa, bao gồm các thành phần:
- eda.py: Khám phá và trực quan hóa dữ liệu.
- preprocess.py: Làm sạch văn bản, chuyển đổi TF-IDF và chia tập Train/Test.
- train.py: Huấn luyện các thuật toán Machine Learning (LR, NB, SVM).
- evaluate.py: Đánh giá mô hình và vẽ Confusion Matrix.
- utils.py: Chứa các hàm tiện ích dùng chung.
- pipeline.py: Điều phối luồng thực thi tự động.
- predict.py: Chạy thử nghiệm dự đoán với email mới.
"""

# Khai báo phiên bản của gói mã nguồn
__version__ = "1.0.0"

# Ghi chú: File __init__.py giúp Python nhận diện thư mục 'src' là một package.
# Không nên viết các hàm xử lý logic phức tạp vào file này.