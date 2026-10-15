# TÓM TẮT DỰ ÁN
**1. Tên dự án:** 
Lab 01 - Phân loại Email Rác (Spam Email Classifier).

**2. Mục tiêu cốt lõi:** 
Xây dựng một hệ thống Machine Learning có khả năng tự động đọc nội dung văn bản của một email và phán đoán xem đó là Email Rác (Spam) hay Email An toàn (Ham).

**3. Dữ liệu & Kỹ thuật:**
- **Dữ liệu đầu vào:** Sử dụng bộ dữ liệu `SpamAssassin.csv` (đặt tại thư mục `data/raw`).
- **Kỹ thuật trích xuất:** Làm sạch văn bản và sử dụng phương pháp TF-IDF để chuyển đổi chữ viết thành các ma trận số học.
- **Thuật toán sử dụng:** Hồi quy Logistic (Logistic Regression), Naive Bayes và Máy học véc-tơ hỗ trợ (SVM).

**4. Kiến trúc mã nguồn:** 
Dự án TUYỆT ĐỐI KHÔNG viết dồn code vào một file duy nhất. Mã nguồn được chia nhỏ (module hóa) thành các tệp `.py` thực hiện các nhiệm vụ độc lập bên trong thư mục `src/`.