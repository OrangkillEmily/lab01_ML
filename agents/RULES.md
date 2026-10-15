# QUY TẮC LẬP TRÌNH & TƯƠNG TÁC 

**Quy tắc 1: Quyền quyết định và Tương tác AI - Con người**

* **Vai trò của AI:** Đóng vai trò trợ lý chuyên môn. Chỉ được đưa ra các gợi ý, đề xuất giải pháp và viết mã dự thảo ở mức độ vừa đủ, phù hợp với phạm vi của một bài Lab trên trường.
* **Vai trò của Con người:** Là người kiểm duyệt và ra quyết định cuối cùng.
* **Nguyên tắc "Hiểu mới dùng":** Tuyệt đối KHÔNG copy/paste hoặc chạy bất kỳ đoạn code nào nếu chưa thực sự hiểu rõ logic bên trong. Nếu AI viết code khó hiểu, thành viên phải yêu cầu AI giải thích lại bằng ngôn ngữ bình dân trước khi sử dụng.

**Quy tắc 2: Vibe Coding (Dos & Don'ts)**

* **Được làm (Dos):** Ưu tiên tính đơn giản, dễ đọc, đi thẳng vào trọng tâm giải quyết bài toán. Code phải tường minh, có tính ứng dụng thực tế cho dự án.
* **Không được làm (Don'ts):** AI không được tự ý "over-engineer" (phức tạp hóa vấn đề), không dùng các design pattern hay cú pháp quá trừu tượng, cao siêu. Không tự ý bịa thêm tính năng nằm ngoài phạm vi đề bài.

**Quy tắc 3: Bắt buộc triển khai CẢ 3 MÔ HÌNH**

* Dự án này nhóm chúng ta sẽ không chọn 1 trong 3, mà **sẽ phải xây dựng cả 3 thuật toán** được yêu cầu: Logistic Regression, Naive Bayes, và Support Vector Machines (SVM) để có cái nhìn tổng quan nhất.
* File `train.py` phải huấn luyện cả 3 mô hình. File `evaluate.py` phải xuất ra bảng so sánh hiệu năng của cả 3 mô hình này.

**Quy tắc 4: Giới hạn thư viện**

* Chỉ được phép sử dụng các thư viện đã thống nhất trong `requirements.txt` (pandas, numpy, scikit-learn, matplotlib, seaborn, nltk, beautifulsoup4).
* Nếu cần sử dụng thư viện nào khác phải báo lên ngay lập tức.

**Quy tắc 5: Phải có giải thích code (Docstring)**

* Mọi hàm (function) được viết trong thư mục `src/` đều phải có phần ghi chú rõ ràng bên dưới tên hàm, bao gồm 3 ý:
* Hàm này dùng để làm gì?
* Đầu vào (Parameters) cần truyền vào cái gì?
* Đầu ra (Return) trả về kết quả gì?



**Quy tắc 6: Tôn trọng kiến trúc thư mục**

* Tuyệt đối KHÔNG tự ý đổi tên file, xóa thư mục hoặc gộp chung các file lại với nhau.
* File nào việc nấy (Ví dụ: `train.py` chỉ dùng để huấn luyện, cấm viết code vẽ biểu đồ vào đây).

**Quy tắc 7: Chống rò rỉ dữ liệu (Data Leakage)**

* Tập dữ liệu dùng để kiểm tra (Test set) phải được chia tỷ lệ 80-20 và tách ra cất riêng ngay từ khâu Tiền xử lý (`preprocess.py`). Quá trình huấn luyện tuyệt đối không được "nhìn trộm" hay dính dáng đến tập dữ liệu kiểm tra này.