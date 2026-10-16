import numpy as np

class LogisticRegression:
    def __init__(self,learning_rate,n_iterations): 
        self.learning_rate = 0.01
        self.n_iterations = 1000
        self.weights = None
        self.bias = 0
    def _sigmoid(self, x):
    # đưa giá trị về khoảng (0, 1)
        return 1 / (1 + np.exp(-x))
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 

        for _ in range(self.n_iterations):
            linear_model = np.dot(X, self.weights) + self.bias
            # dùng sigmoid function 
            y_predicted = self._sigmoid(linear_model)

            # gradient descent
            dw = (1/n_samples) * np.dot(X.T, y_predicted - y) # đạo hàm theo weights
            db = (1/n_samples) * np.sum(y_predicted - y) #  đạo hàm theo bias   

            # cập nhật weights và bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

class SVM:
    def __init__(self,learning_rate, lambda_param, n_iterations):
        self.learning_rate = 0.0001
        self.lambda_param = 0.01
        self.n_iterations = 1000 
        self.weights = None
        self.bias = 0

    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape
        # chuyển đổi y từ {0,1} sang {-1,1}
        y_ = np.where(y <= 0, -1, 1)
        self.weights = np.zeros(n_features)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                # kiểm tra dữ liệu nằm đúng phía của lề (margin) hay không
                condition = y_[idx] * (np.dot(x_i,self.weights) - self.bias) >= 1
                if condition:
                    # nếu đúng, cập nhật theo regularization term
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights)
                else:
                    # nếu sai, cập nhật theo cả loss và regularization
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(x_i,y_[idx]))
                    self.bias -= self.learning_rate * y_[idx]

    def predict(self, X):
        # đưa giá trị về khoảng (0,1)
        approx = np.dot(X, self.weights) - self.bias
        y_pred = np.sign(approx)
        # nếu y_pred[idx] có giá trị là -1 trả về 0, nếu không thì trả về 1 
        return np.where(y_pred == -1, 0, 1)

class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None
    def fit(self,X,y):
        X = np.array(X)
        y = np.array(y).flatten()
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # tạo ma trận chứa mean, var, priors
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.var = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            # ma trận con X_c chỉ chứa các mẫu dữ liệu thuộc nhãn c hiện tại
            X_c = X[y==c]
            # tính giá trị trung bình cho từng đặc trưng của nhóm c
            self.mean[idx,:] = X_c.mean(axis=0)
            # tính phương sai cho từng đặc trưng của nhóm c và cộng thêm một epsilon nhỏ để tránh lỗi chia cho 0
            self.var[idx,:] = X_c.var(axis=0) + 1e-9
            # tính xác suất tiên nghiệm
            self.priors[idx] = X_c.shape[0] / float(n_samples)
    
    """Hàm mật độ xác suất của phân phối chuẩn"""
    def _pdf(self, class_idx, x):
        mean = self.mean[class_idx]
        var = self.var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)        

        return numerator / denominator

    def _predict_single(self,x):
        posteriors = []

        # tính xác suất hậu nghiệm cho mỗi class
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx] + 1e-9)
            # tính tổng logarit để tránh underflow
            posterior = np.sum(np.log(self._pdf(idx,x) + 1e-9))           
            posterior = prior + posterior
            posteriors.append(posterior)

        # trả về class có xác suất lớn nhất
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._predict_single(x) for x in X]
        return np.array(y_pred)

