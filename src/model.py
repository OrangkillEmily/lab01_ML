
import numpy as np


class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def _sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        y = np.asarray(y).flatten()

        for _ in range(self.n_iterations):
            linear_model = np.asarray(
                X @ self.weights
            ).flatten() + self.bias

            y_predicted = self._sigmoid(linear_model)

            error = y_predicted - y

            dw = np.asarray(
                X.T @ error
            ).flatten() / n_samples

            db = np.sum(error) / n_samples

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X, threshold=0.3):
        linear_model = np.asarray(
            X @ self.weights
        ).flatten() + self.bias

        y_predicted = self._sigmoid(linear_model)

        return (y_predicted >= threshold).astype(int)



class SVM:
    def __init__(self, learning_rate, lambda_param, n_iterations):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        # Chuyển nhãn về mảng NumPy
        y = np.asarray(y).flatten()

        n_samples, n_features = X.shape

        # Chuyển nhãn {0, 1} thành {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        # Khởi tạo weights và bias
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Huấn luyện SVM
        for _ in range(self.n_iterations):
            for idx in range(n_samples):

                # Lấy một mẫu dữ liệu
                if hasattr(X, "getrow"):
                    x_i = X.getrow(idx)
                    x_weighted = x_i.toarray().flatten()
                    score = float(x_i @ self.weights) + self.bias
                else:
                    x_i = X[idx]
                    x_weighted = np.asarray(x_i).flatten()
                    score = float(x_weighted @ self.weights) + self.bias

                # Kiểm tra điều kiện margin
                if y_[idx] * score >= 1:
                    # Chỉ cập nhật regularization
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                    )
                else:
                    # Cập nhật weights theo hinge loss
                    self.weights -= self.learning_rate * (
                        2 * self.lambda_param * self.weights
                        - y_[idx] * x_weighted
                    )

                    # Cập nhật bias
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        # Tính điểm dự đoán
        scores = np.asarray(
            X @ self.weights
        ).flatten() + self.bias

        # Chuyển điểm thành nhãn 0 hoặc 1
        return np.where(scores >= 0, 1, 0)


class NaiveBayes:
    def __init__(self):
        self.classes = None
        self.mean = None
        self.var = None
        self.priors = None

    def fit(self, X, y):
        # Chuyển nhãn về mảng NumPy
        y = np.asarray(y).flatten()

        n_samples, n_features = X.shape

        # Xác định các lớp
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Khởi tạo tham số cho từng lớp
        self.mean = np.zeros(
            (n_classes, n_features),
            dtype=np.float64
        )

        self.var = np.zeros(
            (n_classes, n_features),
            dtype=np.float64
        )

        self.priors = np.zeros(
            n_classes,
            dtype=np.float64
        )

        # Tính trung bình, phương sai và xác suất tiên nghiệm
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]

            # Tính trung bình
            self.mean[idx, :] = np.asarray(
                X_c.mean(axis=0)
            ).flatten()

            # Tính trung bình bình phương
            if hasattr(X_c, "power"):
                mean_square = np.asarray(
                    X_c.power(2).mean(axis=0)
                ).flatten()
            else:
                mean_square = np.mean(
                    np.asarray(X_c) ** 2,
                    axis=0
                ).flatten()

            # Phương sai = E[X^2] - E[X]^2
            self.var[idx, :] = np.maximum(
                mean_square - self.mean[idx, :] ** 2,
                1e-9
            )

            # Xác suất tiên nghiệm của lớp
            self.priors[idx] = (
                X_c.shape[0] / float(n_samples)
            )

    def _log_likelihood(self, class_idx, x):
        # Lấy trung bình và phương sai của lớp
        mean = self.mean[class_idx]
        var = self.var[class_idx]

        # Tính log xác suất Gaussian
        log_probability = -0.5 * np.sum(
            np.log(2 * np.pi * var)
            + ((x - mean) ** 2) / var
        )

        return log_probability

    def _predict_single(self, x):
        posteriors = []

        # Tính log xác suất hậu nghiệm cho từng lớp
        for idx, c in enumerate(self.classes):
            prior = np.log(self.priors[idx] + 1e-12)

            likelihood = self._log_likelihood(
                idx,
                x
            )

            posterior = prior + likelihood
            posteriors.append(posterior)

        # Chọn lớp có điểm hậu nghiệm cao nhất
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = []

        # Dự đoán từng mẫu dữ liệu
        for idx in range(X.shape[0]):
            if hasattr(X, "getrow"):
                x = X.getrow(idx).toarray().flatten()
            else:
                x = np.asarray(X[idx]).flatten()

            y_pred.append(
                self._predict_single(x)
            )

        return np.asarray(y_pred)
