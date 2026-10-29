
import sys
import time
from pathlib import Path
from datetime import datetime

import yaml
import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.model import LogisticRegression, NaiveBayes, SVM


PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_FILE = ROOT_DIR / "docs" / "experiment_logs.csv"


def load_train_data():
    """Đọc dữ liệu huấn luyện từ data/processed/."""
    print("[INFO] Đang tải dữ liệu huấn luyện...")

    x_joblib_path = PROCESSED_DIR / "X_train.joblib"
    x_pkl_path = PROCESSED_DIR / "X_train.pkl"

    if x_joblib_path.exists():
        X_train = joblib.load(x_joblib_path)
    elif x_pkl_path.exists():
        X_train = joblib.load(x_pkl_path)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy X_train trong {PROCESSED_DIR}"
        )

    y_csv_path = PROCESSED_DIR / "y_train.csv"
    y_pkl_path = PROCESSED_DIR / "y_train.pkl"

    if y_csv_path.exists():
        y_train = pd.read_csv(y_csv_path).values
    elif y_pkl_path.exists():
        y_train = joblib.load(y_pkl_path)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy y_train trong {PROCESSED_DIR}"
        )

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    else:
        X_train = np.asarray(X_train)

    y_train = np.asarray(y_train).ravel().astype(int)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "Số lượng mẫu X_train và y_train không khớp."
        )

    print(
        f"[INFO] Tải thành công! "
        f"X_train: {X_train.shape}, y_train: {y_train.shape}"
    )

    return X_train, y_train


def load_config():
    """Đọc cấu hình từ config.yaml."""
    config_path = ROOT_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def log_experiment(log_file, hyperparams, execution_times):
    """Ghi lịch sử thực nghiệm vào experiment_logs.csv."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "LR_Params": str(
            hyperparams.get("logistic_regression", {})
        ),
        "NB_Params": str(
            hyperparams.get("naive_bayes", {})
        ),
        "SVM_Params": str(
            hyperparams.get("svm", {})
        ),
        "LR_Train_Time_Sec": round(
            execution_times.get("Logistic Regression", 0), 2
        ),
        "NB_Train_Time_Sec": round(
            execution_times.get("Naive Bayes", 0), 2
        ),
        "SVM_Train_Time_Sec": round(
            execution_times.get("SVM", 0), 2
        ),
        "Total_Train_Time_Sec": round(
            sum(execution_times.values()), 2
        )
    }

    df_log = pd.DataFrame([log_entry])
    file_exists = log_file.is_file()

    df_log.to_csv(
        log_file,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8"
    )

    print(f"[LOG] Đã lưu lịch sử thực nghiệm tại: {log_file}")


def main():
    X_train, y_train = load_train_data()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config()
    model_config = config["models"]

    execution_times = {}

    models = [
        (
            "Logistic Regression",
            LogisticRegression(
                learning_rate=model_config["logistic_regression"]["learning_rate"],
                n_iterations=model_config["logistic_regression"]["n_iterations"]
            ),
            MODEL_DIR / "model_lr.pkl",
        ),
        (
            "Naive Bayes",
            NaiveBayes(),
            MODEL_DIR / "model_nb.pkl",
        ),
        (
            "SVM",
            SVM(
                learning_rate=model_config["svm"]["learning_rate"],
                lambda_param=model_config["svm"]["lambda_param"],
                n_iterations=model_config["svm"]["n_iterations"]
            ),
            MODEL_DIR / "model_svm.pkl",
        ),
    ]

    print("\n" + "=" * 55)
    print("BẮT ĐẦU HUẤN LUYỆN 3 MÔ HÌNH (FROM SCRATCH)")
    print("=" * 55)

    for model_name, model, save_path in models:
        print(f"\n[TRAIN] Đang huấn luyện mô hình: {model_name}...")

        start_time = time.time()

        model.fit(X_train, y_train)

        elapsed_time = time.time() - start_time
        execution_times[model_name] = elapsed_time

        joblib.dump(model, save_path)

        print(
            f"[DONE] Hoàn thành {model_name} "
            f"trong {elapsed_time:.2f} giây."
        )
        print(f"[SAVE] Đã lưu mô hình tại: {save_path}")

    log_experiment(LOG_FILE, model_config, execution_times)

    print("\n" + "=" * 55)
    print("HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN VÀ LƯU MÔ HÌNH!")
    print(
        f"Tổng thời gian huấn luyện: "
        f"{sum(execution_times.values()):.2f} giây"
    )
    print("=" * 55)


if __name__ == "__main__":
    main()
