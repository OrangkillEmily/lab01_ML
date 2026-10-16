import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.model import LogisticRegression, NaiveBayes, SVM

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


def load_train_data():
    """Đọc dữ liệu X_train và y_train từ thư mục data/processed/."""
    print("[INFO] Đang tải dữ liệu huấn luyện từ data/processed/...")

    # 1. Đọc X_train (hỗ trợ cả .joblib của ThuMinh29 và .pkl dự phòng)
    x_joblib_path = PROCESSED_DIR / "X_train.joblib"
    x_pkl_path = PROCESSED_DIR / "X_train.pkl"
    if x_joblib_path.exists():
        X_train = joblib.load(x_joblib_path)
    elif x_pkl_path.exists():
        X_train = joblib.load(x_pkl_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy X_train trong {PROCESSED_DIR}")

  
    y_csv_path = PROCESSED_DIR / "y_train.csv"
    y_pkl_path = PROCESSED_DIR / "y_train.pkl"
    if y_csv_path.exists():
        y_train = pd.read_csv(y_csv_path).values
    elif y_pkl_path.exists():
        y_train = joblib.load(y_pkl_path)
    else:
        raise FileNotFoundError(f"Không tìm thấy y_train trong {PROCESSED_DIR}")

   
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    else:
        X_train = np.asarray(X_train)

    # Đưa nhãn y_train về mảng 1 chiều (0 và 1)
    y_train = np.asarray(y_train).ravel().astype(int)

    print(f"[INFO] Tải thành công! X_train: {X_train.shape}, y_train: {y_train.shape}")
    return X_train, y_train


def main():
    X_train, y_train = load_train_data()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    
    models = [
        (
            "Logistic Regression",
            LogisticRegression(learning_rate=0.1, n_iterations=500),
            MODEL_DIR / "model_lr.pkl",
        ),
        (
            "Naive Bayes",
            NaiveBayes(),
            MODEL_DIR / "model_nb.pkl",
        ),
        (
            "SVM",
            SVM(learning_rate=0.001, lambda_param=0.01, n_iterations=100),
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
        joblib.dump(model, save_path)
        print(f"[DONE] Hoàn thành {model_name} trong {elapsed_time:.2f} giây.")
        print(f"[SAVE] Đã lưu mô hình tại: {save_path}")

    print("\n" + "=" * 55)
    print("HOÀN TẤT QUÁ TRÌNH HUẤN LUYỆN VÀ LƯU MÔ HÌNH!")
    print("=" * 55)


if __name__ == "__main__":
    main()