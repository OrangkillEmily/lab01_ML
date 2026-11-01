import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import LogisticRegression, NaiveBayes, SVM  # noqa: F401

PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
IMAGE_DIR = ROOT_DIR / "docs" / "images"


def compute_metrics(y_true, y_pred):
    """Tính các chỉ số phân loại nhị phân, xem nhãn 1 là Spam."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true và y_pred phải có cùng số phần tử.")

    TN = int(np.count_nonzero((y_true == 0) & (y_pred == 0)))
    FP = int(np.count_nonzero((y_true == 0) & (y_pred == 1)))
    FN = int(np.count_nonzero((y_true == 1) & (y_pred == 0)))
    TP = int(np.count_nonzero((y_true == 1) & (y_pred == 1)))

    total = TP + TN + FP + FN

    accuracy = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if TP + FP else 0.0
    recall = TP / (TP + FN) if TP + FN else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    tpr = TP / (TP + FN) if TP + FN else 0.0
    fpr = FP / (FP + TN) if FP + TN else 0.0

    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TPR": tpr,
        "FPR": fpr,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "TP": TP,
        "cm": np.array([[TN, FP], [FN, TP]]),
    }


def plot_confusion_matrix(cm, model_name, save_path):
    """Vẽ và lưu ma trận nhầm lẫn của một mô hình."""
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham (0)", "Spam (1)"],
        yticklabels=["Ham (0)", "Spam (1)"],
    )

    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_confusion_matrices(results, save_path):
    """Vẽ confusion matrix của các mô hình trên cùng một hình."""
    if not results:
        return

    _, axes = plt.subplots(
        1,
        len(results),
        figsize=(6 * len(results), 5),
        squeeze=False,
    )

    for ax, (model_name, metrics) in zip(axes[0], results.items()):
        sns.heatmap(
            metrics["cm"],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Ham (0)", "Spam (1)"],
            yticklabels=["Ham (0)", "Spam (1)"],
            ax=ax,
        )

        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(f"Confusion Matrix - {model_name}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def load_data(split):
    """Đọc dữ liệu Train hoặc Test."""
    x_joblib_path = PROCESSED_DIR / f"X_{split}.joblib"
    x_pkl_path = PROCESSED_DIR / f"X_{split}.pkl"

    y_joblib_path = PROCESSED_DIR / f"y_{split}.joblib"
    y_pkl_path = PROCESSED_DIR / f"y_{split}.pkl"

    if x_joblib_path.exists():
        X = joblib.load(x_joblib_path)
    elif x_pkl_path.exists():
        X = joblib.load(x_pkl_path)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy X_{split}.joblib hoặc X_{split}.pkl "
            f"trong {PROCESSED_DIR}"
        )

    if y_joblib_path.exists():
        y = joblib.load(y_joblib_path)
    elif y_pkl_path.exists():
        y = joblib.load(y_pkl_path)
    else:
        raise FileNotFoundError(
            f"Không tìm thấy y_{split}.joblib hoặc y_{split}.pkl "
            f"trong {PROCESSED_DIR}"
        )

    y = np.asarray(y).ravel().astype(int)

    if X.shape[0] != len(y):
        raise ValueError(
            f"Số mẫu X_{split} ({X.shape[0]}) khác số nhãn "
            f"y_{split} ({len(y)})."
        )

    return X, y


def load_models():
    """Tải ba mô hình đã được huấn luyện."""
    model_paths = {
        "Logistic Regression": MODEL_DIR / "model_lr.pkl",
        "Naive Bayes": MODEL_DIR / "model_nb.pkl",
        "SVM": MODEL_DIR / "model_svm.pkl",
    }

    missing_paths = [
        path for path in model_paths.values()
        if not path.exists()
    ]

    if missing_paths:
        missing_files = ", ".join(
            path.name for path in missing_paths
        )

        raise FileNotFoundError(
            f"Không tìm thấy tệp mô hình: {missing_files}. "
            "Vui lòng chạy train.py trước."
        )

    return {
        name: joblib.load(path)
        for name, path in model_paths.items()
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 55)
    print("BẮT ĐẦU ĐÁNH GIÁ CÁC MÔ HÌNH")
    print("=" * 55)

    print("[INFO] Đang tải dữ liệu Train và Test...")

    X_train, y_train = load_data("train")
    X_test, y_test = load_data("test")

    print(
        f"[INFO] X_train: {X_train.shape}, "
        f"y_train: {y_train.shape}"
    )

    print(
        f"[INFO] X_test: {X_test.shape}, "
        f"y_test: {y_test.shape}"
    )

    print("[INFO] Đang tải các mô hình...")
    models = load_models()

    model_suffixes = {
        "Logistic Regression": "lr",
        "Naive Bayes": "nb",
        "SVM": "svm",
    }

    metric_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TPR",
        "FPR",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    summary_rows = []
    test_results = {}

    datasets = {
        "Train": (X_train, y_train),
        "Test": (X_test, y_test),
    }

    for model_name, model in models.items():
        for dataset_name, (X, y) in datasets.items():
            print(
                f"\n[INFO] Đang đánh giá {model_name} "
                f"trên tập {dataset_name}..."
            )

            y_pred = model.predict(X)
            metrics = compute_metrics(y, y_pred)

            print(
                f"{dataset_name} | "
                f"Accuracy: {metrics['Accuracy']:.4f} | "
                f"Precision: {metrics['Precision']:.4f} | "
                f"Recall: {metrics['Recall']:.4f} | "
                f"F1: {metrics['F1']:.4f} | "
                f"TPR: {metrics['TPR']:.4f} | "
                f"FPR: {metrics['FPR']:.4f}"
            )

            row = {
                "Mô hình": model_name,
                "Tập dữ liệu": dataset_name,
            }

            for column in metric_columns:
                row[column] = metrics[column]

            summary_rows.append(row)

            # Chỉ lưu Confusion Matrix trên tập Test.
            if dataset_name == "Test":
                test_results[model_name] = metrics

                confusion_matrix_path = IMAGE_DIR / (
                    f"confusion_matrix_"
                    f"{model_suffixes[model_name]}.png"
                )

                plot_confusion_matrix(
                    metrics["cm"],
                    model_name,
                    confusion_matrix_path,
                )

                print(
                    f"[DONE] Đã lưu confusion matrix tại: "
                    f"{confusion_matrix_path}"
                )

    summary = pd.DataFrame(summary_rows)

    columns = [
        "Mô hình",
        "Tập dữ liệu",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "TPR",
        "FPR",
        "TN",
        "FP",
        "FN",
        "TP",
    ]

    summary = summary[columns]

    print("\n" + "=" * 55)
    print("BẢNG SO SÁNH CÁC MÔ HÌNH TRÊN TRAIN VÀ TEST")
    print("=" * 55)

    print(summary.round(4).to_string(index=False))

    summary_path = REPORT_DIR / "metrics_summary.csv"

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[DONE] Đã lưu bảng metrics tại: {summary_path}")

    all_confusion_matrices_path = (
        IMAGE_DIR / "confusion_matrix_all.png"
    )

    plot_all_confusion_matrices(
        test_results,
        all_confusion_matrices_path,
    )

    print(
        f"[DONE] Đã lưu các confusion matrix tại: "
        f"{all_confusion_matrices_path}"
    )

    test_summary = summary[
        summary["Tập dữ liệu"] == "Test"
    ]

    best_row = test_summary.loc[
        test_summary["F1"].idxmax()
    ]

    print(
        f"[DONE] Mô hình tốt nhất theo F1 trên tập Test: "
        f"{best_row['Mô hình']} "
        f"(F1 = {best_row['F1']:.4f})"
    )

    print("\n" + "=" * 55)
    print("HOÀN TẤT QUÁ TRÌNH ĐÁNH GIÁ!")
    print("=" * 55)


if __name__ == "__main__":
    main()