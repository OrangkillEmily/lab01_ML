import sys
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_curve, auc


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.model import LogisticRegression, NaiveBayes, SVM  # noqa: F401
from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)


PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
IMAGE_DIR = ROOT_DIR / "docs" / "images"


def compute_metrics(y_true, y_pred):
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


def get_roc_scores(model, X, model_name):
    if model_name == "Logistic Regression":
        linear_model = np.asarray(
            X @ model.weights
        ).flatten() + model.bias

        linear_model = np.clip(linear_model, -500, 500)
        scores = 1 / (1 + np.exp(-linear_model))

    elif model_name == "SVM":
        scores = np.asarray(
            X @ model.weights
        ).flatten() + model.bias

    elif model_name == "Naive Bayes":
        spam_idx = np.where(model.classes == 1)[0][0]
        ham_idx = np.where(model.classes == 0)[0][0]

        def calculate_log_posterior(class_idx):
            mean = model.mean[class_idx]
            var = model.var[class_idx]

            inverse_var = 1.0 / var

            constant = np.sum(
                np.log(2 * np.pi * var)
                + (mean ** 2) / var
            )

            if hasattr(X, "power"):
                squared_term = np.asarray(
                    X.power(2) @ inverse_var
                ).flatten()
            else:
                X_array = np.asarray(X)
                squared_term = (X_array ** 2) @ inverse_var

            linear_term = np.asarray(
                X @ (mean * inverse_var)
            ).flatten()

            log_likelihood = -0.5 * (
                constant + squared_term - 2 * linear_term
            )

            log_prior = np.log(
                model.priors[class_idx] + 1e-12
            )

            return log_prior + log_likelihood

        spam_scores = calculate_log_posterior(spam_idx)
        ham_scores = calculate_log_posterior(ham_idx)

        scores = spam_scores - ham_scores

    else:
        raise ValueError(
            f"Không hỗ trợ ROC cho mô hình: {model_name}"
        )

    return np.asarray(scores).ravel()


def plot_roc_curve(y_true, models, X_test, save_path):
    y_true = np.asarray(y_true).ravel()

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "ROC Curve yêu cầu tập dữ liệu có đủ hai lớp Spam và Ham."
        )

    plt.figure(figsize=(9, 7))

    auc_results = {}

    for model_name, model in models.items():
        scores = get_roc_scores(model, X_test, model_name)

        fpr, tpr, _ = roc_curve(
            y_true,
            scores,
            pos_label=1
        )

        roc_auc = auc(fpr, tpr)
        auc_results[model_name] = roc_auc

        plt.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{model_name} (AUC = {roc_auc:.4f})"
        )

        print(f"[ROC] {model_name}: AUC = {roc_auc:.4f}")

    plt.plot(
        [0, 1],
        [0, 1],
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier (AUC = 0.5000)"
    )

    plt.xlabel("False Positive Rate (FPR)", fontsize=12)
    plt.ylabel("True Positive Rate (TPR)", fontsize=12)
    plt.title("ROC Curve - Model Comparison", fontsize=15)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[DONE] Đã lưu ROC Curve tại: {save_path}")

    return auc_results


def plot_confusion_matrix(cm, model_name, save_path):
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


def main(run_id=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Tao run moi neu chay rieng evaluate.py
    own_run = run_id is None

    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()

    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        write_log(
            run_id,
            "EVALUATION",
            "START",
            "Bat dau danh gia cac mo hinh."
        )

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
            "AUC",
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

        # Danh gia tung mo hinh tren Train va Test
        for model_name, model in models.items():
            for dataset_name, (X, y) in datasets.items():

                step_name = (
                    f"EVALUATION - {model_name} - {dataset_name}"
                )

                print(
                    f"\n[INFO] Đang đánh giá {model_name} "
                    f"trên tập {dataset_name}..."
                )

                eval_start = time.perf_counter()

                try:
                    y_pred = model.predict(X)
                    metrics = compute_metrics(y, y_pred)

                    roc_scores = get_roc_scores(
                        model,
                        X,
                        model_name
                    )

                    if len(np.unique(y)) == 2:
                        fpr_values, tpr_values, _ = roc_curve(
                            y,
                            roc_scores,
                            pos_label=1
                        )

                        roc_auc = auc(
                            fpr_values,
                            tpr_values
                        )
                    else:
                        roc_auc = np.nan

                    metrics["AUC"] = roc_auc

                    eval_time = (
                        time.perf_counter() - eval_start
                    )

                    print(
                        f"{dataset_name} | "
                        f"Accuracy: {metrics['Accuracy']:.4f} | "
                        f"Precision: {metrics['Precision']:.4f} | "
                        f"Recall: {metrics['Recall']:.4f} | "
                        f"F1: {metrics['F1']:.4f} | "
                        f"TPR: {metrics['TPR']:.4f} | "
                        f"FPR: {metrics['FPR']:.4f} | "
                        f"AUC: {metrics['AUC']:.4f}"
                    )

                    # Luu ket qua vao bang tong hop
                    row = {
                        "Mô hình": model_name,
                        "Tập dữ liệu": dataset_name,
                    }

                    for column in metric_columns:
                        row[column] = metrics[column]

                    summary_rows.append(row)

                    # Chi luu confusion matrix cho tap Test
                    if dataset_name == "Test":
                        test_results[model_name] = metrics

                        confusion_matrix_path = IMAGE_DIR / (
                            f"confusion_matrix_"
                            f"{model_suffixes[model_name]}.png"
                        )

                        plot_confusion_matrix(
                            metrics["cm"],
                            model_name,
                            confusion_matrix_path
                        )

                        print(
                            f"[DONE] Đã lưu confusion matrix tại: "
                            f"{confusion_matrix_path}"
                        )

                    # Ghi ket qua tung lan danh gia
                    metric_data = {
                        "Accuracy": float(metrics["Accuracy"]),
                        "Precision": float(metrics["Precision"]),
                        "Recall": float(metrics["Recall"]),
                        "F1": float(metrics["F1"]),
                        "TPR": float(metrics["TPR"]),
                        "FPR": float(metrics["FPR"]),
                        "AUC": (
                            float(metrics["AUC"])
                            if np.isfinite(metrics["AUC"])
                            else None
                        ),
                        "TN": int(metrics["TN"]),
                        "FP": int(metrics["FP"]),
                        "FN": int(metrics["FN"]),
                        "TP": int(metrics["TP"]),
                        "confusion_matrix": metrics["cm"].tolist(),
                        "evaluation_time_seconds": round(
                            eval_time, 4
                        ),
                    }

                    update_progress(
                        run_id,
                        step_name,
                        "SUCCESS",
                        input_data={
                            "model_name": model_name,
                            "dataset": dataset_name,
                            "X_shape": list(X.shape),
                            "y_shape": list(y.shape),
                            "samples": int(len(y)),
                            "features": int(X.shape[1]),
                        },
                        output_data=metric_data
                    )

                    write_log(
                        run_id,
                        step_name,
                        "SUCCESS",
                        f"Danh gia {model_name} tren {dataset_name} "
                        f'hoan tat. F1={metrics["F1"]:.4f}, '
                        f'AUC={metrics["AUC"]:.4f}'
                    )

                except Exception as error:
                    update_progress(
                        run_id,
                        step_name,
                        "FAILED",
                        input_data={
                            "model_name": model_name,
                            "dataset": dataset_name,
                        },
                        error=str(error)
                    )

                    write_log(
                        run_id,
                        step_name,
                        "FAILED",
                        str(error)
                    )

                    raise

        # Tao bang tong hop ket qua
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
            "AUC",
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

        # Ve confusion matrix tong hop
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

        # Ve ROC Curve va tinh AUC
        roc_curve_path = IMAGE_DIR / "roc_curve_all.png"

        auc_results = plot_roc_curve(
            y_test,
            models,
            X_test,
            roc_curve_path
        )

        # Tim mo hinh co F1 cao nhat tren Test
        test_summary = summary[
            summary["Tập dữ liệu"] == "Test"
        ]

        best_row = test_summary.loc[
            test_summary["F1"].idxmax()
        ]

        print("\n" + "=" * 55)
        print("KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG")
        print("=" * 55)

        print(
            f"[DONE] Mô hình tốt nhất theo F1 trên tập Test: "
            f"{best_row['Mô hình']} "
            f"(F1 = {best_row['F1']:.4f})"
        )

        print("\nSO SÁNH AUC:")

        for model_name, roc_auc in auc_results.items():
            print(f"{model_name}: AUC = {roc_auc:.4f}")

        best_auc_model = max(
            auc_results,
            key=auc_results.get
        )

        print(
            f"\n[DONE] Mô hình có AUC cao nhất: "
            f"{best_auc_model} "
            f"(AUC = {auc_results[best_auc_model]:.4f})"
        )

        # Tong hop cac file dau ra
        saved_files = [
            "reports/metrics_summary.csv",
            "docs/images/confusion_matrix_lr.png",
            "docs/images/confusion_matrix_nb.png",
            "docs/images/confusion_matrix_svm.png",
            "docs/images/confusion_matrix_all.png",
            "docs/images/roc_curve_all.png",
        ]

        total_time = time.perf_counter() - start_time

        # Ghi ket qua danh gia tong the
        update_progress(
            run_id,
            "EVALUATION",
            "SUCCESS",
            input_data={
                "train_samples": int(len(y_train)),
                "test_samples": int(len(y_test)),
                "features": int(X_test.shape[1]),
                "models": list(models.keys()),
            },
            output_data={
                "best_model_by_f1": str(best_row["Mô hình"]),
                "best_f1": float(best_row["F1"]),
                "best_model_by_auc": best_auc_model,
                "auc_results": {
                    name: float(value)
                    for name, value in auc_results.items()
                },
                "summary_file": str(summary_path),
                "saved_files": saved_files,
                "evaluation_time_seconds": round(
                    total_time, 4
                ),
            }
        )

        write_log(
            run_id,
            "EVALUATION",
            "SUCCESS",
            f"Danh gia hoan tat trong {total_time:.2f} giay."
        )

        print("\n" + "=" * 55)
        print("HOÀN TẤT QUÁ TRÌNH ĐÁNH GIÁ!")
        print("=" * 55)

        if own_run:
            finish_run(run_id, "SUCCESS")

    except Exception as error:
        write_log(
            run_id,
            "EVALUATION",
            "FAILED",
            str(error)
        )

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()
