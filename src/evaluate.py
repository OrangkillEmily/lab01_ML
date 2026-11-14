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
from src.visualize import create_stage_dir
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

METRICS_FILE = REPORT_DIR / "metrics_summary.csv"

METRIC_COLUMNS = [
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

SUMMARY_COLUMNS = ["Run_ID"] + METRIC_COLUMNS


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


def plot_metric_comparison(test_results, save_path):
    """So sánh Accuracy, Precision, Recall và F1 trên tập Test."""

    if not test_results:
        return

    metric_names = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1"
    ]

    model_names = list(test_results.keys())

    x = np.arange(len(metric_names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6))

    for index, model_name in enumerate(model_names):
        values = [
            test_results[model_name][metric]
            for metric in metric_names
        ]

        positions = x + (
            index - (len(model_names) - 1) / 2
        ) * width

        bars = ax.bar(
            positions,
            values,
            width,
            label=model_name
        )

        ax.bar_label(
            bars,
            fmt="%.3f",
            fontsize=8,
            padding=2
        )

    ax.set_title("Test Set Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_tpr_fpr_comparison(test_results, save_path):
    """So sánh TPR và FPR trên tập Test."""

    if not test_results:
        return

    model_names = list(test_results.keys())

    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    tpr_values = [
        test_results[name]["TPR"]
        for name in model_names
    ]

    fpr_values = [
        test_results[name]["FPR"]
        for name in model_names
    ]

    tpr_bars = ax.bar(
        x - width / 2,
        tpr_values,
        width,
        label="TPR"
    )

    fpr_bars = ax.bar(
        x + width / 2,
        fpr_values,
        width,
        label="FPR"
    )

    ax.bar_label(tpr_bars, fmt="%.3f", padding=2)
    ax.bar_label(fpr_bars, fmt="%.3f", padding=2)

    ax.set_title("TPR and FPR Comparison on Test Set")
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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


def save_metrics(summary, run_id):
    """
    Lưu metrics theo từng Run_ID.

    - Giữ lại kết quả của các lần chạy trước.
    - Dữ liệu cũ chưa có Run_ID được gán LEGACY.
    - Nếu Run_ID đã tồn tại thì thay thế kết quả của lần chạy đó.
    """

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    current_summary = summary.copy()
    current_summary.insert(0, "Run_ID", str(run_id))

    # Đọc dữ liệu cũ nếu file đã tồn tại
    if METRICS_FILE.exists():
        try:
            old_summary = pd.read_csv(
                METRICS_FILE,
                encoding="utf-8-sig"
            )

            if not old_summary.empty:
                # Dữ liệu cũ chưa có Run_ID
                if "Run_ID" not in old_summary.columns:
                    old_summary.insert(
                        0,
                        "Run_ID",
                        "LEGACY"
                    )

                # Bổ sung những cột còn thiếu để thống nhất schema
                for column in SUMMARY_COLUMNS:
                    if column not in old_summary.columns:
                        old_summary[column] = np.nan

                old_summary = old_summary[SUMMARY_COLUMNS]

            else:
                old_summary = pd.DataFrame(
                    columns=SUMMARY_COLUMNS
                )

        except pd.errors.EmptyDataError:
            old_summary = pd.DataFrame(
                columns=SUMMARY_COLUMNS
            )

    else:
        old_summary = pd.DataFrame(
            columns=SUMMARY_COLUMNS
        )

    # Loại bỏ kết quả cũ của chính Run_ID hiện tại
    if not old_summary.empty:
        old_summary = old_summary[
            old_summary["Run_ID"].astype(str) != str(run_id)
        ]

    # Gộp dữ liệu cũ và dữ liệu mới
    combined_summary = pd.concat(
        [
            old_summary,
            current_summary
        ],
        ignore_index=True
    )

    combined_summary = combined_summary[SUMMARY_COLUMNS]

    # Lưu lại toàn bộ lịch sử metrics
    combined_summary.to_csv(
        METRICS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"[DONE] Đã lưu metrics cho Run_ID: {run_id}"
    )

    print(
        f"[INFO] Tổng số dòng metrics hiện tại: "
        f"{len(combined_summary)}"
    )

    return METRICS_FILE


def main(run_id=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    own_run = run_id is None

    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()

    try:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        evaluation_figure_dir = create_stage_dir(
            run_id,
            "evaluation"
        )

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

        summary_rows = []
        test_results = {}

        datasets = {
            "Train": (X_train, y_train),
            "Test": (X_test, y_test),
        }

        # Đánh giá từng mô hình trên Train và Test
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

                    row = {
                        "Mô hình": model_name,
                        "Tập dữ liệu": dataset_name,
                    }

                    for column in METRIC_COLUMNS[2:]:
                        row[column] = metrics[column]

                    summary_rows.append(row)

                    # Chỉ lưu Confusion Matrix của tập Test
                    if dataset_name == "Test":
                        test_results[model_name] = metrics

                        confusion_matrix_name = (
                            f"confusion_matrix_"
                            f"{model_suffixes[model_name]}.png"
                        )

                        confusion_matrix_path = (
                            evaluation_figure_dir
                            / confusion_matrix_name
                        )

                        legacy_confusion_matrix_path = (
                            IMAGE_DIR / confusion_matrix_name
                        )

                        plot_confusion_matrix(
                            metrics["cm"],
                            model_name,
                            confusion_matrix_path
                        )

                        plot_confusion_matrix(
                            metrics["cm"],
                            model_name,
                            legacy_confusion_matrix_path
                        )

                        print(
                            f"[DONE] Đã lưu confusion matrix tại: "
                            f"{confusion_matrix_path}"
                        )

                    # Lưu kết quả vào progress_results
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
                        f"hoan tat. F1={metrics['F1']:.4f}, "
                        f"AUC={metrics['AUC']:.4f}"
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

        # Tạo bảng tổng hợp kết quả
        summary = pd.DataFrame(
            summary_rows,
            columns=METRIC_COLUMNS
        )

        print("\n" + "=" * 55)
        print("BẢNG SO SÁNH CÁC MÔ HÌNH TRÊN TRAIN VÀ TEST")
        print("=" * 55)

        print(summary.round(4).to_string(index=False))

        # Lưu metrics theo Run_ID, không ghi đè lịch sử
        summary_path = save_metrics(
            summary,
            run_id
        )

        print(f"[DONE] Đã lưu bảng metrics tại: {summary_path}")

        # Vẽ Confusion Matrix tổng hợp
        all_confusion_matrices_path = (
            evaluation_figure_dir
            / "confusion_matrix_all.png"
        )

        legacy_all_confusion_matrices_path = (
            IMAGE_DIR / "confusion_matrix_all.png"
        )

        plot_all_confusion_matrices(
            test_results,
            all_confusion_matrices_path
        )

        plot_all_confusion_matrices(
            test_results,
            legacy_all_confusion_matrices_path
        )

        print(
            f"[DONE] Đã lưu các confusion matrix tại: "
            f"{all_confusion_matrices_path}"
        )

        # Vẽ ROC Curve và tính AUC
        roc_curve_path = (
            evaluation_figure_dir / "roc_curve_all.png"
        )

        legacy_roc_curve_path = (
            IMAGE_DIR / "roc_curve_all.png"
        )

        auc_results = plot_roc_curve(
            y_test,
            models,
            X_test,
            roc_curve_path
        )

        plot_roc_curve(
            y_test,
            models,
            X_test,
            legacy_roc_curve_path
        )

        # Vẽ biểu đồ so sánh các chỉ số
        metric_comparison_path = (
            evaluation_figure_dir
            / "metrics_comparison.png"
        )

        tpr_fpr_comparison_path = (
            evaluation_figure_dir
            / "tpr_fpr_comparison.png"
        )

        plot_metric_comparison(
            test_results,
            metric_comparison_path
        )

        plot_tpr_fpr_comparison(
            test_results,
            tpr_fpr_comparison_path
        )

        print(
            f"[DONE] Đã lưu biểu đồ so sánh metrics: "
            f"{metric_comparison_path}"
        )

        print(
            f"[DONE] Đã lưu biểu đồ TPR/FPR: "
            f"{tpr_fpr_comparison_path}"
        )

        # Tìm mô hình tốt nhất theo F1
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

        # Tổng hợp danh sách file đầu ra
        saved_files = [
            "reports/metrics_summary.csv",
            "docs/images/confusion_matrix_lr.png",
            "docs/images/confusion_matrix_nb.png",
            "docs/images/confusion_matrix_svm.png",
            "docs/images/confusion_matrix_all.png",
            "docs/images/roc_curve_all.png",
        ]

        saved_files.extend(
            str(path.relative_to(ROOT_DIR)).replace("\\", "/")
            for path in sorted(
                evaluation_figure_dir.glob("*.png")
            )
        )

        total_time = time.perf_counter() - start_time

        # Ghi kết quả đánh giá tổng thể
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
                "figure_directory": str(evaluation_figure_dir),
                "saved_figures": [
                    str(path)
                    for path in sorted(
                        evaluation_figure_dir.glob("*.png")
                    )
                ],
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
