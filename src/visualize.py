from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT_DIR / "reports" / "figures"


def create_stage_dir(run_id, stage):
    """Tao thu muc luu hinh theo tung run va tung giai doan."""
    output_dir = FIGURE_DIR / str(run_id) / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


# =========================================================
# EDA VISUALIZATIONS
# =========================================================

def save_label_distribution(label_distribution, output_dir):
    """Ve bieu do phan bo nhan email."""

    fig, ax = plt.subplots(figsize=(8, 5))

    label_distribution.plot(kind="bar", ax=ax)

    ax.set_title("Email Label Distribution")
    ax.set_xlabel("Email Label (0 = Ham, 1 = Spam)")
    ax.set_ylabel("Number of Emails")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    output_path = output_dir / "label_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_missing_values(missing_values, output_dir):
    """Ve bieu do so luong gia tri thieu."""

    fig, ax = plt.subplots(figsize=(8, 5))

    missing_values.plot(kind="bar", ax=ax)

    ax.set_title("Missing Values by Column")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    ax.tick_params(axis="x", rotation=0)

    fig.tight_layout()

    output_path = output_dir / "missing_values.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_email_length_distribution(email_lengths, output_dir):
    """Ve phan phoi do dai email."""

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(email_lengths, bins=50, edgecolor="black")

    ax.set_title("Email Length Distribution")
    ax.set_xlabel("Email Length (characters)")
    ax.set_ylabel("Number of Emails")

    fig.tight_layout()

    output_path = output_dir / "email_length_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =========================================================
# PREPROCESSING VISUALIZATIONS
# =========================================================

def save_cleaning_comparison(raw_text, cleaned_text, output_dir):
    """So sanh noi dung email truoc va sau khi lam sach."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    raw_display = str(raw_text)[:1000]
    cleaned_display = str(cleaned_text)[:1000]

    axes[0].text(
        0.02,
        0.95,
        raw_display if raw_display else "(Empty)",
        transform=axes[0].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        wrap=True,
        fontsize=9
    )

    axes[0].set_title("Before Cleaning")
    axes[0].axis("off")

    axes[1].text(
        0.02,
        0.95,
        cleaned_display if cleaned_display else "(Empty)",
        transform=axes[1].transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        wrap=True,
        fontsize=9
    )

    axes[1].set_title("After Cleaning")
    axes[1].axis("off")

    fig.suptitle("Email Text Cleaning Comparison", fontsize=14)
    fig.tight_layout()

    output_path = output_dir / "cleaning_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_preprocessing_steps(step_counts, output_dir):
    """Ve so luong dong du lieu sau tung buoc tien xu ly."""

    steps = list(step_counts.keys())
    counts = list(step_counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(steps, counts)

    ax.set_title("Number of Emails Through Preprocessing Steps")
    ax.set_xlabel("Preprocessing Step")
    ax.set_ylabel("Number of Emails")
    ax.tick_params(axis="x", rotation=15)

    ax.bar_label(bars, padding=3)

    fig.tight_layout()

    output_path = output_dir / "preprocessing_steps.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_train_test_distribution(y_train, y_test, output_dir):
    """So sanh phan bo nhan trong tap Train va Test."""

    train_counts = y_train.value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    test_counts = y_test.value_counts().reindex(
        [0, 1],
        fill_value=0
    )

    distribution = pd.DataFrame({
        "Train": train_counts,
        "Test": test_counts
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    distribution.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title("Ham/Spam Distribution in Train and Test")
    ax.set_xlabel("Email Label (0 = Ham, 1 = Spam)")
    ax.set_ylabel("Number of Emails")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="Dataset")

    fig.tight_layout()

    output_path = output_dir / "train_test_distribution.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_tfidf_sparsity(X_train_tfidf, X_test_tfidf, output_dir):
    """Ve ty le gia tri 0 va khac 0 trong ma tran TF-IDF."""

    matrices = {
        "Train": X_train_tfidf,
        "Test": X_test_tfidf
    }

    densities = []
    sparsities = []

    for matrix in matrices.values():
        total_values = matrix.shape[0] * matrix.shape[1]

        if total_values == 0:
            density = 0
        else:
            density = matrix.nnz / total_values

        densities.append(density * 100)
        sparsities.append((1 - density) * 100)

    x = np.arange(len(matrices))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(
        x - width / 2,
        densities,
        width,
        label="Non-zero values"
    )

    ax.bar(
        x + width / 2,
        sparsities,
        width,
        label="Zero values"
    )

    ax.set_title("TF-IDF Matrix Density and Sparsity")
    ax.set_ylabel("Percentage (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(list(matrices.keys()))
    ax.set_ylim(0, 100)
    ax.legend()

    fig.tight_layout()

    output_path = output_dir / "tfidf_sparsity.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_tfidf_top_features(X_train_tfidf, vectorizer, output_dir):
    """Ve 20 tu co TF-IDF trung binh cao nhat tren tap Train."""

    feature_names = vectorizer.get_feature_names_out()

    mean_tfidf = np.asarray(
        X_train_tfidf.mean(axis=0)
    ).ravel()

    top_indices = np.argsort(mean_tfidf)[-20:]

    top_features = feature_names[top_indices]
    top_scores = mean_tfidf[top_indices]

    fig, ax = plt.subplots(figsize=(10, 7))

    bars = ax.barh(top_features, top_scores)

    ax.set_title("Top 20 Features by Mean TF-IDF Score")
    ax.set_xlabel("Mean TF-IDF Score")
    ax.set_ylabel("Terms")

    ax.bar_label(
        bars,
        fmt="%.4f",
        padding=3,
        fontsize=8
    )

    fig.tight_layout()

    output_path = output_dir / "tfidf_top_features.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


# =========================================================
# TRAINING VISUALIZATIONS
# =========================================================

def save_training_times(execution_times, output_dir):
    """So sanh thoi gian huan luyen cac mo hinh."""

    model_names = list(execution_times.keys())
    times = list(execution_times.values())

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(model_names, times)

    ax.set_title("Training Time Comparison")
    ax.set_xlabel("Machine Learning Models")
    ax.set_ylabel("Training Time (seconds)")
    ax.tick_params(axis="x", rotation=10)

    ax.bar_label(
        bars,
        labels=[f"{value:.2f}s" for value in times],
        padding=3
    )

    fig.tight_layout()

    output_path = output_dir / "training_time_comparison.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_training_configuration(model_config, output_dir):
    """Truc quan hoa tham so cua cac mo hinh."""

    configurations = [
        {
            "Model": "Logistic Regression",
            "Learning Rate": model_config.get(
                "logistic_regression", {}
            ).get("learning_rate", "N/A"),
            "Iterations": model_config.get(
                "logistic_regression", {}
            ).get("n_iterations", "N/A"),
            "Lambda": "N/A"
        },
        {
            "Model": "Naive Bayes",
            "Learning Rate": "N/A",
            "Iterations": "N/A",
            "Lambda": "N/A"
        },
        {
            "Model": "SVM",
            "Learning Rate": model_config.get(
                "svm", {}
            ).get("learning_rate", "N/A"),
            "Iterations": model_config.get(
                "svm", {}
            ).get("n_iterations", "N/A"),
            "Lambda": model_config.get(
                "svm", {}
            ).get("lambda_param", "N/A")
        }
    ]

    columns = [
        "Model",
        "Learning Rate",
        "Iterations",
        "Lambda"
    ]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table = ax.table(
        cellText=[
            [str(item[column]) for column in columns]
            for item in configurations
        ],
        colLabels=columns,
        cellLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.set_title("Model Training Configuration", pad=20)

    fig.tight_layout()

    output_path = output_dir / "training_configuration.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path


def save_training_dataset_overview(
    train_samples,
    features,
    output_dir
):
    """Truc quan kich thuoc du lieu huan luyen."""

    labels = [
        "Training Samples",
        "TF-IDF Features"
    ]

    values = [
        train_samples,
        features
    ]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, values)

    ax.set_title("Training Dataset Overview")
    ax.set_ylabel("Count")

    ax.bar_label(
        bars,
        labels=[f"{value:,}" for value in values],
        padding=3
    )

    fig.tight_layout()

    output_path = output_dir / "training_dataset_overview.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return output_path