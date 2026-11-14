import re
import pickle
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)

from src.visualize import (
    create_stage_dir,
    save_cleaning_comparison,
    save_preprocessing_steps,
    save_train_test_distribution,
    save_tfidf_sparsity,
    save_tfidf_top_features
)


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
CONFIG_PATH = ROOT_DIR / "config.yaml"


def clean_text(text):
    text = str(text).lower()

    # Xoa the HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Bo URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Chi giu chu cai tieng Anh va khoang trang
    text = re.sub(r"[^a-z\s]", " ", text)

    # Chuan hoa khoang trang
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main(run_id=None):
    own_run = run_id is None

    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()

    write_log(
        run_id,
        "PREPROCESSING",
        "START",
        "Bat dau tien xu ly du lieu"
    )

    try:
        # Doc cau hinh tu config.yaml
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}

        data_config = config.get("data", {})
        tfidf_config = config.get("tfidf", {})

        # Lay tham so chia Train/Test
        test_size = float(data_config.get("test_size", 0.2))
        random_state = int(data_config.get("random_state", 42))

        # Lay tham so TF-IDF
        max_features = int(tfidf_config.get("max_features", 10000))
        ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))

        # Doc du lieu
        df = pd.read_csv(DATA_PATH)

        original_rows = len(df)

        # Tao thu muc luu hinh cho run hien tai
        figure_dir = create_stage_dir(run_id, "preprocessing")

        # -------------------------------------------------
        # STEP 1: LOAI BO GIA TRI THIEU
        # -------------------------------------------------

        df = df.dropna(
            subset=["label", "data"]
        ).copy()

        rows_after_dropna = len(df)

        # -------------------------------------------------
        # STEP 2: LOAI BO EMAIL TRUNG LAP
        # -------------------------------------------------

        df = df.drop_duplicates(
            subset=["label", "data"]
        )

        rows_after_duplicates = len(df)

        # Luu lai email goc de so sanh truoc/sau lam sach
        raw_texts = df["data"].copy()

        # -------------------------------------------------
        # STEP 3: LAM SACH NOI DUNG EMAIL
        # -------------------------------------------------

        df["data"] = df["data"].apply(clean_text)

        # Chon mot email hop le de minh hoa
        valid_indices = df.index[df["data"].str.len() > 0]

        if len(valid_indices) > 0:
            sample_index = valid_indices[0]

            raw_sample = raw_texts.loc[sample_index]
            cleaned_sample = df.loc[sample_index, "data"]
        else:
            raw_sample = ""
            cleaned_sample = ""

        # -------------------------------------------------
        # STEP 4: LOAI BO EMAIL RONG
        # -------------------------------------------------

        df = df[df["data"].str.len() > 0].copy()

        rows_after_cleaning = len(df)

        X = df["data"]
        y = df["label"]

        # -------------------------------------------------
        # VISUALIZATION: CAC BUOC TIEN XU LY
        # -------------------------------------------------

        step_counts = {
            "Original": original_rows,
            "After Drop NA": rows_after_dropna,
            "After Deduplication": rows_after_duplicates,
            "After Cleaning": rows_after_cleaning
        }

        save_preprocessing_steps(
            step_counts,
            figure_dir
        )

        save_cleaning_comparison(
            raw_sample,
            cleaned_sample,
            figure_dir
        )

        # -------------------------------------------------
        # STEP 5: CHIA TRAIN / TEST
        # -------------------------------------------------

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        save_train_test_distribution(
            y_train,
            y_test,
            figure_dir
        )

        # -------------------------------------------------
        # STEP 6: CHUYEN VAN BAN THANH TF-IDF
        # -------------------------------------------------

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )

        # Chi fit tren Train de tranh ro ri du lieu
        X_train_tfidf = vectorizer.fit_transform(X_train)

        # Test chi transform, khong fit lai
        X_test_tfidf = vectorizer.transform(X_test)

        # -------------------------------------------------
        # VISUALIZATION: KET QUA TF-IDF
        # -------------------------------------------------

        save_tfidf_sparsity(
            X_train_tfidf,
            X_test_tfidf,
            figure_dir
        )

        save_tfidf_top_features(
            X_train_tfidf,
            vectorizer,
            figure_dir
        )

        # -------------------------------------------------
        # STEP 7: LUU DU LIEU VA MODEL
        # -------------------------------------------------

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        output_data = {
            "X_train.pkl": X_train_tfidf,
            "X_test.pkl": X_test_tfidf,
            "y_train.pkl": y_train,
            "y_test.pkl": y_test
        }

        for filename, data in output_data.items():
            with open(OUTPUT_DIR / filename, "wb") as file:
                pickle.dump(data, file)

        # Luu du lieu da lam sach
        pd.DataFrame({
            "data": X_train,
            "label": y_train
        }).to_csv(
            OUTPUT_DIR / "train_clean.csv",
            index=False
        )

        pd.DataFrame({
            "data": X_test,
            "label": y_test
        }).to_csv(
            OUTPUT_DIR / "test_clean.csv",
            index=False
        )

        # Luu vectorizer TF-IDF
        with open(
            MODEL_DIR / "tfidf_vectorizer.pkl",
            "wb"
        ) as file:
            pickle.dump(vectorizer, file)

        print("Da tien xu ly xong.")
        print("So mau train:", X_train_tfidf.shape[0])
        print("So mau test:", X_test_tfidf.shape[0])
        print("So dac trung TF-IDF:", X_train_tfidf.shape[1])
        print("Test size:", test_size)
        print("Random state:", random_state)
        print("Max features:", max_features)
        print("N-gram range:", ngram_range)
        print("Da luu 4 tap du lieu dang .pkl trong data/processed/")
        print("Da luu TF-IDF trong models/tfidf_vectorizer.pkl")
        print("Da luu hinh tien xu ly tai:", figure_dir)

        duration = round(
            time.perf_counter() - start_time,
            4
        )

        # Danh sach hinh da tao
        saved_figures = [
            str(figure_dir / "cleaning_comparison.png"),
            str(figure_dir / "preprocessing_steps.png"),
            str(figure_dir / "train_test_distribution.png"),
            str(figure_dir / "tfidf_sparsity.png"),
            str(figure_dir / "tfidf_top_features.png")
        ]

        # -------------------------------------------------
        # TRACKING: PROGRESS RESULTS
        # -------------------------------------------------

        update_progress(
            run_id,
            "PREPROCESSING",
            "SUCCESS",
            input_data={
                "dataset": str(DATA_PATH),
                "config": str(CONFIG_PATH),
                "original_rows": original_rows
            },
            output_data={
                "rows_after_dropna": rows_after_dropna,
                "rows_after_duplicates": rows_after_duplicates,
                "rows_after_cleaning": rows_after_cleaning,
                "train_samples": X_train_tfidf.shape[0],
                "test_samples": X_test_tfidf.shape[0],
                "train_shape": list(X_train_tfidf.shape),
                "test_shape": list(X_test_tfidf.shape),
                "tfidf_features": X_train_tfidf.shape[1],
                "parameters": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "max_features": max_features,
                    "ngram_range": list(ngram_range)
                },
                "saved_files": [
                    "data/processed/X_train.pkl",
                    "data/processed/X_test.pkl",
                    "data/processed/y_train.pkl",
                    "data/processed/y_test.pkl",
                    "data/processed/train_clean.csv",
                    "data/processed/test_clean.csv",
                    "models/tfidf_vectorizer.pkl"
                ],
                "saved_figures": saved_figures,
                "duration_seconds": duration
            }
        )

        # Ghi log
        write_log(
            run_id,
            "PREPROCESSING",
            "SUCCESS",
            f"Hoan thanh tien xu ly trong {duration} giay"
        )

        if own_run:
            finish_run(run_id)

    except Exception as error:
        update_progress(
            run_id,
            "PREPROCESSING",
            "FAILED",
            error=str(error)
        )

        write_log(
            run_id,
            "PREPROCESSING",
            "FAILED",
            str(error)
        )

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()