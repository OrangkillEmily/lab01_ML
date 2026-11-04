import re
import pickle
import time
from pathlib import Path

import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from tracking import create_run, update_progress, write_log, finish_run

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

    write_log(run_id, "PREPROCESSING", "START", "Bat dau tien xu ly du lieu")

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

        # Chi giu cac dong co nhan va noi dung
        df = df.dropna(subset=["label", "data"]).copy()
        rows_after_dropna = len(df)

        # Xoa email trung lap theo noi dung va nhan
        df = df.drop_duplicates(subset=["label", "data"])
        rows_after_duplicates = len(df)

        # Lam sach noi dung email
        df["data"] = df["data"].apply(clean_text)

        # Bo email rong sau khi lam sach
        df = df[df["data"].str.len() > 0]
        rows_after_cleaning = len(df)

        X = df["data"]
        y = df["label"]

        # Chia du lieu Train/Test theo cau hinh
        # Stratify giu ty le nhan gan voi du lieu ban dau
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )

        # Tao vectorizer TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english"
        )

        # Chi fit tren Train de tranh ro ri du lieu
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # Tao thu muc dau ra
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Luu 4 tap du lieu dang PKL
        output_data = {
            "X_train.pkl": X_train_tfidf,
            "X_test.pkl": X_test_tfidf,
            "y_train.pkl": y_train,
            "y_test.pkl": y_test
        }

        for filename, data in output_data.items():
            with open(OUTPUT_DIR / filename, "wb") as file:
                pickle.dump(data, file)

        # Luu du lieu da lam sach de kiem tra
        pd.DataFrame({
            "data": X_train,
            "label": y_train
        }).to_csv(OUTPUT_DIR / "train_clean.csv", index=False)

        pd.DataFrame({
            "data": X_test,
            "label": y_test
        }).to_csv(OUTPUT_DIR / "test_clean.csv", index=False)

        # Luu vectorizer TF-IDF
        with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as file:
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

        duration = round(time.perf_counter() - start_time, 4)

        # Luu ket qua tien xu ly vao progress_results/
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

        write_log(run_id, "PREPROCESSING", "FAILED", str(error))

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()