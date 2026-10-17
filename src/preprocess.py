
import re
import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


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


def main():
    df = pd.read_csv(DATA_PATH)

    # Chi giu cac dong co nhan va noi dung
    df = df.dropna(subset=["label", "data"]).copy()

    # Xoa email trung lap theo noi dung va nhan
    df = df.drop_duplicates(subset=["label", "data"])

    # Lam sach noi dung email
    df["data"] = df["data"].apply(clean_text)

    # Bo email rong sau khi lam sach
    df = df[df["data"].str.len() > 0]

    X = df["data"]
    y = df["label"]

    # Chia du lieu train/test theo ty le 80/20
    # Thuc hien truoc khi huan luyen TF-IDF de tranh ro ri du lieu
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Chi fit TF-IDF tren tap train
    # Loai bo stop-words tieng Anh
    vectorizer = TfidfVectorizer(
        max_features=10000,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Tao thu muc dau ra
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Luu du 4 tap du lieu dang PKL
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

    # Luu mo hinh TF-IDF dang PKL
    with open(MODEL_DIR / "tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file)

    print("Da tien xu ly xong.")
    print("So mau train:", X_train_tfidf.shape[0])
    print("So mau test:", X_test_tfidf.shape[0])
    print("So dac trung TF-IDF:", X_train_tfidf.shape[1])
    print("Da luu 4 tap du lieu dang .pkl trong data/processed/")
    print("Da luu TF-IDF trong models/tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
