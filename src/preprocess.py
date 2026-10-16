
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


def clean_text(text):
    text = str(text).lower()

    # Bo cac ky tu xuong dong va khoang trang thua
    text = re.sub(r"\s+", " ", text)

    # Bo URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Chi giu chu cai va khoang trang
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

    df["data"] = df["data"].apply(clean_text)

    # Bo email rong sau khi lam sach
    df = df[df["data"].str.len() > 0]

    X = df["data"]
    y = df["label"]

    # Chia du lieu truoc khi hoc TF-IDF
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # TF-IDF chi fit tren tap train
    vectorizer = TfidfVectorizer(max_features=10000)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Luu ma tran dac trung
    joblib.dump(X_train_tfidf, OUTPUT_DIR / "X_train.joblib")
    joblib.dump(X_test_tfidf, OUTPUT_DIR / "X_test.joblib")

    # Luu nhan
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    # Luu du lieu da lam sach de kiem tra
    pd.DataFrame({
        "data": X_train,
        "label": y_train
    }).to_csv(OUTPUT_DIR / "train_clean.csv", index=False)

    pd.DataFrame({
        "data": X_test,
        "label": y_test
    }).to_csv(OUTPUT_DIR / "test_clean.csv", index=False)

    # Luu vectorizer de dung lai khi du doan
    joblib.dump(
        vectorizer,
        MODEL_DIR / "tfidf_vectorizer.joblib"
    )

    print("Da tien xu ly xong.")
    print("So mau train:", X_train_tfidf.shape[0])
    print("So mau test:", X_test_tfidf.shape[0])
    print("So dac trung TF-IDF:", X_train_tfidf.shape[1])


if __name__ == "__main__":
    main()
