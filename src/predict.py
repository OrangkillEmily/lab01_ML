import re
import sys
from pathlib import Path

import joblib


# Thu muc goc cua project
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_DIR = ROOT_DIR / "models"


def clean_text(text):
    """Lam sach email giong preprocess.py."""
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


def load_model(filename):
    """Tai vectorizer hoac mo hinh."""
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {path}"
        )

    print(f"[INFO] Dang tai: {filename}")
    return joblib.load(path)


def load_models():
    """Tai TF-IDF va ba mo hinh."""
    vectorizer = load_model("tfidf_vectorizer.pkl")

    models = {
        "Logistic Regression": load_model("model_lr.pkl"),
        "Naive Bayes": load_model("model_nb.pkl"),
        "SVM": load_model("model_svm.pkl"),
    }

    return vectorizer, models


def predict_email(email, vectorizer, models):
    """Du doan email la Spam hay Ham."""
    cleaned_email = clean_text(email)

    if not cleaned_email:
        print("[WARN] Email rong sau khi lam sach.")
        return

    # Chuyen email sang TF-IDF
    X = vectorizer.transform([cleaned_email])

    print("\nNoi dung sau khi lam sach:")
    print(cleaned_email)

    print("\nKet qua du doan:")

    for model_name, model in models.items():
        prediction = model.predict(X)[0]

        # Gia dinh: 0 = Ham, 1 = Spam
        if prediction == 1:
            result = "Spam"
        elif prediction == 0:
            result = "Ham"
        else:
            result = f"Nhãn không xác định: {prediction}"

        print(f"{model_name}: {result}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 50)
    print("SPAMASSASSIN - EMAIL SPAM PREDICTION")
    print("=" * 50)

    try:
        vectorizer, models = load_models()
    except Exception as error:
        print(f"[ERROR] Khong the tai mo hinh: {error}")
        return

    while True:
        email = input(
            "\nNhap email (go 'exit' de thoat): "
        ).strip()

        if email.lower() == "exit":
            print("Ket thuc chuong trinh.")
            break

        if not email:
            print("Vui long nhap noi dung email.")
            continue

        try:
            predict_email(email, vectorizer, models)
        except Exception as error:
            print(f"[ERROR] Khong the du doan: {error}")


if __name__ == "__main__":
    main()
