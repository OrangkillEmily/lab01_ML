import re
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)


MODEL_DIR = ROOT_DIR / "models"


def clean_text(text):
    """Lam sach email giong preprocess.py."""
    text = str(text).lower()

    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
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


def decode_label(prediction):
    """Chuyen nhan so thanh Spam hoac Ham."""
    prediction = int(prediction)

    if prediction == 1:
        return "Spam"
    elif prediction == 0:
        return "Ham"
    else:
        return f"Nhãn không xác định: {prediction}"


def predict_email(email, vectorizer, models, run_id=None):
    """Du doan email bang ba mo hinh va ghi tracking."""

    is_new_run = run_id is None

    if is_new_run:
        run_id = create_run()

    start_time = time.perf_counter()

    try:
        # Step 1: Kiem tra va lam sach email
        step_start = time.perf_counter()

        if not isinstance(email, str) or not email.strip():
            raise ValueError(
                "Noi dung email khong duoc de trong."
            )

        cleaned_email = clean_text(email)

        if not cleaned_email:
            raise ValueError(
                "Email rong sau khi lam sach."
            )

        cleaning_time = time.perf_counter() - step_start

        update_progress(
            run_id,
            "PREDICTION - TEXT CLEANING",
            "SUCCESS",
            input_data={
                "raw_text": email,
                "input_length": len(email)
            },
            output_data={
                "cleaned_text": cleaned_email,
                "cleaned_length": len(cleaned_email),
                "cleaning_time_seconds": round(
                    cleaning_time, 4
                )
            }
        )

        write_log(
            run_id,
            "TEXT CLEANING",
            "SUCCESS",
            "Da lam sach email thanh cong."
        )

        # Step 2: Chuyen email sang TF-IDF
        step_start = time.perf_counter()

        X = vectorizer.transform([cleaned_email])

        tfidf_time = time.perf_counter() - step_start

        update_progress(
            run_id,
            "PREDICTION - TF-IDF",
            "SUCCESS",
            input_data={
                "cleaned_text": cleaned_email
            },
            output_data={
                "feature_shape": list(X.shape),
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "tfidf_time_seconds": round(
                    tfidf_time, 4
                )
            }
        )

        write_log(
            run_id,
            "TF-IDF",
            "SUCCESS",
            f"Vector TF-IDF co kich thuoc {X.shape}."
        )

        print("\nNoi dung sau khi lam sach:")
        print(cleaned_email)

        print("\nKich thuoc vector TF-IDF:")
        print(X.shape)

        # Step 3: Du doan bang cac mo hinh
        predictions = {}

        print("\nKet qua du doan:")

        for model_name, model in models.items():
            step_start = time.perf_counter()

            prediction = model.predict(X)[0]
            result = decode_label(prediction)

            prediction_time = time.perf_counter() - step_start

            predictions[model_name] = {
                "prediction": int(prediction),
                "label": result
            }

            update_progress(
                run_id,
                f"PREDICTION - {model_name}",
                "SUCCESS",
                input_data={
                    "model_name": model_name,
                    "feature_shape": list(X.shape)
                },
                output_data={
                    "prediction": int(prediction),
                    "label": result,
                    "prediction_time_seconds": round(
                        prediction_time, 4
                    )
                }
            )

            write_log(
                run_id,
                model_name,
                "SUCCESS",
                f"Du doan: {result}."
            )

            print(f"{model_name}: {result}")

        # Step 4: Tong hop ket qua
        total_time = time.perf_counter() - start_time

        result = {
            "run_id": run_id,
            "status": "SUCCESS",
            "timestamp": datetime.now().astimezone().isoformat(
                timespec="seconds"
            ),
            "input": {
                "raw_text": email,
                "cleaned_text": cleaned_email
            },
            "output": {
                "feature_shape": list(X.shape),
                "predictions": predictions,
                "total_prediction_time_seconds": round(
                    total_time, 4
                )
            }
        }

        update_progress(
            run_id,
            "PREDICTION",
            "SUCCESS",
            input_data={
                "models": list(models.keys()),
                "feature_shape": list(X.shape)
            },
            output_data={
                "predictions": predictions,
                "total_prediction_time_seconds": round(
                    total_time, 4
                )
            }
        )

        write_log(
            run_id,
            "PREDICTION",
            "SUCCESS",
            "Hoan thanh du doan email bang tat ca mo hinh."
        )

        if is_new_run:
            finish_run(run_id, "SUCCESS")

        return result

    except Exception as error:
        error_time = time.perf_counter() - start_time

        update_progress(
            run_id,
            "PREDICTION",
            "FAILED",
            input_data={
                "email_length": len(email)
                if isinstance(email, str) else None
            },
            output_data={
                "total_time_seconds": round(
                    error_time, 4
                )
            },
            error=str(error)
        )

        write_log(
            run_id,
            "PREDICTION",
            "FAILED",
            str(error)
        )

        if is_new_run:
            finish_run(run_id, "FAILED")

        raise


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
            result = predict_email(
                email,
                vectorizer,
                models
            )

            print("\nTrang thai:", result["status"])
            print("Run ID:", result["run_id"])

        except Exception as error:
            print(f"[ERROR] Khong the du doan: {error}")


if __name__ == "__main__":
    main()
