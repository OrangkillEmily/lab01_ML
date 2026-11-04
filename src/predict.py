import argparse
import csv
import json
import sys
import time
from pathlib import Path
from datetime import datetime

import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocess import clean_text
from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)

MODEL_DIR = ROOT_DIR / "models"
REPORT_DIR = ROOT_DIR / "reports"
HISTORY_PATH = REPORT_DIR / "inference_history.csv"


def load_model(filename, verbose=True):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file: {path}"
        )

    if verbose:
        print(f"[INFO] Dang tai: {filename}")

    return joblib.load(path)


def load_models(verbose=True):
    vectorizer = load_model(
        "tfidf_vectorizer.pkl", verbose
    )

    models = {
        "Logistic Regression": load_model(
            "model_lr.pkl", verbose
        ),
        "Naive Bayes": load_model(
            "model_nb.pkl", verbose
        ),
        "SVM": load_model(
            "model_svm.pkl", verbose
        ),
    }

    return vectorizer, models


def decode_label(prediction):
    prediction = int(prediction)

    if prediction == 1:
        return "Spam"

    if prediction == 0:
        return "Ham"

    return f"Nhãn không xác định: {prediction}"


def save_inference_history(result):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    predictions = result["output"]["predictions"]

    row = {
        "timestamp": result["timestamp"],
        "run_id": result["run_id"],
        "email": result["input"]["raw_text"],
        "cleaned_email": result["input"]["cleaned_text"],
        "logistic_regression": predictions[
            "Logistic Regression"
        ]["label"],
        "naive_bayes": predictions[
            "Naive Bayes"
        ]["label"],
        "svm": predictions["SVM"]["label"],
        "total_time_seconds": result["output"][
            "total_prediction_time_seconds"
        ],
    }

    file_exists = HISTORY_PATH.exists()

    with open(
        HISTORY_PATH,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(row.keys())
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def predict_email(email, vectorizer, models, run_id=None):
    is_new_run = run_id is None

    if is_new_run:
        run_id = create_run()

    start_time = time.perf_counter()

    try:
        if not isinstance(email, str) or not email.strip():
            raise ValueError(
                "Noi dung email khong duoc de trong."
            )

        # Làm sạch email
        step_start = time.perf_counter()

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
                "input_length": len(email)
            },
            output_data={
                "cleaned_text": cleaned_email,
                "cleaned_length": len(cleaned_email),
                "cleaning_time_seconds": round(
                    cleaning_time, 6
                )
            }
        )

        write_log(
            run_id,
            "TEXT CLEANING",
            "SUCCESS",
            "Da lam sach email thanh cong."
        )

        # Chuyển email sang TF-IDF
        step_start = time.perf_counter()

        X = vectorizer.transform([cleaned_email])

        tfidf_time = time.perf_counter() - step_start

        update_progress(
            run_id,
            "PREDICTION - TF-IDF",
            "SUCCESS",
            output_data={
                "feature_shape": list(X.shape),
                "samples": int(X.shape[0]),
                "features": int(X.shape[1]),
                "tfidf_time_seconds": round(
                    tfidf_time, 6
                )
            }
        )

        write_log(
            run_id,
            "TF-IDF",
            "SUCCESS",
            f"Vector TF-IDF co kich thuoc {X.shape}."
        )

        # Dự đoán bằng từng mô hình
        predictions = {}

        for model_name, model in models.items():
            step_start = time.perf_counter()

            prediction = model.predict(X)[0]

            prediction_time = (
                time.perf_counter() - step_start
            )

            predictions[model_name] = {
                "prediction": int(prediction),
                "label": decode_label(prediction),
                "prediction_time_seconds": round(
                    prediction_time, 6
                )
            }

            update_progress(
                run_id,
                f"PREDICTION - {model_name}",
                "SUCCESS",
                input_data={
                    "model_name": model_name,
                    "feature_shape": list(X.shape)
                },
                output_data=predictions[model_name]
            )

            write_log(
                run_id,
                model_name,
                "SUCCESS",
                f"Du doan: {decode_label(prediction)}."
            )

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
                "cleaning_time_seconds": round(
                    cleaning_time, 6
                ),
                "tfidf_time_seconds": round(
                    tfidf_time, 6
                ),
                "predictions": predictions,
                "total_prediction_time_seconds": round(
                    total_time, 6
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
            output_data=result["output"]
        )

        write_log(
            run_id,
            "PREDICTION",
            "SUCCESS",
            "Hoan thanh du doan email bang tat ca mo hinh."
        )

        save_inference_history(result)

        if is_new_run:
            finish_run(run_id, "SUCCESS")

        return result

    except Exception as error:
        error_time = time.perf_counter() - start_time

        update_progress(
            run_id,
            "PREDICTION",
            "FAILED",
            output_data={
                "total_time_seconds": round(
                    error_time, 6
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


def display_result(result):
    print("\nNoi dung sau khi lam sach:")
    print(result["input"]["cleaned_text"])

    print("\nKich thuoc vector TF-IDF:")
    print(tuple(result["output"]["feature_shape"]))

    print("\nKet qua du doan:")

    for model_name, prediction in result["output"][
        "predictions"
    ].items():
        print(
            f"{model_name}: {prediction['label']} "
            f"({prediction['prediction_time_seconds']:.6f}s)"
        )

    print("\nTrang thai:", result["status"])
    print("Run ID:", result["run_id"])

    print(
        "Tong thoi gian du doan:",
        result["output"]["total_prediction_time_seconds"],
        "giay"
    )


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Du doan email Spam/Ham bang 3 mo hinh."
    )

    parser.add_argument(
        "--text",
        type=str,
        help="Noi dung email can du doan."
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Xuat ket qua duoi dang JSON."
    )

    args = parser.parse_args()

    if not args.json:
        print("=" * 50)
        print("SPAMASSASSIN - EMAIL SPAM PREDICTION")
        print("=" * 50)

    try:
        vectorizer, models = load_models(
            verbose=not args.json
        )

    except Exception as error:
        if args.json:
            print(json.dumps({
                "status": "FAILED",
                "error": str(error)
            }, ensure_ascii=False))
        else:
            print(f"[ERROR] Khong the tai mo hinh: {error}")

        sys.exit(1)

    def process_email(email):
        try:
            result = predict_email(
                email,
                vectorizer,
                models
            )

            if args.json:
                print(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        indent=4
                    )
                )
            else:
                display_result(result)

        except Exception as error:
            if args.json:
                print(json.dumps({
                    "status": "FAILED",
                    "error": str(error)
                }, ensure_ascii=False))
            else:
                print(f"[ERROR] Khong the du doan: {error}")

    # Chế độ dự đoán một email
    if args.text is not None:
        process_email(args.text)
        return

    # Chế độ nhập email liên tục
    while True:
        try:
            email = input(
                "\nNhap email (go 'exit' de thoat): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nKet thuc chuong trinh.")
            break

        if email.lower() == "exit":
            print("Ket thuc chuong trinh.")
            break

        if not email:
            print("Vui long nhap noi dung email.")
            continue

        process_email(email)


if __name__ == "__main__":
    main()
