import sys
import csv
import time
from pathlib import Path
from datetime import datetime

import yaml
import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.model import LogisticRegression, NaiveBayes, SVM
from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)

from src.visualize import (
    create_stage_dir,
    save_training_times,
    save_training_configuration,
    save_training_dataset_overview
)


PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"
LOG_FILE = ROOT_DIR / "docs" / "experiment_logs.csv"


def load_train_data():
    """Doc du lieu huan luyen tu data/processed/."""

    print("[INFO] Dang tai du lieu huan luyen...")

    x_joblib_path = PROCESSED_DIR / "X_train.joblib"
    x_pkl_path = PROCESSED_DIR / "X_train.pkl"

    if x_joblib_path.exists():
        X_train = joblib.load(x_joblib_path)

    elif x_pkl_path.exists():
        X_train = joblib.load(x_pkl_path)

    else:
        raise FileNotFoundError(
            f"Khong tim thay X_train trong {PROCESSED_DIR}"
        )

    y_csv_path = PROCESSED_DIR / "y_train.csv"
    y_pkl_path = PROCESSED_DIR / "y_train.pkl"

    if y_csv_path.exists():
        y_train = pd.read_csv(y_csv_path).values

    elif y_pkl_path.exists():
        y_train = joblib.load(y_pkl_path)

    else:
        raise FileNotFoundError(
            f"Khong tim thay y_train trong {PROCESSED_DIR}"
        )

    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    else:
        X_train = np.asarray(X_train)

    y_train = np.asarray(y_train).ravel().astype(int)

    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(
            "So luong mau X_train va y_train khong khop."
        )

    print(
        f"[INFO] Tai thanh cong! "
        f"X_train: {X_train.shape}, y_train: {y_train.shape}"
    )

    return X_train, y_train


def load_config():
    """Doc cau hinh tu config.yaml."""

    config_path = ROOT_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config


def log_experiment(log_file, run_id, hyperparams, execution_times):
    """
    Ghi lich su thuc nghiem.
    Moi lan chay duoc gan mot Run_ID rieng.
    Du lieu cu duoc giu lai voi nhan LEGACY.
    """

    log_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Run_ID",
        "Timestamp",
        "LR_Params",
        "NB_Params",
        "SVM_Params",
        "LR_Train_Time_Sec",
        "NB_Train_Time_Sec",
        "SVM_Train_Time_Sec",
        "Total_Train_Time_Sec"
    ]

    new_row = {
        "Run_ID": str(run_id),
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "LR_Params": str(
            hyperparams.get("logistic_regression", {})
        ),

        "NB_Params": str(
            hyperparams.get("naive_bayes", {})
        ),

        "SVM_Params": str(
            hyperparams.get("svm", {})
        ),

        "LR_Train_Time_Sec": round(
            execution_times.get("Logistic Regression", 0), 2
        ),

        "NB_Train_Time_Sec": round(
            execution_times.get("Naive Bayes", 0), 2
        ),

        "SVM_Train_Time_Sec": round(
            execution_times.get("SVM", 0), 2
        ),

        "Total_Train_Time_Sec": round(
            sum(execution_times.values()), 2
        )
    }

    old_rows = []

    # -------------------------------------------------
    # DOC DU LIEU LICH SU CU
    # -------------------------------------------------

    if log_file.exists() and log_file.stat().st_size > 0:

        with open(
            log_file,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            rows = list(csv.reader(file))

        if rows:

            # Truong hop file da co header moi
            if rows[0] == fieldnames:

                for values in rows[1:]:

                    if not values:
                        continue

                    values = (
                        values + [""] * len(fieldnames)
                    )[:len(fieldnames)]

                    old_rows.append(
                        dict(zip(fieldnames, values))
                    )

            else:

                old_fieldnames = fieldnames[1:]

                # Truong hop file cu co header nhung chua co Run_ID
                if rows[0] == old_fieldnames:
                    rows = rows[1:]

                # Chuyen du lieu cu sang schema moi
                for values in rows:

                    if not values:
                        continue

                    values = (
                        values + [""] * len(old_fieldnames)
                    )[:len(old_fieldnames)]

                    old_rows.append({
                        "Run_ID": "LEGACY",
                        **dict(zip(old_fieldnames, values))
                    })

    # -------------------------------------------------
    # LOAI BO DU LIEU TRUNG CUNG RUN_ID
    # -------------------------------------------------

    old_rows = [
        row for row in old_rows
        if row.get("Run_ID") != str(run_id)
    ]

    # -------------------------------------------------
    # GHI LAI CSV VOI HEADER CHUAN
    # -------------------------------------------------

    with open(
        log_file,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(new_row)

    print(
        f"[LOG] Da luu lich su thuc nghiem: {log_file}"
    )


def main(run_id=None):

    own_run = run_id is None

    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()

    write_log(
        run_id,
        "TRAINING",
        "START",
        "Bat dau huan luyen mo hinh"
    )

    try:

        # -------------------------------------------------
        # STEP 1: LOAD TRAINING DATA
        # -------------------------------------------------

        X_train, y_train = load_train_data()

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        figure_dir = create_stage_dir(
            run_id,
            "training"
        )

        # -------------------------------------------------
        # STEP 2: LOAD CONFIGURATION
        # -------------------------------------------------

        config = load_config()
        model_config = config["models"]

        execution_times = {}

        # -------------------------------------------------
        # STEP 3: INITIALIZE MODELS
        # -------------------------------------------------

        models = [

            (
                "Logistic Regression",

                LogisticRegression(
                    learning_rate=model_config[
                        "logistic_regression"
                    ]["learning_rate"],

                    n_iterations=model_config[
                        "logistic_regression"
                    ]["n_iterations"]
                ),

                MODEL_DIR / "model_lr.pkl",

                model_config["logistic_regression"]
            ),

            (
                "Naive Bayes",

                NaiveBayes(),

                MODEL_DIR / "model_nb.pkl",

                {}
            ),

            (
                "SVM",

                SVM(
                    learning_rate=model_config[
                        "svm"
                    ]["learning_rate"],

                    lambda_param=model_config[
                        "svm"
                    ]["lambda_param"],

                    n_iterations=model_config[
                        "svm"
                    ]["n_iterations"]
                ),

                MODEL_DIR / "model_svm.pkl",

                model_config["svm"]
            )
        ]

        print("\n" + "=" * 55)
        print("BAT DAU HUAN LUYEN 3 MO HINH (FROM SCRATCH)")
        print("=" * 55)

        # -------------------------------------------------
        # STEP 4: TRAIN MODELS
        # -------------------------------------------------

        for model_name, model, save_path, parameters in models:

            print(
                f"\n[TRAIN] Dang huan luyen mo hinh: {model_name}..."
            )

            model_start = time.perf_counter()

            try:

                model.fit(X_train, y_train)

                elapsed_time = time.perf_counter() - model_start

                execution_times[model_name] = elapsed_time

                joblib.dump(model, save_path)

                print(
                    f"[DONE] Hoan thanh {model_name} "
                    f"trong {elapsed_time:.2f} giay."
                )

                print(
                    f"[SAVE] Da luu mo hinh tai: {save_path}"
                )

                update_progress(
                    run_id,
                    f"TRAINING - {model_name}",
                    "SUCCESS",

                    input_data={
                        "train_shape": list(X_train.shape),
                        "label_shape": list(y_train.shape),
                        "parameters": parameters
                    },

                    output_data={
                        "model_name": model_name,
                        "status": "TRAINED",
                        "train_samples": X_train.shape[0],
                        "features": X_train.shape[1],

                        "training_time_seconds": round(
                            elapsed_time, 4
                        ),

                        "saved_model": str(save_path)
                    }
                )

                write_log(
                    run_id,
                    f"TRAINING - {model_name}",
                    "SUCCESS",
                    f"Training completed in {elapsed_time:.2f} seconds"
                )

            except Exception as error:

                update_progress(
                    run_id,
                    f"TRAINING - {model_name}",
                    "FAILED",

                    input_data={
                        "train_shape": list(X_train.shape),
                        "parameters": parameters
                    },

                    error=str(error)
                )

                write_log(
                    run_id,
                    f"TRAINING - {model_name}",
                    "FAILED",
                    str(error)
                )

                raise

        # -------------------------------------------------
        # STEP 5: VISUALIZE TRAINING RESULTS
        # -------------------------------------------------

        print("\n[INFO] Dang tao bieu do Training...")

        save_training_times(
            execution_times,
            figure_dir
        )

        save_training_configuration(
            model_config,
            figure_dir
        )

        save_training_dataset_overview(
            train_samples=X_train.shape[0],
            features=X_train.shape[1],
            output_dir=figure_dir
        )

        saved_figures = [
            str(figure_dir / "training_time_comparison.png"),
            str(figure_dir / "training_configuration.png"),
            str(figure_dir / "training_dataset_overview.png")
        ]

        print(
            f"[DONE] Da luu 3 bieu do tai: {figure_dir}"
        )

        # -------------------------------------------------
        # STEP 6: SAVE EXPERIMENT HISTORY
        # -------------------------------------------------

        log_experiment(
            LOG_FILE,
            run_id,
            model_config,
            execution_times
        )

        total_time = round(
            time.perf_counter() - start_time,
            4
        )

        # -------------------------------------------------
        # STEP 7: TRACK TRAINING RESULTS
        # -------------------------------------------------

        update_progress(
            run_id,
            "TRAINING",
            "SUCCESS",

            input_data={
                "train_samples": X_train.shape[0],
                "features": X_train.shape[1],
                "models": list(execution_times.keys())
            },

            output_data={
                "execution_times": {
                    name: round(value, 4)
                    for name, value in execution_times.items()
                },

                "total_training_time": total_time,

                "saved_models": [
                    "models/model_lr.pkl",
                    "models/model_nb.pkl",
                    "models/model_svm.pkl"
                ],

                "saved_figures": saved_figures,

                "experiment_log": "docs/experiment_logs.csv"
            }
        )

        write_log(
            run_id,
            "TRAINING",
            "SUCCESS",
            f"Hoan thanh training trong {total_time} giay"
        )

        # -------------------------------------------------
        # STEP 8: DISPLAY SUMMARY
        # -------------------------------------------------

        print("\n" + "=" * 55)
        print("HOAN TAT QUA TRINH HUAN LUYEN VA LUU MO HINH!")
        print("=" * 55)

        for name, elapsed in execution_times.items():
            print(f"{name}: {elapsed:.2f} giay")

        print(
            f"Tong thoi gian huan luyen: "
            f"{sum(execution_times.values()):.2f} giay"
        )

        print(
            f"Thu muc bieu do: {figure_dir}"
        )

        print("=" * 55)

        if own_run:
            finish_run(run_id)

    except Exception as error:

        write_log(
            run_id,
            "TRAINING",
            "FAILED",
            str(error)
        )

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()
