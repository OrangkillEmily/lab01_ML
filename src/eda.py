from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from tracking import create_run, update_progress, write_log, finish_run

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
REPORT_DIR = ROOT_DIR / "reports"
IMAGE_DIR = ROOT_DIR / "docs" / "images"


def main(run_id=None):
    own_run = run_id is None

    if own_run:
        run_id = create_run()

    start_time = time.perf_counter()

    write_log(run_id, "EDA", "START", "Bat dau phan tich du lieu")

    try:
        df = pd.read_csv(DATA_PATH)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        print("Kich thuoc du lieu:", df.shape)
        print("\nCac cot:", df.columns.tolist())

        print("\nSo gia tri thieu:")
        missing_values = df[["label", "data", "filename"]].isnull().sum()
        print(missing_values)

        duplicate_rows = int(df.duplicated().sum())
        print("\nSo dong trung lap:", duplicate_rows)

        print("\nPhan bo nhan:")
        label_distribution = df["label"].value_counts().sort_index()
        print(label_distribution)

        # Luu thong ke nhan vao reports/
        label_distribution.to_csv(
            REPORT_DIR / "label_distribution.csv",
            header=["count"]
        )

        # Ve bieu do phan bo nhan
        label_distribution.plot(
            kind="bar",
            figsize=(7, 5)
        )

        plt.title("Phân bố email theo nhãn trong dữ liệu thô")
        plt.xlabel("Nhãn email")
        plt.ylabel("Số lượng email")
        plt.xticks(rotation=0)
        plt.tight_layout()

        # Luu bieu do vao docs/images/
        plt.savefig(
            IMAGE_DIR / "label_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

        # Thong ke do dai email
        df["email_length"] = df["data"].fillna("").astype(str).str.len()

        print("\nThong ke do dai email:")
        email_length = df["email_length"].describe()
        print(email_length)

        print("\nDa luu file CSV trong thu muc reports/")
        print("Da luu bieu do trong thu muc docs/images/")

        duration = round(time.perf_counter() - start_time, 4)

        # Luu ket qua EDA vao progress_results/
        update_progress(
            run_id,
            "EDA",
            "SUCCESS",
            input_data={
                "dataset": str(DATA_PATH)
            },
            output_data={
                "dataset_shape": list(df.shape),
                "columns": df.columns.tolist(),
                "missing_values": missing_values.to_dict(),
                "duplicate_rows": duplicate_rows,
                "label_distribution": label_distribution.to_dict(),
                "email_length_statistics": email_length.to_dict(),
                "saved_files": [
                    "reports/label_distribution.csv",
                    "docs/images/label_distribution.png"
                ],
                "duration_seconds": duration
            }
        )

        # Ghi log
        write_log(
            run_id,
            "EDA",
            "SUCCESS",
            f"Hoan thanh EDA trong {duration} giay"
        )

        if own_run:
            finish_run(run_id)

    except Exception as error:
        update_progress(
            run_id,
            "EDA",
            "FAILED",
            error=str(error)
        )

        write_log(run_id, "EDA", "FAILED", str(error))

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()