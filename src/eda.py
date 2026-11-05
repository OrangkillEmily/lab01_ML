from pathlib import Path
import time

import pandas as pd
import matplotlib.pyplot as plt

from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)
from src.visualize import (
    create_stage_dir,
    save_label_distribution,
    save_missing_values,
    save_email_length_distribution
)

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
        # Doc du lieu
        df = pd.read_csv(DATA_PATH)

        # Luu kich thuoc du lieu goc
        dataset_shape = list(df.shape)

        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        # Tao thu muc hinh rieng cho run hien tai
        figure_dir = create_stage_dir(run_id, "eda")

        print("Kich thuoc du lieu:", df.shape)
        print("\nCac cot:", df.columns.tolist())

        # Thong ke gia tri thieu
        print("\nSo gia tri thieu:")

        missing_values = df[
            ["label", "data", "filename"]
        ].isnull().sum()

        print(missing_values)

        # Thong ke dong trung lap
        duplicate_rows = int(df.duplicated().sum())

        print("\nSo dong trung lap:", duplicate_rows)

        # Thong ke phan bo nhan
        print("\nPhan bo nhan:")

        label_distribution = df["label"].value_counts().sort_index()

        print(label_distribution)

        # Luu thong ke nhan vao reports/
        label_distribution.to_csv(
            REPORT_DIR / "label_distribution.csv",
            header=["count"]
        )

        # Hinh 1: Phan bo nhan
        save_label_distribution(
            label_distribution,
            figure_dir
        )

        # Hinh 2: Gia tri thieu
        save_missing_values(
            missing_values,
            figure_dir
        )

        # Hinh 3: Phan phoi do dai email
        df["email_length"] = (
            df["data"]
            .fillna("")
            .astype(str)
            .str.len()
        )

        print("\nThong ke do dai email:")

        email_length = df["email_length"].describe()

        print(email_length)

        save_email_length_distribution(
            df["email_length"],
            figure_dir
        )

        # Giu lai bieu do phan bo nhan o vi tri cu
        label_distribution.plot(
            kind="bar",
            figsize=(7, 5)
        )

        plt.title("Phân bố email theo nhãn trong dữ liệu thô")
        plt.xlabel("Nhãn email")
        plt.ylabel("Số lượng email")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            IMAGE_DIR / "label_distribution.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("\nDa luu file CSV trong thu muc reports/")
        print("Da luu bieu do trong thu muc docs/images/")
        print("Da luu bo hinh EDA tai:", figure_dir)

        duration = round(time.perf_counter() - start_time, 4)

        # Danh sach hinh da tao
        saved_figures = [
            str(figure_dir / "label_distribution.png"),
            str(figure_dir / "missing_values.png"),
            str(figure_dir / "email_length_distribution.png")
        ]

        # Luu ket qua EDA vao progress_results/
        update_progress(
            run_id,
            "EDA",
            "SUCCESS",
            input_data={
                "dataset": str(DATA_PATH)
            },
            output_data={
                "dataset_shape": dataset_shape,
                "columns": [
                    "label",
                    "data",
                    "filename"
                ],
                "missing_values": missing_values.to_dict(),
                "duplicate_rows": duplicate_rows,
                "label_distribution": label_distribution.to_dict(),
                "email_length_statistics": email_length.to_dict(),
                "saved_files": [
                    "reports/label_distribution.csv",
                    "docs/images/label_distribution.png"
                ],
                "saved_figures": saved_figures,
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

        write_log(
            run_id,
            "EDA",
            "FAILED",
            str(error)
        )

        if own_run:
            finish_run(run_id, "FAILED")

        raise


if __name__ == "__main__":
    main()