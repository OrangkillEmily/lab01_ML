from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
REPORT_DIR = ROOT_DIR / "reports"
IMAGE_DIR = ROOT_DIR / "docs" / "images"


def main():
    df = pd.read_csv(DATA_PATH)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    print("Kich thuoc du lieu:", df.shape)
    print("\nCac cot:", df.columns.tolist())

    print("\nSo gia tri thieu:")
    print(df[["label", "data", "filename"]].isnull().sum())

    print("\nSo dong trung lap:", df.duplicated().sum())

    print("\nPhan bo nhan:")
    print(df["label"].value_counts().sort_index())

    # Luu thong ke nhan vao reports/
    df["label"].value_counts().sort_index().to_csv(
        REPORT_DIR / "label_distribution.csv",
        header=["count"]
    )

    # Ve bieu do phan bo nhan
    df["label"].value_counts().sort_index().plot(
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
    print(df["email_length"].describe())

    print("\nDa luu file CSV trong thu muc reports/")
    print("Da luu bieu do trong thu muc docs/images/")


if __name__ == "__main__":
    main()
