
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "raw" / "SpamAssassin.csv"
OUTPUT_DIR = ROOT_DIR / "reports"


def main():
    df = pd.read_csv(DATA_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Kich thuoc du lieu:", df.shape)
    print("\nCac cot:", df.columns.tolist())

    print("\nSo gia tri thieu:")
    print(df[["label", "data", "filename"]].isnull().sum())

    print("\nSo dong trung lap:", df.duplicated().sum())

    print("\nPhan bo nhan:")
    print(df["label"].value_counts().sort_index())

    # Luu thong ke nhan
    df["label"].value_counts().sort_index().to_csv(
        OUTPUT_DIR / "label_distribution.csv",
        header=["count"]
    )

    # Ve bieu do
    df["label"].value_counts().sort_index().plot(
        kind="bar",
        figsize=(7, 5)
    )

    plt.title("Phan bo email theo nhan")
    plt.xlabel("Label")
    plt.ylabel("So luong")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "label_distribution.png", dpi=300)
    plt.close()

    # Thong ke do dai email
    df["email_length"] = df["data"].fillna("").astype(str).str.len()

    print("\nThong ke do dai email:")
    print(df["email_length"].describe())

    print("\nDa luu ket qua trong thu muc reports/")


if __name__ == "__main__":
    main()
