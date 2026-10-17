import argparse
import sys
import time
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import evaluate, preprocess, train


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Chạy tuần tự tiền xử lý, huấn luyện và đánh giá mô hình."
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Bỏ qua bước tiền xử lý dữ liệu.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Bỏ qua bước huấn luyện mô hình.",
    )
    args = parser.parse_args()

    steps = [
        ("Tiền xử lý dữ liệu", preprocess.main, args.skip_preprocess),
        ("Huấn luyện mô hình", train.main, args.skip_train),
        ("Đánh giá mô hình", evaluate.main, False),
    ]

    total_start_time = time.time()
    print("\n" + "=" * 55)
    print("BẮT ĐẦU QUY TRÌNH MÁY HỌC")
    print("=" * 55)

    for step_number, (step_name, step_function, skip) in enumerate(steps, start=1):
        if skip:
            print(f"[BỎ QUA] {step_name}")
            continue

        print("\n" + "=" * 55)
        print(f"[BƯỚC {step_number}/3] {step_name}")
        print("=" * 55)
        step_start_time = time.time()
        try:
            step_function()
        except Exception as error:
            print(f"[LỖI] Bước '{step_name}' thất bại: {error}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

        elapsed_time = time.time() - step_start_time
        print(f"[DONE] Hoàn thành trong {elapsed_time:.2f} giây")

    total_elapsed_time = time.time() - total_start_time
    print("\n" + "=" * 55)
    print(f"[DONE] Tổng thời gian: {total_elapsed_time:.2f} giây")
    print(f"[INFO] Kết quả nằm trong thư mục: {ROOT_DIR / 'reports'}")
    print("=" * 55)


if __name__ == "__main__":
    main()