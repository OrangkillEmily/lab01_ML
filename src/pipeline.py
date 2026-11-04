import argparse
import sys
import time
import traceback
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from src import eda, evaluate, preprocess, train
from src.tracking import (
    create_run,
    update_progress,
    write_log,
    finish_run
)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    # Cau hinh cac tuy chon khi chay pipeline
    parser = argparse.ArgumentParser(
        description="Chay toan bo quy trinh may hoc."
    )

    parser.add_argument(
        "--skip-eda",
        action="store_true",
        help="Bo qua buoc phan tich du lieu."
    )

    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Bo qua buoc tien xu ly du lieu."
    )

    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Bo qua buoc huan luyen mo hinh."
    )

    args = parser.parse_args()

    # Tao run_id chung cho toan bo pipeline
    run_id = create_run()

    steps = [
        ("EDA", eda.main, args.skip_eda),
        ("PREPROCESSING", preprocess.main, args.skip_preprocess),
        ("TRAINING", train.main, args.skip_train),
        ("EVALUATION", evaluate.main, False)
    ]

    total_start_time = time.perf_counter()

    print("\n" + "=" * 55)
    print("BAT DAU QUY TRINH MAY HOC")
    print("=" * 55)

    write_log(
        run_id,
        "PIPELINE",
        "START",
        "Bat dau quy trinh may hoc."
    )

    try:
        # Chay lan luot cac buoc
        for step_number, (step_name, step_function, skip) in enumerate(
            steps, start=1
        ):

            if skip:
                print(f"\n[BO QUA] {step_name}")

                update_progress(
                    run_id,
                    step_name,
                    "SKIPPED",
                    output_data={
                        "message": "Buoc nay da duoc bo qua."
                    }
                )

                write_log(
                    run_id,
                    step_name,
                    "SKIPPED",
                    "Bo qua buoc theo cau hinh."
                )

                continue

            print("\n" + "=" * 55)
            print(
                f"[BUOC {step_number}/{len(steps)}] {step_name}"
            )
            print("=" * 55)

            step_start_time = time.perf_counter()

            write_log(
                run_id,
                step_name,
                "START",
                f"Bat dau buoc {step_name}."
            )

            try:
                # Truyen run_id chung vao tung buoc
                step_function(run_id)

            except Exception as error:
                elapsed_time = time.perf_counter() - step_start_time

                update_progress(
                    run_id,
                    step_name,
                    "FAILED",
                    output_data={
                        "execution_time_seconds": round(
                            elapsed_time, 4
                        )
                    },
                    error=str(error)
                )

                write_log(
                    run_id,
                    step_name,
                    "FAILED",
                    str(error)
                )

                raise

            elapsed_time = time.perf_counter() - step_start_time

            print(
                f"[DONE] Hoan thanh trong {elapsed_time:.2f} giay"
            )

            write_log(
                run_id,
                step_name,
                "SUCCESS",
                f"Hoan thanh trong {elapsed_time:.2f} giay."
            )

        # Tong hop ket qua pipeline
        total_elapsed_time = time.perf_counter() - total_start_time

        update_progress(
            run_id,
            "PIPELINE",
            "SUCCESS",
            output_data={
                "total_execution_time_seconds": round(
                    total_elapsed_time, 4
                ),
                "completed_steps": [
                    step_name
                    for step_name, _, skip in steps
                    if not skip
                ]
            }
        )

        finish_run(run_id, "SUCCESS")

        write_log(
            run_id,
            "PIPELINE",
            "SUCCESS",
            f"Hoan thanh toan bo quy trinh trong "
            f"{total_elapsed_time:.2f} giay."
        )

        print("\n" + "=" * 55)
        print("[DONE] QUY TRINH MAY HOC HOAN TAT")
        print(f"[INFO] Run ID: {run_id}")
        print(
            f"[INFO] Tong thoi gian: "
            f"{total_elapsed_time:.2f} giay"
        )
        print(
            f"[INFO] Ket qua: {ROOT_DIR / 'reports'}"
        )
        print("=" * 55)

    except Exception as error:
        total_elapsed_time = time.perf_counter() - total_start_time

        update_progress(
            run_id,
            "PIPELINE",
            "FAILED",
            output_data={
                "total_execution_time_seconds": round(
                    total_elapsed_time, 4
                )
            },
            error=str(error)
        )

        finish_run(run_id, "FAILED")

        write_log(
            run_id,
            "PIPELINE",
            "FAILED",
            str(error)
        )

        print(
            f"\n[ERROR] Pipeline that bai: {error}",
            file=sys.stderr
        )

        traceback.print_exc()

        print(f"[INFO] Run ID: {run_id}")
        sys.exit(1)


if __name__ == "__main__":
    main()
