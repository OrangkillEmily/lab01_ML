import json
import logging
import uuid
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT_DIR / "logs"
RESULT_DIR = ROOT_DIR / "progress_results"

LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def create_run():
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    run_data = {
        "run_id": run_id,
        "start_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "RUNNING",
        "steps": []
    }

    result_path = RESULT_DIR / f"progress_{run_id}.json"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(run_data, file, ensure_ascii=False, indent=4)

    return run_id


def update_progress(run_id, step_name, status, input_data=None, output_data=None, error=None):
    result_path = RESULT_DIR / f"progress_{run_id}.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Khong tim thay run_id: {run_id}")

    with open(result_path, "r", encoding="utf-8") as file:
        run_data = json.load(file)

    step_data = {
        "step": step_name,
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": status,
        "input": input_data,
        "output": output_data,
        "error": error
    }

    run_data["steps"].append(step_data)

    if status == "FAILED":
        run_data["status"] = "FAILED"

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(run_data, file, ensure_ascii=False, indent=4, default=str)


def write_log(run_id, step_name, status, message):
    log_path = LOG_DIR / f"pipeline_{run_id}.log"

    with open(log_path, "a", encoding="utf-8") as file:
        timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
        file.write(f"[{timestamp}] [{status}] [{step_name}] {message}\n")


def finish_run(run_id, status="SUCCESS"):
    result_path = RESULT_DIR / f"progress_{run_id}.json"

    if not result_path.exists():
        raise FileNotFoundError(f"Khong tim thay run_id: {run_id}")

    with open(result_path, "r", encoding="utf-8") as file:
        run_data = json.load(file)

    run_data["end_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
    run_data["status"] = status

    with open(result_path, "w", encoding="utf-8") as file:
        json.dump(run_data, file, ensure_ascii=False, indent=4, default=str)