import argparse
import base64
import csv
import html
import json
import mimetypes
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT_DIR / "progress_results"
LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

METRICS_FILE = REPORT_DIR / "metrics_summary.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. MO TA CAC GIAI DOAN CUA PIPELINE
# ============================================================

STEP_GUIDES = {
    "EDA": {
        "title": "Exploratory Data Analysis - Phân tích dữ liệu",
        "purpose": (
            "Khảo sát dữ liệu email ban đầu để hiểu cấu trúc, "
            "phân bố nhãn, độ dài email và phát hiện các vấn đề "
            "như dữ liệu thiếu hoặc dữ liệu bất thường."
        ),
        "input": (
            "Tập dữ liệu email SpamAssassin ban đầu, gồm nhãn "
            "phân loại và nội dung email."
        ),
        "method": (
            "Đọc dữ liệu, kiểm tra số lượng dòng và cột, kiểu dữ liệu, "
            "giá trị thiếu, dữ liệu trùng lặp, phân bố nhãn và "
            "thống kê độ dài nội dung email."
        ),
        "output": (
            "Các thống kê mô tả và biểu đồ giúp đánh giá chất lượng "
            "dữ liệu trước khi tiền xử lý."
        ),
    },

    "PREPROCESSING": {
        "title": "Data Preprocessing - Tiền xử lý dữ liệu",
        "purpose": (
            "Làm sạch dữ liệu email và chuyển nội dung văn bản "
            "thành dạng số để các thuật toán Machine Learning "
            "có thể xử lý."
        ),
        "input": (
            "Dữ liệu email sau khi khảo sát ở giai đoạn EDA."
        ),
        "method": (
            "Làm sạch nội dung, xử lý dữ liệu trùng lặp, "
            "tách dữ liệu thành tập Train và Test, sau đó sử dụng "
            "TF-IDF để biểu diễn văn bản thành vector đặc trưng."
        ),
        "output": (
            "Dữ liệu đã làm sạch, các tập X_train, X_test, "
            "y_train, y_test và ma trận đặc trưng TF-IDF."
        ),
    },

    "TRAINING": {
        "title": "Model Training - Huấn luyện mô hình",
        "purpose": (
            "Xây dựng các mô hình phân loại email Spam và Ham "
            "từ dữ liệu huấn luyện đã được vector hóa."
        ),
        "input": (
            "X_train, y_train và cấu hình tham số của từng mô hình."
        ),
        "method": (
            "Huấn luyện lần lượt Logistic Regression, Naive Bayes "
            "và Support Vector Machine. Trong quá trình huấn luyện, "
            "mỗi mô hình học các tham số từ dữ liệu Train."
        ),
        "output": (
            "Các mô hình đã huấn luyện, tham số đã học và "
            "thời gian huấn luyện."
        ),
    },

    "EVALUATION": {
        "title": "Model Evaluation - Đánh giá mô hình",
        "purpose": (
            "Đánh giá khả năng phân loại của từng mô hình "
            "trên dữ liệu kiểm tra chưa được sử dụng để huấn luyện."
        ),
        "input": (
            "X_test, y_test và các mô hình đã huấn luyện."
        ),
        "method": (
            "Dự đoán nhãn trên tập Test, sau đó tính Accuracy, "
            "Precision, Recall, F1-score, TPR, FPR và AUC. "
            "Đồng thời phân tích Confusion Matrix và ROC Curve."
        ),
        "output": (
            "Bảng metrics, ma trận nhầm lẫn, đường ROC và "
            "kết quả so sánh giữa các mô hình."
        ),
    },

    "INFERENCE": {
        "title": "Inference - Dự đoán dữ liệu mới",
        "purpose": (
            "Sử dụng mô hình đã huấn luyện để phân loại "
            "một email mới do người dùng cung cấp."
        ),
        "input": (
            "Nội dung email mới, TF-IDF Vectorizer đã lưu "
            "và các mô hình đã huấn luyện."
        ),
        "method": (
            "Nhận nội dung email, áp dụng cùng quy trình "
            "vector hóa TF-IDF đã sử dụng khi huấn luyện, "
            "sau đó đưa vector vào từng mô hình để dự đoán."
        ),
        "output": (
            "Nhãn dự đoán Spam hoặc Ham của từng mô hình, "
            "cùng lịch sử dự đoán nếu hệ thống có lưu lại."
        ),
    },
}


# ============================================================
# 2. MO TA CAC BIEU DO
# ============================================================

IMAGE_GUIDES = {
    "label_distribution": (
        "Phân bố nhãn Spam và Ham",
        "Biểu đồ thể hiện số lượng email thuộc từng lớp. "
        "Dùng để kiểm tra mức độ cân bằng giữa hai lớp dữ liệu. "
        "Nếu số lượng hai lớp chênh lệch lớn, mô hình có thể "
        "thiên lệch về lớp chiếm đa số."
    ),

    "email_length_distribution": (
        "Phân bố độ dài email",
        "Biểu đồ thể hiện độ dài nội dung email trong tập dữ liệu. "
        "Giúp quan sát email thường ngắn hay dài, đồng thời phát hiện "
        "các email có độ dài bất thường."
    ),

    "missing_values": (
        "Kiểm tra dữ liệu thiếu",
        "Biểu đồ thể hiện số lượng giá trị thiếu trong từng cột. "
        "Mục đích là xác định dữ liệu cần xử lý trước khi đưa "
        "vào các bước tiếp theo."
    ),

    "cleaning_comparison": (
        "So sánh dữ liệu trước và sau làm sạch",
        "Biểu đồ so sánh số lượng dữ liệu trước và sau quá trình "
        "làm sạch. Qua đó đánh giá ảnh hưởng của các thao tác "
        "xử lý dữ liệu."
    ),

    "preprocessing_steps": (
        "Các bước tiền xử lý dữ liệu",
        "Biểu đồ tóm tắt các thao tác tiền xử lý. "
        "Giúp theo dõi quá trình chuyển dữ liệu email thô "
        "thành dữ liệu phù hợp cho huấn luyện."
    ),

    "tfidf_sparsity": (
        "Độ thưa của ma trận TF-IDF",
        "Biểu đồ thể hiện tỷ lệ phần tử khác 0 trong ma trận "
        "TF-IDF. Văn bản thường có nhiều đặc trưng bằng 0, "
        "do mỗi email chỉ chứa một phần nhỏ trong toàn bộ "
        "từ vựng của tập dữ liệu."
    ),

    "tfidf_top_features": (
        "Các đặc trưng TF-IDF nổi bật",
        "Biểu đồ thể hiện những đặc trưng văn bản có giá trị "
        "TF-IDF nổi bật. Giúp quan sát các từ hoặc cụm từ "
        "được biểu diễn trong không gian đặc trưng."
    ),

    "train_test_distribution": (
        "Phân bố dữ liệu Train và Test",
        "Biểu đồ so sánh phân bố nhãn giữa tập huấn luyện "
        "và tập kiểm tra. Mục đích là kiểm tra hai tập có "
        "phân bố tương đối nhất quán hay không."
    ),

    "training_configuration": (
        "Cấu hình huấn luyện",
        "Biểu đồ thể hiện các cấu hình hoặc tham số được sử dụng "
        "trong quá trình huấn luyện mô hình."
    ),

    "training_dataset_overview": (
        "Tổng quan dữ liệu huấn luyện",
        "Biểu đồ tóm tắt dữ liệu được sử dụng để huấn luyện "
        "các mô hình phân loại."
    ),

    "training_time_comparison": (
        "So sánh thời gian huấn luyện",
        "Biểu đồ so sánh thời gian huấn luyện giữa các mô hình. "
        "Giúp đánh giá chi phí tính toán của từng thuật toán."
    ),

    "confusion_matrix_all": (
        "Confusion Matrix của các mô hình",
        "Ma trận nhầm lẫn tổng hợp số lượng dự đoán đúng "
        "và sai của các mô hình. Với bài toán Spam, cần chú ý "
        "False Positive và False Negative vì hai loại lỗi "
        "này có ý nghĩa khác nhau."
    ),

    "confusion_matrix_lr": (
        "Confusion Matrix - Logistic Regression",
        "Thể hiện số lượng dự đoán đúng và sai của Logistic "
        "Regression trên tập đánh giá."
    ),

    "confusion_matrix_nb": (
        "Confusion Matrix - Naive Bayes",
        "Thể hiện số lượng dự đoán đúng và sai của Naive Bayes "
        "trên tập đánh giá."
    ),

    "confusion_matrix_svm": (
        "Confusion Matrix - Support Vector Machine",
        "Thể hiện số lượng dự đoán đúng và sai của SVM "
        "trên tập đánh giá."
    ),

    "metrics_comparison": (
        "So sánh các chỉ số đánh giá",
        "Biểu đồ so sánh các chỉ số như Accuracy, Precision, "
        "Recall và F1-score. Giúp nhận biết mô hình có hiệu quả "
        "phân loại tốt hơn theo từng tiêu chí."
    ),

    "roc_curve_all": (
        "Đường ROC của các mô hình",
        "Đường ROC thể hiện mối quan hệ giữa TPR và FPR "
        "ở các ngưỡng phân loại khác nhau. AUC càng cao "
        "thì khả năng phân biệt hai lớp của mô hình càng tốt."
    ),

    "tpr_fpr_comparison": (
        "So sánh TPR và FPR",
        "Biểu đồ so sánh tỷ lệ phát hiện đúng Spam và tỷ lệ "
        "nhận diện nhầm Ham thành Spam giữa các mô hình."
    ),
}


# ============================================================
# 3. DOC DU LIEU TRACKING, LOG VA METRICS
# ============================================================

def load_progress(run_id):
    """Đọc dữ liệu tracking JSON của một lần chạy."""

    progress_path = RESULT_DIR / f"progress_{run_id}.json"

    if not progress_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file tracking: {progress_path}"
        )

    with open(progress_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_log(run_id):
    """Đọc file log của Pipeline."""

    log_path = LOG_DIR / f"pipeline_{run_id}.log"

    if not log_path.exists():
        return "Không tìm thấy file log."

    with open(log_path, "r", encoding="utf-8") as file:
        return file.read()


def load_metrics(run_id):
    """Chỉ lấy metrics thuộc đúng Run_ID."""

    if not METRICS_FILE.exists():
        print("[WARNING] Không tìm thấy metrics_summary.csv.")
        return []

    with open(
        METRICS_FILE,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        if not reader.fieldnames or "Run_ID" not in reader.fieldnames:
            print(
                "[WARNING] metrics_summary.csv chưa có cột Run_ID."
            )
            return []

        metrics = [
            row
            for row in reader
            if row.get("Run_ID", "").strip() == str(run_id)
        ]

    print(
        f"[INFO] Đã tải {len(metrics)} dòng metrics "
        f"cho Run_ID: {run_id}"
    )

    if not metrics:
        print(
            f"[WARNING] Không tìm thấy metrics cho Run_ID: {run_id}"
        )

    return metrics


def get_latest_run():
    """Lấy Pipeline SUCCESS mới nhất."""

    pipeline_runs = []

    for progress_path in RESULT_DIR.glob("progress_*.json"):
        try:
            with open(
                progress_path,
                "r",
                encoding="utf-8"
            ) as file:
                progress = json.load(file)

            if progress.get("status") != "SUCCESS":
                continue

            has_pipeline = any(
                step.get("step") == "PIPELINE"
                and step.get("status") == "SUCCESS"
                for step in progress.get("steps", [])
            )

            if has_pipeline:
                pipeline_runs.append(progress_path)

        except (json.JSONDecodeError, OSError):
            continue

    if not pipeline_runs:
        raise FileNotFoundError(
            "Không tìm thấy Pipeline nào có trạng thái SUCCESS."
        )

    latest_file = max(
        pipeline_runs,
        key=lambda path: path.stat().st_mtime
    )

    return latest_file.stem.replace("progress_", "", 1)


# ============================================================
# 4. HAM HO TRO
# ============================================================

def format_value(value):
    """Chuyển dữ liệu thành chuỗi để hiển thị."""

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    if value is None:
        return "N/A"

    return str(value)


def get_pipeline_steps(progress):
    return progress.get("steps", [])


def get_metrics_table(metrics):
    """Tạo bảng Markdown từ metrics."""

    if not metrics:
        return "Chưa có dữ liệu metrics cho Run_ID này."

    columns = list(metrics[0].keys())

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    rows = []

    for item in metrics:
        values = []

        for column in columns:
            value = str(item.get(column, ""))
            value = value.replace("|", "\\|").replace("\n", " ")
            values.append(value)

        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def get_best_model(metrics):
    """Tìm mô hình có F1 cao nhất trên tập Test."""

    best_model = None
    best_f1 = -1

    for item in metrics:
        if item.get("Tập dữ liệu") != "Test":
            continue

        try:
            f1 = float(item.get("F1", 0))
        except (ValueError, TypeError):
            continue

        if f1 > best_f1:
            best_f1 = f1
            best_model = item.get("Mô hình", "Unknown")

    if best_model is None:
        return "Chưa có dữ liệu", None

    return best_model, best_f1


def get_images(run_id):
    """Lấy biểu đồ theo từng giai đoạn của một Pipeline run."""

    figure_dir = FIGURE_DIR / run_id

    image_groups = {
        "EDA": figure_dir / "eda",
        "Preprocessing": figure_dir / "preprocessing",
        "Training": figure_dir / "training",
        "Evaluation": figure_dir / "evaluation",
        "Inference": figure_dir / "inference",
    }

    images = {}

    for group_name, folder in image_groups.items():
        if folder.exists():
            images[group_name] = sorted(
                list(folder.glob("*.png"))
                + list(folder.glob("*.jpg"))
                + list(folder.glob("*.jpeg"))
            )
        else:
            images[group_name] = []

    return images


def get_image_relative_path(run_id, group_name, image_path):
    """Tạo đường dẫn ảnh tương đối từ thư mục reports."""

    return (
        f"figures/{run_id}/{group_name.lower()}/"
        f"{image_path.name}"
    )


def get_image_guide(image_path):
    """Lấy mô tả biểu đồ theo tên file."""

    stem = image_path.stem.lower()

    if stem in IMAGE_GUIDES:
        return IMAGE_GUIDES[stem]

    # Tìm tên gần đúng nếu file có thêm hậu tố.
    for key, guide in IMAGE_GUIDES.items():
        if key in stem:
            return guide

    readable_name = re.sub(r"[_\-]+", " ", image_path.stem)
    readable_name = readable_name.strip().title()

    return (
        readable_name,
        "Biểu đồ được tạo trong giai đoạn này của Pipeline. "
        "Quan sát tiêu đề, trục và chú giải để xác định "
        "các thông tin mà biểu đồ thể hiện."
    )


def get_step_guide(step_name):
    """Lấy mô tả giai đoạn theo tên step."""

    normalized = str(step_name).upper().replace(" ", "_")

    if normalized in STEP_GUIDES:
        return STEP_GUIDES[normalized]

    for key, guide in STEP_GUIDES.items():
        if key in normalized:
            return guide

    return {
        "title": str(step_name),
        "purpose": "Thực hiện chức năng tương ứng trong Pipeline.",
        "input": "Dữ liệu được chuyển từ giai đoạn trước.",
        "method": "Thực hiện xử lý theo logic được cài đặt trong mã nguồn.",
        "output": "Kết quả được chuyển cho giai đoạn tiếp theo.",
    }


def image_to_base64(image_path):
    """Chuyển ảnh thành Base64 để nhúng trực tiếp vào HTML."""

    mime_type, _ = mimetypes.guess_type(str(image_path))

    if not mime_type:
        mime_type = "image/png"

    with open(image_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


# ============================================================
# 5. TAO NOI DUNG CHI TIET CHO BIEU DO
# ============================================================

def build_image_description(image_path):
    """Tạo tên và phần giải thích cho từng biểu đồ."""

    title, description = get_image_guide(image_path)

    return {
        "title": title,
        "description": description,
        "filename": image_path.name,
    }


# ============================================================
# 6. TAO BAO CAO MARKDOWN
# ============================================================

def generate_markdown(run_id, progress, metrics, log_content):
    """Tạo báo cáo Markdown."""

    status = progress.get("status", "UNKNOWN")
    start_time = progress.get("start_time", "N/A")
    end_time = progress.get("end_time", "N/A")

    best_model, best_f1 = get_best_model(metrics)
    steps = get_pipeline_steps(progress)
    images = get_images(run_id)

    lines = [
        "# BÁO CÁO THỰC NGHIỆM LAB01",
        "",
        "## 1. Thông tin lần chạy",
        "",
        f"- Run ID: `{run_id}`",
        f"- Trạng thái: **{status}**",
        f"- Thời gian bắt đầu: {start_time}",
        f"- Thời gian kết thúc: {end_time}",
        "",
        "## 2. Tổng quan Pipeline",
        "",
        "Pipeline thực hiện các giai đoạn chính:",
        "",
        "EDA → Preprocessing → Training → Evaluation → Inference",
        "",
        "Mỗi giai đoạn nhận dữ liệu từ bước trước, thực hiện "
        "chức năng riêng và tạo đầu ra phục vụ giai đoạn tiếp theo.",
        "",
        "## 3. Tiến trình thực hiện",
        "",
        "| STT | Bước thực hiện | Trạng thái |",
        "|---:|---|---|"
    ]

    for index, step in enumerate(steps, start=1):
        lines.append(
            f"| {index} | {step.get('step', 'Unknown')} "
            f"| {step.get('status', 'UNKNOWN')} |"
        )

    lines.extend([
        "",
        "## 4. Giải thích các giai đoạn",
        ""
    ])

    for index, step in enumerate(steps, start=1):
        step_name = step.get("step", "Unknown")
        guide = get_step_guide(step_name)

        lines.extend([
            f"### 4.{index}. {guide['title']}",
            "",
            f"**Mục đích:** {guide['purpose']}",
            "",
            f"**Input:** {guide['input']}",
            "",
            f"**Cách thực hiện:** {guide['method']}",
            "",
            f"**Output:** {guide['output']}",
            "",
            f"**Trạng thái thực tế:** {step.get('status', 'UNKNOWN')}",
            "",
            "**Input tracking thực tế:**",
            "",
            "```json",
            json.dumps(
                step.get("input"),
                ensure_ascii=False,
                indent=2,
                default=str
            ),
            "```",
            "",
            "**Output tracking thực tế:**",
            "",
            "```json",
            json.dumps(
                step.get("output"),
                ensure_ascii=False,
                indent=2,
                default=str
            ),
            "```",
        ])

        if step.get("error"):
            lines.extend([
                "",
                f"**Lỗi:** {step.get('error')}"
            ])

        lines.append("")

    lines.extend([
        "## 5. Kết quả đánh giá mô hình",
        "",
        get_metrics_table(metrics),
        "",
        "## 6. Mô hình tốt nhất",
        "",
        f"- Mô hình: **{best_model}**",
        f"- F1-score: **{best_f1 if best_f1 is not None else 'N/A'}**",
        "",
        "Mô hình tốt nhất được xác định dựa trên F1-score "
        "trên tập Test. F1-score cân bằng giữa Precision và Recall.",
        "",
        "## 7. Biểu đồ và giải thích kết quả",
        ""
    ])

    total_images = sum(len(group) for group in images.values())

    if total_images:
        for group_name, image_paths in images.items():
            if not image_paths:
                continue

            group_key = group_name.upper()
            guide = STEP_GUIDES.get(group_key)

            lines.extend([
                f"### 7.{list(images.keys()).index(group_name) + 1}. "
                f"{group_name}",
                ""
            ])

            if guide:
                lines.extend([
                    guide["purpose"],
                    ""
                ])

            for image_path in image_paths:
                image_info = build_image_description(image_path)

                relative_path = get_image_relative_path(
                    run_id,
                    group_name,
                    image_path
                )

                lines.extend([
                    f"#### {image_info['title']}",
                    "",
                    f"![{image_info['title']}]({relative_path})",
                    "",
                    f"**Tên file:** `{image_info['filename']}`",
                    "",
                    f"**Ý nghĩa biểu đồ:** {image_info['description']}",
                    "",
                    "**Cách đọc:** Quan sát các trục, nhãn, chú giải "
                    "và sự khác biệt giữa các giá trị trên biểu đồ "
                    "để rút ra nhận xét phù hợp.",
                    ""
                ])
    else:
        lines.append("Chưa có biểu đồ cho lần chạy này.")

    lines.extend([
        "## 8. Nhật ký thực thi",
        "",
        "Log ghi lại các thông tin trong quá trình Pipeline chạy, "
        "bao gồm các bước thực hiện, thời điểm và lỗi nếu có.",
        "",
        "```text",
        log_content,
        "```",
        "",
        "---",
        "",
        "Báo cáo được tạo tự động từ dữ liệu tracking, metrics, "
        "biểu đồ và log của Lab01."
    ])

    return "\n".join(lines)


# ============================================================
# 7. TAO BAO CAO HTML
# ============================================================

def generate_html(run_id, progress, metrics, log_content):
    """Tạo báo cáo HTML, nhúng ảnh trực tiếp bằng Base64."""

    status = progress.get("status", "UNKNOWN")
    best_model, best_f1 = get_best_model(metrics)
    steps = get_pipeline_steps(progress)
    images = get_images(run_id)

    status_color = (
        "#16a34a" if status == "SUCCESS" else "#dc2626"
    )

    # Bảng metrics
    metric_html = ""

    if metrics:
        headers = list(metrics[0].keys())

        metric_html = "<table><thead><tr>"

        for header in headers:
            metric_html += (
                f"<th>{html.escape(str(header))}</th>"
            )

        metric_html += "</tr></thead><tbody>"

        for item in metrics:
            metric_html += "<tr>"

            for header in headers:
                value = str(item.get(header, ""))
                metric_html += (
                    f"<td>{html.escape(value)}</td>"
                )

            metric_html += "</tr>"

        metric_html += "</tbody></table>"
    else:
        metric_html = (
            "<p>Chưa có dữ liệu metrics cho Run_ID này.</p>"
        )

    # Các bước Pipeline
    step_html = ""

    for index, step in enumerate(steps, start=1):
        step_name = step.get("step", "Unknown")
        guide = get_step_guide(step_name)

        input_data = html.escape(
            format_value(step.get("input"))
        )

        output_data = html.escape(
            format_value(step.get("output"))
        )

        error = step.get("error")

        step_html += f"""
        <div class="step">
            <h3>{index}. {html.escape(guide['title'])}</h3>

            <p><b>Mục đích:</b>
            {html.escape(guide['purpose'])}</p>

            <p><b>Input:</b>
            {html.escape(guide['input'])}</p>

            <p><b>Cách thực hiện:</b>
            {html.escape(guide['method'])}</p>

            <p><b>Output:</b>
            {html.escape(guide['output'])}</p>

            <p><b>Trạng thái:</b>
            {html.escape(str(step.get('status', 'UNKNOWN')))}</p>

            <h4>Input tracking thực tế</h4>
            <pre>{input_data}</pre>

            <h4>Output tracking thực tế</h4>
            <pre>{output_data}</pre>
        """

        if error:
            step_html += (
                f"<p class='error'>Lỗi: "
                f"{html.escape(str(error))}</p>"
            )

        step_html += "</div>"

    # Biểu đồ và mô tả
    image_html = ""
    image_count = 0

    for group_name, image_paths in images.items():
        if not image_paths:
            continue

        image_html += (
            f"<h3>{html.escape(group_name)}</h3>"
        )

        for image_path in image_paths:
            image_count += 1

            image_info = build_image_description(image_path)
            image_data = image_to_base64(image_path)

            image_html += f"""
            <div class="image-card">
                <h4>{html.escape(image_info['title'])}</h4>

                <img src="{image_data}"
                     alt="{html.escape(image_info['title'])}">

                <p><b>Tên file:</b>
                {html.escape(image_info['filename'])}</p>

                <p><b>Ý nghĩa biểu đồ:</b>
                {html.escape(image_info['description'])}</p>

                <p><b>Cách đọc:</b>
                Quan sát các trục, nhãn, chú giải và sự khác biệt
                giữa các giá trị để rút ra nhận xét phù hợp.</p>
            </div>
            """

    if image_count == 0:
        image_html = "<p>Chưa có biểu đồ cho lần chạy này.</p>"

    escaped_log = html.escape(log_content)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">

<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Báo cáo Lab01 - {html.escape(run_id)}</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    font-family: Arial, sans-serif;
    max-width: 1150px;
    margin: 30px auto;
    padding: 0 22px;
    line-height: 1.7;
    color: #222;
    background: #fff;
}}

h1, h2 {{
    border-bottom: 1px solid #ddd;
    padding-bottom: 9px;
}}

h3 {{
    margin-top: 25px;
    color: #263b5a;
}}

h4 {{
    margin-bottom: 8px;
}}

.info, .step, .image-card {{
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 18px;
    margin: 18px 0;
}}

.status {{
    color: {status_color};
    font-weight: bold;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    display: block;
    overflow-x: auto;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 9px;
    text-align: center;
}}

th {{
    background: #f1f1f1;
}}

pre {{
    background: #f5f5f5;
    padding: 13px;
    border-radius: 5px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}

.image-card {{
    margin: 22px 0 30px;
}}

.image-card img {{
    display: block;
    max-width: 100%;
    height: auto;
    margin: 12px auto;
    border: 1px solid #ddd;
    border-radius: 5px;
}}

.error {{
    color: #dc2626;
}}

@media print {{
    body {{
        max-width: 100%;
        margin: 0;
    }}

    .step, .image-card {{
        break-inside: avoid;
    }}
}}
</style>
</head>

<body>

<h1>BÁO CÁO THỰC NGHIỆM LAB01</h1>

<div class="info">
    <p><b>Run ID:</b> {html.escape(run_id)}</p>

    <p><b>Trạng thái:</b>
       <span class="status">{html.escape(str(status))}</span></p>

    <p><b>Bắt đầu:</b>
       {html.escape(str(progress.get('start_time', 'N/A')))}</p>

    <p><b>Kết thúc:</b>
       {html.escape(str(progress.get('end_time', 'N/A')))}</p>
</div>

<h2>1. Tổng quan Pipeline</h2>

<p>
Pipeline thực hiện các giai đoạn:
<b>EDA → Preprocessing → Training → Evaluation → Inference</b>.
Mỗi giai đoạn nhận dữ liệu đầu vào, xử lý theo chức năng
riêng và tạo đầu ra cho các bước tiếp theo.
</p>

<h2>2. Chi tiết các giai đoạn</h2>

{step_html}

<h2>3. Kết quả đánh giá mô hình</h2>

{metric_html}

<h2>4. Mô hình tốt nhất</h2>

<p><b>Mô hình:</b> {html.escape(str(best_model))}</p>

<p><b>F1-score:</b> {html.escape(str(best_f1))}</p>

<p>
Mô hình tốt nhất được xác định dựa trên F1-score
trên tập Test.
</p>

<h2>5. Biểu đồ và giải thích kết quả</h2>

{image_html}

<h2>6. Nhật ký thực thi</h2>

<pre>{escaped_log}</pre>

<hr>

<p>
Báo cáo được tạo tự động từ dữ liệu tracking,
metrics, biểu đồ và log của Lab01.
</p>

</body>
</html>
"""


# ============================================================
# 8. TAO BAO CAO PDF
# ============================================================


def generate_pdf(
    run_id,
    progress,
    metrics,
    log_content,
    output_path
):
    """Tạo PDF có nội dung đồng nhất với Markdown và HTML."""

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import (
        getSampleStyleSheet,
        ParagraphStyle
    )
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak,
        Preformatted
    )
    from xml.sax.saxutils import escape

    # --------------------------------------------------------
    # 1. FONT TIENG VIET
    # --------------------------------------------------------

    font_regular = "Helvetica"
    font_bold = "Helvetica-Bold"

    font_candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf")
        ),
        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        )
    ]

    for regular_path, bold_path in font_candidates:
        if regular_path.exists():
            try:
                pdfmetrics.registerFont(
                    TTFont("VietnameseFont", str(regular_path))
                )
                font_regular = "VietnameseFont"

                if bold_path.exists():
                    pdfmetrics.registerFont(
                        TTFont("VietnameseFontBold", str(bold_path))
                    )
                    font_bold = "VietnameseFontBold"
                else:
                    font_bold = font_regular

                break
            except Exception:
                continue

    # --------------------------------------------------------
    # 2. KHOI TAO DOCUMENT VA STYLE
    # --------------------------------------------------------

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        name="ReportTitleVI",
        parent=styles["Title"],
        fontName=font_bold,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        spaceAfter=18
    ))

    styles.add(ParagraphStyle(
        name="HeadingVI",
        parent=styles["Heading2"],
        fontName=font_bold,
        fontSize=13,
        leading=17,
        spaceBefore=12,
        spaceAfter=8
    ))

    styles.add(ParagraphStyle(
        name="SubHeadingVI",
        parent=styles["Heading3"],
        fontName=font_bold,
        fontSize=10,
        leading=14,
        spaceBefore=8,
        spaceAfter=5
    ))

    styles.add(ParagraphStyle(
        name="BodyVI",
        parent=styles["BodyText"],
        fontName=font_regular,
        fontSize=9,
        leading=13,
        spaceAfter=6
    ))

    styles.add(ParagraphStyle(
        name="SmallVI",
        parent=styles["BodyText"],
        fontName=font_regular,
        fontSize=7,
        leading=9
    ))

    styles.add(ParagraphStyle(
        name="CodeVI",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=6.5,
        leading=8,
        leftIndent=5,
        rightIndent=5,
        spaceAfter=8
    ))

    story = []

    def add_paragraph(value, style="BodyVI"):
        story.append(Paragraph(
            escape(str(value)).replace("\n", "<br/>"),
            styles[style]
        ))

    def add_code(value):
        """Hiển thị dữ liệu tracking hoặc log."""

        content = format_value(value)

        story.append(Preformatted(
            content,
            styles["CodeVI"],
            maxLineLength=100
        ))
        story.append(Spacer(1, 6))

    def add_section(title):
        story.append(Paragraph(
            escape(title),
            styles["HeadingVI"]
        ))

    # --------------------------------------------------------
    # 3. THONG TIN LAN CHAY
    # --------------------------------------------------------

    story.append(Paragraph(
        "BÁO CÁO THỰC NGHIỆM LAB01",
        styles["ReportTitleVI"]
    ))

    add_paragraph(f"Run ID: {run_id}")
    add_paragraph(
        f"Trạng thái: {progress.get('status', 'UNKNOWN')}"
    )
    add_paragraph(
        f"Thời gian bắt đầu: "
        f"{progress.get('start_time', 'N/A')}"
    )
    add_paragraph(
        f"Thời gian kết thúc: "
        f"{progress.get('end_time', 'N/A')}"
    )

    # --------------------------------------------------------
    # 4. TONG QUAN PIPELINE
    # --------------------------------------------------------

    add_section("1. TỔNG QUAN PIPELINE")

    add_paragraph(
        "Pipeline thực hiện các giai đoạn: "
        "EDA → Preprocessing → Training → Evaluation → Inference. "
        "Mỗi giai đoạn nhận dữ liệu đầu vào, xử lý theo chức năng "
        "riêng và tạo đầu ra phục vụ giai đoạn tiếp theo."
    )

    # --------------------------------------------------------
    # 5. TIEN TRINH THUC HIEN
    # --------------------------------------------------------

    add_section("2. TIẾN TRÌNH THỰC HIỆN")

    steps = get_pipeline_steps(progress)

    step_data = [
        ["STT", "Bước thực hiện", "Trạng thái"]
    ]

    for index, step in enumerate(steps, start=1):
        step_data.append([
            str(index),
            str(step.get("step", "Unknown")),
            str(step.get("status", "UNKNOWN"))
        ])

    step_table = Table(
        step_data,
        colWidths=[1.2 * cm, 9.5 * cm, 5 * cm],
        repeatRows=1
    )

    step_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, -1), font_regular),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))

    story.append(step_table)
    story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # 6. GIAI THICH CHI TIET CAC GIAI DOAN
    # --------------------------------------------------------

    add_section("3. GIẢI THÍCH CÁC GIAI ĐOẠN")

    for index, step in enumerate(steps, start=1):

        step_name = step.get("step", "Unknown")
        guide = get_step_guide(step_name)

        story.append(Paragraph(
            f"{index}. {escape(guide['title'])}",
            styles["SubHeadingVI"]
        ))

        add_paragraph(
            f"Mục đích: {guide['purpose']}"
        )

        add_paragraph(
            f"Input: {guide['input']}"
        )

        add_paragraph(
            f"Cách thực hiện: {guide['method']}"
        )

        add_paragraph(
            f"Output: {guide['output']}"
        )

        add_paragraph(
            f"Trạng thái thực tế: "
            f"{step.get('status', 'UNKNOWN')}"
        )

        add_paragraph("Input tracking thực tế:")

        add_code(step.get("input"))

        add_paragraph("Output tracking thực tế:")

        add_code(step.get("output"))

        if step.get("error"):
            add_paragraph(
                f"Lỗi: {step.get('error')}"
            )

    # --------------------------------------------------------
    # 7. KET QUA DANH GIA
    # --------------------------------------------------------

    add_section("4. KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH")

    if metrics:
        headers = list(metrics[0].keys())

        metric_data = [[
            Paragraph(
                escape(str(header)),
                styles["SmallVI"]
            )
            for header in headers
        ]]

        for item in metrics:
            metric_data.append([
                Paragraph(
                    escape(str(item.get(header, ""))),
                    styles["SmallVI"]
                )
                for header in headers
            ])

        available_width = A4[0] - 3 * cm
        column_width = available_width / len(headers)

        metric_table = Table(
            metric_data,
            colWidths=[column_width] * len(headers),
            repeatRows=1
        )

        metric_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTNAME", (0, 0), (-1, -1), font_regular),
            ("PADDING", (0, 0), (-1, -1), 4)
        ]))

        story.append(metric_table)
    else:
        add_paragraph(
            "Chưa có dữ liệu metrics cho Run_ID này."
        )

    # --------------------------------------------------------
    # 8. MO HINH TOT NHAT
    # --------------------------------------------------------

    add_section("5. MÔ HÌNH TỐT NHẤT")

    best_model, best_f1 = get_best_model(metrics)

    add_paragraph(f"Mô hình: {best_model}")

    add_paragraph(
        f"F1-score: "
        f"{best_f1 if best_f1 is not None else 'N/A'}"
    )

    add_paragraph(
        "Mô hình tốt nhất được xác định dựa trên F1-score "
        "trên tập Test. F1-score cân bằng giữa Precision "
        "và Recall."
    )

    # --------------------------------------------------------
    # 9. BIEU DO VA GIAI THICH
    # --------------------------------------------------------

    add_section("6. BIỂU ĐỒ VÀ GIẢI THÍCH KẾT QUẢ")

    images = get_images(run_id)
    total_images = sum(
        len(image_paths)
        for image_paths in images.values()
    )

    if total_images == 0:
        add_paragraph(
            "Chưa có biểu đồ cho lần chạy này."
        )

    for group_name, image_paths in images.items():

        if not image_paths:
            continue

        add_section(group_name.upper())

        group_key = group_name.upper()
        group_guide = STEP_GUIDES.get(group_key)

        if group_guide:
            add_paragraph(group_guide["purpose"])

        for image_path in image_paths:

            image_info = build_image_description(
                image_path
            )

            story.append(Paragraph(
                escape(image_info["title"]),
                styles["SubHeadingVI"]
            ))

            report_image = Image(str(image_path))

            max_width = A4[0] - 3 * cm
            max_height = 11 * cm

            scale = min(
                max_width / report_image.imageWidth,
                max_height / report_image.imageHeight,
                1
            )

            report_image.drawWidth *= scale
            report_image.drawHeight *= scale

            story.append(report_image)
            story.append(Spacer(1, 6))

            add_paragraph(
                f"Tên file: {image_info['filename']}"
            )

            add_paragraph(
                f"Ý nghĩa biểu đồ: "
                f"{image_info['description']}"
            )

            add_paragraph(
                "Cách đọc: Quan sát các trục, nhãn, chú giải "
                "và sự khác biệt giữa các giá trị trên biểu đồ "
                "để rút ra nhận xét phù hợp."
            )

            story.append(Spacer(1, 12))

    # --------------------------------------------------------
    # 10. NHAT KY THUC THI
    # --------------------------------------------------------

    add_section("7. NHẬT KÝ THỰC THI")

    add_paragraph(
        "Log ghi lại quá trình thực thi Pipeline, "
        "bao gồm các bước thực hiện, thông tin xử lý "
        "và lỗi nếu có."
    )

    add_code(log_content)

    # --------------------------------------------------------
    # 11. KET THUC
    # --------------------------------------------------------

    story.append(Spacer(1, 15))

    add_paragraph(
        "Báo cáo được tạo tự động từ dữ liệu tracking, "
        "metrics, biểu đồ và log của Lab01."
    )

    document.build(story)



# ============================================================
# 9. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tạo báo cáo Markdown, HTML và PDF cho Lab01."
        )
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID cần tạo báo cáo."
    )

    args = parser.parse_args()

    try:
        run_id = args.run_id or get_latest_run()

        print("=" * 55)
        print("BẮT ĐẦU TẠO BÁO CÁO LAB01")
        print("=" * 55)

        print(f"[INFO] Run ID: {run_id}")

        progress = load_progress(run_id)
        metrics = load_metrics(run_id)
        log_content = load_log(run_id)

        print(
            f"[INFO] Số bước Pipeline: "
            f"{len(get_pipeline_steps(progress))}"
        )

        print(
            f"[INFO] Số dòng metrics: {len(metrics)}"
        )

        images = get_images(run_id)

        for group_name, image_paths in images.items():
            print(
                f"[INFO] {group_name}: {len(image_paths)} biểu đồ"
            )

        # Tạo tên file
        markdown_path = REPORT_DIR / f"report_{run_id}.md"
        html_path = REPORT_DIR / f"report_{run_id}.html"
        pdf_path = REPORT_DIR / f"report_{run_id}.pdf"

        # Markdown
        markdown_content = generate_markdown(
            run_id,
            progress,
            metrics,
            log_content
        )

        with open(
            markdown_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(markdown_content)

        print(f"[DONE] Markdown: {markdown_path}")

        # HTML
        html_content = generate_html(
            run_id,
            progress,
            metrics,
            log_content
        )

        with open(
            html_path,
            "w",
            encoding="utf-8"
        ) as file:
            file.write(html_content)

        print(f"[DONE] HTML: {html_path}")

        # PDF
        generate_pdf(
            run_id,
            progress,
            metrics,
            log_content,
            pdf_path
        )

        print(f"[DONE] PDF: {pdf_path}")

        print("\n" + "=" * 55)
        print("[DONE] TẠO BÁO CÁO THÀNH CÔNG")
        print("=" * 55)

    except Exception as error:
        print(f"[ERROR] Không thể tạo báo cáo: {error}")
        raise


if __name__ == "__main__":
    main()