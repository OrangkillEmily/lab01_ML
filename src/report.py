import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = ROOT_DIR / "progress_results"
LOG_DIR = ROOT_DIR / "logs"
REPORT_DIR = ROOT_DIR / "reports"
IMAGE_DIR = ROOT_DIR / "docs" / "images"

METRICS_FILE = REPORT_DIR / "metrics_summary.csv"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_progress(run_id):
    """Tai du lieu tracking JSON."""
    progress_path = RESULT_DIR / f"progress_{run_id}.json"

    if not progress_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file tracking: {progress_path}"
        )

    with open(progress_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_log(run_id):
    """Tai file log cua pipeline."""
    log_path = LOG_DIR / f"pipeline_{run_id}.log"

    if not log_path.exists():
        return "Khong tim thay file log."

    with open(log_path, "r", encoding="utf-8") as file:
        return file.read()


def load_metrics():
    """Tai bang ket qua danh gia mo hinh."""
    if not METRICS_FILE.exists():
        return []

    with open(METRICS_FILE, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)


def get_latest_run():
    """Lay run_id moi nhat trong progress_results."""
    progress_files = list(
        RESULT_DIR.glob("progress_*.json")
    )

    if not progress_files:
        raise FileNotFoundError(
            "Chua co du lieu tracking trong progress_results/."
        )

    latest_file = max(
        progress_files,
        key=lambda path: path.stat().st_mtime
    )

    return latest_file.stem.replace("progress_", "", 1)


def format_value(value):
    """Chuyen du lieu thanh chuoi de hien thi."""
    if isinstance(value, dict):
        return json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            default=str
        )

    if isinstance(value, list):
        return ", ".join(str(item) for item in value)

    if value is None:
        return "N/A"

    return str(value)


def get_pipeline_steps(progress):
    """Lay danh sach cac buoc tu tracking."""
    return progress.get("steps", [])


def get_metrics_table(metrics):
    """Tao bang Markdown tu ket qua CSV."""
    if not metrics:
        return "Chua co du lieu metrics_summary.csv."

    columns = list(metrics[0].keys())

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    rows = []

    for item in metrics:
        values = []

        for column in columns:
            value = item.get(column, "")
            values.append(str(value))

        rows.append("| " + " | ".join(values) + " |")

    return "\n".join([header, separator] + rows)


def get_best_model(metrics):
    """Tim mo hinh co F1 cao nhat tren tap Test."""
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
        return "Chua co du lieu", None

    return best_model, best_f1


def get_images():
    """Lay cac bieu do da duoc tao."""
    image_names = [
        "label_distribution.png",
        "confusion_matrix_lr.png",
        "confusion_matrix_nb.png",
        "confusion_matrix_svm.png",
        "confusion_matrix_all.png",
        "roc_curve_all.png"
    ]

    images = []

    for image_name in image_names:
        image_path = IMAGE_DIR / image_name

        if image_path.exists():
            images.append(image_path)

    return images


def generate_markdown(run_id, progress, metrics, log_content):
    """Tao bao cao Markdown."""
    status = progress.get("status", "UNKNOWN")
    start_time = progress.get("start_time", "N/A")
    end_time = progress.get("end_time", "N/A")

    best_model, best_f1 = get_best_model(metrics)

    steps = get_pipeline_steps(progress)

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
        "## 2. Tiến trình thực hiện",
        "",
        "| STT | Bước thực hiện | Trạng thái |",
        "|---:|---|---|"
    ]

    for index, step in enumerate(steps, start=1):
        step_name = step.get("step", "Unknown")
        step_status = step.get("status", "UNKNOWN")

        lines.append(
            f"| {index} | {step_name} | {step_status} |"
        )

    lines.extend([
        "",
        "## 3. Chi tiết tracking",
        ""
    ])

    for index, step in enumerate(steps, start=1):
        step_name = step.get("step", "Unknown")

        lines.extend([
            f"### 3.{index}. {step_name}",
            "",
            f"- Trạng thái: **{step.get('status', 'UNKNOWN')}**",
            "",
            "**Input:**",
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
            "**Output:**",
            "",
            "```json",
            json.dumps(
                step.get("output"),
                ensure_ascii=False,
                indent=2,
                default=str
            ),
            "```"
        ])

        if step.get("error"):
            lines.extend([
                "",
                f"**Lỗi:** {step.get('error')}"
            ])

        lines.append("")

    lines.extend([
        "## 4. Kết quả đánh giá mô hình",
        "",
        get_metrics_table(metrics),
        "",
        "## 5. Mô hình tốt nhất",
        "",
        f"- Mô hình: **{best_model}**",
        f"- F1-score: **{best_f1 if best_f1 is not None else 'N/A'}**",
        "",
        "## 6. Biểu đồ kết quả",
        ""
    ])

    images = get_images()

    if images:
        for image_path in images:
            relative_path = "../docs/images/" + image_path.name

            lines.extend([
                f"### {image_path.stem}",
                "",
                f"![{image_path.stem}]({relative_path})",
                ""
            ])
    else:
        lines.append("Chua co bieu do.")

    lines.extend([
        "## 7. Nhật ký thực thi",
        "",
        "```text",
        log_content,
        "```",
        "",
        "---",
        "",
        "Báo cáo được tạo tự động từ dữ liệu tracking của Lab01."
    ])

    return "\n".join(lines)


def generate_html(run_id, progress, metrics, log_content):
    """Tao bao cao HTML."""
    status = progress.get("status", "UNKNOWN")
    best_model, best_f1 = get_best_model(metrics)
    steps = get_pipeline_steps(progress)

    status_color = (
        "#16a34a" if status == "SUCCESS" else "#dc2626"
    )

    metric_headers = list(metrics[0].keys()) if metrics else []

    metric_html = ""

    if metrics:
        metric_html += "<table><thead><tr>"

        for header in metric_headers:
            metric_html += f"<th>{html.escape(header)}</th>"

        metric_html += "</tr></thead><tbody>"

        for item in metrics:
            metric_html += "<tr>"

            for header in metric_headers:
                value = str(item.get(header, ""))
                metric_html += (
                    f"<td>{html.escape(value)}</td>"
                )

            metric_html += "</tr>"

        metric_html += "</tbody></table>"
    else:
        metric_html = "<p>Chua co du lieu metrics.</p>"

    step_html = ""

    for index, step in enumerate(steps, start=1):
        step_name = html.escape(
            str(step.get("step", "Unknown"))
        )

        step_status = html.escape(
            str(step.get("status", "UNKNOWN"))
        )

        input_data = html.escape(
            format_value(step.get("input"))
        )

        output_data = html.escape(
            format_value(step.get("output"))
        )

        error = step.get("error")

        step_html += f"""
        <div class="step">
            <h3>{index}. {step_name}</h3>
            <p>Trạng thái: <b>{step_status}</b></p>
            <h4>Input</h4>
            <pre>{input_data}</pre>
            <h4>Output</h4>
            <pre>{output_data}</pre>
        """

        if error:
            step_html += (
                f"<p class='error'>Lỗi: "
                f"{html.escape(str(error))}</p>"
            )

        step_html += "</div>"

    image_html = ""

    for image_path in get_images():
        relative_path = "../docs/images/" + image_path.name

        image_html += f"""
        <div class="image">
            <h3>{html.escape(image_path.stem)}</h3>
            <img src="{html.escape(relative_path)}"
                 alt="{html.escape(image_path.stem)}">
        </div>
        """

    escaped_log = html.escape(log_content)

    return f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="UTF-8">
<title>Báo cáo Lab01 - {html.escape(run_id)}</title>
<style>
body {{
    font-family: Arial, sans-serif;
    max-width: 1100px;
    margin: 30px auto;
    padding: 0 20px;
    line-height: 1.6;
    color: #222;
}}
h1, h2 {{
    border-bottom: 1px solid #ddd;
    padding-bottom: 8px;
}}
.info, .step {{
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
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
    padding: 8px;
    text-align: center;
}}
th {{
    background: #f1f1f1;
}}
pre {{
    background: #f5f5f5;
    padding: 12px;
    border-radius: 5px;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
}}
img {{
    max-width: 100%;
    height: auto;
    border: 1px solid #ddd;
}}
.error {{
    color: #dc2626;
}}
</style>
</head>
<body>

<h1>BÁO CÁO THỰC NGHIỆM LAB01</h1>

<div class="info">
    <p><b>Run ID:</b> {html.escape(run_id)}</p>
    <p><b>Trạng thái:</b>
       <span class="status">{html.escape(status)}</span></p>
    <p><b>Bắt đầu:</b>
       {html.escape(str(progress.get("start_time", "N/A")))}</p>
    <p><b>Kết thúc:</b>
       {html.escape(str(progress.get("end_time", "N/A")))}</p>
</div>

<h2>1. Tiến trình thực hiện</h2>
{step_html}

<h2>2. Kết quả đánh giá mô hình</h2>
{metric_html}

<h2>3. Mô hình tốt nhất</h2>
<p><b>{html.escape(str(best_model))}</b></p>
<p>F1-score: {html.escape(str(best_f1))}</p>

<h2>4. Biểu đồ kết quả</h2>
{image_html}

<h2>5. Nhật ký thực thi</h2>
<pre>{escaped_log}</pre>

</body>
</html>"""


def generate_pdf(run_id, progress, metrics, log_content, output_path):
    """Tao bao cao PDF bang ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        Image,
        PageBreak
    )
    from xml.sax.saxutils import escape

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.5 * cm,
        leftMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm
    )

    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            fontSize=18,
            leading=23,
            spaceAfter=15
        )
    )

    styles.add(
        ParagraphStyle(
            name="SmallText",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11
        )
    )

    story = []

    story.append(
        Paragraph("BAO CAO THUC NGHIEM LAB01", styles["ReportTitle"])
    )

    story.append(
        Paragraph(
            f"<b>Run ID:</b> {escape(run_id)}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Trang thai:</b> "
            f"{escape(str(progress.get('status', 'UNKNOWN')))}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Thoi gian bat dau:</b> "
            f"{escape(str(progress.get('start_time', 'N/A')))}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>Thoi gian ket thuc:</b> "
            f"{escape(str(progress.get('end_time', 'N/A')))}",
            styles["Normal"]
        )
    )

    story.append(Spacer(1, 15))

    # Tien trinh pipeline
    story.append(
        Paragraph("1. TIEN TRINH THUC HIEN", styles["Heading2"])
    )

    step_data = [
        ["STT", "Buoc thuc hien", "Trang thai"]
    ]

    for index, step in enumerate(get_pipeline_steps(progress), start=1):
        step_data.append([
            str(index),
            str(step.get("step", "Unknown")),
            str(step.get("status", "UNKNOWN"))
        ])

    step_table = Table(
        step_data,
        colWidths=[1.2 * cm, 10 * cm, 5 * cm],
        repeatRows=1
    )

    step_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("PADDING", (0, 0), (-1, -1), 6)
        ])
    )

    story.append(step_table)
    story.append(Spacer(1, 15))

    # Ket qua metrics
    story.append(
        Paragraph("2. KET QUA DANH GIA MO HINH", styles["Heading2"])
    )

    if metrics:
        headers = list(metrics[0].keys())
        metric_data = [[
            Paragraph(escape(str(item)), styles["SmallText"])
            for item in headers
        ]]

        for item in metrics:
            metric_data.append([
                Paragraph(
                    escape(str(item.get(header, ""))),
                    styles["SmallText"]
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

        metric_table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("PADDING", (0, 0), (-1, -1), 4)
            ])
        )

        story.append(metric_table)
    else:
        story.append(
            Paragraph("Chua co du lieu metrics.", styles["Normal"])
        )

    best_model, best_f1 = get_best_model(metrics)

    story.append(Spacer(1, 12))

    story.append(
        Paragraph("3. MO HINH TOT NHAT", styles["Heading2"])
    )

    story.append(
        Paragraph(
            f"<b>Mo hinh:</b> {escape(str(best_model))}",
            styles["Normal"]
        )
    )

    story.append(
        Paragraph(
            f"<b>F1-score:</b> {escape(str(best_f1))}",
            styles["Normal"]
        )
    )

    # Bieu do
    images = get_images()

    if images:
        story.append(PageBreak())

        story.append(
            Paragraph("4. BIEU DO KET QUA", styles["Heading2"])
        )

        for image_path in images:
            story.append(
                Paragraph(
                    escape(image_path.stem),
                    styles["Heading3"]
                )
            )

            report_image = Image(str(image_path))

            max_width = A4[0] - 3 * cm
            max_height = 13 * cm

            scale = min(
                max_width / report_image.imageWidth,
                max_height / report_image.imageHeight,
                1
            )

            report_image.drawWidth *= scale
            report_image.drawHeight *= scale

            story.append(report_image)
            story.append(Spacer(1, 12))

    # Log
    story.append(PageBreak())

    story.append(
        Paragraph("5. NHAT KY THUC THI", styles["Heading2"])
    )

    for line in log_content.splitlines():
        story.append(
            Paragraph(
                escape(line) if line.strip() else " ",
                styles["SmallText"]
            )
        )

    document.build(story)


def main():
    parser = argparse.ArgumentParser(
        description="Tao bao cao Markdown, HTML va PDF cho Lab01."
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID can tao bao cao."
    )

    args = parser.parse_args()

    try:
        # Lay run_id
        run_id = args.run_id or get_latest_run()

        print("=" * 55)
        print("BAT DAU TAO BAO CAO LAB01")
        print("=" * 55)

        print(f"[INFO] Run ID: {run_id}")

        # Doc du lieu
        progress = load_progress(run_id)
        metrics = load_metrics()
        log_content = load_log(run_id)

        # Tao ten file
        markdown_path = REPORT_DIR / f"report_{run_id}.md"
        html_path = REPORT_DIR / f"report_{run_id}.html"
        pdf_path = REPORT_DIR / f"report_{run_id}.pdf"

        # Tao Markdown
        markdown_content = generate_markdown(
            run_id,
            progress,
            metrics,
            log_content
        )

        with open(
            markdown_path, "w", encoding="utf-8"
        ) as file:
            file.write(markdown_content)

        print(f"[DONE] Da tao Markdown: {markdown_path}")

        # Tao HTML
        html_content = generate_html(
            run_id,
            progress,
            metrics,
            log_content
        )

        with open(
            html_path, "w", encoding="utf-8"
        ) as file:
            file.write(html_content)

        print(f"[DONE] Da tao HTML: {html_path}")

        # Tao PDF
        generate_pdf(
            run_id,
            progress,
            metrics,
            log_content,
            pdf_path
        )

        print(f"[DONE] Da tao PDF: {pdf_path}")

        print("\n" + "=" * 55)
        print("[DONE] TAO BAO CAO THANH CONG")
        print("=" * 55)

    except Exception as error:
        print(f"[ERROR] Khong the tao bao cao: {error}")
        raise


if __name__ == "__main__":
    main()
