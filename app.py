
import re
import sys
from pathlib import Path

import joblib
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Đăng ký các lớp mô hình để tải file pickle
from src.model import LogisticRegression, NaiveBayes, SVM


MODEL_DIR = ROOT_DIR / "models"


def clean_text(text):
    """Làm sạch email giống src/predict.py."""
    text = str(text).lower()

    # Xóa thẻ HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Loại bỏ URL
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Chỉ giữ chữ cái tiếng Anh và khoảng trắng
    text = re.sub(r"[^a-z\s]", " ", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    return text


@st.cache_resource
def load_models():
    """Tải TF-IDF và ba mô hình đã huấn luyện."""
    vectorizer_path = MODEL_DIR / "tfidf_vectorizer.pkl"

    model_paths = {
        "Logistic Regression": MODEL_DIR / "model_lr.pkl",
        "Naive Bayes": MODEL_DIR / "model_nb.pkl",
        "SVM": MODEL_DIR / "model_svm.pkl",
    }

    required_files = [vectorizer_path, *model_paths.values()]

    missing_files = [
        str(path.relative_to(ROOT_DIR))
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Thiếu các tệp cần thiết:\n"
            + "\n".join(missing_files)
            + "\nHãy chạy pipeline để tạo mô hình trước."
        )

    vectorizer = joblib.load(vectorizer_path)

    models = {
        name: joblib.load(path)
        for name, path in model_paths.items()
    }

    return vectorizer, models


def predict_email(email, vectorizer, model):
    """Dự đoán email bằng mô hình được chọn."""
    cleaned_email = clean_text(email)

    if not cleaned_email:
        raise ValueError(
            "Nội dung email không còn chữ cái tiếng Anh "
            "sau khi làm sạch."
        )

    features = vectorizer.transform([cleaned_email])
    prediction = int(model.predict(features)[0])

    if prediction == 1:
        return "Spam"

    if prediction == 0:
        return "Ham"

    return f"Nhãn không xác định: {prediction}"


# =========================
# GIAO DIỆN STREAMLIT
# =========================

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Classifier")
st.write(
    "Ứng dụng phân loại email rác bằng Machine Learning "
    "với ba mô hình được cài đặt trong dự án Lab01."
)

st.caption(
    "Spam = email rác (nhãn 1) | "
    "Ham = email bình thường (nhãn 0)"
)

st.divider()

# Tải mô hình
try:
    vectorizer, models = load_models()
    st.success("Đã tải thành công TF-IDF và 3 mô hình.")
except Exception as error:
    st.error(f"Không thể tải mô hình: {error}")
    st.stop()


# Chọn mô hình
model_choice = st.selectbox(
    "Chọn mô hình phân loại",
    [
        "So sánh cả 3 mô hình",
        "Logistic Regression",
        "Naive Bayes",
        "SVM",
    ]
)

# Nhập email
email = st.text_area(
    "Nội dung email cần kiểm tra",
    placeholder=(
        "Ví dụ: Congratulations! You have won a free prize. "
        "Click here to claim your offer now."
    ),
    height=200
)

if st.button(
    "🔍 Phân loại email",
    type="primary",
    use_container_width=True
):
    if not email.strip():
        st.warning("Vui lòng nhập nội dung email.")
    else:
        try:
            if model_choice == "So sánh cả 3 mô hình":
                st.subheader("Kết quả dự đoán")

                columns = st.columns(3)

                for column, (name, model) in zip(
                    columns, models.items()
                ):
                    result = predict_email(
                        email, vectorizer, model
                    )

                    with column:
                        st.markdown(f"**{name}**")

                        if result == "Spam":
                            st.error("🚨 Spam — Email rác")
                        elif result == "Ham":
                            st.success(
                                "✅ Ham — Email bình thường"
                            )
                        else:
                            st.warning(result)

            else:
                model = models[model_choice]
                result = predict_email(
                    email, vectorizer, model
                )

                st.subheader("Kết quả dự đoán")
                st.write(f"Mô hình: **{model_choice}**")

                if result == "Spam":
                    st.error("🚨 Email được phân loại là SPAM")
                elif result == "Ham":
                    st.success(
                        "✅ Email được phân loại là HAM "
                        "(bình thường)"
                    )
                else:
                    st.warning(result)

            with st.expander("Xem nội dung sau khi làm sạch"):
                st.write(clean_text(email))

        except Exception as error:
            st.error(f"Không thể dự đoán email: {error}")


st.divider()

st.caption(
    "Lab01 – Spam Email Classification | "
    "Logistic Regression • Naive Bayes • SVM"
)
