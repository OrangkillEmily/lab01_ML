import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.predict import load_models, predict_email


st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Classifier")
st.write(
    "Ứng dụng phân loại email rác bằng Machine Learning "
    "với ba mô hình Logistic Regression, Naive Bayes và SVM."
)

st.caption(
    "Spam = Email rác (nhãn 1) | "
    "Ham = Email bình thường (nhãn 0)"
)

st.divider()


@st.cache_resource
def get_models():
    return load_models(verbose=False)


try:
    vectorizer, models = get_models()
    st.success("Đã tải thành công TF-IDF và 3 mô hình.")
except Exception as error:
    st.error(f"Không thể tải mô hình: {error}")
    st.stop()


model_choice = st.selectbox(
    "Chọn mô hình phân loại",
    [
        "So sánh cả 3 mô hình",
        "Logistic Regression",
        "Naive Bayes",
        "SVM"
    ]
)

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
            with st.spinner("Đang phân tích email..."):
                result = predict_email(
                    email,
                    vectorizer,
                    models
                )

            predictions = result["output"]["predictions"]
            output = result["output"]

            st.subheader("Kết quả dự đoán")

            if model_choice == "So sánh cả 3 mô hình":
                columns = st.columns(3)

                for column, (name, prediction) in zip(
                    columns,
                    predictions.items()
                ):
                    with column:
                        st.markdown(f"**{name}**")

                        if prediction["label"] == "Spam":
                            st.error("🚨 Spam — Email rác")
                        elif prediction["label"] == "Ham":
                            st.success("✅ Ham — Email bình thường")
                        else:
                            st.warning(prediction["label"])

                        st.caption(
                            f"Thời gian: "
                            f"{prediction['prediction_time_seconds']:.6f}s"
                        )

            else:
                prediction = predictions[model_choice]

                st.write(f"Mô hình: **{model_choice}**")

                if prediction["label"] == "Spam":
                    st.error("🚨 Email được phân loại là SPAM")
                elif prediction["label"] == "Ham":
                    st.success("✅ Email được phân loại là HAM")
                else:
                    st.warning(prediction["label"])

                st.caption(
                    f"Thời gian dự đoán: "
                    f"{prediction['prediction_time_seconds']:.6f}s"
                )

            st.divider()

            st.subheader("Thông tin xử lý")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Thời gian làm sạch",
                f"{output['cleaning_time_seconds']:.6f}s"
            )

            col2.metric(
                "Thời gian TF-IDF",
                f"{output['tfidf_time_seconds']:.6f}s"
            )

            col3.metric(
                "Tổng thời gian",
                f"{output['total_prediction_time_seconds']:.6f}s"
            )

            with st.expander("Chi tiết quá trình xử lý"):
                st.markdown("**Email sau khi làm sạch:**")
                st.write(result["input"]["cleaned_text"])

                st.markdown("**Kích thước vector TF-IDF:**")
                st.write(tuple(output["feature_shape"]))

                st.markdown("**Run ID:**")
                st.code(result["run_id"])

                st.markdown("**Thời gian dự đoán từng mô hình:**")

                for name, prediction in predictions.items():
                    st.write(
                        f"- {name}: "
                        f"{prediction['prediction_time_seconds']:.6f}s"
                    )

            st.success(
                "Đã lưu kết quả vào reports/inference_history.csv"
            )

        except Exception as error:
            st.error(f"Không thể dự đoán email: {error}")


st.divider()

st.caption(
    "Lab01 – Spam Email Classification | "
    "Logistic Regression • Naive Bayes • SVM"
)
