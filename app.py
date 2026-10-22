
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="ConnectTel Churn Prediction",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("connecttel_churn_pipeline.joblib")

model = load_model()

st.title("📊 ConnectTel Customer Churn Prediction")
st.write(
    "Identify customers at risk of leaving "
    "and support proactive retention."
)

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple = st.sidebar.selectbox(
    "Multiple Lines", ["No", "Yes", "No phone service"]
)
internet = st.sidebar.selectbox(
    "Internet Service", ["DSL", "Fiber optic", "No"]
)
security = st.sidebar.selectbox(
    "Online Security", ["Yes", "No", "No internet service"]
)
backup = st.sidebar.selectbox(
    "Online Backup", ["Yes", "No", "No internet service"]
)
device = st.sidebar.selectbox(
    "Device Protection", ["Yes", "No", "No internet service"]
)
tech = st.sidebar.selectbox(
    "Tech Support", ["Yes", "No", "No internet service"]
)
tv = st.sidebar.selectbox(
    "Streaming TV", ["Yes", "No", "No internet service"]
)
movies = st.sidebar.selectbox(
    "Streaming Movies", ["Yes", "No", "No internet service"]
)
contract = st.sidebar.selectbox(
    "Contract", ["Month-to-month", "One year", "Two year"]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)
monthly = st.sidebar.number_input(
    "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
)
total = st.sidebar.number_input(
    "Total Charges", min_value=0.0, max_value=20000.0, value=840.0
)

if st.button("Predict Churn Risk", type="primary"):
    service_cols = [
        phone, multiple, security, backup,
        device, tech, tv, movies
    ]

    service_count = sum(value == "Yes" for value in service_cols)

    total_per_tenure = total / max(tenure, 1)

    tenure_group = pd.cut(
        pd.Series([tenure]),
        bins=[-1, 12, 24, 48, 72],
        labels=["New", "Medium", "Long", "Loyal"]
    ).iloc[0]

    charge_category = pd.cut(
        pd.Series([monthly]),
        bins=[-float("inf"), 35, 70, float("inf")],
        labels=["Low", "Medium", "High"]
    ).iloc[0]

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "TotalChargesPerTenure": total_per_tenure,
        "ServiceCount": service_count,
        "TenureGroup": tenure_group,
        "MonthlyChargeCategory": charge_category
    }])

    probability = model.predict_proba(customer)[0, 1]

    st.subheader("Prediction Result")
    st.metric("Estimated Churn Probability", f"{probability:.1%}")

    if probability >= 0.60:
        st.error("High risk: Consider a personalized retention offer.")
    elif probability >= 0.30:
        st.warning("Medium risk: Monitor this customer and consider engagement.")
    else:
        st.success("Low risk: Continue regular customer engagement.")

    st.caption(
        "Predictions are estimates from a machine learning model, "
        "not guarantees of customer behavior."
    )
