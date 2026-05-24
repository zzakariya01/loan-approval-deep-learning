import streamlit as st
import numpy as np
import pickle
import tensorflow as tf

st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="🏦",
    layout="centered"
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("loan_approval_dnn.keras")
    with open("loan_artifacts.pkl", "rb") as f:
        arts = pickle.load(f)
    return model, arts

model, arts = load_artifacts()

# ── Header ────────────────────────────────────────
st.title("🏦 Loan Sanction Predictor")
st.caption("Deep Learning — Binary Classification (DNN)")
st.markdown("---")

# ── Input Form ───────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    gender         = st.selectbox("Gender",          ["Male", "Female"])
    married        = st.selectbox("Married",          ["Yes", "No"])
    dependents     = st.selectbox("Dependents",       ["0", "1", "2", "3+"])
    education      = st.selectbox("Education",        ["Graduate", "Not Graduate"])
    self_employed  = st.selectbox("Self Employed",    ["No", "Yes"])
    credit_history = st.selectbox("Credit History",   [1, 0],
                                   format_func=lambda x: "Good (1)" if x == 1 else "Bad (0)")

with col2:
    applicant_income   = st.number_input("Applicant Income (₹)",
                                          min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Income (₹)",
                                          min_value=0, value=0, step=500)
    loan_amount        = st.number_input("Loan Amount (thousands)",
                                          min_value=1, value=150, step=5)
    loan_amount_term   = st.number_input("Loan Term (months)",
                                          min_value=12, value=360, step=12)
    property_area      = st.selectbox("Property Area",
                                       ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# ── Prediction ───────────────────────────────────
if st.button("🔍  Predict Loan Status", use_container_width=True):

    dep_val = 3 if dependents == "3+" else int(dependents)

    gender_enc   = arts["le_gender"].transform([gender])[0]
    married_enc  = arts["le_married"].transform([married])[0]
    edu_enc      = arts["le_education"].transform([education])[0]
    se_enc       = arts["le_self_employed"].transform([self_employed])[0]
    prop_enc     = arts["le_property"].transform([property_area])[0]

    features = np.array([[gender_enc, married_enc, dep_val,
                           edu_enc, se_enc,
                           applicant_income, coapplicant_income,
                           loan_amount, loan_amount_term,
                           credit_history, prop_enc]], dtype=float)

    features_scaled = arts["scaler"].transform(features)
    prob      = float(model.predict(features_scaled, verbose=0)[0][0])
    approved  = prob >= 0.5

    st.subheader("📊 Prediction Result")

    if approved:
        st.success(f"✅  APPROVED — Confidence: {prob*100:.1f}%")
    else:
        st.error(f"❌  REJECTED — Confidence: {(1-prob)*100:.1f}%")

    st.progress(float(prob))
    st.caption(f"Approval probability score: {prob:.4f}")

    with st.expander("ℹ️ About this model"):
        st.markdown("""
        **Model:** Deep Neural Network (4 hidden layers)  
        **Concepts:** Batch Normalization · Dropout · Early Stopping  
        · Learning Rate Scheduler · ROC-AUC evaluation  
        **Dataset:** Kaggle — Loan Sanction Dataset  
        **Framework:** TensorFlow / Keras
        """)
