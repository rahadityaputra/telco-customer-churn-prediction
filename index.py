import sys
from sklearn.base import BaseEstimator, TransformerMixin
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import sklearn
import imblearn
import sklearn.compose._column_transformer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer, OrdinalEncoder
from imblearn.pipeline import Pipeline 

class _RemainderColsList:
    """Mock class to satisfy pickle reference."""
    pass
    
try:
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
    print("Injected _RemainderColsList mock class successfully.")
except Exception as e:
    print(f"Warning: Failed to inject mock class: {e}")
    




print("--- INFORMASI LINGKUNGAN INDEX.PY ---")
print(f"Versi Python: {sys.version.split()[0]}")
print(f"Versi Scikit-learn: {sklearn.__version__}")
print("--------------------------------------")

BINS = [0, 12, 36, 100]
LABELS = ['New/At-Risk', 'Mid-Term', 'Long-Term/Loyal']

def custom_tenure_binning(X):
    tenure_df = pd.DataFrame(X, columns=['tenure'])
    
    binned_data = pd.cut(
        tenure_df['tenure'],
        bins=BINS,
        labels=LABELS,
        right=False,
        include_lowest=True
    )
    
    return binned_data.values.reshape(-1, 1)

class BinaryEncoder(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        for col in X_copy.columns:
            if col == 'gender':
                 X_copy[col] = X_copy[col].replace({'Female': 1, 'Male': 0})
            else:
                 X_copy[col] = X_copy[col].replace({'Yes': 1, 'No': 0})
                 
        return X_copy.astype(float)

FILE_PIPELINE = 'gradient_bosting_model.pkl'
model_pipeline = None

try:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Versi SCiKit-Learn:** `{sklearn.__version__}`")
    st.sidebar.markdown(f"**Versi Imblearn:** `{imblearn.__version__}`")
    st.sidebar.markdown("---")

    with open(FILE_PIPELINE, 'rb') as file:
        model_pipeline = pickle.load(file)
except FileNotFoundError:
    st.error("Error: 'random_forest_model.pkl' tidak ditemukan. Pastikan file ada di direktori yang sama.")
    sys.exit() 
except Exception as e:
    st.error(f"❌ ERROR Fatal saat memuat model: {type(e).__name__}: {e}")
    st.warning("Ini mungkin disebabkan ketidakcocokan versi Python/Pustaka. Latih ulang model Anda di lingkungan v1.7.")
    sys.exit()


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Telco Customer Churn Prediction")
st.markdown("Masukkan detail pelanggan untuk memprediksi kemungkinan Churn.")

with st.sidebar:
    st.header("Customer Information")

    gender = st.radio("Gender", ['Male', 'Female'])
    senior_citizen_bool = st.checkbox("Senior Citizen (65+ years old)") 
    partner = st.radio("Partner", ['No', 'Yes'])
    dependents = st.radio("Dependents", ['No', 'Yes'])
    phone_service = st.radio("Phone Service", ['Yes', 'No'])
    paperless_billing = st.radio("Paperless Billing", ['No', 'Yes'])

    # Numerical features
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=50.0, step=0.1)
  
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=8000.0, value=500.0, step=0.1) 

    st.subheader("Service Information")
    # Categorical features
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", ['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", ['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", ['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])

input_data = pd.DataFrame({
    'gender': [gender],
    'SeniorCitizen': [1 if senior_citizen_bool else 0], 
    'Partner': [partner],
    'Dependents': [dependents],
    'tenure': [tenure],
    'PhoneService': [phone_service],
    'MultipleLines': [multiple_lines],
    'OnlineSecurity': [online_security],
    'OnlineBackup': [online_backup],
    'DeviceProtection': [device_protection],
    'TechSupport': [tech_support],
    'StreamingTV': [streaming_tv],
    'StreamingMovies': [streaming_movies],
    'Contract': [contract],
    'PaperlessBilling': [paperless_billing],
    'PaymentMethod': [payment_method],
    'MonthlyCharges': [monthly_charges],
    'TotalCharges': [total_charges], 
    'InternetService': [internet_service],
})

if st.button("Predict Churn"):
    if model_pipeline is None:
        st.error("Model belum dimuat. Tidak dapat melakukan prediksi.")
    else:
        try:
            prediction = model_pipeline.predict(input_data)
            prediction_proba = model_pipeline.predict_proba(input_data)

            churn_probability = prediction_proba[0][1] 
            
            st.subheader("Prediction Results:")
            if prediction[0] == 1:
                st.markdown(f"## 🚨 Pelanggan diprediksi **CHURN**")
                st.error(f"Probabilitas Churn: **{churn_probability:.2%}**")
                st.balloons()
            else:
                st.markdown(f"## ✅ Pelanggan diprediksi **TIDAK CHURN**")
                st.success(f"Probabilitas Tidak Churn: **{1-churn_probability:.2%}**")

            st.write("---")
            st.write("Detail Input yang Dipakai untuk Prediksi:")
            st.dataframe(input_data.T, width=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi. Pastikan data input konsisten dengan pelatihan.")
            st.code(f"Error: {e}")