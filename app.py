import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from utils import plot_conf_matrix, plot_roc, plot_fraud_probability

# Load model
model = joblib.load("fraud_model.pkl")
scaler = StandardScaler()

st.set_page_config(page_title="💳 Credit Card Fraud Detection", layout="wide")
st.title("💳 Credit Card Fraud Detection")

st.markdown("Upload a CSV of transactions. This app predicts fraudulent transactions using a trained ML model.")

uploaded_file = st.file_uploader("📂 Upload CSV File", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    if 'Time' in df.columns and 'Amount' in df.columns:
        df_scaled = df.copy()

        # Drop 'Class' if it exists (was not used during training)
        if 'Class' in df_scaled.columns:
            df_scaled = df_scaled.drop('Class', axis=1)

        # Scale relevant columns
        df_scaled[['Time', 'Amount']] = scaler.fit_transform(df_scaled[['Time', 'Amount']])

        # Predict
        y_probs = model.predict_proba(df_scaled)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)
        df['Fraud_Probability'] = y_probs
        df['Prediction'] = y_pred

        fraud_count = (df['Prediction'] == 1).sum()
        legit_count = (df['Prediction'] == 0).sum()

        st.subheader("📊 Prediction Summary")
        col1, col2 = st.columns(2)
        col1.metric("⚠️ Fraudulent Transactions", fraud_count)
        col2.metric("✅ Legitimate Transactions", legit_count)

        # If 'Class' exists in uploaded file, evaluate metrics
        if 'Class' in df.columns:
            st.subheader("📈 Evaluation Metrics")
            st.pyplot(plot_conf_matrix(df['Class'], df['Prediction']))
            st.pyplot(plot_roc(df['Class'], df['Fraud_Probability']))
        
        st.subheader("🔮 Fraud Probability Distribution")
        st.pyplot(plot_fraud_probability(df['Fraud_Probability']))

        st.subheader("📑 Full Results")
        st.dataframe(df[['Time', 'Amount', 'Fraud_Probability', 'Prediction']])
    else:
        st.warning("Uploaded file must contain 'Time' and 'Amount' columns.")
