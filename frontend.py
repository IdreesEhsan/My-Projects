# frontend.py - Beautiful & Functional Fraud Detection Dashboard

import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import xgboost as xgb
import mlflow.xgboost
import os

st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beauty
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { background-color: #ff4b4b; color: white; border-radius: 8px; padding: 0.5rem 1rem; }
    .fraud-alert { background-color: #330000; padding: 1.5rem; border-radius: 12px; border-left: 6px solid #ff0000; margin: 1rem 0; }
    .safe-transaction { background-color: #003300; padding: 1.5rem; border-radius: 12px; border-left: 6px solid #00ff00; margin: 1rem 0; }
    h1 { color: #ffaa00; text-align: center; }
    .stProgress > div > div > div > div { background-color: #ff4444; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ Advanced Credit Card Fraud Detection System")
st.markdown("<h3 style='text-align: center; color: #aaaaaa;'>Real-time XGBoost-powered fraud analysis on 284K+ transactions</h3>", unsafe_allow_html=True)

# Load model for feature importance
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.xgboost.load_model("models:/CreditCardFraudXGBoost/latest")

# Sidebar: Feature Importance
st.sidebar.header("🔍 Model Insights")
if st.sidebar.button("Show Top Feature Importance"):
    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(model, max_num_features=10, ax=ax, color='#ffaa00')
    ax.set_title("Top 10 Most Important Features")
    st.sidebar.pyplot(fig)

st.sidebar.markdown("---")

# Sidebar: Input form
with st.sidebar.form("transaction_form"):
    st.subheader("💳 Transaction Details")
    
    col1, col2 = st.columns(2)
    with col1:
        Time = st.number_input("Time (seconds)", value=0.0, step=1.0)
    with col2:
        Amount = st.number_input("Amount ($)", value=100.0, min_value=0.0)

    st.markdown("**Anonymized Features (V1–V28)**")
    V = {}
    for i in range(1, 29):
        default = st.session_state.get(f"V{i}", 0.0)
        V[f"V{i}"] = st.slider(f"V{i}", min_value=-10.0, max_value=10.0, value=float(default), step=0.01, key=f"V{i}")

    submitted = st.form_submit_button("🚨 Analyze Transaction", use_container_width=True)

# Quick sample buttons
st.sidebar.markdown("### ⚡ Quick Test Samples")
col_s1, col_s2 = st.sidebar.columns(2)

if col_s1.button("Normal Tx", use_container_width=True):
    df = pd.read_csv("../data/creditcard.csv")
    sample = df[df['Class'] == 0].drop('Class', axis=1).sample(1).iloc[0]
    for col, val in sample.items():
        st.session_state[col] = float(val)
    st.rerun()

if col_s2.button("Fraud Tx", use_container_width=True):
    df = pd.read_csv("../data/creditcard.csv")
    sample = df[df['Class'] == 1].drop('Class', axis=1).sample(1).iloc[0]
    for col, val in sample.items():
        st.session_state[col] = float(val)
    st.rerun()

# Main prediction
if submitted:
    payload = {"Time": Time, "Amount": Amount, **V}
    
    with st.spinner("🔍 Analyzing transaction..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            
            prob = result["fraud_probability"]
            is_fraud = result["alert"]
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud Probability", f"{prob:.1%}")
            with col2:
                st.metric("Risk Level", "HIGH ⚠️" if is_fraud else "LOW ✅")
            with col3:
                st.progress(prob)
            
            if is_fraud:
                st.markdown(f"<div class='fraud-alert'><h2>🚨 FRAUD DETECTED!</h2><p>Probability: <strong>{prob:.1%}</strong><br>Recommendation: <strong>BLOCK TRANSACTION</strong></p></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='safe-transaction'><h2>✅ Transaction Approved</h2><p>Probability: <strong>{prob:.1%}</strong><br>Appears legitimate</p></div>", unsafe_allow_html=True)
                
        except requests.exceptions.ConnectionError:
            st.error("❌ FastAPI server not running! Start with: `uvicorn src.app:app --reload` in another terminal")
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

st.caption("Built with ❤️ using XGBoost • MLflow • FastAPI • Streamlit | January 2026")