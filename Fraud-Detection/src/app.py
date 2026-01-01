# src/app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import mlflow
import mlflow.xgboost
import pandas as pd
import xgboost as xgb

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Load registered model
model = mlflow.xgboost.load_model("models:/CreditCardFraudXGBoost/latest")

app = FastAPI(title="Credit Card Fraud Detection API")

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.get("/")
def home():
    return {"message": "API is running! Go to /docs"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        input_dict = transaction.dict()
        input_df = pd.DataFrame([input_dict])
        
        # Fix column order to match training
        expected_columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        input_df = input_df[expected_columns]
        
        dmatrix = xgb.DMatrix(input_df)
        prob = float(model.predict(dmatrix)[0])
        
        return {
            "fraud_probability": round(prob, 4),
            "predicted_class": "Fraud" if prob > 0.5 else "Normal",
            "alert": prob > 0.5
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))