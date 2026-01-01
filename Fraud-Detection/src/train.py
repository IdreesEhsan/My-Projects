# src/train.py

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import mlflow
import mlflow.xgboost

print("Script started")

# =====================================================
# Project paths (DEFINE FIRST — VERY IMPORTANT)
# =====================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(script_dir, "..", "data", "creditcard.csv")

artifact_dir = os.path.join(script_dir, "..", "artifacts")
os.makedirs(artifact_dir, exist_ok=True)

# =====================================================
# MLflow configuration
# =====================================================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Credit Card Fraud Detection")

# =====================================================
# Load data
# =====================================================
print(f"Loading data from: {data_path}")
df = pd.read_csv(data_path)
print(f"Data loaded — shape: {df.shape}")

X = df.drop("Class", axis=1)
y = df["Class"]

# =====================================================
# Train / Test split (stratified)
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42,
)

print("Train/test split done")

# =====================================================
# Handle class imbalance
# =====================================================
neg, pos = y_train.value_counts()
scale_pos_weight = neg / pos
print(f"Scale pos weight: {scale_pos_weight:.2f}")

# =====================================================
# MLflow run
# =====================================================
with mlflow.start_run(run_name="XGBoost_CreditFraud_v1"):
    print("MLflow run started")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 6,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
        "tree_method": "hist",
    }

    mlflow.log_params(params)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    print("Starting training...")
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=[(dtest, "test")],
        early_stopping_rounds=50,
        verbose_eval=100,
    )
    print("Training finished!")

    # =================================================
    # Predictions
    # =================================================
    y_pred_proba = model.predict(dtest)

    # =================================================
    # Threshold tuning (IMPORTANT for fraud)
    # =================================================
    thresholds = np.linspace(0.01, 0.99, 99)
    best_f1 = 0
    best_threshold = 0

    for t in thresholds:
        preds = (y_pred_proba >= t).astype(int)
        report = classification_report(y_test, preds, output_dict=True)
        f1 = report["1"]["f1-score"]

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Best Threshold: {best_threshold:.2f}")
    print(f"Best F1 Score: {best_f1:.4f}")

    # =================================================
    # Final metrics
    # =================================================
    y_pred = (y_pred_proba >= best_threshold).astype(int)

    auc_roc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("pr_auc", pr_auc)
    mlflow.log_metric("best_f1", best_f1)
    mlflow.log_metric("best_threshold", best_threshold)
    mlflow.log_metric("best_iteration", model.best_iteration)

    # =================================================
    # Log model to MLflow Registry
    # =================================================
    mlflow.xgboost.log_model(
        model,
        artifact_path="xgboost_model",
        registered_model_name="CreditCardFraudXGBoost",
    )

    # =================================================
    # Feature importance plot
    # =================================================
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, max_num_features=15)
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()

    fi_path = os.path.join(artifact_dir, "feature_importance.png")
    plt.savefig(fi_path)
    plt.close()

    mlflow.log_artifact(fi_path)

    # =================================================
    # Confusion matrix plot
    # =================================================
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    cm_path = os.path.join(artifact_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    mlflow.log_artifact(cm_path)

    print("All done! Check MLflow UI 🚀")
