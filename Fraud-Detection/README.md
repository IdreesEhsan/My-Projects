
# Credit Card Fraud Detection Pipeline 🛡️

**End-to-end machine learning system** for real-time detection of fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate).

Built in **January 2026** as a professional-grade portfolio project demonstrating modern MLOps practices.

## 🚀 Key Features

- **Exploratory Data Analysis** with visualizations (class imbalance, feature correlations, distributions)
- **XGBoost classifier** with:
  - Class imbalance handling via `scale_pos_weight`
  - Threshold tuning to optimize F1-score on fraud class
  - Early stopping and histogram tree method for fast CPU training
- **MLflow integration**:
  - Experiment tracking
  - Model registry (`CreditCardFraudXGBoost`)
  - Logging of parameters, metrics, feature importance, and confusion matrix
- **FastAPI backend** for real-time predictions
- **Beautiful Streamlit frontend** with:
  - Dark theme + custom styling
  - Fraud probability progress bar and color-coded alerts
  - Quick sample buttons (normal/fraud transactions)
  - On-demand feature importance plot

## 📊 Results (Typical Run)

- **AUC-ROC**: ~0.97
- **PR-AUC**: High performance on imbalanced data
- **High recall** on fraud class (critical for fraud detection)
- Top predictive features: **V14, V17, V12, V10, V11**

## 📁 Project Structure

fraud-detection-pipeline/
├── data/
│ └── creditcard.csv # Dataset
├── notebooks/
│ └── eda.ipynb # Exploratory Data Analysis
├── src/
│ ├── train.py # Training + MLflow logging
│ └── app.py # FastAPI backend
├── artifacts/ # Generated plots
│ ├── feature_importance.png
│ └── confusion_matrix.png
├── frontend.py # Streamlit interactive dashboard
├── mlflow.db # MLflow tracking database
├── requirements.txt
└── README.md

## 🛠 How to Run

### 1. Setup

```bash
python -m venv fraud-env
fraud-env\Scripts\activate          # Windows
pip install -r requirements.txt

2. Train the Model

python src/train.py

3. View Experiments in MLflow

mlflow ui --backend-store-uri sqlite:///mlflow.db
Open http://localhost:5000 → Check Experiments and Models tab.

4. Start the API Backend

uvicorn src.app:app --reload
Test API at http://127.0.0.1:8000/docs (Swagger UI)

5. Launch the Interactive Frontend

streamlit run frontend.py

🏗 Tech Stack

Python 3.11+
Pandas, NumPy, Scikit-learn
XGBoost
MLflow (tracking + model registry)
FastAPI + Uvicorn
Streamlit (frontend)

🌟 Why This Project Stands Out

Handles real-world imbalanced classification
Production-ready design with model registry
Full observability via MLflow
Interactive user interface for non-technical stakeholders
Clean, modular, and well-documented code
```
=======
# My-Projects
Projects for understanding core concepts
>>>>>>> b30fecfc3c7486d3e34fa781e29a911f51016ab8
