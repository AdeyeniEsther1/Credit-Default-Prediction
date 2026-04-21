# 💳 Credit Score Card System

An end-to-end, AI-powered credit risk underwriting platform. This system utilizes advanced Machine Learning to predict borrower default probabilities, translates them into a classic 300–850 Credit Score metric, and outputs actionable lending decisions backed by explainable AI (SHAP).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Framework-FF4B4B)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-orange)

## 🔗 Live Demo
**Try the app live here:** [Credit Default Predictor Dashboard](https://credit-default-prediction-bn6z4qf5tmkjwmn5g2tzct.streamlit.app/)

---

##  Live Demo
**Try the app live here:** [Credit Default Predictor Dashboard](https://credit-default-prediction-ewatyvhafgqrjkvtrkvl8q.streamlit.app/)

##  Features

- **Credit Score Transformation:** Seamlessly converts raw machine learning probability outputs into a familiar, scalable FICO-style credit score range (300 to 850).
- **Automated Lending Decisions:** Evaluates applications instantly and issues a structured directive: 
  - ✅ **Approve** (Score $\ge$ 700 / Low Risk)
  - ⚠️ **Manual Review** (Score 600-699 / Medium Risk)
  - 🚨 **Reject** (Score $<$ 600 / High Risk)
- **Interactive Explainability (SHAP):** Empowers lenders to understand *why* a decision was made by displaying beautiful, dynamic waterfall plots showing exact feature contributions (e.g., how a borrower's high debt ratio negatively impacted their final score).
- **Batch Processing Dashboard:** Upload a `.csv` of hundreds of borrowers to instantly receive a processed sheet with calculated Credit Scores, Risk Tiers, and Decisions, visualised interactively in Plotly histograms.
- **Optimized Model Pipeline:** The backend training engine tests Logisitic Regression, Random Forests, XGBoost, and LightGBM using parallel hyperparameter tuning and native class weighting for lightning-fast retraining without heavy sampling distortion.

---

##  Quick Start

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm imbalanced-learn plotly streamlit shap matplotlib
```

### 2. Train the Model
The repository expects a highly optimized ML Pipeline built to handle severe class imbalance out of the box. 
*Note: Make sure your `cs-training.csv` and `cs-test.csv` datasets are placed in the root directory.*

```bash
python credit_model.py
```
This script will benchmark the models, locate the mathematically ideal model (evaluating ROC-AUC and PR-AUC), apply hyperparameters, and save it natively as `best_model.pkl`.

### 3. Launch the Dashboard
Once `best_model.pkl` is successfully generated, spin up the Streamlit UI!

```bash
streamlit run app.py
```

---

##  Usage Guide

The Streamlit interface is divided into two primary tabs:

### 👤 Single Prediction
Perfect for interacting with the application via simulated numbers.
- Manually enter standard borrower metrics (Age, Utilization, Number of Dependents, Delinquency History, etc.).
- **Info Tooltips:** Every input includes a tiny `(?)` marker mathematically explaining the specific feature definition to guarantee error-free entries.
- Receive the **Credit Score**, **Lending Decision**, and view the dynamically rendered **SHAP Explanation Chart**.

### 📁 Batch Prediction
Perfect for assessing bulk applications efficiently.
- Upload a standard `.csv` file detailing borrowers.
- Instantly review Risk distributions across the entire dataset.
- Click **Download Results** to pull a `.csv` appended tightly with the new Credit Score, Risk Tier, and Decision attached to every single applicant profile.

---

## 🧠 Code Architecture

- `credit_model.py`: The data-cleaning, model tuning, and evaluation powerhouse. Utilizes RandomizedSearchCV heavily optimized avoiding legacy SMOTE loops in favor of scaled class weights resulting in phenomenal speed hikes.
- `app.py`: The Streamlit dashboard frontend handling realtime UI deployment, UI configurations, dynamic probability-to-score math scaling logic, and SHAP graphic generation.
