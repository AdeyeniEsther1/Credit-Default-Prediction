"""
Credit Score Default Prediction Pipeline
=========================================
Predicts P(SeriousDlqin2yrs=1) for each borrower.

Usage:
    python credit_model.py

Outputs:
    best_model.pkl      — trained sklearn Pipeline
    submission.csv      — predictions for cs-test.csv
    model_metrics.json  — evaluation metrics
"""

import os
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, confusion_matrix, classification_report
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "cs-training.csv")
TEST_PATH  = os.path.join(BASE_DIR, "cs-test.csv")

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
print("=" * 60)
print("1. Loading data …")
df_train = pd.read_csv(TRAIN_PATH, index_col=0)
df_test  = pd.read_csv(TEST_PATH,  index_col=0)

print(f"   Training rows : {len(df_train):,}")
print(f"   Test rows     : {len(df_test):,}")
print(f"   Columns       : {list(df_train.columns)}")
print(f"   Default rate  : {df_train[TARGET].mean():.2%}")

# ─────────────────────────────────────────────
# 2. Preprocessing helper
# ─────────────────────────────────────────────
SKEWED_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberRealEstateLoansOrLines",
]

def preprocess(df: pd.DataFrame, caps: dict = None, income_median: float = None):
    df = df.copy()

    # Imputation
    if income_median is None:
        income_median = df["MonthlyIncome"].median()
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # Outlier capping
    if caps is None:
        caps = {}
        for col in SKEWED_COLS:
            if col in df.columns:
                caps[col] = df[col].quantile(0.99)

    for col, cap in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(upper=cap)

    # Feature engineering
    df["TotalLatePayments"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
    )
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    df["DebtToIncome"] = df["DebtRatio"] * df["MonthlyIncome"]

    return df, caps, income_median

# ─────────────────────────────────────────────
# 3. Split + preprocess
# ─────────────────────────────────────────────
print("\n2. Preprocessing …")
y = df_train[TARGET]
X_raw = df_train.drop(columns=[TARGET])

X_proc, caps, income_median = preprocess(X_raw)
X_test_proc, _, _ = preprocess(df_test, caps=caps, income_median=income_median)

FEATURES = [c for c in X_proc.columns]

X = X_proc[FEATURES]
X_test = X_test_proc[FEATURES]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# ─────────────────────────────────────────────
# 4. Skipping SMOTE for Speed (Using Class Weights)
# ─────────────────────────────────────────────
print("\n3. Using native Class Weights instead of SMOTE for optimization …")
X_train_res, y_train_res = X_train, y_train

# ─────────────────────────────────────────────
# 5. Models
# ─────────────────────────────────────────────
pos_weight = int((y_train == 0).sum() / (y_train == 1).sum())

models = {
    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000, C=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE))
    ]),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, max_depth=8,
        class_weight="balanced",
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, scale_pos_weight=pos_weight,
        eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=6, is_unbalance=True,
        n_jobs=-1, random_state=RANDOM_STATE,
        verbose=-1
    ),
}

# ─────────────────────────────────────────────
# 6. Train & evaluate
# ─────────────────────────────────────────────
print("\n4. Training & evaluating all models …")
print(f"\n{'Model':<22} {'ROC-AUC':>9} {'PR-AUC':>9} {'F1':>8}")
print("-" * 52)

results = {}

for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.30).astype(int)

    roc = roc_auc_score(y_val, y_prob)
    pr  = average_precision_score(y_val, y_prob)
    f1  = f1_score(y_val, y_pred)

    results[name] = {"roc_auc": roc, "pr_auc": pr, "f1": f1, "model": model}

    print(f"  {name:<20} {roc:9.4f} {pr:9.4f} {f1:8.4f}")

# ─────────────────────────────────────────────
# 7. Best model
# ─────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = results[best_name]["model"]

print(f"\n   ✓ Best model: {best_name}")

# ─────────────────────────────────────────────
# 8. Hyperparameter tuning
# ─────────────────────────────────────────────
param_grids = {
    "XGBoost": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "LightGBM": {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [200, 400],
        "max_depth": [6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "LogisticRegression": {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
        "clf__solver": ["liblinear"],
    },
}

if best_name in param_grids:
    search = RandomizedSearchCV(
        estimator=best_model,
        param_distributions=param_grids[best_name],
        n_iter=5,  # Optimized for faster execution
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(X_train_res, y_train_res)
    best_model = search.best_estimator_

    y_prob_tuned = best_model.predict_proba(X_val)[:, 1]
    print(f"Tuned ROC-AUC: {roc_auc_score(y_val, y_prob_tuned):.4f}")

# ─────────────────────────────────────────────
# 9. Final evaluation
# ─────────────────────────────────────────────
y_prob_final = best_model.predict_proba(X_val)[:, 1]
y_pred_final = (y_prob_final >= 0.30).astype(int)

print("\nFinal Evaluation")
print("ROC-AUC:", roc_auc_score(y_val, y_prob_final))
print("PR-AUC :", average_precision_score(y_val, y_prob_final))
print("F1     :", f1_score(y_val, y_pred_final))

print("\nConfusion Matrix:\n", confusion_matrix(y_val, y_pred_final))
print("\nReport:\n", classification_report(y_val, y_pred_final))

# ─────────────────────────────────────────────
# 10. Save model
# ─────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, "best_model.pkl")

with open(model_path, "wb") as f:
    pickle.dump({
        "model": best_model,
        "features": FEATURES,
        "caps": caps,
        "income_median": income_median,
        "best_name": best_name,
    }, f)

print("Model saved →", model_path)

# ─────────────────────────────────────────────
# 11. Submission
# ─────────────────────────────────────────────
probs = best_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "Id": df_test.index,
    "Probability": probs,
})

sub_path = os.path.join(BASE_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)

print("Submission saved →", sub_path)

# ─────────────────────────────────────────────
# 12. Metrics
# ─────────────────────────────────────────────
metrics = {
    name: {k: v for k, v in vals.items() if k != "model"}
    for name, vals in results.items()
}

metrics_path = os.path.join(BASE_DIR, "model_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print("Metrics saved →", metrics_path)

print("\nPipeline complete ✓")