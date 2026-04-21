"""
Credit Score Default Prediction — Streamlit Dashboard (FIXED VERSION)
"""

import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# SHAP optional
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ─────────────────────────────
# PAGE CONFIG
# ─────────────────────────────
st.set_page_config(
    page_title="Credit Score card System",
    page_icon="💳",
    layout="wide",
)

# Hide Streamlit's default headers and footers to make it look professional
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            
            /* Add some premium styling to the button and metrics */
            .stButton>button {
                border-radius: 8px;
                font-weight: bold;
                transition: all 0.3s ease;
            }
            .stButton>button:hover {
                transform: scale(1.02);
            }
            div[data-testid="stMetricValue"] {
                font-size: 3rem;
                font-weight: 700;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ─────────────────────────────
# LOAD MODEL
# ─────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if not os.path.exists(MODEL_PATH):
    st.error("⚠️ Model not found. Run your credit_model.py first.")
    st.stop()

bundle = load_model()
model = bundle["model"]
features = bundle["features"]


# ─────────────────────────────
# HELPERS
# ─────────────────────────────
def prob_to_score(prob):
    """
    Converts default probability into a classic Credit Score (300-850 scale)
    A standard base FICO score structure aligns around 300 to 850.
    """
    # Clip prob to prevent log(0) or log(1)
    prob = max(min(prob, 0.999), 0.001)
    
    # Calculate odds of GOOD (not defaulting)
    odds = (1 - prob) / prob
    
    # Calibrated so that:
    # prob = 0.50 (odds = 1) -> Score = 600
    # prob = 0.20 (odds = 4) -> Score = 700
    factor = 72.13
    offset = 600.0
    
    score = offset + factor * np.log(odds)
    
    # Round and scale to FICO range tightly
    return int(max(min(round(score), 850), 300))

def lending_decision(score):
    if score >= 700:
        return "Approve", "Low Risk", "#2ecc71", "✅"
    elif score >= 600:
        return "Manual Review", "Medium Risk", "#f39c12", "⚠️"
    else:
        return "Reject", "High Risk", "#e74c3c", "🚨"


def gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Credit Score"},
        gauge={
            "axis": {"range": [300, 850]},
            "bar": {"color": "#7c83fd"},
            "steps": [
                {"range": [300, 600], "color": "#e74c3c"},
                {"range": [600, 700], "color": "#f39c12"},
                {"range": [700, 850], "color": "#2ecc71"},
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_pdf_report(score, decision, risk, prob, age, income):
    if not FPDF_AVAILABLE:
        return None
        
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, txt="Credit Risk Decision Report", ln=1, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Generated Score: {score} / 850", ln=1)
    pdf.cell(200, 10, txt=f"Final Decision:  {decision} ({risk} Risk)", ln=1)
    pdf.cell(200, 10, txt=f"Default Prob:    {prob:.2%}", ln=1)
    
    pdf.line(10, 50, 200, 50)
    pdf.cell(200, 10, txt="Applicant Summary", ln=1)
    pdf.cell(200, 8, txt=f"- Age: {age}", ln=1)
    pdf.cell(200, 8, txt=f"- Monthly Income: ${income:,.2f}", ln=1)
    
    pdf.line(10, 80, 200, 80)
    pdf.cell(200, 15, txt="Authorized By: _________________________", ln=1)
    
    pdf_out = pdf.output(dest='S')
    return pdf_out.encode('latin-1') if isinstance(pdf_out, str) else bytes(pdf_out)


def get_actionable_insights(utilization, late_30, score):
    insights = []
    if utilization > 0.3:
        insights.append(f"💡 **Reduce Utilization**: Lowering revolving utilization from {utilization:.0%} to below 30% could significantly boost the score.")
    if late_30 > 0:
        insights.append("💡 **Payment History**: Eliminating short-term delinquencies (30-59 days late) is the fastest path to score rehabilitation.")
    if score >= 750:
        insights.append("🌟 **Excellent Profile**: Maintain current credit habits. Profile outperforms 80% of regional benchmarks.")
    elif score >= 650:
        insights.append("📈 **Fair Profile**: Minor adjustments to unsecured lines can move this to a 'Low Risk' tier.")
    return insights


# ─────────────────────────────
# PREPROCESS (FIXED)
# ─────────────────────────────
def preprocess(df):
    df = df.copy()

    # numeric conversion
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.fillna(0)

    # ───────── FEATURE ENGINEERING ─────────

    if "MonthlyIncome" in df.columns and "NumberOfDependents" in df.columns:
        df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)
    else:
        df["IncomePerDependent"] = 0

    df["DebtToIncome"] = df.get("DebtRatio", 0)

    df["TotalLatePayments"] = (
        df.get("NumberOfTime30-59DaysPastDueNotWorse", 0) +
        df.get("NumberOfTime60-89DaysPastDueNotWorse", 0) +
        df.get("NumberOfTimes90DaysLate", 0)
    )

    # ───────── ENSURE ALL FEATURES EXIST ─────────
    for col in features:
        if col not in df.columns:
            df[col] = 0

    return df[features]


# ─────────────────────────────
# UI
# ─────────────────────────────
st.title("💳 Credit Score Card System")

tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])


# ─────────────────────────────
# TAB 1
# ─────────────────────────────
with tab1:
    st.subheader("Borrower Details")

    with st.form("form"):
        st.markdown("### 📋 Borrower Profile")
        
        col_profile, col_history, col_wealth = st.columns(3)
        
        with col_profile:
            st.markdown("#### Personal Specs")
            age = st.number_input("Age", 18, 100, 40, help="Borrower's age in years.")
            income = st.number_input("Monthly Income", 0, 1_000_000, 5000, help="Monthly gross income in dollars.")
            debt_ratio = st.number_input("Debt Ratio", 0.0, 10.0, 0.3, help="Monthly debt payments divided by monthly gross income.")
            utilization = st.slider("Revolving Utilization", 0.0, 1.0, 0.3, help="Total balance on credit cards divided by the sum of credit limits.")

        with col_history:
            st.markdown("#### Delinquency History")
            late_30 = st.number_input("30–59 Days Late", 0, 50, 0, help="Number of times borrower was 30-59 days past due.")
            late_60 = st.number_input("60–89 Days Late", 0, 50, 0, help="Number of times borrower was 60-89 days past due.")
            late_90 = st.number_input("90+ Days Late", 0, 50, 0, help="Number of times borrower was 90+ days past due.")

        with col_wealth:
            st.markdown("#### Assets & Family")
            dependents = st.number_input("Dependents", 0, 20, 0, help="Number of dependents in family.")
            open_loans = st.number_input("Open Credit Lines", 0, 50, 1, help="Number of open loans and lines of credit (e.g., auto loans, credit cards).")
            real_estate = st.number_input("Real Estate Loans", 0, 20, 0, help="Number of mortgage and real estate loans.")

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("Assess Credit Risk 🚀")

    if submit:
        input_df = pd.DataFrame([{
            "age": age,
            "MonthlyIncome": income,
            "DebtRatio": debt_ratio,
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfTimes90DaysLate": late_90,
            "NumberOfDependents": dependents,
            "NumberOfOpenCreditLinesAndLoans": open_loans,
            "NumberRealEstateLoansOrLines": real_estate,
        }])

        X = preprocess(input_df)
        prob = model.predict_proba(X)[:, 1][0]
        score = prob_to_score(prob)
        decision, risk, color, icon = lending_decision(score)
        
        # Celebrate excellent outcomes
        if score >= 750:
            st.toast("Analysis Complete! Exceptional applicant.", icon="🎉")
            st.balloons()
        else:
            st.toast("Analysis Complete!", icon="✅")

        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown(f"## {icon} {decision} ({risk})")
            st.metric("Credit Score", f"{score}", delta=f"P(Default): {prob:.2%}", delta_color="inverse")
            st.plotly_chart(gauge(score), use_container_width=True)
            
            # Application Actionable Insights
            st.markdown("### Actionable Adjustments")
            insights = get_actionable_insights(utilization, late_30, score)
            for insight in insights:
                st.info(insight)
            
            # PDF Report Export
            if FPDF_AVAILABLE:
                pdf_data = create_pdf_report(score, decision, risk, prob, age, income)
                if pdf_data:
                    st.download_button(
                        label="📥 Download Official Report (PDF)",
                        data=pdf_data,
                        file_name="Credit_Decision_Report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
            
        with col2:
            st.markdown("### Feature Impacts (SHAP)")
            if SHAP_AVAILABLE:
                with st.spinner("Generating explanation..."):
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X)
                    
                    fig, ax = plt.subplots(figsize=(8, 4))
                    
                    # Handle SHAP dimensionality for models that output multiple classes (e.g., shape (N, features, classes))
                    if len(shap_values.shape) == 3:
                        explanation = shap_values[0, :, 1]  # Select positive class
                    else:
                        explanation = shap_values[0]
                        
                    # Pass the matplotlib subplot to shap
                    shap.plots.waterfall(explanation, show=False)
                    plt.tight_layout()
                    st.pyplot(fig)
            else:
                st.warning("SHAP is not installed. Install with `pip install shap` to see explanations.")


# ─────────────────────────────
# TAB 2
# ─────────────────────────────
with tab2:
    st.subheader("Upload CSV for Batch Prediction")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)

        X = preprocess(df)
        probs = model.predict_proba(X)[:, 1]
        
        scores = [prob_to_score(p) for p in probs]
        df["DefaultProbability"] = probs
        df["CreditScore"] = scores
        df["RiskTier"] = ["Low Risk" if s >= 700 else "Medium Risk" if s >= 600 else "High Risk" for s in scores]
        df["Decision"] = ["Approve" if s >= 700 else "Manual Review" if s >= 600 else "Reject" for s in scores]

        st.write(df.head())

        fig = px.histogram(df, x="CreditScore", nbins=30, title="Credit Score Distribution", color="Decision")
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            "predictions.csv",
            "text/csv"
        )
    else:
        st.info("Upload a CSV file to begin prediction.")


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888888; padding: 10px;'>
        <p><b>Advanced Credit Default Prediction System</b></p>
        <p><i>Developed independently for professional risk decisioning</i></p>
    </div>
    """,
    unsafe_allow_html=True
)