"""Streamlit credit decision app: scores one loan application and shows its PD, risk band and decision."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
bundle = joblib.load(ROOT / "models" / "credit_risk_model.joblib")
metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
BANDS = ["A (lowest)", "B", "C", "D", "E (highest)"]

st.set_page_config(page_title="Credit Risk Scoring", page_icon="📊")
st.title("Credit Risk Scoring")
st.caption(f"Model: {metrics['best_model']} · test ROC-AUC {metrics['model_comparison'][metrics['best_model']]['test_auc']}")

c1, c2 = st.columns(2)
age = c1.number_input("Age", 18, 80, 30)
sex = c2.selectbox("Sex", ["male", "female"])
job = c1.selectbox("Job skill level (0 = unskilled non-resident, 3 = highly skilled)", [0, 1, 2, 3], index=2)
housing = c2.selectbox("Housing", ["own", "rent", "free"])
saving = c1.selectbox("Saving account", ["little", "moderate", "quite rich", "rich", "unknown"])
checking = c2.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
amount = c1.number_input("Credit amount (DM)", 100, 20000, 3000)
duration = c2.number_input("Duration (months)", 4, 72, 24)
purpose = st.selectbox("Purpose", ["car", "radio/TV", "furniture/equipment", "business", "education",
                                   "repairs", "domestic appliances", "vacation/others"])

if st.button("Score application", type="primary"):
    row = pd.DataFrame([{"sex": sex, "housing": housing, "saving_accounts": saving, "checking_account": checking,
                         "purpose": purpose, "age": age, "job": job, "credit_amount": amount,
                         "duration": duration, "monthly_burden": amount / duration}])[bundle["features"]]
    pd_score = float(bundle["model"].predict_proba(row)[0, 1])
    band = BANDS[int(np.searchsorted(bundle["band_edges"], pd_score))]
    flagged = pd_score >= bundle["threshold"]

    st.metric("Probability of default", f"{pd_score:.1%}")
    st.write(f"**Risk band:** {band}")
    if flagged:
        st.error(f"Early-warning flag: PD above policy threshold ({bundle['threshold']:.1%}) — refer to credit review")
    else:
        st.success("Within risk appetite — eligible for straight-through approval")
