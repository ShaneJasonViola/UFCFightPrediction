# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load Model
# -------------------------------
try:
    model = joblib.load("random_forest_model.pkl")  # Ensure this matches your GitHub model name
except Exception as e:
    st.error("🚫 Failed to load model.")
    st.text(traceback.format_exc())
    st.stop()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="🥊 UFC Fight Outcome Predictor", layout="wide")
st.title("🥊 UFC Fight Outcome Predictor")
st.markdown("Predict the outcome of a UFC fight based on fighter statistics.")

# -------------------------------
# Input Form
# -------------------------------
st.sidebar.header("👤 Enter Fighter Stats")

with st.sidebar.form("input_form"):
    red_age = st.number_input("Red Fighter Age", 18, 60, value=30)
    blue_age = st.number_input("Blue Fighter Age", 18, 60, value=30)

    red_odds = st.number_input("Red Fighter Odds", value=-150)
    blue_odds = st.number_input("Blue Fighter Odds", value=130)

    red_reach = st.slider("Red Reach (in)", 60, 85, value=72)
    blue_reach = st.slider("Blue Reach (in)", 60, 85, value=75)

    red_td = st.number_input("Red Avg TD Landed", min_value=0.0, value=1.5)
    blue_td = st.number_input("Blue Avg TD Landed", min_value=0.0, value=1.0)

    red_sig_str_pct = st.slider("Red Sig. Strike %", 0.0, 1.0, 0.45)
    blue_sig_str_pct = st.slider("Blue Sig. Strike %", 0.0, 1.0, 0.40)

    red_td_pct = st.slider("Red Takedown %", 0.0, 1.0, 0.35)
    blue_td_pct = st.slider("Blue Takedown %", 0.0, 1.0, 0.30)

    red_sub_att = st.number_input("Red Sub Attempts", min_value=0.0, value=0.5)
    blue_sub_att = st.number_input("Blue Sub Attempts", min_value=0.0, value=0.3)

    red_sig_str_landed = st.number_input("Red Sig. Strikes Landed", min_value=0.0, value=30.0)

    red_ko = st.number_input("Red Wins by KO", 0, 100, value=5)
    red_sub = st.number_input("Red Wins by Submission", 0, 100, value=3)
    red_wins = st.number_input("Red Total Wins", 1, 100, value=10)  # Prevent division by zero

    blue_total_fights = st.number_input("Blue Total Fights", 1, 100, value=15)
    blue_win_loss = st.slider("Blue Win/Loss Ratio", 0.0, 2.0, 1.0)
    red_win_loss = st.slider("Red Win/Loss Ratio", 0.0, 2.0, 1.2)

    submit = st.form_submit_button("Predict Outcome")

# -------------------------------
# Feature Engineering
# -------------------------------
def compute_features():
    return pd.DataFrame([{
        "RedWinLossRatio": red_win_loss,
        "BlueAge": blue_age,
        "RedAvgTDLanded": red_td,
        "RedAvgSigStrPct": red_sig_str_pct,
        "RedAvgTDPct": red_td_pct,
        "RedAvgSubAtt": red_sub_att,
        "SubAttDif": red_sub_att - blue_sub_att,
        "RedWinsBySubmission": red_sub,
        "ReachDif": red_reach - blue_reach,
        "RedAvgSigStrLanded": red_sig_str_landed,
        "SubPctDiff": (red_sub / red_wins) - 0,  # Assuming 0 blue submissions for now
        "AgeDif": red_age - blue_age,
        "BlueTotalFights": blue_total_fights,
        "RedWins": red_wins,
        "TDPctDiff": red_td_pct - blue_td_pct,
        "SigStrPctDif": red_sig_str_pct - blue_sig_str_pct,
        "KOPctDiff": (red_ko / red_wins) - 0,  # Assuming 0 blue KOs for now
        "BlueAvgSigStrPct": blue_sig_str_pct,
        "BlueAvgTDLanded": blue_td,
        "RedAge": red_age,
        "BlueOdds": blue_odds,
        "RedOdds": red_odds,
        "BlueWinLossRatio": blue_win_loss
    }])

# -------------------------------
# Prediction
# -------------------------------
if submit:
    try:
        X_input = compute_features()
        prediction = model.predict(X_input)[0]
        prob = model.predict_proba(X_input)[0]

        result = "🏆 **Red Fighter Wins**" if prediction == 1 else "🔵 **Blue Fighter Wins**"
        st.subheader("Prediction Result:")
        st.markdown(result)
        st.markdown(f"Confidence: **Red** {prob[1]*100:.1f}% | **Blue** {prob[0]*100:.1f}%")
    except Exception as e:
        st.error("Error during prediction:")
        st.text(traceback.format_exc())

# -------------------------------
# Dashboard - Optional Chart
# -------------------------------
st.markdown("---")
st.subheader("📈 UFC Win Method Distribution (Sample Chart)")
try:
    data = pd.DataFrame({
        "Method": ["KO", "Submission", "Decision"],
        "Percentage": [42, 26, 32]
    })
    fig, ax = plt.subplots()
    sns.barplot(data=data, x="Method", y="Percentage", ax=ax)
    ax.set_title("Win Method Distribution")
    st.pyplot(fig)
except Exception as e:
    st.error("Could not generate visualization:")
    st.text(traceback.format_exc())
