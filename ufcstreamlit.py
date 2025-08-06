import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model (1).pkl")

model = load_model()

# Title
st.title("UFC Fight Outcome Predictor")
st.markdown("Enter both fighters' statistics below. The app will predict the winner.")

# Info Resources
st.markdown("## Resources to Look Up Information")
col1, col2 = st.columns(2)

with col1:
    st.markdown("[Upcoming UFC Events](https://www.ufc.com/events)")
with col2:
    st.markdown("[UFC Fighter Statistics](http://ufcstats.com/statistics/fighters)")

# Input Form
with st.form("fight_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Red Fighter Stats")
        red_age = st.number_input("Red Age", min_value=18, max_value=60, value=28, step=1)
        red_wins = st.number_input("Red Wins", min_value=0, value=10, step=1)
        red_losses = st.number_input("Red Losses", min_value=0, value=5, step=1)
        red_td_landed = st.number_input("Red Avg TD Landed", min_value=0.0, value=1.5)
        red_sig_str_pct = st.number_input("Red Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.45)
        red_reach_in = st.number_input("Red Reach (inches)", min_value=48, max_value=100, value=71, step=1, format="%d")
        red_reach = red_reach_in * 2.54

    with col2:
        st.subheader("Blue Fighter Stats")
        blue_age = st.number_input("Blue Age", min_value=18, max_value=60, value=30, step=1)
        blue_wins = st.number_input("Blue Wins", min_value=0, value=12, step=1)
        blue_losses = st.number_input("Blue Losses", min_value=0, value=6, step=1)
        blue_td_landed = st.number_input("Blue Avg TD Landed", min_value=0.0, value=1.2)
        blue_sig_str_pct = st.number_input("Blue Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.43)
        blue_reach_in = st.number_input("Blue Reach (inches)", min_value=48, max_value=100, value=71, step=1, format="%d")
        blue_reach = blue_reach_in * 2.54

    # Odds section with message
    st.markdown("### ⚠️ Important Note About Odds")
    st.warning("Odds are set to equal by default (+100). Changing them can significantly influence the model's prediction, as odds are a heavily weighted feature.")

    col1, col2 = st.columns(2)
    with col1:
        red_odds = st.number_input("Red Odds (e.g., -120 for favorite)", value=100, step=1, format="%d")
    with col2:
        blue_odds = st.number_input("Blue Odds (e.g., +110 for underdog)", value=100, step=1, format="%d")

    submitted = st.form_submit_button("Predict Winner")

# Prediction
if submitted:
    age_dif = blue_age - red_age
    red_win_loss_ratio = red_wins / (red_wins + red_losses + 1)
    blue_win_loss_ratio = blue_wins / (blue_wins + blue_losses + 1)
    reach_dif = red_reach - blue_reach

    input_vector = np.array([[ 
        red_odds, blue_odds,
        blue_age, red_age, age_dif,
        red_win_loss_ratio, blue_win_loss_ratio,
        red_td_landed, blue_td_landed,
        red_sig_str_pct, blue_sig_str_pct,
        reach_dif
    ]])

    pred = model.predict(input_vector)[0]
    confidence = model.predict_proba(input_vector)[0][pred] * 100
    winner = "Red Fighter" if pred == 1 else "Blue Fighter"

    st.subheader("Prediction Result")
    st.write(f"Predicted Winner: **{winner}**")
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")

# Feature Importances
features = [
    'RedOdds', 'BlueOdds', 'BlueAge', 'RedAge', 'AgeDif',
    'RedWinLossRatio', 'BlueWinLossRatio',
    'RedAvgTDLanded', 'BlueAvgTDLanded',
    'RedAvgSigStrPct', 'BlueAvgSigStrPct',
    'ReachDif'
]

importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names_sorted = np.array(features)[indices]

st.subheader("Feature Importances")
fig


