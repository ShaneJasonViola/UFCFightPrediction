import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from PIL import Image

# =========================
# Load SVM model
# =========================
@st.cache_resource
def load_model():
    return joblib.load("svm_model.pkl")  # Change to your saved SVM model filename

model = load_model()

# =========================
# Title & Info
# =========================
st.set_page_config(page_title="UFC Fight Outcome Predictor", layout="wide")
st.title("UFC Fight Outcome Predictor")
st.markdown("Enter both fighters' statistics below. The app will predict the winner.")

st.markdown("## Resources to Look Up Information")
col1, col2 = st.columns(2)
with col1:
    st.markdown("[Upcoming UFC Events](https://www.ufc.com/events)")
with col2:
    st.markdown("[UFC Fighter Statistics](http://ufcstats.com/statistics/fighters)")

# =========================
# Prediction Form
# =========================
with st.form("fight_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Red Fighter Stats")
        red_age = st.number_input("Red Age", 18, 60, 28)
        red_wins = st.number_input("Red Wins", 0, 100, 10)
        red_losses = st.number_input("Red Losses", 0, 100, 5)
        red_td_landed = st.number_input("Red Avg TD Landed", 0.0, 20.0, 1.5)
        red_td_pct = st.number_input("Red Avg TD % (0-1)", 0.0, 1.0, 0.45)
        red_sig_str_pct = st.number_input("Red Sig Str % (0-1)", 0.0, 1.0, 0.45)
        red_sig_str_landed = st.number_input("Red Sig Strikes Landed", 0.0, 20.0, 4.5)
        red_reach_in = st.number_input("Red Reach (inches)", 48, 100, 71)
        red_height_in = st.number_input("Red Height (inches)", 48, 90, 70)
        red_win_streak = st.number_input("Red Current Win Streak", 0, 20, 2)
        red_lose_streak = st.number_input("Red Current Lose Streak", 0, 20, 0)
        red_sub_att = st.number_input("Red Sub Attempts", 0, 20, 1)

    with col2:
        st.subheader("Blue Fighter Stats")
        blue_age = st.number_input("Blue Age", 18, 60, 30)
        blue_wins = st.number_input("Blue Wins", 0, 100, 12)
        blue_losses = st.number_input("Blue Losses", 0, 100, 6)
        blue_td_landed = st.number_input("Blue Avg TD Landed", 0.0, 20.0, 1.2)
        blue_td_pct = st.number_input("Blue Avg TD % (0-1)", 0.0, 1.0, 0.43)
        blue_sig_str_pct = st.number_input("Blue Sig Str % (0-1)", 0.0, 1.0, 0.43)
        blue_sig_str_landed = st.number_input("Blue Sig Strikes Landed", 0.0, 20.0, 4.0)
        blue_reach_in = st.number_input("Blue Reach (inches)", 48, 100, 71)
        blue_height_in = st.number_input("Blue Height (inches)", 48, 90, 71)
        blue_win_streak = st.number_input("Blue Current Win Streak", 0, 20, 3)
        blue_lose_streak = st.number_input("Blue Current Lose Streak", 0, 20, 0)
        blue_sub_att = st.number_input("Blue Sub Attempts", 0, 20, 1)

    st.markdown("### Important Note About Odds")
    st.warning("Odds are set to equal by default (+100). Changing them can significantly influence predictions.")
    col1, col2 = st.columns(2)
    with col1:
        red_odds = st.number_input("Red Odds", value=100)
    with col2:
        blue_odds = st.number_input("Blue Odds", value=100)

    submitted = st.form_submit_button("Predict Winner")

# =========================
# Prediction Logic
# =========================
if submitted:
    # Derived features
    age_dif = blue_age - red_age
    reach_dif = red_reach_in * 2.54 - blue_reach_in * 2.54
    height_dif = red_height_in * 2.54 - blue_height_in * 2.54
    td_landed_dif = red_td_landed - blue_td_landed
    td_pct_dif = red_td_pct - blue_td_pct
    sig_str_pct_dif = red_sig_str_pct - blue_sig_str_pct
    sig_str_landed_dif = red_sig_str_landed - blue_sig_str_landed
    sub_att_dif = red_sub_att - blue_sub_att
    red_win_loss_ratio = red_wins / (red_wins + red_losses + 1)
    blue_win_loss_ratio = blue_wins / (blue_wins + blue_losses + 1)

    # Match model feature order
    input_vector = np.array([[
        red_win_loss_ratio, blue_win_loss_ratio,
        red_age, blue_age, age_dif,
        red_td_landed, blue_td_landed, td_landed_dif,
        red_td_pct, blue_td_pct, td_pct_dif,
        red_sig_str_pct, blue_sig_str_pct, sig_str_pct_dif,
        red_sig_str_landed, blue_sig_str_landed, sig_str_landed_dif,
        reach_dif, sub_att_dif, height_dif,
        red_win_streak, blue_win_streak,
        red_lose_streak, blue_lose_streak
    ]])

    pred = model.predict(input_vector)[0]
    confidence = model.predict_proba(input_vector)[0][pred] * 100
    winner = "Red Fighter" if pred == 1 else "Blue Fighter"

    st.subheader("Prediction Result")
    st.write(f"Predicted Winner: **{winner}**")
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")

# =========================
# Load dataset for analytics
# =========================
@st.cache_resource
def load_data():
    df = pd.read_csv("ufc-master.csv")
    if 'WinnerBinary' not in df.columns:
        if 'Winner' in df.columns:
            df['WinnerBinary'] = df['Winner'].apply(lambda x: 1 if str(x).strip().lower() == 'red' else 0)
        else:
            st.error("Dataset missing both 'WinnerBinary' and 'Winner' columns.")
            st.stop()
    return df

df = load_data()

# =========================
# Analytics
# =========================
st.header("UFC Analytics Dashboard")



# Feature Correlation with WinnerBinary
st.subheader("Feature Correlation with WinnerBinary")

# List of all relevant features including target
features_corr = [
    'RedWinLossRatio', 'BlueWinLossRatio', 'RedWins', 'BlueWins',
    'RedAge', 'BlueAge', 'AgeDif',
    'RedAvgTDLanded', 'BlueAvgTDLanded', 'RedAvgTDPct', 'BlueAvgTDPct', 'TDPctDiff',
    'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'SigStrPctDif',
    'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 'SigStrLandedDif', 
    'SubAttDif', 'WinnerBinary', 'HeightDif',
    'ReachDif', 'FightExperienceDiff',
    'RedCurrentWinStreak', 'BlueCurrentWinStreak',
    'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
]

# Only keep features that actually exist in dataset
features_corr = [f for f in features_corr if f in df.columns]

# Compute correlations
if 'WinnerBinary' in df.columns:
    corr_df = df[features_corr].corr()['WinnerBinary'].drop('WinnerBinary').sort_values(ascending=False)

    # Plot in Streamlit
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_df.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_title('Feature Correlation with WinnerBinary')
    ax.set_xlabel('Correlation Coefficient')
    ax.set_ylabel('Features')
    ax.grid(True)
    ax.invert_yaxis()  # Highest correlation at top
    st.pyplot(fig)

    # Also show as table
    st.dataframe(corr_df.to_frame(name='Correlation'))
else:
    st.error("WinnerBinary column is missing from dataset.")

