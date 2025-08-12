import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix

# ----------------------------
# Load SVM model
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("svm_model.pkl")

model = load_model()

# ----------------------------
# Title & Info
# ----------------------------
st.title("UFC Fight Outcome Predictor (SVM Model)")
st.markdown("Enter both fighters' statistics below. The app will predict the winner based on the SVM model.")

# Info Links
st.markdown("## Resources to Look Up Information")
col1, col2 = st.columns(2)
with col1:
    st.markdown("[Upcoming UFC Events](https://www.ufc.com/events)")
with col2:
    st.markdown("[UFC Fighter Statistics](http://ufcstats.com/statistics/fighters)")

# ----------------------------
# Input Form
# ----------------------------
with st.form("fight_input_form"):
    col1, col2 = st.columns(2)

    # Red Fighter Stats
    with col1:
        st.subheader("Red Fighter Stats")
        red_age = st.number_input("Red Age", 18, 60, 28)
        red_wins = st.number_input("Red Wins", 0, 100, 10)
        red_losses = st.number_input("Red Losses", 0, 100, 5)
        red_avg_td_landed = st.number_input("Red Avg TD Landed", 0.0, 10.0, 1.5)
        red_avg_td_pct = st.number_input("Red Avg TD % (0-1)", 0.0, 1.0, 0.45)
        red_avg_sig_str_pct = st.number_input("Red Sig Str % (0-1)", 0.0, 1.0, 0.45)
        red_avg_sig_str_landed = st.number_input("Red Avg Sig Str Landed", 0.0, 20.0, 5.0)
        red_sub_att = st.number_input("Red Avg Submission Attempts", 0.0, 10.0, 0.5)
        red_reach_in = st.number_input("Red Reach (inches)", 48, 100, 71)
        red_height_in = st.number_input("Red Height (inches)", 48, 90, 70)
        red_win_streak = st.number_input("Red Current Win Streak", 0, 20, 2)
        red_lose_streak = st.number_input("Red Current Lose Streak", 0, 20, 0)

    # Blue Fighter Stats
    with col2:
        st.subheader("Blue Fighter Stats")
        blue_age = st.number_input("Blue Age", 18, 60, 30)
        blue_wins = st.number_input("Blue Wins", 0, 100, 12)
        blue_losses = st.number_input("Blue Losses", 0, 100, 6)
        blue_avg_td_landed = st.number_input("Blue Avg TD Landed", 0.0, 10.0, 1.2)
        blue_avg_td_pct = st.number_input("Blue Avg TD % (0-1)", 0.0, 1.0, 0.43)
        blue_avg_sig_str_pct = st.number_input("Blue Sig Str % (0-1)", 0.0, 1.0, 0.43)
        blue_avg_sig_str_landed = st.number_input("Blue Avg Sig Str Landed", 0.0, 20.0, 4.8)
        blue_sub_att = st.number_input("Blue Avg Submission Attempts", 0.0, 10.0, 0.4)
        blue_reach_in = st.number_input("Blue Reach (inches)", 48, 100, 71)
        blue_height_in = st.number_input("Blue Height (inches)", 48, 90, 72)
        blue_win_streak = st.number_input("Blue Current Win Streak", 0, 20, 3)
        blue_lose_streak = st.number_input("Blue Current Lose Streak", 0, 20, 1)

    submitted = st.form_submit_button("Predict Winner")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    # Derived features
    red_reach = red_reach_in * 2.54
    blue_reach = blue_reach_in * 2.54
    red_height = red_height_in * 2.54
    blue_height = blue_height_in * 2.54

    red_win_loss_ratio = red_wins / (red_wins + red_losses + 1)
    blue_win_loss_ratio = blue_wins / (blue_wins + blue_losses + 1)

    age_dif = blue_age - red_age
    td_landed_dif = red_avg_td_landed - blue_avg_td_landed
    td_pct_dif = red_avg_td_pct - blue_avg_td_pct
    sig_str_pct_dif = red_avg_sig_str_pct - blue_avg_sig_str_pct
    sig_str_landed_dif = red_avg_sig_str_landed - blue_avg_sig_str_landed
    reach_dif = red_reach - blue_reach
    sub_att_dif = red_sub_att - blue_sub_att
    height_dif = red_height - blue_height

    # Arrange in model feature order
    input_vector = np.array([[
        red_win_loss_ratio, blue_win_loss_ratio,
        red_age, blue_age, age_dif,
        red_avg_td_landed, blue_avg_td_landed, td_landed_dif,
        red_avg_td_pct, blue_avg_td_pct, td_pct_dif,
        red_avg_sig_str_pct, blue_avg_sig_str_pct, sig_str_pct_dif,
        red_avg_sig_str_landed, blue_avg_sig_str_landed, sig_str_landed_dif,
        reach_dif, sub_att_dif, height_dif,
        red_win_streak, blue_win_streak,
        red_lose_streak, blue_lose_streak
    ]])

    # Predict
    pred = model.predict(input_vector)[0]
    confidence = model.predict_proba(input_vector)[0][pred] * 100
    winner = "Red Fighter" if pred == 1 else "Blue Fighter"

    st.subheader("Prediction Result")
    st.write(f"Predicted Winner: **{winner}**")
    st.write(f"Prediction Confidence: **{confidence:.2f}%**")

# ----------------------------
# Model Performance Summary
# ----------------------------
st.subheader("SVM Model Performance (Example Scores)")
test_data = {
    'SVM': [0.5972, 0.60, 0.60, 0.60],  # Accuracy, Precision, Recall, F1
}
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
df_perf = pd.DataFrame(test_data, index=metrics)

fig, ax = plt.subplots(figsize=(6, 4))
df_perf.T.plot(kind='bar', ax=ax, color='skyblue')
plt.ylim(0, 1)
plt.title("SVM Test Performance")
st.pyplot(fig)

# ----------------------------
# Confusion Matrix Example
# ----------------------------
st.subheader("Confusion Matrix - SVM")
cm = np.array([[560, 390], [410, 599]])  # Replace with real CM if available
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
plt.tight_layout()
st.pyplot(fig)

# ==============================
# UFC Analytics Dashboard
# ==============================

# Create a tab layout: first tab for Prediction, second for Analytics
tab1, tab2 = st.tabs(["Prediction", "Analytics"])

with tab1:
    st.header("Predict UFC Fight Outcome")
    # Your existing prediction form logic goes here
    # ...

with tab2:
    st.header("UFC Analytics Dashboard")

    # Load UFC dataset
    @st.cache_resource
    def load_data():
        return pd.read_csv("ufc-master.csv")
    df = load_data()

    # Ensure required columns exist
    if 'WinnerBinary' not in df.columns:
        st.error("Dataset is missing 'WinnerBinary' column.")
    else:
        # Chart 1: Distribution of Fight Outcomes by Method
        st.subheader("Win Method Distribution")
        if 'win_by' in df.columns:
            win_method_counts = df['win_by'].value_counts().reset_index()
            win_method_counts.columns = ['Method', 'Count']
            fig1, ax1 = plt.subplots()
            ax1.bar(win_method_counts['Method'], win_method_counts['Count'], color=['crimson', 'darkblue', 'gray'])
            ax1.set_ylabel("Number of Wins")
            ax1.set_xlabel("Win Method")
            ax1.set_title("Distribution of Fight Outcomes")
            st.pyplot(fig1)
        else:
            st.warning("Column 'win_by' not found in dataset.")

        # Chart 2: Age Difference vs Win Probability
        st.subheader("Age Difference vs Win Probability")
        if 'BlueAge' in df.columns and 'RedAge' in df.columns:
            df['AgeDif'] = df['BlueAge'] - df['RedAge']
            age_win_prob = df.groupby(pd.cut(df['AgeDif'], bins=range(-15, 16, 2)))['WinnerBinary'].mean().reset_index()
            fig2, ax2 = plt.subplots()
            ax2.plot(age_win_prob['AgeDif'].astype(str), age_win_prob['WinnerBinary'], marker='o')
            ax2.set_ylabel("Win Probability (Red Fighter)")
            ax2.set_xlabel("Age Difference (Blue - Red)")
            ax2.set_title("Effect of Age Difference on Win Rate")
            plt.xticks(rotation=45)
            st.pyplot(fig2)
        else:
            st.warning("Age columns not found in dataset.")

        # Chart 3: Reach Difference Impact on Win Rate
        st.subheader("Reach Difference Impact on Win Rate")
        if 'RedReachCms' in df.columns and 'BlueReachCms' in df.columns:
            df['ReachDif'] = df['RedReachCms'] - df['BlueReachCms']
            reach_win_prob = df.groupby(pd.cut(df['ReachDif'], bins=range(-30, 31, 5)))['WinnerBinary'].mean().reset_index()
            fig3, ax3 = plt.subplots()
            ax3.bar(reach_win_prob['ReachDif'].astype(str), reach_win_prob['WinnerBinary'], color='orange')
            ax3.set_ylabel("Win Probability (Red Fighter)")
            ax3.set_xlabel("Reach Difference (cm)")
            ax3.set_title("Reach Advantage vs Win Rate")
            plt.xticks(rotation=45)
            st.pyplot(fig3)
        else:
            st.warning("Reach columns not found in dataset.")

        # Chart 4: Correlation Heatmap of Selected Features
        st.subheader("Correlation Heatmap of Selected Features")
        selected_features = [
            'RedWinLossRatio', 'BlueWinLossRatio',  
            'RedAge', 'BlueAge', 'AgeDif',
            'RedAvgTDLanded', 'BlueAvgTDLanded', 'TDLandedDif',
            'RedAvgTDPct', 'BlueAvgTDPct', 'TDPctDiff',
            'RedAvgSigStrPct', 'BlueAvgSigStrPct', 'SigStrPctDif',
            'RedAvgSigStrLanded', 'BlueAvgSigStrLanded', 'SigStrLandedDif', 
            'ReachDif', 'SubAttDif', 'HeightDif',   
            'RedCurrentWinStreak', 'BlueCurrentWinStreak',
            'RedCurrentLoseStreak', 'BlueCurrentLoseStreak',
        ]
        available_features = [f for f in selected_features if f in df.columns]
        if available_features:
            fig4, ax4 = plt.subplots(figsize=(12, 9))
            sns.heatmap(df[available_features].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax4)
            ax4.set_title("Correlation Heatmap")
            st.pyplot(fig4)
        else:
            st.warning("Selected features not found in dataset for correlation heatmap.")

