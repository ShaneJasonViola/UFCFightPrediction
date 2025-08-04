import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

with st.sidebar:
    st.markdown("## 📚 Resources")
    st.markdown("- 🗓️ [UFC Events](https://www.ufc.com/events)")
    st.markdown("- 🥋 [Fighter Stats](http://ufcstats.com/statistics/fighters)")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model (1).pkl")

model = load_model()

# Title
st.title("UFC Fight Outcome Predictor")
st.markdown("Enter both fighters' statistics below. The app will predict the winner.")

# Form for user input
with st.form("fight_input_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Red Fighter Stats")
        red_age = st.number_input("Red Age", min_value=18, max_value=60, value=28)
        red_wins = st.number_input("Red Wins", min_value=0, value=10)
        red_losses = st.number_input("Red Losses", min_value=0, value=5)
        red_td_landed = st.number_input("Red Avg TD Landed", min_value=0.0, value=1.5)
        red_sig_str_pct = st.number_input("Red Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.45)
        red_odds = st.number_input("Red Odds", value=-120.0)
        red_reach_in = st.number_input("Red Reach (inches)", min_value=30.0, max_value=100.0, value=71.0)
        red_reach = red_reach_in * 2.54 # Convert to CM
        
    with col2:
        st.subheader("Blue Fighter Stats")
        blue_age = st.number_input("Blue Age", min_value=18, max_value=60, value=30)
        blue_wins = st.number_input("Blue Wins", min_value=0, value=12)
        blue_losses = st.number_input("Blue Losses", min_value=0, value=6)
        blue_td_landed = st.number_input("Blue Avg TD Landed", min_value=0.0, value=1.2)
        blue_sig_str_pct = st.number_input("Blue Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.43)
        blue_odds = st.number_input("Blue Odds", value=110.0)
        blue_reach_in = st.number_input("Blue Reach (inches)", min_value=30.0, max_value=100.0, value=71.0)
        blue_reach = blue_reach_in * 2.54 # Convert to CM
        
    submitted = st.form_submit_button("Predict Winner")

# Prediction logic
if submitted:
    # Derived features
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

features = [
    'RedOdds', 'BlueOdds', 'BlueAge', 'RedAge', 'AgeDif',
    'RedWinLossRatio', 'BlueWinLossRatio',
    'RedAvgTDLanded', 'BlueAvgTDLanded',
    'RedAvgSigStrPct', 'BlueAvgSigStrPct',
    'ReachDif'
]

# Get importances and sort
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names_sorted = np.array(features)[indices]

# Plot in Streamlit
st.subheader(" Feature Importances")
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(len(importances)), importances[indices], align="center")
ax.set_xticks(range(len(importances)))
ax.set_xticklabels(feature_names_sorted, rotation=45, ha='right')
ax.set_title("Feature Importances - Random Forest")
ax.set_ylabel("Importance Score")
plt.tight_layout()
st.pyplot(fig)



# Predict and calculate confusion matrix
st.subheader("Confusion Matrix - Random Forest")

# Static example confusion matrix
static_cm = np.array([[602, 348],
                      [332,677]])

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(static_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix (Testing Data)')
plt.tight_layout()
st.pyplot(fig)



# Static values for 6528 total fights
ko_wins = 2205
sub_wins = 1313
total_fights = 6528
decision_wins = total_fights - ko_wins - sub_wins

# Create DataFrame
win_data = pd.DataFrame({
    'Method': ['KO', 'Submission', 'Decision'],
    'Wins': [ko_wins, sub_wins, decision_wins]
})
win_data['Percentage'] = (win_data['Wins'] / total_fights) * 100

# Plot in Streamlit
st.subheader("Win Method Distribution (% of All UFC Fights)")
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(win_data['Method'], win_data['Percentage'], color=['crimson', 'darkblue', 'gray'])

# Add text labels
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.1f}%', ha='center', fontsize=12)

ax.set_ylim(0, 100)
ax.set_ylabel('Percentage of Total Fights')
ax.set_title('Win Method Distribution (KO vs Submission vs Decision)')
st.pyplot(fig)

