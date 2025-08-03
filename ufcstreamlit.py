# streamlit_app.py
import streamlit as st
import numpy as np
import joblib

# Load model
@st.cache_resource
def load_model():
    return joblib.load("random_forest_model (1).pkl")

model = load_model()

# App layout
st.title("UFC Fight Outcome Predictor")
st.markdown("Enter fighter stats below. The app will calculate derived features and predict the winner.")

# Fighter input form
with st.form("fight_form"):
    st.header("Fighter Stats")

    col1, col2 = st.columns(2)
    with col1:
        red_age = st.number_input("Red Fighter Age", min_value=18, max_value=60)
        red_wins = st.number_input("Red Fighter Wins", min_value=0)
        red_losses = st.number_input("Red Fighter Losses", min_value=0)
        red_td_landed = st.number_input("Red Avg TD Landed", min_value=0.0)
        red_sig_str_pct = st.number_input("Red Sig Str %", min_value=0.0, max_value=1.0)
        red_odds = st.number_input("Red Odds", value=100.0)

    with col2:
        blue_age = st.number_input("Blue Fighter Age", min_value=18, max_value=60)
        blue_wins = st.number_input


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
