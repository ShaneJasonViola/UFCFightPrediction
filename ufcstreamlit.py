import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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
        red_wins = st.number_input("
