import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# -------------------------------
# Page & constants
# -------------------------------
st.set_page_config(page_title="UFC Fight Outcome Predictor (SVM)", layout="wide")
st.title("UFC Fight Outcome Predictor (SVM)")
st.markdown("Enter both fighters' statistics below. The app will predict the winner using your SVM model.")

# Resource links
st.markdown("## Resources to Look Up Information")
c1, c2 = st.columns(2)
with c1:
    st.markdown("[Upcoming UFC Events](https://www.ufc.com/events)")
with c2:
    st.markdown("[UFC Fighter Statistics](http://ufcstats.com/statistics/fighters)")

CSV_PATH = "ufc-master.csv"
MODEL_PATH = "svm_model.pkl"
PREPROC_PATH = "svm_preprocess.pkl"   # optional saved tuple: (imputer, scaler)

# The exact feature order used to train the SVM
FEATURE_ORDER = [
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

# -------------------------------
# Helpers
# -------------------------------
def ensure_winner_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure WinnerBinary exists (1 = Red, 0 = Blue)."""
    if 'WinnerBinary' not in df.columns:
        if 'Winner' in df.columns:
            df['WinnerBinary'] = df['Winner'].apply(
                lambda x: 1 if str(x).strip().lower() == 'red' else 0
            )
        else:
            st.error("Dataset is missing both 'WinnerBinary' and 'Winner' columns.")
            st.stop()
    return df

def compute_feature_columns_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all features the model expects, for a whole dataset."""
    for col in ['RedWins', 'RedLosses', 'BlueWins', 'BlueLosses']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if {'RedWins', 'RedLosses'}.issubset(df.columns):
        df['RedWinLossRatio'] = df['RedWins'] / (df['RedWins'] + df['RedLosses'] + 1)
    if {'BlueWins', 'BlueLosses'}.issubset(df.columns):
        df['BlueWinLossRatio'] = df['BlueWins'] / (df['BlueWins'] + df['BlueLosses'] + 1)

    if {'RedAge', 'BlueAge'}.issubset(df.columns):
        df['AgeDif'] = df['BlueAge'] - df['RedAge']
    if {'RedAvgTDLanded', 'BlueAvgTDLanded'}.issubset(df.columns):
        df['TDLandedDif'] = df['RedAvgTDLanded'] - df['BlueAvgTDLanded']
    if {'RedAvgTDPct', 'BlueAvgTDPct'}.issubset(df.columns):
        df['TDPctDiff'] = df['RedAvgTDPct'] - df['BlueAvgTDPct']
    if {'RedAvgSigStrPct', 'BlueAvgSigStrPct'}.issubset(df.columns):
        df['SigStrPctDif'] = df['RedAvgSigStrPct'] - df['BlueAvgSigStrPct']
    if {'RedAvgSigStrLanded', 'BlueAvgSigStrLanded'}.issubset(df.columns):
        df['SigStrLandedDif'] = df['RedAvgSigStrLanded'] - df['BlueAvgSigStrLanded']
    if {'RedReachCms', 'BlueReachCms'}.issubset(df.columns):
        df['ReachDif'] = df['RedReachCms'] - df['BlueReachCms']
    if {'RedAvgSubAtt', 'BlueAvgSubAtt'}.issubset(df.columns):
        df['SubAttDif'] = df['RedAvgSubAtt'] - df['BlueAvgSubAtt']
    if {'RedHeightCms', 'BlueHeightCms'}.issubset(df.columns):
        df['HeightDif'] = df['RedHeightCms'] - df['BlueHeightCms']

    for f in FEATURE_ORDER:
        if f not in df.columns:
            df[f] = np.nan
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_dataset():
    if not os.path.exists(CSV_PATH):
        st.error(f"Could not find {CSV_PATH}. Place the dataset next to this app.")
        st.stop()
    df = pd.read_csv(CSV_PATH)
    df = ensure_winner_binary(df)
    df = compute_feature_columns_from_df(df)
    return df

def load_or_fit_preprocessor(df_features: pd.DataFrame):
    """Load saved (imputer, scaler) or fit them from the dataset features."""
    if os.path.exists(PREPROC_PATH):
        try:
            imputer, scaler = joblib.load(PREPROC_PATH)
            return imputer, scaler
        except Exception:
            pass
    imputer = SimpleImputer(strategy='mean')
    scaler = StandardScaler()
    X = df_features[FEATURE_ORDER].values
    X_imp = imputer.fit_transform(X)
    scaler.fit(X_imp)
    try:
        joblib.dump((imputer, scaler), PREPROC_PATH)
    except Exception:
        pass
    return imputer, scaler

def build_input_row(vals: dict) -> np.ndarray:
    """Compute derived features from the form inputs and return a single row in FEATURE_ORDER."""
    red_reach_cm  = vals['red_reach_in']  * 2.54
    blue_reach_cm = vals['blue_reach_in'] * 2.54
    red_height_cm  = vals['red_height_in']  * 2.54
    blue_height_cm = vals['blue_height_in'] * 2.54

    red_wlr  = vals['red_wins']  / (vals['red_wins']  + vals['red_losses']  + 1)
    blue_wlr = vals['blue_wins'] / (vals['blue_wins'] + vals['blue_losses'] + 1)

    age_dif              = vals['blue_age'] - vals['red_age']
    td_landed_dif        = vals['red_td_landed'] - vals['blue_td_landed']
    td_pct_dif           = vals['red_td_pct'] - vals['blue_td_pct']
    sig_str_pct_dif      = vals['red_sig_str_pct'] - vals['blue_sig_str_pct']
    sig_str_landed_dif   = vals['red_sig_str_landed'] - vals['blue_sig_str_landed']
    reach_dif            = red_reach_cm - blue_reach_cm
    sub_att_dif          = vals['red_sub_att'] - vals['blue_sub_att']
    height_dif           = red_height_cm - blue_height_cm

    row = [
        red_wlr, blue_wlr,
        vals['red_age'], vals['blue_age'], age_dif,
        vals['red_td_landed'], vals['blue_td_landed'], td_landed_dif,
        vals['red_td_pct'], vals['blue_td_pct'], td_pct_dif,
        vals['red_sig_str_pct'], vals['blue_sig_str_pct'], sig_str_pct_dif,
        vals['red_sig_str_landed'], vals['blue_sig_str_landed'], sig_str_landed_dif,
        reach_dif, sub_att_dif, height_dif,
        vals['red_win_streak'], vals['blue_win_streak'],
        vals['red_lose_streak'], vals['blue_lose_streak'],
    ]
    return np.array([row], dtype=float)

# -------------------------------
# Load resources
# -------------------------------
model = load_model()
df_all = load_dataset()
imputer, scaler = load_or_fit_preprocessor(df_all)

# -------------------------------
# Prediction form (no odds; mutually exclusive streaks)
# -------------------------------
with st.form("fight_input_form"):
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Red Fighter")
        red_age   = st.number_input("Red Age",   min_value=18, max_value=60, value=28, step=1)
        red_wins  = st.number_input("Red Wins",  min_value=0,  max_value=100, value=10, step=1)
        red_losses= st.number_input("Red Losses",min_value=0,  max_value=100, value=5,  step=1)
        red_td_landed = st.number_input("Red Avg TD Landed", min_value=0.0, max_value=20.0, value=1.5)
        red_td_pct    = st.number_input("Red Avg TD % (0-1)", min_value=0.0, max_value=1.0, value=0.45)
        red_sig_str_pct     = st.number_input("Red Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.45)
        red_sig_str_landed  = st.number_input("Red Sig Strikes Landed", min_value=0.0, max_value=20.0, value=4.5)
        red_reach_in  = st.number_input("Red Reach (inches)",  min_value=48, max_value=100, value=71, step=1)
        red_height_in = st.number_input("Red Height (inches)", min_value=48, max_value=90,  value=70, step=1)
        # Mutually exclusive streak control
        red_streak_type = st.radio("Red Streak Type", ["None", "Win", "Lose"], index=0, horizontal=True)
        red_streak_len  = st.number_input("Red Streak Length", min_value=0, max_value=20, value=0, step=1)
        red_win_streak  = red_streak_len if red_streak_type == "Win"  else 0
        red_lose_streak = red_streak_len if red_streak_type == "Lose" else 0
        red_sub_att     = st.number_input("Red Sub Attempts", min_value=0, max_value=20, value=1, step=1)

    with c2:
        st.subheader("Blue Fighter")
        blue_age   = st.number_input("Blue Age",   min_value=18, max_value=60, value=30, step=1)
        blue_wins  = st.number_input("Blue Wins",  min_value=0,  max_value=100, value=12, step=1)
        blue_losses= st.number_input("Blue Losses",min_value=0,  max_value=100, value=6,  step=1)
        blue_td_landed = st.number_input("Blue Avg TD Landed", min_value=0.0, max_value=20.0, value=1.2)
        blue_td_pct    = st.number_input("Blue Avg TD % (0-1)", min_value=0.0, max_value=1.0, value=0.43)
        blue_sig_str_pct     = st.number_input("Blue Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.43)
        blue_sig_str_landed  = st.number_input("Blue Sig Strikes Landed", min_value=0.0, max_value=20.0, value=4.0)
        blue_reach_in  = st.number_input("Blue Reach (inches)",  min_value=48, max_value=100, value=71, step=1)
        blue_height_in = st.number_input("Blue Height (inches)", min_value=48, max_value=90,  value=71, step=1)
        # Mutually exclusive streak control
        blue_streak_type = st.radio("Blue Streak Type", ["None", "Win", "Lose"], index=0, horizontal=True)
        blue_streak_len  = st.number_input("Blue Streak Length", min_value=0, max_value=20, value=0, step=1)
        blue_win_streak  = blue_streak_len if blue_streak_type == "Win"  else 0
        blue_lose_streak = blue_streak_len if blue_streak_type == "Lose" else 0
        blue_sub_att     = st.number_input("Blue Sub Attempts", min_value=0, max_value=20, value=1, step=1)

    submitted = st.form_submit_button("Predict Winner")

# -------------------------------
# Prediction
# -------------------------------
if submitted:
    vals = {
        'red_age': red_age, 'blue_age': blue_age,
        'red_wins': red_wins, 'red_losses': red_losses,
        'blue_wins': blue_wins, 'blue_losses': blue_losses,
        'red_td_landed': red_td_landed, 'blue_td_landed': blue_td_landed,
        'red_td_pct': red_td_pct, 'blue_td_pct': blue_td_pct,
        'red_sig_str_pct': red_sig_str_pct, 'blue_sig_str_pct': blue_sig_str_pct,
        'red_sig_str_landed': red_sig_str_landed, 'blue_sig_str_landed': blue_sig_str_landed,
        'red_reach_in': red_reach_in, 'blue_reach_in': blue_reach_in,
        'red_height_in': red_height_in, 'blue_height_in': blue_height_in,
        'red_win_streak': red_win_streak, 'blue_win_streak': blue_win_streak,
        'red_lose_streak': red_lose_streak, 'blue_lose_streak': blue_lose_streak,
        'red_sub_att': red_sub_att, 'blue_sub_att': blue_sub_att,
    }

    X_row = build_input_row(vals)
    X_row_imp = imputer.transform(X_row)
    X_row_scaled = scaler.transform(X_row_imp)

    pred = model.predict(X_row_scaled)[0]
    proba = model.predict_proba(X_row_scaled)[0][pred] * 100.0
    winner = "Red Fighter" if pred == 1 else "Blue Fighter"

    st.subheader("Prediction Result")
    st.write(f"Predicted Winner: {winner}")
    st.write(f"Prediction Confidence: {proba:.2f}%")

# -------------------------------
# Single analytics chart: Feature Correlation with WinnerBinary
# -------------------------------
st.header("Feature Correlation with WinnerBinary")

available_for_corr = ['WinnerBinary'] + [f for f in FEATURE_ORDER if f in df_all.columns]
df_corr = df_all[available_for_corr].copy()

corr_series = (
    df_corr.corr()['WinnerBinary']
    .drop('WinnerBinary')
    .sort_values(ascending=False)
)

fig, ax = plt.subplots(figsize=(10, 8))
corr_series.plot(kind='barh', color='skyblue', ax=ax)
ax.set_title('Feature Correlation with WinnerBinary')
ax.set_xlabel('Correlation Coefficient')
ax.set_ylabel('Features')
ax.grid(True, axis='x', linestyle='--', alpha=0.4)
ax.invert_yaxis()
st.pyplot(fig)


