import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


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
        red_reach_in = st.number_input("Red Reach (inches)", min_value=48, max_value=100, value=71, step=1)
        red_reach = red_reach_in * 2.54

    with col2:
        st.subheader("Blue Fighter Stats")
        blue_age = st.number_input("Blue Age", min_value=18, max_value=60, value=30, step=1)
        blue_wins = st.number_input("Blue Wins", min_value=0, value=12, step=1)
        blue_losses = st.number_input("Blue Losses", min_value=0, value=6, step=1)
        blue_td_landed = st.number_input("Blue Avg TD Landed", min_value=0.0, value=1.2)
        blue_sig_str_pct = st.number_input("Blue Sig Str % (0-1)", min_value=0.0, max_value=1.0, value=0.43)
        blue_reach_in = st.number_input("Blue Reach (inches)", min_value=48, max_value=100, value=71, step=1)
        blue_reach = blue_reach_in * 2.54

    # Odds section with message
    st.markdown("### Important Note About Odds")
    st.warning("Odds are set to equal by default (+100). Changing them can significantly influence the model's prediction, as odds are a heavily weighted feature.")

    col1, col2 = st.columns(2)
    with col1:
        red_odds = st.number_input("Red Odds (e.g., -120 for favorite)", value=100, step=1)
    with col2:
        blue_odds = st.number_input("Blue Odds (e.g., +110 for underdog)", value=100, step=1)

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



# Load dataset from your repo directory
@st.cache_resource
def load_data():
    return pd.read_csv("ufc-master.csv")

df = load_data()

# Define selected features
features = [
    'RedWinLossRatio', 'BlueAge', 'RedAvgTDLanded', 'RedAvgSigStrPct', 'RedAvgTDPct',
    'RedAvgSubAtt', 'SubAttDif', 'RedWinsBySubmission', 'ReachDif',
    'RedAvgSigStrLanded', 'SubPctDiff', 'AgeDif', 'BlueTotalFights',
    'RedWins', 'TDPctDiff', 'SigStrPctDif', 'KOPctDiff',
    'BlueAvgSigStrPct', 'BlueAvgTDLanded', 'RedAge', 'BlueOdds', 'RedOdds', 'BlueWinLossRatio',
    'WinnerBinary'
]

# Filter to available features in the DataFrame
available_features = [f for f in features if f in df.columns]

st.subheader("Correlation Heatmap of Selected UFC Features")

if available_features:
    fig, ax = plt.subplots(figsize=(12, 9))
    corr_matrix = df[available_features].corr()
    sns.heatmap(corr_matrix, annot=True, fmt=".1f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Heatmap of Selected UFC Features")
    st.pyplot(fig)
else:
    st.warning("None of the specified features were found in the dataset.")

# Feature Importances 
try:
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
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], align="center")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels(feature_names_sorted, rotation=45, ha='right')
    ax.set_title("Feature Importances - Random Forest")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.error(f"Could not generate feature importances. Reason: {str(e)}")

# Model Performance Data

train_data = {
    'LOG-R':      [0.6660, 0.67, 0.67, 0.67],
    'GNB':        [0.6535, 0.65, 0.65, 0.65],
    'DT':         [0.6772, 0.68, 0.68, 0.68],
    'RF':         [0.8759, 0.88, 0.88, 0.88],
    'SVM':        [0.6680, 0.67, 0.67, 0.67],
    'KNN':        [0.6708, 0.67, 0.67, 0.67],
    'KMeans_k=5': [None,   None, None, None]  # Not applicable for train
}

test_data = {
    'LOG-R':      [0.6437, 0.64, 0.64, 0.64],
    'GNB':        [0.6253, 0.63, 0.62, 0.63],
    'DT':         [0.6269, 0.63, 0.62, 0.62],
    'RF':         [0.6524, 0.65, 0.65, 0.65],
    'SVM':        [0.6452, 0.65, 0.64, 0.64],
    'KNN':        [0.6314, 0.63, 0.63, 0.63],
    'KMeans_k=5': [0.5942, 0.60, 0.60, 0.59]
}

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']

# Streamlit UI

st.title("🔍 UFC Model Evaluation Dashboard")

# Toggle between Train and Test
data_type = st.radio("Select Data Type:", ["Test Set", "Train Set"])

# Select View Mode
view_mode = st.radio("Select View Mode:", ["All Metrics", "Single Metric"])

# Get corresponding data
if data_type == "Test Set":
    selected_data = test_data
else:
    selected_data = train_data

df = pd.DataFrame(selected_data, index=metrics)

# Plotting

st.subheader(f"Model Performance on {data_type}")

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['lightcoral', 'goldenrod', 'mediumseagreen', 'lightseagreen']

if view_mode == "All Metrics":
    df.T.plot(kind='bar', ax=ax, color=colors)
    plt.title(f"{data_type} - All Metrics")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.legend(title="Metric", loc='upper right')
else:
    selected_metric = st.selectbox("Select Metric", metrics)
    df.loc[selected_metric].plot(kind='bar', ax=ax, color='skyblue')
    plt.title(f"{data_type} - {selected_metric}")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)

plt.xlabel("Model")
st.pyplot(fig)

# Confusion Matrix (Static Example)
st.subheader("Confusion Matrix - Random Forest")
static_cm = np.array([[602, 348], [332, 677]])
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(static_cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Loss', 'Win'], yticklabels=['Loss', 'Win'], ax=ax)
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix (Testing Data)')
plt.tight_layout()
st.pyplot(fig)

# Win Method Distribution
ko_wins = 2205
sub_wins = 1313
total_fights = 6528
decision_wins = total_fights - ko_wins - sub_wins

win_data = pd.DataFrame({
    'Method': ['KO', 'Submission', 'Decision'],
    'Wins': [ko_wins, sub_wins, decision_wins]
})
win_data['Percentage'] = (win_data['Wins'] / total_fights) * 100

st.subheader("Win Method Distribution (% of All UFC Fights)")
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(win_data['Method'], win_data['Percentage'], color=['crimson', 'darkblue', 'gray'])
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, height + 1, f'{height:.1f}%', ha='center', fontsize=12)

ax.set_ylim(0, 100)
ax.set_ylabel('Percentage of Total Fights')
ax.set_title('Win Method Distribution (KO vs Submission vs Decision)')
st.pyplot(fig)

x = df[model_features]  # Replace model_features with your actual feature list
scaler = StandardScaler()
x_scaled = scaler.fit_transform(X)

# Elbow method
inertia = []
K = range(1, 15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the elbow chart
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances (Inertia)')
plt.title('Elbow Method For Optimal k')
plt.grid(True)
st.pyplot(plt)  # If you're in Streamlit
