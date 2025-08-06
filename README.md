# UFCFightPrediction
Predict the outcome of UFC fights


UFC Fight Outcome Predictor

This Streamlit web application predicts the winner of a UFC fight based on fighter statistics. It uses a pre-trained Random Forest model and allows users to input stats for two fighters, then view the predicted winner along with confidence and visual insights.


Features

- User inputs for Red and Blue fighter stats
- Computes derived features like win/loss ratios and reach difference
- Predicts winner with a confidence score using a trained Random Forest classifier
- Displays feature importance chart
- Includes confusion matrix from test set
- Win method distribution chart based on historical UFC fights
- Links to external UFC data sources

How to Run

1. Clone the repository
   git clone https://github.com/yourusername/ufc-fight-predictor.git
   cd ufc-fight-predictor

2. Install dependencies
   pip install -r requirements.txt

3. Add the following file to the project root:
   - random_forest_model (1).pkl

4. Run the app
   streamlit run app.py

Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

Model Features

- RedOdds, BlueOdds
- BlueAge, RedAge, AgeDif
- RedWinLossRatio, BlueWinLossRatio
- RedAvgTDLanded, BlueAvgTDLanded
- RedAvgSigStrPct, BlueAvgSigStrPct
- ReachDif

Notes

- All inputs are validated with minimum thresholds (e.g. reach cannot be below 48 inches)
- Reach is entered in inches and converted to centimeters internally
- Odds can be positive or negative and increase in whole number steps
- The application includes static evaluation metrics and win method distribution from 6,528 historical fights from 2010-2024

License

This project was created for academic and demonstration purposes.
Author: Shane Viola
