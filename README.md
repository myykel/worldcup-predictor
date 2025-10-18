# ⚽ World Cup Match Predictor  

**Live App:** [https://myykel-worldcup-predictor-app-tbopl0.streamlit.app/](https://myykel-worldcup-predictor-app-tbopl0.streamlit.app/)  

---

## 🧠 Overview  
The **World Cup Match Predictor** is a Streamlit web app that predicts the outcome of football (soccer) matches based on average team performance statistics.  
It uses a **machine learning model (Random Forest Classifier)** trained on historical FIFA World Cup data to forecast whether the **home team** will win, lose, or draw.  

This project demonstrates how **data science and football analytics** can merge to make data-driven predictions.

---

## 🚀 Features  
- Predicts match outcomes based on team performance statistics  
- Clean and intuitive **Streamlit** interface  
- Input custom stats for any match scenario  
- Model trained using historical World Cup results  
- Deployed live on **Streamlit Cloud**

---

## 🏗️ Project Structure  

worldcup-predictor/
│
├── app.py # Streamlit app
├── model.pkl # Trained Random Forest model
├── worldcup_results.csv # Historical World Cup dataset
├── requirements.txt # Python dependencies
└── README.md # Project documentation

## ⚙️ Installation & Local Setup  

### 1️⃣ Clone the repository  
```bash
git clone https://github.com/myykel/worldcup-predictor.git
cd worldcup-predictor

2️⃣ Create a virtual environment

python -m venv .venv

3️⃣ Activate the environment
.venv\Scripts\activate

4️⃣ Install dependencies
pip install -r requirements.txt

5️⃣ Run the app locally
streamlit run app.py

Model Training (optional)

If you want to retrain the model using the dataset:

from sklearn.ensemble import RandomForestClassifier
import pandas as pd, joblib

df = pd.read_csv('WorldCupMatches.csv')

# Define features and target
X = df[['home_avg_goals', 'home_avg_conceded', 'away_avg_goals', 'away_avg_conceded']]
y = df['home_win']

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, 'model.pkl', compress=3)

Dataset

The dataset used contains historical FIFA World Cup match results (1930–2022).
You can download a similar dataset from Kaggle:
🔗 FIFA World Cup Results 1930–2018

Deployment

The app is deployed live on Streamlit Cloud here:
👉 https://myykel-worldcup-predictor-app-tbopl0.streamlit.app/

Requirements

Python 3.9+

streamlit

pandas

numpy

scikit-learn

joblib

These are automatically installed when you run:
pip install -r requirements.txt

Future Improvements

Fetch live data from football APIs (e.g., FIFA, ESPN)

Add dropdowns for team selection

Include charts for prediction probabilities

Implement more advanced models like Gradient Boosting or XGBoost

Streamlit App
https://myykel-worldcup-predictor-app-tbopl0.streamlit.app/
