# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="World Cup Match Predictor", layout="centered")

st.title("⚽ World Cup Match Predictor")
st.write("Enter pre-match stats and get a predicted outcome.")

# --- Load model ---
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # make sure model.pkl is in the repo
    return model

model = load_model()

# --- User inputs ---
st.markdown("### Match info")
home_team = st.text_input("Home team", "Argentina")
away_team = st.text_input("Away team", "France")

st.markdown("### Team performance (averages)")
home_avg_goals = st.number_input("Home avg goals scored", value=1.8, format="%.2f")
home_avg_conceded = st.number_input("Home avg goals conceded", value=0.9, format="%.2f")
away_avg_conceded = st.number_input("Away avg goals conceded", value=1.2, format="%.2f")

# --- Optional home advantage (only used if your model trained with it) ---
home_advantage = st.selectbox("Venue", ["Home", "Away", "Neutral"])
if home_advantage == "Home":
    home_adv = 1
elif home_advantage == "Away":
    home_adv = 0
else:
    home_adv = 0  # neutral

# --- Build dataframe exactly like training ---
X_new = pd.DataFrame([{
    'home_avg_goals': home_avg_goals,
    'home_avg_conceded': home_avg_conceded,
    'away_avg_conceded': away_avg_conceded
    # do NOT include home_advantage unless model was trained with it
}])

# --- Make prediction ---
if st.button("Predict"):
    try:
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0] if hasattr(model, "predict_proba") else None

        # Map result to human-readable
        if pred == 1:
            result = "Home Win 🏅"
        else:
            result = "Home Not Win (Draw/Loss) ❌"

        st.subheader(result)

        if proba is not None:
            st.write("Prediction probabilities (Home Not Win / Home Win):")
            st.write(np.round(proba, 2))
    except Exception as e:
        st.error(f"Prediction failed — check feature names and model pipeline. Error: {e}")
