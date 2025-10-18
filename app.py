# app.py
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="World Cup Match Predictor", layout="centered")
st.title("⚽ World Cup Match Predictor")
st.write("Enter pre-match stats to predict if the home team will win.")

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    # Make sure 'model.pkl' is in the repo
    model = joblib.load("model.pkl")
    return model

model = load_model()

# -----------------------------
# User inputs
# -----------------------------
st.markdown("### Match info")
home_team = st.text_input("Home team", "Argentina")
away_team = st.text_input("Away team", "France")

st.markdown("### Team stats (averages per match)")
home_avg_goals = st.number_input("Home avg goals scored", value=1.8, format="%.2f")
home_avg_conceded = st.number_input("Home avg goals conceded", value=0.9, format="%.2f")
away_avg_goals = st.number_input("Away avg goals scored", value=1.5, format="%.2f")
away_avg_conceded = st.number_input("Away avg goals conceded", value=1.2, format="%.2f")

# -----------------------------
# Build DataFrame with exact features
# -----------------------------
X_new = pd.DataFrame([{
    'home_avg_goals': home_avg_goals,
    'home_avg_conceded': home_avg_conceded,
    'away_avg_goals': away_avg_goals,
    'away_avg_conceded': away_avg_conceded
}])

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):
    try:
        pred = model.predict(X_new)[0]
        proba = model.predict_proba(X_new)[0] if hasattr(model, "predict_proba") else None

        # Human-readable result
        result = "Home Win 🏅" if pred == 1 else "Home Not Win (Draw/Loss) ❌"
        st.subheader(result)

        if proba is not None:
            st.write("Prediction probabilities (Home Not Win / Home Win):")
            st.write(np.round(proba, 2))

        st.write(f"Match: {home_team} vs {away_team}")

    except Exception as e:
        st.error(f"Prediction failed — check feature names and model pipeline. Error: {e}")
