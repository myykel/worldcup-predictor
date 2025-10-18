# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="World Cup Match Predictor", layout="centered")

st.title("⚽ World Cup Match Predictor")
st.write("Enter pre-match stats (team and opponent) and get a predicted outcome.")

# Load model & scaler
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    try:
        scaler = joblib.load("scaler.pkl")
    except:
        scaler = None
    return model, scaler

model, scaler = load_model()

st.markdown("### Match info")
home_team = st.text_input("Home team (label only)", "Argentina")
away_team = st.text_input("Away team (label only)", "France")

st.markdown("### Team performance (averages)")
col1, col2 = st.columns(2)
with col1:
    home_avg_goals = st.number_input("Home avg goals scored", value=1.8, format="%.2f")
    home_avg_conceded = st.number_input("Home avg goals conceded", value=0.9, format="%.2f")
with col2:
    away_avg_goals = st.number_input("Away avg goals scored", value=1.7, format="%.2f")
    away_avg_conceded = st.number_input("Away avg goals conceded", value=1.2, format="%.2f")

home_advantage = st.selectbox("Venue", ["Home", "Away", "Neutral"])
if home_advantage == "Home":
    home_adv = 1.0
elif home_advantage == "Away":
    home_adv = 0.0
else:
    home_adv = 0.5

# Build dataframe with the exact columns your model expects
# IMPORTANT: this must match the pipeline/features used during training
X_new = pd.DataFrame([{
    'home_avg_goals': home_avg_goals,
    'home_avg_conceded': home_avg_conceded,
    'away_avg_goals': away_avg_goals,
    'away_avg_goals_conceded': away_avg_conceded,  # if your training used a different name adapt here
    'home_advantage': home_adv   # include only if model expects it
}])

# If your training pipeline used different column names, rename appropriately before scaling/predict
# Example assumes same columns are used; adapt as necessary.

if st.button("Predict"):
    # align/order columns if needed
    try:
        # scale if scaler exists
        if scaler is not None:
            X_scaled = scaler.transform(X_new)
        else:
            X_scaled = X_new.values

        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0] if hasattr(model, "predict_proba") else None

        # Map result to human-readable
        if pred == 1:
            result = "Home Win 🏅"
        else:
            result = "Home Not Win (Draw/Loss) ❌"

        st.subheader(result)
        if proba is not None:
            st.write("Model confidence / probabilities:")
            st.write(proba)
    except Exception as e:
        st.error(f"Prediction failed — check app column names and model pipeline. Error: {e}")
