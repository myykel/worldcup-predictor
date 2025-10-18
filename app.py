# # app.py
# import streamlit as st
# import pandas as pd
# import joblib
# import numpy as np

# # -----------------------------
# # Page config
# # -----------------------------
# st.set_page_config(page_title="World Cup Match Predictor", layout="centered")
# st.title("⚽ World Cup Match Predictor")
# st.write("Enter pre-match stats to predict if the home team will win.")

# # -----------------------------
# # Load model
# # -----------------------------
# @st.cache_resource
# def load_model():
#     # Make sure 'model.pkl' is in the repo
#     model = joblib.load("model.pkl")
#     return model

# model = load_model()

# # -----------------------------
# # User inputs
# # -----------------------------
# st.markdown("### Match info")
# home_team = st.text_input("Home team", "Argentina")
# away_team = st.text_input("Away team", "France")

# st.markdown("### Team stats (averages per match)")
# home_avg_goals = st.number_input("Home avg goals scored", value=1.8, format="%.2f")
# home_avg_conceded = st.number_input("Home avg goals conceded", value=0.9, format="%.2f")
# away_avg_goals = st.number_input("Away avg goals scored", value=1.5, format="%.2f")
# away_avg_conceded = st.number_input("Away avg goals conceded", value=1.2, format="%.2f")

# # -----------------------------
# # Build DataFrame with exact features
# # -----------------------------
# X_new = pd.DataFrame([{
#     'home_avg_goals': home_avg_goals,
#     'home_avg_conceded': home_avg_conceded,
#     'away_avg_goals': away_avg_goals,
#     'away_avg_conceded': away_avg_conceded
# }])

# # -----------------------------
# # Prediction
# # -----------------------------
# if st.button("Predict"):
#     try:
#         pred = model.predict(X_new)[0]
#         proba = model.predict_proba(X_new)[0] if hasattr(model, "predict_proba") else None

#         # Human-readable result
#         result = "Home Win 🏅" if pred == 1 else "Home Not Win (Draw/Loss) ❌"
#         st.subheader(result)

#         if proba is not None:
#             st.write("Prediction probabilities (Home Not Win / Home Win):")
#             st.write(np.round(proba, 2))

#         st.write(f"Match: {home_team} vs {away_team}")

#     except Exception as e:
#         st.error(f"Prediction failed — check feature names and model pipeline. Error: {e}")




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
st.write("Select teams and get a predicted home win probability based on historical World Cup results.")

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")  # Ensure model.pkl is in repo
    return model

model = load_model()

# -----------------------------
# Load historical match data & compute team stats dynamically
# -----------------------------
@st.cache_data
def load_team_stats():
    df = pd.read_csv("WorldCupMatches.csv")  # Must contain columns: HomeTeam, AwayTeam, HomeGoals, AwayGoals
    
    # Home stats
    home_stats = df.groupby('HomeTeam').agg(
        avg_goals_scored=('HomeGoals', 'mean'),
        avg_goals_conceded=('AwayGoals', 'mean')
    ).reset_index().rename(columns={'HomeTeam': 'team'})
    
    # Away stats
    away_stats = df.groupby('AwayTeam').agg(
        avg_goals_scored=('AwayGoals', 'mean'),
        avg_goals_conceded=('HomeGoals', 'mean')
    ).reset_index().rename(columns={'AwayTeam': 'team'})
    
    # Merge home and away stats
    team_stats = pd.merge(home_stats, away_stats, on='team', how='outer', suffixes=('_home', '_away'))
    
    # Overall averages
    team_stats['avg_goals_scored'] = team_stats[['avg_goals_scored_home', 'avg_goals_scored_away']].mean(axis=1)
    team_stats['avg_goals_conceded'] = team_stats[['avg_goals_conceded_home', 'avg_goals_conceded_away']].mean(axis=1)
    
    return team_stats[['team', 'avg_goals_scored', 'avg_goals_conceded']]

team_stats = load_team_stats()
teams = team_stats['team'].tolist()

# -----------------------------
# User selects teams
# -----------------------------
st.markdown("### Match Selection")
home_team = st.selectbox("Home team", teams)
away_team = st.selectbox("Away team", [t for t in teams if t != home_team])

# -----------------------------
# Fetch stats automatically
# -----------------------------
home_avg_goals = float(team_stats[team_stats['team'] == home_team]['avg_goals_scored'])
home_avg_conceded = float(team_stats[team_stats['team'] == home_team]['avg_goals_conceded'])
away_avg_goals = float(team_stats[team_stats['team'] == away_team]['avg_goals_scored'])
away_avg_conceded = float(team_stats[team_stats['team'] == away_team]['avg_goals_conceded'])

# -----------------------------
# Build dataframe for prediction
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

        # Display human-readable result
        result = "Home Win 🏅" if pred == 1 else "Home Not Win (Draw/Loss) ❌"
        st.subheader(result)

        if proba is not None:
            st.write("Prediction probabilities (Home Not Win / Home Win):")
            st.write(np.round(proba, 2))

        st.write(f"Match: {home_team} vs {away_team}")

    except Exception as e:
        st.error(f"Prediction failed — check feature names and model pipeline. Error: {e}")
