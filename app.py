import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fifa_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load dataset containing international matches from 1872 - 2026. 
df = pd.read_csv("results.csv")

st.title("⚽ FIFA World CupMatch Predictor ⚽")

WC_tournaments = [
    "FIFA World Cup",
    "FIFA World Cup qualification"
]

df = df[df["tournament"].isin(WC_tournaments)]

home_teams = df["home_team"].dropna().astype(str).unique()
away_teams = df["away_team"].dropna().astype(str).unique()

tournaments = sorted(df["tournament"].unique()) 


home_team = st.selectbox(
    "Select Home Team",
    home_teams
)

away_team = st.selectbox(
    "Select Away Team",
    away_teams
)

tournament = st.selectbox(
    "Tournament",
    tournaments
)

neutral = st.checkbox("Neutral Venue")

if st.button("Predict"):

    new_match = pd.DataFrame({
        "home_team": [home_team],
        "away_team": [away_team],
        "tournament": [tournament],
        "neutral": [neutral]
    })

    new_match = pd.get_dummies(
        new_match,
        columns=["home_team", "away_team", "tournament"]
    )

    # Make sure columns match the training data
    new_match = new_match.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(new_match)[0]

    probabilities = model.predict_proba(new_match)[0]

    st.success(f"Predicted Outcome: {prediction}")

    results = pd.DataFrame({
        "Outcome": model.classes_,
        "Probability": probabilities * 100
    })

    st.dataframe(results)

    chart_data = pd.DataFrame({
        "Probability": probabilities * 100
    }, index=model.classes_)

    st.bar_chart(chart_data)
