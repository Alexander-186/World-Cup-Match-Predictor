import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("fifa_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Load dataset containing international matches from 1872 - 2026. 
df = pd.read_csv("results.csv")

# Adding a title and feature for the tournaments. So instead of producing every single tournament for each international match,
my model can only focus on the FIFA World Cup/FIFA World Cup qualification matches. 

st.title("⚽ FIFA World CupMatch Predictor ⚽")

WC_tournaments = [
    "FIFA World Cup",
    "FIFA World Cup qualification"
]

df = df[df["tournament"].isin(WC_tournaments)]

# Making sure teams dont have null values and producing them as unique strings so that they are easily understandable. 

home_teams = df["home_team"].dropna().astype(str).unique()
away_teams = df["away_team"].dropna().astype(str).unique()

# Sorts the tournaments in alphabetical order, avoiding duplicates and storing the list of tournmanents into one of the
features. 

tournaments = sorted(df["tournament"].unique()) 

# Streamlit code so that website is interactive and users can directly enter what team and tournament they are selecting. 
# It also gives them the option of choosing whether the game is on equal ground. 

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

# When the "predict" button is pressed, a dataframe is accessed regarding the home team, away team, tournament, and neutral.
# get_dummies is used for each of the columns so that it operates under the LogisticRegression model. 

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

    # Making sure columns match the training data being used. Then proceeds to make a prediction and used probability for the match.
    
    new_match = new_match.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(new_match)[0]

    probabilities = model.predict_proba(new_match)[0]

    # If successful, the predicted outcome will be generated with a probability for each team's victory or draw. 

    st.success(f"Predicted Outcome: {prediction}")

    results = pd.DataFrame({
        "Outcome": model.classes_,
        "Probability": probabilities * 100
    })

    st.dataframe(results)

    # An extra feature with a bar chart for the DataFrame that gives a bit of visualization to the user. 

    chart_data = pd.DataFrame({
        "Probability": probabilities * 100
    }, index=model.classes_)

    st.bar_chart(chart_data)
