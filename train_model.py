# Necessary libraries.

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

# Random seed so that I can properly reproduce results.

RANDOM_STATE = np.random.seed(42)

df = pd.read_csv("results.csv")

df.head(5)

# First five rows in our data set ranging from 1872 to 2026.

df.shape

df.dtypes

df.info()

df.isnull().sum()

df = df.dropna(subset = "home_team")
df = df.dropna(subset = "away_score")
df = df.dropna(subset = "home_score")
df = df.dropna(subset = "away_team")

target_column = "tournament"

print(df[target_column].value_counts())

df = df[df["tournament"].isin(["FIFA World Cup", "FIFA World Cup qualification"])]


def get_result(outcome):
    if outcome["home_score"] > outcome["away_score"]:
        return "Home Victory"
    elif outcome["home_score"] < outcome["away_score"]:
        return "Away Victory"
    else:
        return "Draw"

df["outcome"] = df.apply(get_result, axis=1)       

X = df[["home_team", "away_team", "tournament", "neutral"]]
y = df["outcome"]

X = pd.get_dummies(
    X,
    columns=["home_team", "away_team", "tournament"]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

print(X.head())

model = LogisticRegression()

model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy:.2%}")


y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test Accuracy: {test_accuracy:.2%}")

y_test_pred = model.predict(X_test)

print(classification_report(y_test, y_test_pred))

y_train_pred = model.predict(X_train)

print(classification_report(y_train, y_train_pred))

y_test_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_test_pred)

print(cm)

y_train_pred = model.predict(X_train)

cm = confusion_matrix(y_train, y_train_pred)

print(cm)

print("Before importing joblib")

import joblib

print("Joblib imported")

joblib.dump(model, "fifa_model.pkl")

print("Model file created")

joblib.dump(X_train.columns.tolist(), "model_columns.pkl")

print("Columns file created")
