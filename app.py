import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["Study_Hours", "Attendance", "Previous_Score", "Sleep_Hours"]]
y = data["Result"]

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

accuracy = model.score(X, y)
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# App title
st.title("🎓 Student Performance Predictor")

st.write("Enter the student's details below to predict academic performance.")

# Input fields
study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=24.0,
    value=6.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=85.0
)

previous_score = st.number_input(
    "Previous Score",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

sleep_hours = st.number_input(
    "Sleep Hours",
    min_value=0.0,
    max_value=24.0,
    value=7.0
)

# Predict button
if st.button("Predict Performance"):
    if attendance < 75 or previous_score < 40:
        st.error("Predicted Result: Fail")
    else:
        st.success("Predicted Result: Pass")
