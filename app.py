import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# ---------- Dataset banana (rule: attendance < 70 ya previous score < 35 => Fail) ----------
np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Study_Hours": np.round(np.random.uniform(0, 12, n), 1),
    "Attendance": np.round(np.random.uniform(30, 100, n), 1),
    "Previous_Score": np.round(np.random.uniform(0, 100, n), 1),
    "Sleep_Hours": np.round(np.random.uniform(3, 10, n), 1),
})

df["Result"] = np.where(
    (df["Attendance"] < 70) | (df["Previous_Score"] < 35), "Fail", "Pass"
)

# 5% rows mein thoda galat result (real life jaisa), taaki accuracy 100% na aaye
flip = np.random.rand(n) < 0.05
df.loc[flip, "Result"] = np.where(df.loc[flip, "Result"] == "Pass", "Fail", "Pass")

# ---------- Features aur target ----------
X = df[["Study_Hours", "Attendance", "Previous_Score", "Sleep_Hours"]]
y = df["Result"]

# ---------- Train model (80% train, 20% test) ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Real accuracy (test data par)
accuracy = model.score(X_test, y_test)
st.write("Model Accuracy:", round(accuracy * 100, 2), "%")

# ---------- App ----------
st.title("🎓 Student Performance Predictor")
st.write("Enter the student's details below to predict academic performance.")

study_hours = st.number_input("Study Hours", min_value=0.0, max_value=24.0, value=6.0)
attendance = st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, value=85.0)
previous_score = st.number_input("Previous Score", min_value=0.0, max_value=100.0, value=70.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)

# Predict button
if st.button("Predict Performance"):
    input_data = pd.DataFrame(
        [[study_hours, attendance, previous_score, sleep_hours]],
        columns=["Study_Hours", "Attendance", "Previous_Score", "Sleep_Hours"],
    )
    prediction = model.predict(input_data)[0]

    if prediction == "Pass":
        st.success("Predicted Result: Pass")
    else:
        st.error("Predicted Result: Fail")
