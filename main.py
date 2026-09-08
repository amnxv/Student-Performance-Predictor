import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["Study_Hours", "Attendance", "Previous_Score", "Sleep_Hours"]]
y = data["Result"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Test model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Student Performance Predictor")
print("Model trained successfully!")
print("Model Accuracy:", accuracy)

#Predict performance for a new student
new_student = [[6,85,70,7]]
prediction = model.predict(new_student)
print("New Student Prediction:", prediction[0])

#Get input from user 

study_Hours= float(input("Enter STudy Hour:"))
Attendance= float(input("Enter attendance:"))
Previous_Score= float(input("Enter Previous score:"))
Sleep_Hours= float(input("Enter Sleep Hours:"))

new_student= [[study_Hours,Attendance,Previous_Score,Sleep_Hours]]

prediction = model.predict(new_student)

print("prediction:", prediction[0])