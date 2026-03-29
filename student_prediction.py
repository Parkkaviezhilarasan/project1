import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
data = pd.read_csv("student_data.csv")

# Features and target
X = data[["study_hours", "attendance", "previous_marks", "sleep_hours", "assignments_completed"]]
y = data["exam_score"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict using test data
y_pred = model.predict(X_test)

# Print results
print("Actual values:")
print(list(y_test))

print("\nPredicted values:")
print([round(i, 2) for i in y_pred])

print("\nMean Absolute Error:", round(mean_absolute_error(y_test, y_pred), 2))
print("R2 Score:", round(r2_score(y_test, y_pred), 2))

# New student prediction
new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [85],
    "previous_marks": [78],
    "sleep_hours": [7],
    "assignments_completed": [8]
})

predicted_score = model.predict(new_student)
print("\nPredicted Exam Score:", round(predicted_score[0], 2))