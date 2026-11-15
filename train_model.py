import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/flood_risk_data.csv')

# 2. Define features (X) and target (y)
X = df[['rainfall_mm', 'river_discharge', 'elevation_m', 'soil_moisture_%']]
y = df['flood_risk']

# 3. Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42)
}

best_model = None
best_accuracy = 0
best_model_name = ""

print("\n--- Training & Evaluating Models ---")
for name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    
    # Predict on test set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Track the best performing model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# 5. Save the best model to the 'models/' folder
os.makedirs("models", exist_ok=True)
model_path = "models/best_flood_model.pkl"
joblib.dump(best_model, model_path)

print(f"\n========================================")
print(f"Best Model: {best_model_name} with {best_accuracy * 100:.2f}% accuracy!")
print(f"Saved best model successfully to {model_path}")
print(f"========================================")