# churn_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import joblib

# Load dataset
df = pd.read_csv("churn_dataa.csv")

# Drop unneeded columns
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Drop rows where target 'churn' is missing
df = df.dropna(subset=['churn'])

# Encode categorical features
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Split into features and target
X = df.drop('churn', axis=1)
y = df['churn']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\n🔍 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n📊 Classification Report for {name}:\n")
    print(classification_report(y_test, y_pred))

# Save the best model
best_model = GradientBoostingClassifier()
best_model.fit(X_train, y_train)
joblib.dump(best_model, "churn_model.pkl")
print("\n✅ Best model (Gradient Boosting) saved to churn_model.pkl")

