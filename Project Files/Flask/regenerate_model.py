"""
Script to regenerate the model in the Flask environment
This ensures compatibility with the Flask app's Python environment
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from joblib import dump
import os

print("Loading dataset...")
df_flood = pd.read_excel("../Dataset/flood dataset.xlsx")

print("Preparing data...")
X = df_flood.drop('flood', axis=1)
y = df_flood['flood']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Scaling features...")
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print("Training XGBoost (Gradient Boosting) model...")
xgb_model = GradientBoostingClassifier()
xgb_model.fit(X_train, y_train)

print("Saving scaler...")
dump(sc, "transform.save")

print("Saving model...")
dump(xgb_model, "floods.save")

print("✓ Model and scaler regenerated successfully!")
print(f"✓ Model training accuracy: {xgb_model.score(X_train, y_train):.4f}")
print(f"✓ Model test accuracy: {xgb_model.score(X_test, y_test):.4f}")
