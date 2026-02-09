import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

df = pd.read_csv("data/dataset.csv")

X = df.drop(columns=["green_time"])
y = df["green_time"]

model = LinearRegression()
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/green_time_model.pkl")

print("Model trained and saved")
