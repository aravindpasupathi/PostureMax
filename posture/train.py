import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib


def train_model(csv_path: str = "data/posture_dataset.csv", model_path: str = "models/posture_model_v3.pkl") -> None:
    df = pd.read_csv(csv_path)
    X = df.drop("label", axis=1)
    y = df["label"]

    model = DecisionTreeClassifier()
    model.fit(X, y)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"✅ Model saved as {model_path}")
