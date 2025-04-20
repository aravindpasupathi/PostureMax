import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib

df = pd.read_csv("posture_dataset.csv")
X = df.drop("label", axis=1)
y = df["label"]

model = DecisionTreeClassifier()
model.fit(X, y)

joblib.dump(model, "posture_model_v3.pkl")
print("✅ Model saved as posture_model_v3.pkl")