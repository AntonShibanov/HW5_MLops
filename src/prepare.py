import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Загружаем Iris датасет
df = pd.read_csv("data/raw/iris.csv", header=None,
                 names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])

X = df.drop("class", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

processed = pd.DataFrame(X_scaled, columns=X.columns)
processed["class"] = df["class"]
os.makedirs("data/processed", exist_ok=True)
processed.to_csv("data/processed/processed.csv", index=False)
