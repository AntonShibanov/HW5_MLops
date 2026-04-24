import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import mlflow
import mlflow.sklearn

data = pd.read_csv("data/processed/processed.csv")
X = data.drop("class", axis=1)
y = data["class"]
test_size = 0.2
random_state = 42
max_iter = 200

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state
)

# настраиваем MLflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Iris Classification")

with mlflow.start_run():
    # логи
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # обучение
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)

    # оценка
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    # сохраненяем модель
    os.makedirs("model", exist_ok=True)
    model_path = "model/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # логирование модели в MLflow
    mlflow.sklearn.log_model(model, "logreg_model")
