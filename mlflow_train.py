"""Lab 3 Task 1: MLflow tracking with 3 hyperparameter sets."""
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

mlflow.set_tracking_uri("sqlite:///mlflow.db")   # DB backend (needed for the registry)
mlflow.set_experiment("wine-lab3")

configs = [{"max_depth": 1, "learning_rate": 0.01},
           {"max_depth": 3, "learning_rate": 0.1},
           {"max_depth": 5, "learning_rate": 0.5}]

for p in configs:
    with mlflow.start_run():
        m = GradientBoostingClassifier(**p, n_estimators=10, random_state=42).fit(Xtr, ytr)
        acc = accuracy_score(yte, m.predict(Xte))
        mlflow.log_params(p)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(m, artifact_path="model", serialization_format="cloudpickle")
        print(p, "-> accuracy", round(acc, 4))
