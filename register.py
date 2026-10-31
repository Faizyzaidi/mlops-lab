"""Lab 3 Task 2: register best model, create new version, move to Staging."""
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

exp = client.get_experiment_by_name("wine-lab3")
best = client.search_runs([exp.experiment_id],
                          order_by=["metrics.accuracy DESC", "attributes.start_time DESC"],
                          max_results=1)[0]
print("Best run:", best.info.run_id, best.data.params, best.data.metrics)

uri = f"runs:/{best.info.run_id}/model"
name = "WineClassifier"

v1 = mlflow.register_model(uri, name)     # version 1
v2 = mlflow.register_model(uri, name)     # version 2 (the "new version")
print("Registered versions:", v1.version, v2.version)

# Stage transition None -> Staging (deprecated in MLflow 2.9+ but still works)
client.transition_model_version_stage(name, v2.version, stage="Staging")
# Modern alternative: alias
client.set_registered_model_alias(name, "staging", v2.version)

mv = client.get_model_version(name, v2.version)
print(f"Version {mv.version} stage: {mv.current_stage}")
