from azure.ai.ml import MLClient, automl, Input
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential
import yaml

with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)
print("Connected to workspace:", ml_client.workspace_name)

# Create compute cluster if it doesn't exist
print("Setting up compute cluster...")
try:
    cpu_cluster = ml_client.compute.get("cpu-cluster")
    print(f"Cluster already exists: {cpu_cluster.name}")
except Exception:
    print("Creating new cluster...")
    cpu_cluster = AmlCompute(
        name="cpu-cluster",
        type="amlcompute",
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=1,
        idle_time_before_scale_down=120,
        tier="Dedicated",
    )
    cpu_cluster = ml_client.compute.begin_create_or_update(cpu_cluster).result()
    print(f"Cluster created: {cpu_cluster.name}")

# Point to registered dataset
training_data = Input(
    type=AssetTypes.URI_FILE,
    path="azureml:creditcard-fraud:1",
)

# Configure AutoML job
automl_job = automl.classification(
    compute="cpu-cluster",
    experiment_name="fraud-detection",
    training_data=training_data,
    target_column_name="Class",
    primary_metric="AUC_weighted",
    n_cross_validations=3,
    enable_model_explainability=True,
)

automl_job.set_limits(
    timeout_minutes=60,
    trial_timeout_minutes=15,
    max_trials=5,
    enable_early_termination=True,
)

automl_job.set_training(
    enable_vote_ensemble=True,
    enable_stack_ensemble=True,
)

# Submit job
returned_job = ml_client.jobs.create_or_update(automl_job)
print(f"Job submitted: {returned_job.name}")
print(f"Track it at: {returned_job.studio_url}")

ml_client.jobs.stream(returned_job.name)
print("Training complete!")
