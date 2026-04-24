from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
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

data_asset = Data(
    path="./data/creditcard.csv",
    type=AssetTypes.URI_FILE,
    name="creditcard-fraud",
    version="1",
    description="Kaggle credit card fraud dataset - 284K transactions",
)

registered = ml_client.data.create_or_update(data_asset)
print(f"Dataset registered: {registered.name}, version {registered.version}")