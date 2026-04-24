import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml
import os

# ── Connect to Azure ──────────────────────────────────────────────
with open("config.yml") as f:
    cfg = yaml.safe_load(f)

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id=cfg["subscription_id"],
    resource_group_name=cfg["resource_group"],
    workspace_name=cfg["workspace_name"],
)
print("Connected to workspace:", ml_client.workspace_name)

# ── Load local dataset ────────────────────────────────────────────
df = pd.read_csv("./data/creditcard.csv")
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Create output folder for charts
os.makedirs("outputs", exist_ok=True)

# ── 1. Basic dataset info ─────────────────────────────────────────
print("\n── Dataset Info ──")
print(df.describe())
print(f"\nMissing values: {df.isnull().sum().sum()}")
print(f"\nClass distribution:\n{df['Class'].value_counts()}")
print(f"\nFraud percentage: {df['Class'].mean()*100:.4f}%")

# ── 2. Class imbalance chart ──────────────────────────────────────
plt.figure(figsize=(8, 5))
class_counts = df['Class'].value_counts()
bars = plt.bar(['Legitimate (0)', 'Fraud (1)'],
               class_counts.values,
               color=['#2196F3', '#F44336'])
plt.title('Class Distribution - Fraud vs Legitimate Transactions')
plt.ylabel('Number of Transactions')
plt.xlabel('Transaction Type')
for bar, count in zip(bars, class_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1000,
             f'{count:,}\n({count/len(df)*100:.2f}%)',
             ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/1_class_distribution.png', dpi=150)
plt.show()
print("Saved: 1_class_distribution.png")

# ── 3. Transaction amount by class ───────────────────────────────
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Amount'].hist(bins=50, color='#2196F3', alpha=0.7)
plt.title('Legitimate Transaction Amounts')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
df[df['Class'] == 1]['Amount'].hist(bins=50, color='#F44336', alpha=0.7)
plt.title('Fraud Transaction Amounts')
plt.xlabel('Amount ($)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/2_amount_distribution.png', dpi=150)
plt.show()
print("Saved: 2_amount_distribution.png")

# ── 4. Correlation heatmap (top features) ────────────────────────
plt.figure(figsize=(14, 10))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5,
            cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/3_correlation_heatmap.png', dpi=150)
plt.show()
print("Saved: 3_correlation_heatmap.png")

# ── 5. Top features correlated with fraud ────────────────────────
plt.figure(figsize=(10, 6))
fraud_corr = df.corr()['Class'].drop('Class').sort_values()
colors = ['#F44336' if x < 0 else '#2196F3' for x in fraud_corr.values]
fraud_corr.plot(kind='barh', color=colors)
plt.title('Feature Correlation with Fraud (Class)')
plt.xlabel('Correlation Coefficient')
plt.axvline(x=0, color='black', linewidth=0.8)
plt.tight_layout()
plt.savefig('outputs/4_feature_correlation_with_fraud.png', dpi=150)
plt.show()
print("Saved: 4_feature_correlation_with_fraud.png")

# ── 6. Time vs fraud pattern ──────────────────────────────────────
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df[df['Class'] == 0]['Time'].hist(bins=50, color='#2196F3',
                                   alpha=0.7, label='Legitimate')
df[df['Class'] == 1]['Time'].hist(bins=50, color='#F44336',
                                   alpha=0.7, label='Fraud')
plt.title('Transaction Time Distribution')
plt.xlabel('Time (seconds)')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
fraud_by_time = df.groupby(pd.cut(df['Time'],
                bins=24))['Class'].mean() * 100
fraud_by_time.plot(kind='bar', color='#F44336', alpha=0.7)
plt.title('Fraud Rate by Time Period')
plt.xlabel('Time Period')
plt.ylabel('Fraud Rate (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/5_time_analysis.png', dpi=150)
plt.show()
print("Saved: 5_time_analysis.png")

# ── 7. Get AutoML results from Azure ─────────────────────────────
print("\n── Fetching AutoML Results from Azure ──")
try:
    jobs = list(ml_client.jobs.list(parent_job_name=None))
    fraud_jobs = [j for j in jobs if 'fraud' in j.display_name.lower()
                  or 'fraud' in str(j.experiment_name).lower()]

    if fraud_jobs:
        latest_job = fraud_jobs[0]
        print(f"Found job: {latest_job.name}")
        print(f"Status: {latest_job.status}")
        print(f"Studio URL: {latest_job.studio_url}")
    else:
        print("No fraud detection jobs found yet.")
        print("Run 2_automl_train.py first, then re-run this script.")
except Exception as e:
    print(f"Could not fetch jobs: {e}")

print("\n── EDA Complete ──")
print("All charts saved to outputs/ folder")

