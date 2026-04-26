# Credit Card Fraud Detection with Azure ML

## Team Members
- Jyotheswar Narayana Narravula
- Mohith Reddy Kovvuri
- Vaishnavi Madduri

## Wright State University 

---

## View Notebook
https://nbviewer.org/github/jyotheswarnarayana/credit-card-fraud-detection-azureml/blob/main/fraud_detection_analysis.ipynb

---

## Project Overview
This project builds a credit card fraud detection system using the Kaggle Credit Card Fraud Detection dataset with 284,807 real transactions. Multiple machine learning models were trained, compared, and evaluated using Azure Machine Learning.

---

## Dataset
- Source: Kaggle Credit Card Fraud Detection
- Total transactions: 284,807
- Fraudulent transactions: 492 (0.17%)
- Features: V1-V28 (PCA transformed), Amount, Time, Class
- Class 0 = Legitimate, Class 1 = Fraud

---

## Project Structure
- fraud_detection_analysis.ipynb - Main notebook
- 1_upload_data.py - Upload dataset to Azure
- 2_automl_train.py - Azure AutoML training
- 3_eda_and_results.py - EDA scripts
- config.yml.example - Azure config template
- requirements.txt - Python dependencies
- data/ - Dataset folder
- outputs/ - Generated charts

---
## Project Architecture

The architecture we designed with two parallel pipelines:
- Local pipeline: EDA → Local Model Training → Results → Best Model Selected
- Azure ML pipeline: Dataset Registration → AutoML Training → Best Model → MLflow Tracking
- Both pipelines merge into final Model Comparison and Analysis

---
## Models Trained

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| Logistic Regression | 0.9722 | 0.0609 | 0.9184 | 0.1141 |
| Random Forest | 0.9733 | 0.8100 | 0.8265 | 0.8182 |
| XGBoost | 0.9747 | 0.7810 | 0.8367 | 0.8079 |
| Gradient Boosting | 0.7418 | 0.7975 | 0.6429 | 0.7119 |

---

## Azure AutoML Results

| Algorithm | AUC Weighted |
|-----------|-------------|
| VotingEnsemble - Best | 0.96592 |
| ExtremeRandomTrees | 0.95848 |
| XGBoostClassifier | 0.95480 |
| RandomForest | 0.95114 |
| LightGBM | 0.51686 |

---

## Best Model
XGBoost achieved the highest AUC of 0.9747
- Caught 82 out of 98 fraud transactions
- Precision: 0.7810
- Recall: 0.8367
- F1 Score: 0.8079

---

## Azure ML Integration
- Workspace: fraud-detection-ws
- Resource group: fraud-detection-project
- Dataset registered: creditcard-fraud
- Experiments tracked using MLflow
- AutoML job: fraud-automl-final - Completed

---

## Key Findings
1. Dataset is highly imbalanced - only 0.17% fraud
2. XGBoost achieved best AUC of 0.9747
3. V14 is the most important fraud indicator feature
4. Fraud transactions average 122 dollars vs 88 dollars for legitimate
5. class_weight=balanced used to handle class imbalance
6. Our XGBoost outperformed Azure AutoML VotingEnsemble

---

## How to Run This Project

### Prerequisites

- Python 3.13+
- Azure account with ML workspace
- Kaggle account to download dataset

### Step 1 - Clone the repo

git clone https://github.com/jyotheswarnarayana/credit-card-fraud-detection-azureml.git

cd credit-card-fraud-detection-azureml

### Step 2 - Install dependencies

pip install -r requirements.txt

### Step 3 - Download dataset

- Go to Kaggle Credit Card Fraud Detection dataset
- Download creditcard.csv
- Place it in the data/ folder

### Step 4 - Configure Azure ML

cp config.yml.example config.yml

### Step 5 - Login to Azure

az login

### Step 6 - Upload dataset to Azure ML

python 1_upload_data.py

### Step 7 - Run the notebook

Open fraud_detection_analysis.ipynb and click Kernel then Restart and Run All

## Results Summary
XGBoost achieved the best AUC of 0.9747, successfully catching 82 out of 98 fraud transactions in the test set with precision of 0.7810 and recall of 0.8367. The high AUC score demonstrates the model's strong ability to distinguish between fraudulent and legitimate transactions.
