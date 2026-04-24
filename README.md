# Credit Card Fraud Detection with Azure ML

## Team Members
- Jyotheswar Narayana Narravula
- Mohith Reddy Kovvuri
- Vaishnavi Madduri

## Wright State University

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

## Models Trained

| Model | AUC | Precision | Recall | F1 |
|-------|-----|-----------|--------|-----|
| Logistic Regression | 0.9698 | 0.0581 | 0.9184 | 0.1094 |
| Random Forest | 0.9828 | 0.4300 | 0.8776 | 0.5772 |
| XGBoost | 0.9760 | 0.3455 | 0.8673 | 0.4942 |
| Gradient Boosting | 0.9691 | 0.2772 | 0.8571 | 0.4190 |

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
Random Forest with SMOTE preprocessing achieved the highest AUC of 0.9828
- Caught 86 out of 98 fraud transactions
- Precision: 0.4300
- Recall: 0.8776
- F1 Score: 0.5772

---

## Azure ML Integration
- Workspace: fraud-detection-ws
- Resource group: fraud-detection-project
- Dataset registered: creditcard-fraud and creditcard-fraud-table
- Experiments tracked using MLflow
- AutoML job: fraud-automl-final - Completed

---

## Key Findings
1. Dataset is highly imbalanced - only 0.17% fraud
2. SMOTE oversampling significantly improved model performance
3. V14 is the most important fraud indicator feature
4. Fraud transactions average 122 dollars vs 88 dollars for legitimate
5. Our Random Forest with SMOTE outperformed Azure AutoML VotingEnsemble
6. High recall of 0.8776 is most important - catching fraud matters more than false alarms

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
Fill in Azure credentials in config.yml
If you want our credintials to review we are ready to give

### Step 5 - Login to Azure
az login

### Step 6 - Upload dataset to Azure ML
python 1_upload_data.py

### Step 7 - Run the notebook
python3 -m jupyter notebook

Open fraud_detection_analysis.ipynb and click Kernel then Restart and Run All

---

## Results Summary
The Random Forest model with SMOTE preprocessing achieved the best performance with precision of 0.4300 and recall of 0.8776, successfully catching 86 out of 98 fraud transactions in the test set. The high recall score demonstrates the model's strong ability to detect fraudulent transactions which is the primary goal in fraud detection systems.
