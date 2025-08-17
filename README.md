# Fraud Detection Project

## Overview

This project tackles the critical challenge of detecting fraudulent transactions in both e-commerce and banking contexts. Using two real-world datasets—e-commerce purchases (`Fraud_Data.csv`) and anonymized bank credit-card records (`creditcard.csv`)—we aim to build an end-to-end pipeline that:

- Cleans and merges raw data (including IP-to-country mapping).  
- Engineers features that capture temporal patterns, geolocation, and transaction velocity.  
- Addresses severe class imbalance through sampling and class weights.  
- Trains and compares Logistic Regression and ensemble models (Random Forest/Gradient Boosting).  
- Validates performance using F1-Score, AUC-PR, and confusion matrices.  
- Prepares for explainability (e.g., SHAP analysis) in downstream notebooks.

## Data Sources

- **data/raw/Fraud_Data.csv**: E-commerce transactions with user demographics, timestamps, device/browser info, IP addresses, and fraud labels (`class`).  
- **data/raw/IpAddress_to_Country.csv**: IP-range mapping to country codes.  
- **data/raw/creditcard.csv**: Bank transaction records with 28 PCA-derived features, transaction amount, and fraud labels (`Class`).

## Project Structure

```
FRAUD-DETECTION-PROJECT
├── data
│   ├── raw
│   └── processed
├── notebooks
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling_logreg.ipynb
│   ├── 04_modeling_ensemble.ipynb
│   └── 05_shap_analysis.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_train_eval.py
│   └── audit_data.py
├── utils
│   └── helpers.py
├── outputs
│   ├── figures
│   └── shap
├── models
│   └── saved_models
├── README.md
└── requirements.txt
```

## Work Done So Far

### 1. Data Ingestion & Initial EDA  
- Loaded all three raw CSV files, parsing date columns for `Fraud_Data`.  
- Checked shapes, data types, missing/null counts, and duplicate rows.  
- Plotted class imbalance:  
  - E-commerce fraud (≈ 9.4% positive).  
  - Credit-card fraud (≈ 0.17% positive).  
- Concatenated fraud and credit datasets to inspect overall skew (≈ 3.4% fraud).

### 2. Geolocation Merge  
- Converted `ip_address` to integer (`ip_int`).  
- Sorted and merged `Fraud_Data` with IP-to-country ranges via `pd.merge_asof`.  
- Visualized top 10 countries by transaction count.  
- Saved merged frame as `fraud_geo.csv`.

### 3. Feature Engineering Preview  
- In `01_eda.ipynb`, added:  
  - `hour_of_day`, `day_of_week`, `time_since_signup`.  
  - Per-user `purchase_count`, `time_since_prev_sec`.  
- Persisted feature-augmented dataset to `data/processed/`.

### 4. Modular Preprocessing Script  
- Created `src/data_preprocessing.py` with functions to:  
  - Load raw data.  
  - Clean and merge IP geolocation.  
  - Drop duplicates in credit data.  
  - Unify target column naming.  
  - Save processed CSVs (`fraud_clean.csv`, `credit_clean.csv`, `combined.csv`).

### 5. Feature Engineering Pipeline  
- In `src/feature_engineering.py`:  
  - `add_time_features()` and `add_freq_velocity()` functions.  
  - `get_preprocessor()` builds a `ColumnTransformer` with:  
    - `SimpleImputer` + `StandardScaler` for numerics.  
    - `SimpleImputer` + `OneHotEncoder` for categoricals.  
- Demonstrated applying the pipeline in `02_feature_engineering.ipynb`.

### 6. Modeling Baselines  
- In `src/model_train_eval.py`:  
  - `train_logreg()`: SMOTE oversampling + balanced Logistic Regression.  
  - `train_rf()`: SMOTE + Random Forest with class weights.  
- Notebooks `03_modeling_logreg.ipynb` and `04_modeling_ensemble.ipynb` run training, evaluate F1-Score, AUC-PR, and display confusion matrices.

### 7. Data Validation & Audit  
- Developed `utils/helpers.py` → `validate_data(df)` returns a dictionary of issues (missing counts, dtypes, duplicates, negative times, summaries).  
- Created `src/audit_data.py` (`audit_all_cleaned_data()`) to loop through `data/processed/*.csv` and collect issue reports.  
- Discovered:  
  - `combined.csv` has massive missing blocks in user and PCA columns (indicating a merge issue).  
  - `fraud_clean.csv` and `fraud_geo.csv` still miss some `country` values and first-purchase gaps in derived features.

## Next Steps

1. **Troubleshoot Raw-to-Clean Pipeline**  
   - Compare raw vs. processed row counts & columns.  
   - Inspect `pd.concat`/`pd.merge` logic for alignment errors.  

2. **Fix Data Processing Script**  
   - Adjust merge keys or concat axis.  
   - Ensure date parsing and column renaming correctly propagate.

3. **Re-run Audit & Validate**  
   - Confirm no missing values in critical fields before modeling.  

4. **Proceed to SHAP Analysis**  
   - Once cleaned, build explainability scripts in `05_shap_analysis.ipynb` and `src/shap_explain.py`.

