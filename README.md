# Credit Card Fraud Detection

[English](README.md) | [Русский](README.ru.md)

Binary classification of fraudulent credit card transactions under severe class imbalance (~0.17% fraud).
 
## Problem
 
Given an anonymized transaction record, predict whether it is fraudulent. The dataset is extremely imbalanced (492 frauds out of 284,807 transactions), so the usual accuracy/ROC-AUC intuition doesn't apply. The project is built around choosing the right metric and the right decision threshold for that imbalance, not just fitting a model.
 
## Data
 
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) (ULB Machine Learning Group), downloaded via `kagglehub`. 284,807 transactions made by European cardholders over two days in September 2013. Features `V1`–`V28` are PCA components (original features are not disclosed for confidentiality), plus `Time` (seconds since the first transaction) and `Amount`. Target: `Class` (1 = fraud).
 
## Approach
 
1. **EDA** (`notebooks/01_eda.ipynb`) — stratified train/val/test split done *before* any preprocessing, to avoid leakage. Tested several outlier-filtering strategies (hard per-feature thresholds, IQR) and rejected all of them: fraud transactions sit in the extreme tail of the PCA features, so filtering "outliers" mostly filters out fraud and hurts validation PR-AUC (0.71 → 0.55 → 0.16 depending on the strategy). Dropped `Time` (a raw counter over ~48 hours with no generalizable signal). Removed duplicate rows from the training set only.
2. **Modeling** (`notebooks/02_modeling.ipynb`) — compared Logistic Regression, Random Forest, CatBoost and XGBoost via grid search, tracking the train/val PR-AUC gap at every step to catch overfitting early. Picked XGBoost (best validation PR-AUC). Tuned the decision threshold on the validation set to satisfy a precision ≥ 0.7 business constraint, maximizing recall subject to that.
3. **Evaluation** (`notebooks/03_evaluation.ipynb`) — single, final check on the held-out test set (never touched before this point).
## Results (test set)
 
| Metric    | Value  |
|-----------|--------|
| PR-AUC    | 0.873  |
| ROC-AUC   | 0.965  |
| Recall    | 0.867  |
| Precision | 0.669  |
 
Confusion matrix (rows = actual, columns = predicted):
 
```
[[56822    42]
 [   13    85]]
```
 
Precision landed slightly under the 0.7 target used to pick the threshold, but with only 127 positive predictions the 95% confidence interval is roughly [0.59, 0.75], which contains 0.7, the gap is within sampling noise, not a sign the threshold was mis-set. PR-AUC and recall on test are actually *higher* than on validation (0.873 vs 0.843, 0.867 vs 0.828), which is evidence against overfitting. Results are in line with, and slightly ahead on PR-AUC of  [this paper](https://www.iieta.org/download/file/fid/172943) using the same dataset.
 
**Why PR-AUC instead of ROC-AUC:** under this level of imbalance, ROC-AUC stayed nearly flat (0.96–0.97) across configurations where PR-AUC ranged from 0.16 to 0.87. ROC-AUC would have made every one of the experiments above look equally good, which is why PR-AUC (and recall at a fixed precision) drove every modeling decision instead.
 
## Repository structure
 
```
CreditCardFraudDetection/
├── data/           # train/val/test CSVs (gitignored — regenerate via 01_eda.ipynb)
├── models/         # fraud_pipeline.joblib + metadata.json (threshold, hyperparameters, val metrics)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_evaluation.ipynb
├── plots/          # interactive Plotly exports from the EDA notebook
├── src/
│   └── predict.py  # CLI to score new transactions with the trained model
├── requirements.txt
└── README.md
```
 
## How to run
 
### Reproduce the full pipeline
 
```bash
pip install -r requirements.txt
```
 
Run the notebooks in order: `01_eda.ipynb` → `02_modeling.ipynb` → `03_evaluation.ipynb`. The first downloads the dataset via `kagglehub` and writes the splits to `data/`; the second trains/tunes the models and saves the final pipeline + metadata to `models/`; the third runs the one-time test-set evaluation.
 
### Score new data with the trained model
 
```bash
pip install -r requirements.txt
python src/predict.py --input path/to/new_transactions.csv --output predictions.csv
```
 
The input CSV must contain the same feature columns the model was trained on (`V1`–`V28`, `Amount`); any extra columns are ignored.
 
## License / data source
 
Dataset provided by the [Machine Learning Group of ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) on Kaggle, released under the Open Database License (ODbL).
